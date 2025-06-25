"""Tests for quantification_utils.py."""

import numpy as np
import pytest
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import Affine

from src.utils.quantification_utils import (
    calc_effective_wind_speed,
    calc_L_IME,
    calc_Q_IME,
    calc_u_eff,
    calc_wind_direction,
    calc_wind_error,
    get_plume_source_location_v2,
)

###############################################################################
# FIXTURES
###############################################################################


@pytest.fixture
def plume_data() -> np.ndarray:
    """2D test array of methane retrieval values in mol/m2."""
    return np.array([[0.0, 0.01, 0.10], [0.05, 0.03, 0.02]], dtype=float)


@pytest.fixture
def transform() -> Affine:
    """
    Affine transform for EPSG:32614 (WGS 84 / UTM zone 14N).

    With 20m x 20m pixels, origin at (x=620000, y=3413000).
    """
    return Affine(20.0, 0.0, 620000.0, 0.0, -20.0, 3413000.0)


@pytest.fixture
def crs_utm14n() -> CRS:
    """CRS EPSG:32614 (WGS 84 / UTM zone 14N)."""
    return CRS.from_epsg(32614)


###############################################################################
# TESTS
###############################################################################


def test_calc_L_IME(plume_data: np.ndarray) -> None:
    """Test calculation of plume area, effective plume size (L), and IME."""
    # pixel_area = 20 * 20 = 400
    # Non-zero values => 5 pixels => area=5*400=2000
    # Sum of non-zero = 0.01+0.10+0.05+0.03+0.02=0.21 => IME=0.21*400=84
    # L= sqrt(2000)= ~44.72136
    area, L, IME = calc_L_IME(plume_data, spatial_res_m=20.0)

    assert np.isclose(area, 2000.0, rtol=1e-7), f"Expected area=2000, got {area}"
    assert np.isclose(L, 44.72136, rtol=1e-5), f"Expected L ~44.72, got {L}"
    assert np.isclose(IME, 84.0, rtol=1e-5), f"Expected IME=84, got {IME}"


def test_calc_Q_IME() -> None:
    """Test methane emission rate Q calculation.

    Q = (u_eff / L) x IME x 57.75286
    """
    u_eff: float = 2.0  # m/s
    L: float = 50.0  # m
    IME: float = 1000.0  # moles
    # Q= (2.0/50.0)*1000*57.75286= 0.04*1000*57.75286= 40*57.75286= ~2310.1144
    expected_Q: float = 2310.1144

    Q: float = calc_Q_IME(u_eff, L, IME)
    assert np.isclose(Q, expected_Q, rtol=1e-5), f"Expected {expected_Q}, got {Q}"


def test_get_plume_source_location_v2(plume_data: np.ndarray, crs_utm14n: CRS, transform: Affine) -> None:
    """Test finding the lat/lon of the maximum retrieval pixel (0,2)."""
    # For row=0,col=2, with origin at (620000, 3413000) & 20m x 20m pixels:
    x_expected = 620000.0 + (2 + 0.5) * 20.0
    y_expected = 3413000.0 - (0 + 0.5) * 20.0

    # Expected lat/lon in WGS 84
    transformer = Transformer.from_crs(crs_utm14n, "EPSG:4326", always_xy=True)
    lon_expected, lat_expected = transformer.transform(x_expected, y_expected)

    # Run the function
    lat, lon = get_plume_source_location_v2(plume_data, crs_utm14n, transform)

    # Compare actual lat/lon to expected
    assert np.isclose(lat, lat_expected, atol=1e-5), f"Latitude mismatch: expected ~{lat_expected}, got {lat}"
    assert np.isclose(lon, lon_expected, atol=1e-5), f"Longitude mismatch: expected ~{lon_expected}, got {lon}"


def test_calc_effective_wind_speed() -> None:
    """Check magnitude of 2D wind vector from (u10m, v10m)."""
    # Test a known vector: u=3, v=4 => speed=5
    speed: float = calc_effective_wind_speed(3.0, 4.0)
    assert np.isclose(speed, 5.0, atol=1e-7), f"Expected 5.0, got {speed}"

    # Also test zero-wind edge case
    speed_zero: float = calc_effective_wind_speed(0.0, 0.0)
    assert speed_zero == 0.0, f"Expected 0.0, got {speed_zero}"


def test_calc_u_eff() -> None:
    """Verify effective wind speed & bounds using Varon (2021) formula."""
    wind_speed: float = 2.0
    wind_low: float = 1.8
    wind_high: float = 2.2
    u_eff, u_eff_high, u_eff_low = calc_u_eff(wind_speed, wind_low, wind_high)
    # Check that none of these are under the thresholds
    assert u_eff >= 0.01, "Effective wind speed < 0.01 m/s"  # noqa: PLR2004 (magic-number-comparison)
    assert u_eff_high >= 0.011, "Effective wind speed (high) < 0.011 m/s"  # noqa: PLR2004 (magic-number-comparison)
    assert u_eff_low >= 0.009, "Effective wind speed (low) < 0.009 m/s"  # noqa: PLR2004 (magic-number-comparison)

    # Since alpha=0.33, beta=0.45 => for 2.0 m/s:
    #   ln(2.0) ~ 0.693147
    #   alpha*ln(2.0) ~ 0.33 * 0.693147 ~ 0.2287
    #   => 0.2287 + 0.45 ~ 0.6787
    assert np.isclose(u_eff, 0.6787, atol=0.01), f"Unexpected central u_eff: {u_eff}"


def test_calc_wind_direction() -> None:
    """Ensure direction= degrees(arctan2(u10m, v10m)) from north."""
    assert np.isclose(calc_wind_direction(0.0, 1.0), 0.0, atol=1e-5), "Expected 0° for north"
    assert np.isclose(calc_wind_direction(1.0, 0.0), 90.0, atol=1e-5), "Expected 90° for east"
    assert np.isclose(calc_wind_direction(0.0, -1.0), 180.0, atol=1e-5), "Expected 180° for south"
    assert np.isclose(calc_wind_direction(-1.0, 0.0), -90.0, atol=1e-5), "Expected -90° for west"


def test_calc_wind_error() -> None:
    """Check lower/upper wind speed given fractional error."""
    wind_speed: float = 10.0
    wind_error: float = 0.1
    wind_low, wind_high = calc_wind_error(wind_speed, wind_error)

    assert np.isclose(wind_low, 9.0, atol=1e-7), f"Expected 9.0, got {wind_low}"
    assert np.isclose(wind_high, 11.0, atol=1e-7), f"Expected 11.0, got {wind_high}"
