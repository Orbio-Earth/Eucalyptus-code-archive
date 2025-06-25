"""Tests for radtran_utils.py."""

import numpy as np
import pytest

from src.tests.generate_test_data import S2_HAPI_DATA_PATH
from src.utils.radtran_utils import RadTranLookupTable

###############################################################################
# FIXTURES
###############################################################################


@pytest.fixture
def min_ch4() -> float:
    """Return the minimum CH4 concentration (mol/m²)."""
    return 0.0


@pytest.fixture
def max_ch4() -> float:
    """Return the maximum CH4 concentration (mol/m²)."""
    return 100


@pytest.fixture
def spacing_resolution() -> int:
    """Return the number of grid steps between min_ch4 and max_ch4."""
    return 10000


@pytest.fixture
def solar_angle() -> float:
    """Return the solar angle for the atmospheric model."""
    return 52.61


@pytest.fixture
def observation_angle() -> float:
    """Return the observation angle for the atmospheric model."""
    return 2.27


@pytest.fixture
def s2b_lookup_table(
    min_ch4: float, max_ch4: float, spacing_resolution: int, solar_angle: float, observation_angle: float
) -> RadTranLookupTable:
    """
    Build a RadTranLookupTable for S2B.

    It retrieves the relevant arrays from ATMOSPHERIC_DATA["B"], then uses
    generate_lookup_tables to create normalized brightness grids for the target
    and reference bands.
    """
    return RadTranLookupTable.from_params(
        instrument="B",
        solar_angle=solar_angle,
        observation_angle=observation_angle,
        hapi_data_path=S2_HAPI_DATA_PATH,
        min_ch4=min_ch4,
        max_ch4=max_ch4,
        spacing_resolution=spacing_resolution,
        ref_band="B11",
        band="B12",
        full_sensor_name="Sentinel2",
    )


###############################################################################
# TESTS
###############################################################################


def test_s2b_lookup_table_shapes_and_values(s2b_lookup_table: RadTranLookupTable) -> None:
    """Test that the nB grids have the expected shape and values in [0, 1]."""
    lut = s2b_lookup_table
    expected_len = lut.spacing_resolution

    # Shape checks
    assert lut.nB_grid_band.shape == (expected_len,)
    assert lut.nB_grid_ref_band.shape == (expected_len,)

    # Value checks: normalized brightness is in [0, 1].
    assert np.all(lut.nB_grid_band >= 0.0)
    assert np.all(lut.nB_grid_band <= 1.0)
    assert np.all(lut.nB_grid_ref_band >= 0.0)
    assert np.all(lut.nB_grid_ref_band <= 1.0)


@pytest.mark.parametrize(
    "retrieval_values",
    [
        np.linspace(-1e-8, 1.5, 50),  # Testing case with small negative values
        np.linspace(0, 1, 50),  # Only positive values
    ],
)
def test_s2b_lookup_method(s2b_lookup_table: RadTranLookupTable, retrieval_values: np.ndarray) -> None:
    """
    Test the lookup() method with various retrieval ranges.

    retrieval -> nB_band, nB_ref
    """
    lut = s2b_lookup_table

    nB_band, nB_ref = lut.lookup(retrieval_values)

    # Output arrays must match input shape
    assert nB_band.shape == retrieval_values.shape
    assert nB_ref.shape == retrieval_values.shape

    # Brightness values between [0,1].
    assert np.all(nB_band >= 0.0) and np.all(nB_band <= 1.0)
    assert np.all(nB_ref >= 0.0) and np.all(nB_ref <= 1.0)


@pytest.mark.parametrize(
    "frac_values",
    [
        np.linspace(-0.3, 0.0, 50),
    ],
)
def test_s2b_reverse_lookup_method(s2b_lookup_table: RadTranLookupTable, frac_values: np.ndarray) -> None:
    """
    Test the reverse_lookup() method with frac values.

    frac -> retrieval
    """
    lut = s2b_lookup_table

    retrieval = lut.reverse_lookup(frac_values)

    # Output shape must match input
    assert retrieval.shape == frac_values.shape

    # Must be clipped to [min_ch4, max_ch4]
    assert np.all(retrieval >= lut.min_ch4)
    assert np.all(retrieval <= lut.max_ch4)


def test_s2b_round_trip_method(s2b_lookup_table: RadTranLookupTable) -> None:
    """
    Check round-trip consistency.

    retrieval -> lookup -> frac -> reverse_lookup
    """
    lut = s2b_lookup_table

    # Retrieval grid
    retrieval_values = np.linspace(0, 1.5, 50)

    nB_band, nB_ref = lut.lookup(retrieval_values)

    # frac = (nB_band / nB_ref) - 1
    frac = (nB_band / nB_ref) - 1.0

    recovered = lut.reverse_lookup(frac)

    # Check near equality
    assert recovered.shape == retrieval_values.shape
    assert np.allclose(retrieval_values, recovered, atol=1e-3, rtol=3e-2)
