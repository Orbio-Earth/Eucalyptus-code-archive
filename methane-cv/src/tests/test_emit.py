"""Tests for the EMIT data handling functionality."""

from datetime import datetime

import pytest
import xarray as xr
from requests.exceptions import HTTPError

from src.data.emit_data import EmitGranuleAccess, EmitTileUris, query_emit_catalog

# Constants for data validation
EXPECTED_DIMENSIONS = 3  # (bands, height, width)


@pytest.fixture(scope="module")
def emit_granule() -> EmitGranuleAccess:
    """Sample EMIT item for testing."""
    # Use the same EMIT tile ID that's used in the test data
    return EmitGranuleAccess("EMIT_L1B_RAD_001_20240127T195840_2402713_006")


def test_init(emit_granule: EmitGranuleAccess) -> None:
    """Test basic initialization and properties."""
    assert emit_granule.id == "EMIT_L1B_RAD_001_20240127T195840_2402713_006"
    assert emit_granule.instrument == "EMIT"
    assert isinstance(emit_granule.time, datetime)
    assert emit_granule.time.strftime("%Y%m%d") == "20240127"
    assert emit_granule.time.strftime("%H%M%S") == "195840"


@pytest.mark.slow
def test_get_radiance(emit_granule: EmitGranuleAccess) -> None:
    """Test radiance data retrieval."""
    # Test full radiance cube
    radiance = emit_granule.get_radiance()
    assert isinstance(radiance, xr.DataArray)
    assert len(radiance.shape) == EXPECTED_DIMENSIONS  # (bands, height, width)


@pytest.mark.slow
def test_get_mask(emit_granule: EmitGranuleAccess) -> None:
    """Test mask data retrieval."""
    mask = emit_granule.get_mask()
    assert isinstance(mask, xr.DataArray)


@pytest.mark.slow
def test_get_obs(emit_granule: EmitGranuleAccess) -> None:
    """Test observation data retrieval."""
    obs = emit_granule.get_obs()
    assert isinstance(obs, xr.DataArray)

    # Check that all expected bands are present
    expected_bands = [
        "path_length",
        "sensor_azimuth",
        "sensor_zenith",
        "solar_azimuth",
        "solar_zenith",
        "phase",
        "slope",
        "aspect",
        "cosine_i",
        "utc_time",
        "earth_sun_dist",
    ]
    assert list(obs.coords["bands"].values) == expected_bands


@pytest.mark.slow
def test_observation_angles(emit_granule: EmitGranuleAccess) -> None:
    """Test observation angle retrieval and validation."""
    obs = emit_granule.get_obs()
    solar_zenith = obs.sel(bands="solar_zenith")
    sensor_zenith = obs.sel(bands="sensor_zenith")

    # Check types
    assert isinstance(solar_zenith, xr.DataArray)
    assert isinstance(sensor_zenith, xr.DataArray)

    # Check angle ranges (0-90 degrees)
    assert solar_zenith.min() >= 0
    assert solar_zenith.max() <= 90  # noqa: PLR2004
    assert sensor_zenith.min() >= 0
    assert sensor_zenith.max() <= 90  # noqa: PLR2004


def test_error_handling() -> None:
    """Test error handling for invalid tile IDs."""
    with pytest.raises(ValueError):
        EmitGranuleAccess("invalid_tile_id")


def test_dataset_loading_error() -> None:
    """Test error handling for dataset loading failures."""
    # Use a valid format but non-existent tile ID
    nonexistent_item = EmitGranuleAccess("EMIT_L1B_RAD_001_20240127T195840_9999999_999")

    with pytest.raises(HTTPError):
        nonexistent_item.get_radiance()


@pytest.mark.slow
def test_query_emit_catalog() -> None:
    """Test EMIT catalog querying functionality."""
    # Test case with known data
    lat, lon = 31.9974, -102.0584  # Midland, TX coordinates
    start_time = datetime(2024, 1, 1)
    end_time = datetime(2024, 12, 31)

    results = query_emit_catalog(lat, lon, start_time, end_time)

    # Basic validation
    assert len(results) > 0
    assert isinstance(results, list)
    assert all(isinstance(id_, str) for id_ in results)

    # Validate EMIT ID format for all results
    for emit_id in results:
        # This will raise ValueError if format is invalid
        EmitTileUris.from_tile_id(emit_id)

    # Test empty result case
    empty_results = query_emit_catalog(
        lat=0,
        lon=0,
        start_time=datetime(2020, 1, 1),  # Before EMIT launch
        end_time=datetime(2020, 2, 1),
    )
    assert isinstance(empty_results, list)
    assert len(empty_results) == 0

    # Test case with overlapping granules
    # motivated by https://git.orbio.earth/orbio/orbio/-/merge_requests/1015#note_80512
    lat, lon = 32.3221, -101.80803

    results = query_emit_catalog(lat, lon, datetime(2024, 7, 23), datetime(2024, 7, 24))
    assert "EMIT_L1B_RAD_001_20240723T213259_2420514_014" in results
    assert len(results) == 1
