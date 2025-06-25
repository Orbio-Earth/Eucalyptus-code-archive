"""Tests for inference functionality including data loading, model input preparation and inference."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.azure_wrap.ml_client_utils import (
    MLClient,
    get_default_blob_storage,
    initialize_blob_service_client,
)
from src.data.common.data_item import MonoTemporalPlumesDataItem
from src.data.landsat_data import (
    LandsatGranuleAccess,
    LandsatImageAttributes,
    LandsatImageMetadata,
    LandsatImageMetadataFile,
    LandsatQAValues,
    LandsatRadiometricRescaling,
    LandsatThermalConstants,
    convert_band_values,
    load_cropped_landsat_items,
    query_landsat_catalog_for_point,
)
from src.inference.inference_functions import (
    fetch_landsat_items_for_point,
    load_and_crop_emit_image_for_point,
    prepare_clouds_shadows_landsat,
    prepare_model_input,
)
from src.training.transformations import ConcatenateSnapshots
from src.utils.exceptions import MissingImageException
from src.utils.parameters import (
    LANDSAT_BANDS,
    REQUIRED_NUM_SNAPSHOTS,
    SatelliteID,
)
from src.utils.utils import S3Client, initialize_s3_client, recreate_band_extractor_for_satellite

EXPECTED_NUM_SNAPSHOTS = 3
# FIXME: Add S2 tests using new chip selection, see https://git.orbio.earth/orbio/orbio/-/issues/1314


@pytest.fixture(scope="module")
def s3_client(ml_client: MLClient) -> S3Client:
    """Initialize and return an S3 client."""
    return initialize_s3_client(ml_client)


def test_roundtrip_concatenator(sat_key: SatelliteID, concatenator_config: dict[SatelliteID, dict]) -> None:
    """Test the roundtrip serialization and deserialization of ConcatenateSnapshots."""
    if sat_key == SatelliteID.EMIT:
        pytest.skip("EMIT does not use ConcatenateSnapshots. We need to implement a test for EMIT's band extractor.")
    concatenator1 = ConcatenateSnapshots(**concatenator_config[sat_key])
    concat1_dict = concatenator1.asdict()
    concatenator2 = recreate_band_extractor_for_satellite(sat_key, **concat1_dict)
    assert concatenator1.snapshots == concatenator2.snapshots
    assert concatenator1.temporal_bands == concatenator2.temporal_bands
    assert concatenator1.main_bands == concatenator2.main_bands
    assert concatenator1.all_available_bands == concatenator2.all_available_bands

    # also check a concatenator can be created from the default arguments
    _ = recreate_band_extractor_for_satellite(sat_key)


def test_load_and_crop_emit_data(ml_client: MLClient) -> None:
    """Test loading and cropping EMIT data using a known ground truth point."""
    # Point from EMIT_ground_truth_plumes.csv (Texas gas plant)
    lat, lon = 31.34677, -101.79872
    date = datetime(2024, 2, 12)
    crop_size = 128

    cache_container_name = get_default_blob_storage(ml_client).container_name

    cropped_data = list(
        load_and_crop_emit_image_for_point(
            lat=lat,
            lon=lon,
            query_datetime=date,
            crop_size=crop_size,
            ml_client=ml_client,
            cache_container_name=cache_container_name,
        )
    )

    assert len(cropped_data) == 1
    assert "crop_arrays" in cropped_data[0]
    assert "crop_params" in cropped_data[0]
    assert "tile_item" in cropped_data[0]

    # Check the crop_arrays structure
    assert "radiance" in cropped_data[0]["crop_arrays"]
    assert "mask" in cropped_data[0]["crop_arrays"]

    # Check spatial dimensions
    assert cropped_data[0]["crop_arrays"]["radiance"].sizes["downtrack"] == crop_size
    assert cropped_data[0]["crop_arrays"]["radiance"].sizes["crosstrack"] == crop_size
    assert all(
        key in cropped_data[0]["crop_params"]
        for key in ["crop_start_x", "crop_start_y", "crop_height", "crop_width", "out_height", "out_width"]
    )


def test_invalid_location_emit(ml_client: MLClient) -> None:
    """Test error handling for invalid location in EMIT."""
    # Point in ocean where we shouldn't find data
    lat, lon = 0.0, 0.0
    date = datetime(2024, 1, 27)

    cache_container_name = get_default_blob_storage(ml_client).container_name

    with pytest.raises(MissingImageException, match="has no valid EMIT imagery"):
        load_and_crop_emit_image_for_point(
            lat=lat,
            lon=lon,
            query_datetime=date,
            crop_size=128,
            ml_client=ml_client,
            cache_container_name=cache_container_name,
        )


def test_prepare_model_input_emit(ml_client: MLClient) -> None:
    """Test preparing model input for EMIT data."""
    # Point from Texas gas plant (known test point)
    lat, lon = 31.34677, -101.79872
    date = datetime(2024, 2, 12)
    crop_size = 128

    cache_container_name = get_default_blob_storage(ml_client).container_name

    # Call the function
    cropped_data, data_item = prepare_model_input(
        lat=lat,
        lon=lon,
        query_datetime=date,
        crop_size=crop_size,
        cache_container_name=cache_container_name,
        ml_client=ml_client,
    )

    # Verify outputs
    assert isinstance(data_item, MonoTemporalPlumesDataItem)
    assert len(cropped_data) > 0  # Should have at least one item
    assert data_item.crop_main.shape[1:] == (crop_size, crop_size)  # Spatial dimensions


def test_fetch_landsat_items_for_point() -> None:
    """Test fetching Landsat items for a known ground truth point."""
    # Point from Casa Grande, AZ
    # TODO: update with correct landsat point
    lat, lon = 32.8218205, -111.785773
    date = datetime(2022, 10, 25)

    items = fetch_landsat_items_for_point(lat=lat, lon=lon, query_datetime=date)

    # Check we got the expected number of items
    assert len(items) >= REQUIRED_NUM_SNAPSHOTS

    # Verify items are LandsatGranuleAccess instances
    assert all(isinstance(item, LandsatGranuleAccess) for item in items)

    # Verify items are sorted by time (most recent first)
    times = [item.time for item in items]
    assert times == sorted(times, reverse=True)

    # Verify first item is from the requested date
    assert items[0].time.date() == date.date()


def test_load_cropped_landsat_items(ml_client: MLClient, s3_client: S3Client) -> None:
    """Test loading and cropping Landsat data using a known ground truth point."""
    abs_client = initialize_blob_service_client(ml_client)
    # Point from Casa Grande, AZ
    # TODO: update with correct landsat point
    lat, lon = 32.8218205, -111.785773
    date = datetime(2022, 10, 25)
    crop_size = 128

    # First get the items
    items = fetch_landsat_items_for_point(lat=lat, lon=lon, query_datetime=date)[:4]
    for item in items[:]:
        item.prefetch_l1(s3_client, abs_client)

    # Load and crop the items
    cropped_data = load_cropped_landsat_items(
        items=items, bands=LANDSAT_BANDS, lat=lat, lon=lon, image_size=crop_size, abs_client=abs_client
    )

    assert len(cropped_data) == len(items)
    for crop_dict in cropped_data:
        assert "crop_arrays" in crop_dict
        assert "crop_params" in crop_dict
        assert "tile_item" in crop_dict

        # Check spatial dimensions
        assert crop_dict["crop_arrays"].shape[-2:] == (crop_size, crop_size)

        # Check crop parameters
        first_band = LANDSAT_BANDS[0]
        first_band_params = crop_dict["crop_params"][first_band]

        # Verify all required keys exist in the first band's parameters
        assert all(key in first_band_params for key in ["crop_start_x", "crop_start_y", "crop_height", "crop_width"])

        # Verify all bands have the same crop parameters
        for band in LANDSAT_BANDS[1:]:
            assert (
                crop_dict["crop_params"][band] == first_band_params
            ), f"Crop parameters for {band} differ from {first_band}"


def test_invalid_location_landsat() -> None:
    """Test error handling for invalid location in Landsat."""
    # Point in ocean where we shouldn't find data
    lat, lon = 0.0, 0.0
    date = datetime(2023, 1, 15)

    start_time = date - timedelta(days=90)
    end_time = date + timedelta(days=1)

    # Should return empty list for ocean location
    items = query_landsat_catalog_for_point(lat, lon, start_time, end_time)
    assert len(items) == 0


def test_convert_band_values() -> None:
    """Test conversion of band values from DN to reflectance/brightness temperature."""
    # Create mock data
    band_data = np.array([[[1000, 2000], [3000, 4000]]], dtype=np.uint16)
    solar_angle = 30.0

    # Create mock MTL data with all required fields
    image_attributes = {
        "SPACECRAFT_ID": "LANDSAT_8",
        "SENSOR_ID": "OLI_TIRS",
        "WRS_TYPE": "2",
        "WRS_PATH": "36",
        "WRS_ROW": "37",
        "NADIR_OFFNADIR": "NADIR",
        "TARGET_WRS_PATH": "36",
        "TARGET_WRS_ROW": "37",
        "DATE_ACQUIRED": "2025-03-24",
        "SCENE_CENTER_TIME": "17:57:30.7470599Z",
        "STATION_ID": "LGN",
        "CLOUD_COVER": "0.01",
        "CLOUD_COVER_LAND": "0.01",
        "IMAGE_QUALITY_OLI": "9",
        "IMAGE_QUALITY_TIRS": "9",
        "SATURATION_BAND_1": "N",
        "SATURATION_BAND_2": "N",
        "SATURATION_BAND_3": "N",
        "SATURATION_BAND_4": "N",
        "SATURATION_BAND_5": "N",
        "SATURATION_BAND_6": "Y",
        "SATURATION_BAND_7": "Y",
        "SATURATION_BAND_8": "N",
        # "SATURATION_BAND_9": "N", # sometimes fields aren't present in metadata
        "ROLL_ANGLE": "-0.000",
        "SUN_AZIMUTH": "140.57896529",
        "SUN_ELEVATION": "51.79079864",
        "EARTH_SUN_DISTANCE": "0.9971460",
        "SENSOR_MODE": "SAM",  # sometimes there are extra fields
    }

    reflectance_mult = 3.0e-4
    reflectance_add = 0.1
    radiance_mult = 3.0e-4
    radiance_add = 0.1

    rescaling_data = {
        # Required reflectance coefficients for all bands
        **{f"REFLECTANCE_MULT_BAND_{i}": reflectance_mult for i in range(1, 12)},
        **{f"REFLECTANCE_ADD_BAND_{i}": reflectance_add for i in range(1, 12)},
        # Required radiance coefficients for all bands
        **{f"RADIANCE_MULT_BAND_{i}": radiance_mult for i in range(1, 12)},
        **{f"RADIANCE_ADD_BAND_{i}": radiance_add for i in range(1, 12)},
    }

    # Create mock MTL data
    mtl_data = LandsatImageMetadataFile(
        LANDSAT_METADATA_FILE=LandsatImageMetadata(
            IMAGE_ATTRIBUTES=LandsatImageAttributes(**image_attributes),
            LEVEL1_RADIOMETRIC_RESCALING=LandsatRadiometricRescaling(**rescaling_data),
            LEVEL1_THERMAL_CONSTANTS=LandsatThermalConstants(
                K1_CONSTANT_BAND_10=774.89,
                K2_CONSTANT_BAND_10=1321.08,
                K1_CONSTANT_BAND_11=480.89,
                K2_CONSTANT_BAND_11=1201.14,
            ),
        )
    )

    # Test reflectance conversion (e.g., SWIR16)
    swir16_result = convert_band_values("swir16", band_data, mtl_data, solar_angle)
    expected_swir16 = LandsatGranuleAccess.convert_band_dn_to_reflectance_values(
        band_data,
        solar_angle,
        reflectance_mult,
        reflectance_add,
    )
    np.testing.assert_array_equal(swir16_result, expected_swir16)

    # Test thermal band conversion (e.g., LWIR11)
    lwir11_result = convert_band_values("lwir11", band_data, mtl_data, solar_angle)
    expected_bt = LandsatGranuleAccess.convert_thermal_band_dn_to_brightness_values(
        band_data,
        radiance_mult,
        radiance_add,
        mtl_data.LANDSAT_METADATA_FILE.LEVEL1_THERMAL_CONSTANTS.K1_CONSTANT_BAND_10,
        mtl_data.LANDSAT_METADATA_FILE.LEVEL1_THERMAL_CONSTANTS.K2_CONSTANT_BAND_10,
    )
    np.testing.assert_array_equal(lwir11_result, expected_bt)

    # Test QA pixel band (should return unchanged)
    qa_data = np.array([[[1, 2], [3, 4]]], dtype=np.uint16)
    qa_result = convert_band_values("qa_pixel", qa_data, mtl_data, solar_angle)
    np.testing.assert_array_equal(qa_result, qa_data)

    # Test handling of zero values (should remain zero)
    band_data_with_zeros = np.array([[[0, 2000], [3000, 0]]], dtype=np.uint16)
    result = convert_band_values("swir16", band_data_with_zeros, mtl_data, solar_angle)
    assert np.all(result[band_data_with_zeros == 0] == 0)


def test_prepare_clouds_shadows_landsat() -> None:
    """Test cloud and shadow mask generation from QA pixel band."""
    # Create a mock QA array with known values
    qa_array = np.zeros((1, 128, 128), dtype=np.uint16)

    # Set different regions with different cloud/shadow flags
    # Cloud regions
    qa_array[0, 0:10, 0:10] = LandsatQAValues.CLOUD.value
    qa_array[0, 10:20, 0:10] = LandsatQAValues.DILATED_CLOUD.value
    qa_array[0, 20:30, 0:10] = LandsatQAValues.CIRRUS.value
    qa_array[0, 30:40, 0:10] = LandsatQAValues.CLOUD_CONFIDENCE_MEDIUM.value
    qa_array[0, 40:50, 0:10] = LandsatQAValues.CLOUD_CONFIDENCE_HIGH.value
    qa_array[0, 50:60, 0:10] = LandsatQAValues.CIRRUS_CONFIDENCE_MEDIUM.value
    qa_array[0, 60:70, 0:10] = LandsatQAValues.CIRRUS_CONFIDENCE_HIGH.value

    # Shadow regions
    qa_array[0, 0:30, 20:30] = LandsatQAValues.CLOUD_SHADOW.value
    qa_array[0, 30:60, 20:30] = LandsatQAValues.CLOUD_SHADOW_CONFIDENCE_MEDIUM.value
    qa_array[0, 60:90, 20:30] = LandsatQAValues.CLOUD_SHADOW_CONFIDENCE_HIGH.value

    # Fill (nodata) values
    qa_array[0, 100:, 100:] = LandsatQAValues.FILL.value

    # Region with both cloud and shadow (should be marked as cloud only)
    qa_array[0, 80:90, 80:90] = LandsatQAValues.CLOUD.value | LandsatQAValues.CLOUD_SHADOW.value

    # Create mock reflectance data with some zero values
    reflectance_bands = np.ones((9, 128, 128), dtype=np.uint16)
    reflectance_bands[:, 90:100, 90:100] = 0  # Add some reflectance nodata

    # Combine reflectance and QA bands
    mock_data = np.concatenate([reflectance_bands, qa_array])

    cloud_mask, shadow_mask, qa = prepare_clouds_shadows_landsat(mock_data)

    # Verify masks shapes
    assert cloud_mask.shape == (128, 128)
    assert shadow_mask.shape == (128, 128)
    assert qa.shape == (128, 128)

    # Check cloud flags
    for i, y_start in enumerate(range(0, 70, 10)):
        assert np.all(cloud_mask[y_start : y_start + 10, 0:10] == 1), f"Cloud flag {i} not detected"

    # Check shadow flags
    for i, y_start in enumerate(range(0, 90, 30)):
        assert np.all(shadow_mask[y_start : y_start + 30, 20:30] == 1), f"Shadow flag {i} not detected"

    # Check that cloud+shadow region is marked as cloud only
    assert np.all(cloud_mask[80:90, 80:90] == 1)
    assert np.all(shadow_mask[80:90, 80:90] == 0)

    # Check that nodata regions (FILL values) are 0 in both masks
    assert np.all(cloud_mask[100:, 100:] == 0)
    assert np.all(shadow_mask[100:, 100:] == 0)

    # Check that reflectance nodata regions are 0 in both masks
    assert np.all(cloud_mask[90:100, 90:100] == 0)
    assert np.all(shadow_mask[90:100, 90:100] == 0)

    # Verify no overlap between cloud and shadow masks
    assert not np.any(np.logical_and(cloud_mask == 1, shadow_mask == 1)), "Found pixels marked as both cloud and shadow"
