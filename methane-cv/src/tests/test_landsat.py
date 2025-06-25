"""Test module for Landsat data access and processing functionality."""

import datetime
from datetime import timedelta

import numpy as np
import pytest
import rasterio.transform
from azure.ai.ml import MLClient
from azure.storage.blob import BlobServiceClient
from mypy_boto3_s3 import S3Client

from src.azure_wrap.ml_client_utils import initialize_blob_service_client
from src.data.landsat_data import (
    LandsatGranuleAccess,
    LandsatQAValues,
    get_aligned_cropped_landsat_band,
    query_landsat_catalog_for_tile,
)
from src.utils.utils import initialize_s3_client

#######################
### SETUP FUNCTIONS ###
#######################


@pytest.fixture(scope="module")
def s3_client(ml_client: MLClient) -> S3Client:
    """Initialize and return an S3 client."""
    return initialize_s3_client(ml_client)


@pytest.fixture(scope="module")
def abs_client(ml_client: MLClient) -> BlobServiceClient:
    """Initialize and return a Blob Service client."""
    return initialize_blob_service_client(ml_client)


@pytest.fixture(params=["LC08_L1TP_030038_20250125_20250130_02_T1", "LC09_L1GT_223128_20250210_20250210_02_T2"])
def tile_id(request: pytest.FixtureRequest) -> str:
    """Landsat tile ID fixture.

    Returns both Landsat 8 and Landsat 9 tile IDs for comprehensive testing.
    """
    return request.param


######################
### TEST FUNCTIONS ###
######################


def test_init(tile_id: str) -> None:
    """Test LandsatGranuleAccess initialization."""
    item = LandsatGranuleAccess.from_id(tile_id)
    assert item.id == tile_id
    if tile_id.startswith("LC08"):
        assert item.instrument == "8"
    else:
        assert item.instrument == "9"
    assert item.observation_angle == 0.0
    assert item.solar_angle > 0.0
    assert item.crs.startswith("EPSG:")
    date_from_id = datetime.datetime.strptime(tile_id.split("_")[3], "%Y%m%d").date()
    assert date_from_id == item.time.date()
    assert isinstance(item.time, datetime.datetime)


def test_get_band(tile_id: str, s3_client: S3Client, abs_client: BlobServiceClient) -> None:
    """Test getting a single band."""
    item = LandsatGranuleAccess.from_id(tile_id)
    item.prefetch_l1(s3_client, abs_client)
    swir16 = item.get_band("swir16", out_height=400, out_width=400, abs_client=abs_client)
    assert swir16.shape == (1, 400, 400)


def test_get_bands(tile_id: str, s3_client: S3Client, abs_client: BlobServiceClient) -> None:
    """Test getting multiple bands."""
    item = LandsatGranuleAccess.from_id(tile_id)
    item.prefetch_l1(s3_client, abs_client)
    arr = item.get_bands(["swir16", "swir22"], out_height=100, out_width=100, abs_client=abs_client)
    assert arr.shape == (2, 100, 100)


def test_get_band_crop(tile_id: str, s3_client: S3Client, abs_client: BlobServiceClient) -> None:
    """Test cropping a band."""
    item = LandsatGranuleAccess.from_id(tile_id)
    item.prefetch_l1(s3_client, abs_client)
    band11 = item.get_band_crop("swir16", 100, 300, 4000, 2000, 400, 200, abs_client=abs_client)
    assert band11.shape == (1, 400, 200)


def test_get_mask(tile_id: str, s3_client: S3Client, abs_client: BlobServiceClient) -> None:
    """Test generating QA masks."""
    item = LandsatGranuleAccess.from_id(tile_id)
    item.prefetch_l1(s3_client, abs_client)

    # Test with single QA value
    mask = item.get_mask([LandsatQAValues.CLOUD], out_height=100, out_width=100, abs_client=abs_client)
    assert mask.shape == (100, 100)
    assert mask.dtype == bool

    # Test with multiple QA values
    mask_multi = item.get_mask(
        [LandsatQAValues.CLOUD, LandsatQAValues.CLOUD_SHADOW], out_height=100, out_width=100, abs_client=abs_client
    )
    assert mask_multi.shape == (100, 100)
    assert mask_multi.dtype == bool

    # The combined mask should have at least as many True values as the cloud-only mask
    assert mask_multi.sum() >= mask.sum()


def test_get_mask_crop(tile_id: str, s3_client: S3Client, abs_client: BlobServiceClient) -> None:
    """Test generating cropped QA masks."""
    item = LandsatGranuleAccess.from_id(tile_id)
    item.prefetch_l1(s3_client, abs_client)

    # Test with default output dimensions
    mask = item.get_mask_crop(
        [LandsatQAValues.CLOUD],
        crop_start_x=100,
        crop_start_y=100,
        crop_height=200,
        crop_width=200,
        abs_client=abs_client,
    )
    assert mask.shape == (200, 200)
    assert mask.dtype == bool

    # Test with custom output dimensions
    mask_resized = item.get_mask_crop(
        [LandsatQAValues.CLOUD],
        crop_start_x=100,
        crop_start_y=100,
        crop_height=200,
        crop_width=200,
        out_height=100,
        out_width=100,
        abs_client=abs_client,
    )
    assert mask_resized.shape == (100, 100)
    assert mask_resized.dtype == bool

    # Test with multiple QA values
    mask_multi = item.get_mask_crop(
        [LandsatQAValues.CLOUD, LandsatQAValues.CLOUD_SHADOW],
        crop_start_x=100,
        crop_start_y=100,
        crop_height=200,
        crop_width=200,
        abs_client=abs_client,
    )
    assert mask_multi.shape == (200, 200)
    assert mask_multi.dtype == bool

    # The combined mask should have at least as many True values as the cloud-only mask
    assert mask_multi.sum() >= mask.sum()


def test_filter_landsat_duplicates() -> None:
    """Test filtering of Landsat duplicates based on processing date."""
    # Simulated item IDs with same acquisition date but different processing dates
    test_items = [
        "LC08_L1TP_030038_20250125_20250130_02_T1",  # Later processing date
        "LC08_L1TP_030038_20250125_20250128_02_T1",  # Earlier processing date
        "LC09_L1GT_223128_20250210_20250215_02_T2",  # Later processing date
        "LC09_L1GT_223128_20250210_20250212_02_T2",  # Earlier processing date
        "LC09_L1GT_027038_20250128_20250128_02_T2",  # same sensor, same date, lower quality (T2)
        "LC09_L1TP_027037_20250128_20250128_02_T1",  # same sensor, same date, higher quality (T1)
    ]

    from src.data.generation.landsat import select_best_landsat_items_by_quality

    filtered_ids = select_best_landsat_items_by_quality(test_items)
    EXPECTED_UNIQUE_SCENES = len(set(item.split("_")[3] for item in test_items))  # Count unique acquisition dates

    # Should keep only the most recent processing date for each unique scene
    assert len(filtered_ids) == EXPECTED_UNIQUE_SCENES

    # Check that we kept the most recent processing dates
    assert "LC08_L1TP_030038_20250125_20250130_02_T1" in filtered_ids
    assert "LC09_L1GT_223128_20250210_20250215_02_T2" in filtered_ids
    # Check that we filtered out the earlier processing dates
    assert "LC08_L1TP_030038_20250125_20250128_02_T1" not in filtered_ids
    assert "LC09_L1GT_223128_20250210_20250212_02_T2" not in filtered_ids

    # Check that we kept the best quality
    assert "LC09_L1TP_027037_20250128_20250128_02_T1" in filtered_ids
    # Check that we filtered out the worse quality
    assert "LC09_L1GT_027038_20250128_20250128_02_T2" not in filtered_ids


def test_query_landsat_catalog_for_tile() -> None:
    """Test querying Landsat catalog for a specific WRS tile."""
    # Known WRS path/row for Casa Grande, AZ
    wrs_path = "037"
    wrs_row = "037"
    date = datetime.datetime(2022, 10, 26)
    start_time = date - timedelta(days=90)
    end_time = date + timedelta(days=1)

    items = query_landsat_catalog_for_tile(wrs_path=wrs_path, wrs_row=wrs_row, start_time=start_time, end_time=end_time)

    assert len(items) > 0
    for item in items:
        assert item.properties["landsat:wrs_path"] == wrs_path
        assert item.properties["landsat:wrs_row"] == wrs_row
        assert item.properties["platform"] in ["LANDSAT_8", "LANDSAT_9"]


def test_get_aligned_cropped_landsat_band(tile_id: str, s3_client: S3Client, abs_client: BlobServiceClient) -> None:
    """Test getting aligned cropped bands from Landsat images."""
    item = LandsatGranuleAccess.from_id(tile_id)
    item.prefetch_l1(s3_client, abs_client)

    # Get metadata for testing
    reference_meta = item.get_raster_meta("swir16", abs_client=abs_client)

    # Test case 1: Normal case - target and reference are the same image
    target_meta = reference_meta.copy()
    target_crop_params = {"crop_start_x": 2000, "crop_start_y": 3000, "crop_height": 200, "crop_width": 200}

    result = get_aligned_cropped_landsat_band(
        item, reference_meta, target_meta, target_crop_params, "swir16", abs_client=abs_client
    )
    assert result.shape == (1, 200, 200)
    assert not np.all(result == 0)  # Should contain actual data

    # Test case 2: Target image has different transform (shifted by 100 pixels)
    shifted_transform = rasterio.Affine(
        reference_meta["transform"].a,
        reference_meta["transform"].b,
        reference_meta["transform"].c + 100 * reference_meta["transform"].a,
        reference_meta["transform"].d,
        reference_meta["transform"].e,
        reference_meta["transform"].f,
    )
    target_meta_shifted = reference_meta.copy()
    target_meta_shifted["transform"] = shifted_transform

    result_shifted = get_aligned_cropped_landsat_band(
        item, reference_meta, target_meta_shifted, target_crop_params, "swir16", abs_client=abs_client
    )
    assert result_shifted.shape == (1, 200, 200)
    # The shifted transform moves the image 100 pixels right,
    # so the overlapping portions (excluding the shifted regions) should be equal
    assert np.allclose(result[:, :, 100:], result_shifted[:, :, :-100])  # Overlapping portions should match

    # Test case 3: Crop coordinates outside image bounds
    out_of_bounds_params = {
        "crop_start_x": -1000,  # Negative coordinates
        "crop_start_y": -1000,
        "crop_height": 200,
        "crop_width": 200,
    }

    result_out_of_bounds = get_aligned_cropped_landsat_band(
        item, reference_meta, target_meta, out_of_bounds_params, "swir16", abs_client=abs_client
    )
    assert result_out_of_bounds.shape == (1, 200, 200)
    assert np.all(result_out_of_bounds == 0)  # Should be all zeros

    # Test case 4: Crop extends beyond image bounds
    edge_params = {
        "crop_start_x": reference_meta["width"] - 100,  # Near right edge
        "crop_start_y": reference_meta["height"] - 100,  # Near bottom edge
        "crop_height": 200,
        "crop_width": 200,
    }

    result_edge = get_aligned_cropped_landsat_band(
        item, reference_meta, target_meta, edge_params, "swir16", abs_client=abs_client
    )
    assert result_edge.shape == (1, 200, 200)
    assert np.any(result_edge == 0)  # Should contain some zeros

    # Test case 5: Different CRS between target and reference
    target_meta_diff_crs = target_meta.copy()
    target_meta_diff_crs["crs"] = "EPSG:4326"  # Different CRS

    with pytest.raises(ValueError, match="CRS mismatch"):
        get_aligned_cropped_landsat_band(
            item, reference_meta, target_meta_diff_crs, target_crop_params, "swir16", abs_client=abs_client
        )
