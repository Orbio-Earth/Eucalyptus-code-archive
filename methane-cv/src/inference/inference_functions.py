"""Utility functions for loading, processing, and preparing Sentinel-2 data for model inference."""

import logging
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import azure.core.exceptions
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import torch
import xarray as xr
from azure.ai.ml import MLClient
from azure.storage.blob import BlobServiceClient, ContainerClient
from mypy_boto3_s3 import S3Client
from torch import nn
from torch.nn import Module

from src.azure_wrap.ml_client_utils import initialize_blob_service_client
from src.data.common.data_item import MonoTemporalPlumesDataItem
from src.data.emit_data import EmitGranuleAccess, query_emit_catalog
from src.data.generation.landsat import select_best_landsat_items_by_quality
from src.data.generation.sentinel2 import OMNICLOUD_NODATA
from src.data.landsat_data import (
    LandsatGranuleAccess,
    LandsatQAValues,
    load_cropped_landsat_items,
    query_landsat_catalog_for_point,
)
from src.data.sentinel2 import (
    BAND_RESOLUTIONS,
    Sentinel2Item,
    load_cropped_s2_items,
    query_sentinel2_catalog_for_point,
)
from src.data.sentinel2 import SceneClassificationLabel as SCLabel
from src.data.sentinel2_l1c import Sentinel2L1CItem, Sentinel2L1CItem_Copernicus
from src.plotting.plotting_functions import (
    get_rgb_from_xarray,
    grid16,
    plot_frac,
)
from src.training.loss_functions import TwoPartLoss
from src.training.transformations import BaseBandExtractor
from src.utils.exceptions import (
    InsufficientCoverageException,
    InsufficientImageryException,
    InsufficientTemporalImageryException,
    MissingImageException,
)
from src.utils.parameters import (
    LANDSAT_BANDS,
    REQUIRED_NUM_PREVIOUS_SNAPSHOTS,
    REQUIRED_NUM_SNAPSHOTS,
    S2_BANDS,
    SatelliteID,
)

logger = logging.getLogger(__name__)

# Prefix within the default storage container for caching EMIT data
EMIT_CACHE_PREFIX = "data/emit/crop_cache"

SC_MASK_LABELS = [
    SCLabel.NO_DATA,
    SCLabel.CLOUD_HIGH_PROBABILITY,
    SCLabel.CLOUD_MEDIUM_PROBABILITY,
]

####################################################
############ DATA LOADING & PREPARATION ############
####################################################


def fetch_sentinel2_items_for_point(
    lat: float,
    lon: float,
    query_datetime: datetime,
    crop_size: int,
    how_many_days_back: int = 90,
    sbr_notebook: bool = False,
) -> list[Sentinel2L1CItem | Sentinel2L1CItem_Copernicus]:
    """Fetch Sentinel-2 items for a given location and time range."""
    start_date = query_datetime - timedelta(days=how_many_days_back)
    end_date = query_datetime + timedelta(days=1)

    stac_items = query_sentinel2_catalog_for_point(lat, lon, start_date, end_date, crop_size, sbr_notebook=sbr_notebook)
    stac_items = sorted(stac_items, key=lambda item: item.properties["datetime"], reverse=True)

    # Choose the appropriate class based on sbr_notebook parameter
    item_class = Sentinel2L1CItem_Copernicus if sbr_notebook else Sentinel2L1CItem
    items = [item_class(item) for item in stac_items]

    if not items:
        raise MissingImageException(f"No valid S2 imagery from {start_date} to {end_date}.")
    if not items[0].time.date() == query_datetime.date():
        # we expect the first image to be the target date
        raise MissingImageException(f"No valid S2 imagery for {query_datetime.date()}. First image is {items[0].time}")
    if len(items) < REQUIRED_NUM_SNAPSHOTS:
        raise InsufficientTemporalImageryException("No valid image stack (t-2, t-1, t)")
    return items


def fetch_landsat_items_for_point(
    lat: float, lon: float, query_datetime: datetime, how_many_days_back: int = 90
) -> list[LandsatGranuleAccess]:
    """Fetch Landsat items for a given location and time range."""
    start_date = query_datetime - timedelta(days=how_many_days_back)
    end_date = query_datetime + timedelta(days=1)

    stac_items = query_landsat_catalog_for_point(lat, lon, start_date, end_date)
    stac_items = sorted(stac_items, key=lambda item: item.properties["datetime"], reverse=True)
    items = [LandsatGranuleAccess(item) for item in stac_items]

    filtered_item_ids = select_best_landsat_items_by_quality([item.id for item in items])
    # Map back to the original objects
    items = [item for item in items if item.id in filtered_item_ids]

    if not items:
        raise MissingImageException(f"No valid Landsat imagery from {start_date} to {end_date}.")
    if not items[0].time.date() == query_datetime.date():
        # we expect the first image to be the target date
        raise MissingImageException(
            f"No valid Landsat imagery for {query_datetime.date()}. First image is {items[0].time}"
        )
    if len(items) < REQUIRED_NUM_SNAPSHOTS:
        raise InsufficientTemporalImageryException("No valid image stack (t-2, t-1, t)")

    return items


def crop_main_data(
    items: list[Sentinel2L1CItem],
    abs_client: BlobServiceClient,
    s3_client: S3Client,
    lat: float,
    lon: float,
    crop_size: int,
    main_idx: int = 0,
) -> dict:
    """Process main and reference items, including cloud predictions."""
    main_item = items[main_idx]
    create_bands_and_omnicloud(main_item, abs_client, s3_client)

    cropped_main_data = load_cropped_s2_items(
        items=[main_item], bands=[*S2_BANDS, "OmniCloud"], lat=lat, lon=lon, image_size=crop_size, abs_client=abs_client
    )[0]

    return cropped_main_data


def crop_main_data_landsat(
    items: list[LandsatGranuleAccess],
    abs_client: BlobServiceClient,
    s3_client: S3Client,
    lat: float,
    lon: float,
    crop_size: int,
    main_idx: int = 0,
) -> dict:
    """Process main item."""
    main_item = items[main_idx]
    main_item.prefetch_l1(s3_client, abs_client)

    cropped_main_data = load_cropped_landsat_items(
        items=[main_item], bands=LANDSAT_BANDS, lat=lat, lon=lon, image_size=crop_size, abs_client=abs_client
    )[0]

    return cropped_main_data


def crop_reference_data(
    items: list[Sentinel2L1CItem],
    main_data: dict,
    abs_client: BlobServiceClient,
    s3_client: S3Client,
    lat: float,
    lon: float,
    crop_size: int,
    required_num_previous_snapshots: int = REQUIRED_NUM_PREVIOUS_SNAPSHOTS,
    max_bad_pixel_perc: float = 5.0,
) -> list[dict]:
    """Process main and reference items, including cloud predictions."""
    main_item = items[0]
    reference_data: list = []
    for item in items[1:]:
        if is_duplicate_item(item, [main_item] + [r["tile_item"] for r in reference_data]):
            continue

        create_bands_and_omnicloud(item, abs_client, s3_client)

        cropped_ref = check_and_crop_reference(item, abs_client, lat, lon, crop_size, main_data, max_bad_pixel_perc)
        if cropped_ref:
            reference_data.append(cropped_ref)
        if len(reference_data) == required_num_previous_snapshots:
            break

    return reference_data


def crop_reference_data_landsat(  # noqa
    items: list[LandsatGranuleAccess],
    main_data: dict,
    abs_client: BlobServiceClient,
    s3_client: S3Client,
    lat: float,
    lon: float,
    crop_size: int,
    required_num_previous_snapshots: int = REQUIRED_NUM_PREVIOUS_SNAPSHOTS,
    max_bad_pixel_perc: float = 5.0,
    item_meta_dict: dict = {},  # noqa
) -> list[dict]:
    """Process reference items."""
    reference_data: list = []
    main_item = items[0]
    for item in items[1:]:
        item.prefetch_l1(s3_client, abs_client)

        cropped_ref = check_and_crop_reference_landsat(
            main_item, item, abs_client, lat, lon, crop_size, main_data, max_bad_pixel_perc, item_meta_dict
        )
        if cropped_ref:
            reference_data.append(cropped_ref)
        if len(reference_data) == required_num_previous_snapshots:
            break

    return reference_data


def create_bands_and_omnicloud(item: Sentinel2L1CItem, abs_client: BlobServiceClient, s3_client: S3Client) -> None:
    """Process cloud predictions for a single item."""
    item.prefetch_l1c(s3_client, abs_client)
    if not item.check_omnicloud_on_abs(abs_client):
        start = time.time()
        logger.info("Generating OmniCloud prediction")
        img = item.get_bands(
            S2_BANDS, out_height=BAND_RESOLUTIONS["B11"], out_width=BAND_RESOLUTIONS["B11"], abs_client=abs_client
        )
        logger.info(f"Loaded bands in {time.time() - start:.1f}s")
        predict_and_save_omnicloud(img, item, abs_client)


def check_and_crop_reference(
    item: Sentinel2L1CItem,
    abs_client: BlobServiceClient,
    lat: float,
    lon: float,
    crop_size: int,
    main_data: dict,
    max_bad_pixel_perc: float = 5.0,
) -> dict | None:
    """Process a reference item and validate its quality."""
    cropped_ref = load_cropped_s2_items(
        items=[item], bands=[*S2_BANDS, "OmniCloud"], lat=lat, lon=lon, image_size=crop_size, abs_client=abs_client
    )[0]

    clouds_ref, shadows_ref, scl_ref = prepare_clouds_shadows(cropped_ref["crop_arrays"])

    # if px are 0 in the reflectance bands, we consider them as nodata
    reflectance_nodata_ref = (cropped_ref["crop_arrays"][:-5] == 0).sum(axis=0) > 0
    scl_ref[reflectance_nodata_ref] = 0

    scl_main = main_data["crop_arrays"][-5].copy()

    # number of px where the main crop and the reference crop have data
    valid_overlap = float(((scl_ref != 0) & (scl_main != 0)).sum())
    # number of px where the main crop has data
    main_valid = float((scl_main != 0).sum())

    # Of the px where both main and reference crop have data, how much % is cloudy?
    cloud_ratio = 100 * float(clouds_ref.sum()) / (valid_overlap + 0.01)
    # Of the px where both main and reference crop have data, how much % is cloud shadows?
    shadow_ratio = 100 * float(shadows_ref.sum()) / (valid_overlap + 0.01)
    # Of the px where the main crop has data, how much nodata % has the reference crop?
    nodata_perc = 100 * float(((scl_ref == 0) & (scl_main != 0)).sum()) / main_valid

    date = f"{item.id.split('_')[2][:4]}-{item.id.split('_')[2][4:6]}-{item.id.split('_')[2][6:8]}"

    if cloud_ratio + shadow_ratio + nodata_perc < max_bad_pixel_perc:
        logger.info(
            f"Reference {date}: {cloud_ratio:5.1f}% clouds, {shadow_ratio:5.1f}% shadows, "
            f"{nodata_perc:5.1f}% nodata - USE      ({item.id})"
        )
        return cropped_ref
    else:
        logger.info(
            f"Reference {date}: {cloud_ratio:5.1f}% clouds, {shadow_ratio:5.1f}% shadows, "
            f"{nodata_perc:5.1f}% nodata - DONT USE ({item.id})"
        )
        return None


def check_and_crop_reference_landsat(
    main_item: LandsatGranuleAccess,
    item: LandsatGranuleAccess,
    abs_client: BlobServiceClient,
    lat: float,
    lon: float,
    crop_size: int,
    main_data: dict,
    max_bad_pixel_perc: float = 5.0,
    item_meta_dict: dict = {},  # noqa
) -> dict | None:
    """Process a reference item and validate its quality."""
    # We pass the main item first as it is used to anchor the crop parameters.
    # We then discard the main crop item.
    cropped_ref = load_cropped_landsat_items(
        items=[main_item, item],
        bands=LANDSAT_BANDS,
        lat=lat,
        lon=lon,
        image_size=crop_size,
        abs_client=abs_client,
        item_meta_dict=item_meta_dict,
    )[1]  # index 1 is the reference crop

    # Get QA band from reference and main
    qa_main = main_data["crop_arrays"][-1].copy()
    clouds_ref, shadows_ref, qa_ref = prepare_clouds_shadows_landsat(cropped_ref["crop_arrays"])
    # Compute valid pixels considering both QA FILL and reflectance/brightness zeros

    # For main data
    qa_nodata_mask_main = LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], qa_main)
    reflectance_nodata_mask_main = (main_data["crop_arrays"][:-1] == 0).sum(axis=0) > 0
    valid_pixels_main = ~(qa_nodata_mask_main | reflectance_nodata_mask_main)

    # For reference data
    qa_nodata_mask_ref = LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], qa_ref)
    reflectance_nodata_mask_ref = (cropped_ref["crop_arrays"][:-1] == 0).sum(axis=0) > 0
    valid_pixels_ref = ~(qa_nodata_mask_ref | reflectance_nodata_mask_ref)

    valid_pixels_main = ~LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], qa_main)
    valid_pixels_ref = ~LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], qa_ref)

    # number of px where the main crop and the reference crop have data
    valid_and_overlapping_px = (valid_pixels_main & valid_pixels_ref).sum()
    # number of px where the main crop has data
    main_valid_px = float(valid_pixels_main.sum())

    # Of the px where both main and reference crop have data, how much % is cloudy?
    cloud_ratio = 100 * float(clouds_ref.sum()) / (valid_and_overlapping_px + 0.01)
    # Of the px where both main and reference crop have data, how much % is cloud shadows?
    shadow_ratio = 100 * float(shadows_ref.sum()) / (valid_and_overlapping_px + 0.01)
    # Of the px where the main crop has data, how much nodata % has the reference crop?
    nodata_mask_ref = ~valid_pixels_ref & valid_pixels_main
    nodata_perc = 100 * nodata_mask_ref.sum() / (main_valid_px + 0.01)

    date = LandsatGranuleAccess.parse_landsat_tile_id(item.id)["acquisition_date"]
    date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    if cloud_ratio + shadow_ratio + nodata_perc < max_bad_pixel_perc:
        logger.info(
            f"Reference {date}: {cloud_ratio:5.1f}% clouds, {shadow_ratio:5.1f}% shadows, "
            f"{nodata_perc:5.1f}% nodata - USE      ({item.id})"
        )
        return cropped_ref
    else:
        logger.info(
            f"Reference {date}: {cloud_ratio:5.1f}% clouds, {shadow_ratio:5.1f}% shadows, "
            f"{nodata_perc:5.1f}% nodata - DONT USE ({item.id})"
        )
        return None


def prepare_clouds_shadows(
    crop_array: np.ndarray, omnicloud_cloud_t: int = 35, omnicloud_shadow_t: int = 30
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum up thin/thick omnicloud predictions, threshold and combine with SCL clouds."""
    probs = crop_array[-4:].copy()
    scl = crop_array[-5].copy()
    probs = probs.astype(np.float32)
    probs[probs == OMNICLOUD_NODATA] = np.nan

    # Sum thin and thick cloud probabilities together to threshold the sum, instead of the separate classes
    probs_3cls = probs[:3, :, :].copy()
    probs_3cls[1, :, :] = probs_3cls[1, :, :] + probs_3cls[2, :, :]  # Add thin + thick probabilities
    probs_3cls[2, :, :] = probs[3, :, :].copy()

    # threshold the sum of thin+thick cloud probabilities
    clouds_omni = (probs_3cls[1, :, :] > omnicloud_cloud_t).astype(np.uint8)
    # Set clouds as Omniclouds OR Thick/Thin SCL Clouds to catch some Omnicloud FNs
    clouds_scl_thick_thin = (
        (scl == SCLabel.CLOUD_MEDIUM_PROBABILITY.value) | (scl == SCLabel.CLOUD_HIGH_PROBABILITY.value)
    ).astype(np.uint8)
    clouds_combined = ((clouds_omni == 1) | (clouds_scl_thick_thin == 1)).astype(np.uint8)
    # If the cloud shadow threshold is reached and it's not a cloud, classify as cloud shadow
    shadows_combined = ((probs_3cls[2, :, :] > omnicloud_shadow_t) & (clouds_combined != 1)).astype(np.uint8)
    return clouds_combined, shadows_combined, scl


def prepare_clouds_shadows_landsat(
    crop_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract clouds and cloud shadows from Landsat QA_PIXEL band.

    Uses the same logic as in LandsatTile.prepare_clouds_shadows().
    """
    qa = crop_array[-1].copy()

    qa_nodata_mask = LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], qa)
    reflectance_nodata_mask = (crop_array[:-1] == 0).sum(axis=0) > 0
    nodata_mask = qa_nodata_mask | reflectance_nodata_mask

    cloud_labels = [
        LandsatQAValues.CLOUD,
        LandsatQAValues.DILATED_CLOUD,
        LandsatQAValues.CIRRUS,
        LandsatQAValues.CLOUD_CONFIDENCE_MEDIUM,
        LandsatQAValues.CLOUD_CONFIDENCE_HIGH,
        LandsatQAValues.CIRRUS_CONFIDENCE_MEDIUM,
        LandsatQAValues.CIRRUS_CONFIDENCE_HIGH,
    ]
    cloud_mask = LandsatGranuleAccess.get_mask_from_qa_pixel(cloud_labels, qa).astype(np.uint8)

    shadow_labels = [
        LandsatQAValues.CLOUD_SHADOW,
        LandsatQAValues.CLOUD_SHADOW_CONFIDENCE_MEDIUM,
        LandsatQAValues.CLOUD_SHADOW_CONFIDENCE_HIGH,
    ]
    shadow_mask = LandsatGranuleAccess.get_mask_from_qa_pixel(shadow_labels, qa).astype(np.uint8)
    shadow_mask[cloud_mask == 1] = 0  # deactivate cloud flags, avoid double counting clouds as shadows

    nodata_mask = LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], qa)
    cloud_mask[nodata_mask] = 0  # deactivate nodata flags
    shadow_mask[nodata_mask] = 0  # deactivate nodata flags

    return cloud_mask, shadow_mask, qa


def find_nearest_emit_pixel(glt_ds: xr.Dataset, target_lon: float, target_lat: float) -> dict[str, int]:
    """Find the pixel indices closest to the target coordinates.

    Args:
        glt_ds: GLT Dataset containing 'lon' and 'lat' variables with dimensions (downtrack, crosstrack)
        target_lon: Target longitude
        target_lat: Target latitude

    Returns
    -------
        Dictionary with 'downtrack' and 'crosstrack' integer indices of the nearest point
    """
    dist_squared = (glt_ds.lon - target_lon) ** 2 + (glt_ds.lat - target_lat) ** 2

    min_idx = dist_squared.argmin(dim=["downtrack", "crosstrack"])

    return {"downtrack": min_idx["downtrack"].item(), "crosstrack": min_idx["crosstrack"].item()}


def load_and_crop_emit_image_for_point(
    lat: float,
    lon: float,
    query_datetime: datetime,
    crop_size: int,
    ml_client: MLClient,
    cache_container_name: str,
) -> list[dict[str, Any]]:
    """
    Query EMIT data and return the cropped image for the given lat, lon, date.

    Args:
        lat: The latitude of the point to query
        lon: The longitude of the point to query
        query_datetime: The time used for the query. The search will query between this timestamp and 24 hours after it.
        crop_size: The size of the crop in pixels
        ml_client: The Azure ML client

    Returns
    -------
        Dictionary containing:
        - crop_arrays: Concatenated array of cropped radiance and observation data
        - crop_params: Parameters used for cropping
        - tile_item: EmitGranuleAccess used for cropping
    """
    # Parse date to datetime for EMIT query
    start_date = query_datetime
    end_date = start_date + timedelta(days=1)

    # Query EMIT catalog for this location and date
    emit_ids = query_emit_catalog(lat, lon, start_date, end_date)

    if not emit_ids:
        raise MissingImageException(f"Location {lat}, {lon} has no valid EMIT imagery for {query_datetime}")

    # Use the first available scene
    emit_item = EmitGranuleAccess(emit_ids[0])

    cache_blob_uri = f"{EMIT_CACHE_PREFIX}/{emit_item.emit_id}/{lat}_{lon}_{crop_size}.joblib.gz"

    # NOTE: we initialize the blob service client every time here, rather than once at a higher
    # level, as we get token expiration issues after one hour of runtime that were causing problems.
    # An alternative solution to recreating the ABS client every time should be reviewed.
    abs_client = initialize_blob_service_client(ml_client)
    container_client = abs_client.get_container_client(cache_container_name)

    try:
        # Get from cache
        crop_dict = _get_crop_from_cache(container_client, cache_blob_uri)
    except azure.core.exceptions.ResourceNotFoundError:
        # Get from NASA and cache
        crop_dict = _load_and_crop_emit_image_for_point_no_cache(emit_item, lat, lon, crop_size)
        _cache_crop_to_abs(crop_dict, cache_blob_uri, container_client)

    # list of length 1 to match the output of load_and_crop_sentinel_images_for_point
    return [crop_dict]


def _load_and_crop_emit_image_for_point_no_cache(
    emit_item: EmitGranuleAccess,
    lat: float,
    lon: float,
    crop_size: int,
) -> dict[str, Any]:
    """
    Retrieve the EMIT crop data without caching.

    This is used to read down EMIT data over a cropping location and does not rely on caching.
    It is used by load_and_crop_emit_image_for_point() in case the crop is not already cached.

    Args:
        emit_item: The EMIT granule of interest.
        lat: The latitude of the point to query
        lon: The longitude of the point to query
        crop_size: The size of the crop in pixels

    Returns
    -------
        Dictionary containing:
        - crop_arrays: Concatenated array of cropped radiance and observation data
        - crop_params: Parameters used for cropping
        - tile_item: EmitGranuleAccess used for cropping

    Raises
    ------
        MissingImageException: If no EMIT scenes are found for the given location and date
        InsufficientCoverageException: If the location is outside the scene bounds
    """
    radiance = emit_item.get_radiance()

    # Get observation data (contains angles and other metadata)
    obs = emit_item.get_obs()

    glt_ds = emit_item.get_glt()

    # Get transform and CRS from dataset attributes
    # transform_tuple = radiance.attrs["geotransform"]
    # Convert GDAL transform to rasterio transform order
    # transform = rasterio.Affine.from_gdal(*transform_tuple)
    # crs = radiance.attrs["spatial_ref"]

    # Validate point is inside image bounds
    bounds = {
        "left": radiance.attrs["westernmost_longitude"],
        "right": radiance.attrs["easternmost_longitude"],
        "bottom": radiance.attrs["southernmost_latitude"],
        "top": radiance.attrs["northernmost_latitude"],
    }

    if not (bounds["left"] < lon < bounds["right"]):
        raise InsufficientCoverageException(
            f"Longitude {lon} is outside image bounds {bounds['left']}-{bounds['right']} (granule {emit_item.emit_id})"
        )
    if not (bounds["bottom"] < lat < bounds["top"]):
        raise InsufficientCoverageException(
            f"Latitude {lat} is outside image bounds {bounds['bottom']}-{bounds['top']} (granule {emit_item.emit_id})"
        )

    nearest_index = find_nearest_emit_pixel(glt_ds, lon, lat)
    # We will create a convention that y=downtrack and x=crosstrack
    px, py = nearest_index["crosstrack"], nearest_index["downtrack"]
    # If the nearest index is right on the edge, probably something went wrong
    # (although theoretically it's not impossible)
    if not (px > 0 and px < radiance.sizes["crosstrack"] - 1):
        raise InsufficientCoverageException(f"Pixel x {px} is an edge pixel (granule {emit_item.emit_id})")
    if not (py > 0 and py < radiance.sizes["downtrack"] - 1):
        raise InsufficientCoverageException(f"Pixel y {py} is an edge pixel (granule {emit_item.emit_id})")

    height, width = (radiance.sizes["downtrack"], radiance.sizes["crosstrack"])
    half_size = crop_size // 2

    # Calculate crop bounds ensuring they're within raster dimensions
    crop_start_x = int(max(0, min(px - half_size, width - crop_size)))
    crop_start_y = int(max(0, min(py - half_size, height - crop_size)))

    crop_params = {
        "crop_start_x": crop_start_x,
        "crop_start_y": crop_start_y,
        "crop_height": crop_size,
        "crop_width": crop_size,
        "out_height": crop_size,
        "out_width": crop_size,
    }

    # Extract crop coordinates
    x_start = int(crop_params["crop_start_x"])
    y_start = int(crop_params["crop_start_y"])
    height = int(crop_params["crop_height"])
    width = int(crop_params["crop_width"])

    # For the data arrays below:
    # We need to load our xarrays or we will keep references to the files (?), or in any
    # case if we don't do this the cache files are 10x the expected volume and our returned
    # data are potentially views (ready to be lazy loaded) rather than pre-loaded data arrays.

    # Crop radiance data - shape should be (bands, height, width)
    cropped_radiance = (
        radiance.isel(
            downtrack=slice(y_start, y_start + height),
            crosstrack=slice(x_start, x_start + width),
        )
        .transpose("bands", "downtrack", "crosstrack")
        .load()
    )

    # Crop observation data - shape should be (params, height, width)
    cropped_obs = obs.isel(
        bands=slice(None),  # all observation parameters
        downtrack=slice(y_start, y_start + height),
        crosstrack=slice(x_start, x_start + width),
    ).load()

    # Create a dictionary of the cropped arrays instead of concatenating
    cropped_data = {"radiance": cropped_radiance, "mask": cropped_obs}
    crop_dict = {"crop_arrays": cropped_data, "crop_params": crop_params, "tile_item": emit_item}

    return crop_dict


def prepare_data_item(cropped_data: list[dict[str, Any]], crop_size: int) -> MonoTemporalPlumesDataItem | None:
    """
    Prepare data item from EMIT satellite data with NO plumes added.

    Args:
        cropped_data: List of dictionaries containing cropped data and crop parameters
        crop_size: Size of the crop in pixels

    Returns
    -------
        MonoTemporalPlumesDataItem or None if the data is invalid
    """
    # Extract crop parameters from EMIT data
    cropped_dict = cropped_data[0]  # EMIT only has one item
    crop_params = cropped_dict["crop_params"]
    crop_x = int(crop_params["crop_start_x"])
    crop_y = int(crop_params["crop_start_y"])

    emit_item = cropped_dict["tile_item"]
    rad_crop = cropped_dict["crop_arrays"]["radiance"]
    mask_crop = cropped_dict["crop_arrays"]["mask"]

    # Create Dataset directly from the DataArrays
    modified_crop = rad_crop
    mask = mask_crop[0]  # Using first mask layer

    data_item = MonoTemporalPlumesDataItem.create_data_item(
        modified_crop=modified_crop,
        target=np.zeros((crop_size, crop_size)),  # No target for inference
        mask=mask,
        granule_id=emit_item.emit_id,
        plume_files=[],
        plume_emissions=[],
        bands=list(range(rad_crop.shape[0])),  # Using band indices as names
        size=crop_size,
        crop_x=crop_x,
        crop_y=crop_y,
        main_cloud_ratio=0.0,  # Not used for inference
        transformation_params={},
    )

    if data_item is None:
        raise InsufficientImageryException("No valid image stack for model to run.")

    return data_item


def prepare_model_input(
    lat: float,
    lon: float,
    query_datetime: datetime,
    crop_size: int,
    ml_client: MLClient,
    cache_container_name: str = "",
) -> tuple[list[dict[str, Any]], MonoTemporalPlumesDataItem]:
    """
    Load satellite time series data, crop it, and prepare the data item.

    Raises
    ------
        InsufficientImageryException: If no valid satellite imagery is found
    """
    cropped_data = load_and_crop_emit_image_for_point(
        lat=lat,
        lon=lon,
        query_datetime=query_datetime,
        crop_size=crop_size,
        ml_client=ml_client,
        cache_container_name=cache_container_name,
    )

    try:
        data_item = prepare_data_item(cropped_data=cropped_data, crop_size=crop_size)
    except InsufficientImageryException as err:
        raise InsufficientImageryException(
            f"Location {lat}, {lon} for {query_datetime}: no valid image stack for model to run."
        ) from err

    return cropped_data, data_item


####################################################
################# INFERENCE LOGIC ##################
####################################################


def predict(
    model: Module,
    device: torch.device | str,
    band_extractor: BaseBandExtractor,
    data_item: MonoTemporalPlumesDataItem,
    lossFn: TwoPartLoss,
) -> dict:
    """Process a data_item into the correct shape and applies the model to generate predictions."""
    x_dict = {
        "crop_main": torch.Tensor(data_item.crop_main[np.newaxis, ...]),
    }
    for snap in band_extractor.snapshots:
        x_dict[snap] = torch.Tensor(getattr(data_item, snap)[np.newaxis, ...])
    y = torch.Tensor(data_item.target)
    x, y = band_extractor((x_dict, y))
    x = x.to(device)

    model.eval()
    with torch.no_grad():
        pred = model(x)

    marginal_pred, binary_probability, conditional_pred, _ = lossFn.get_prediction_parts(pred)

    # Apply squeeze and assert that the result is 2-dimensional
    marginal_pred_squeezed = marginal_pred.squeeze()
    binary_probability_squeezed = binary_probability.squeeze()
    conditional_pred_squeezed = conditional_pred.squeeze()

    expected_dim = 2
    assert (
        marginal_pred_squeezed.dim() == expected_dim
    ), f"Expected marginal_pred to be 2D when not using batch processing, got {marginal_pred_squeezed.dim()}D"
    assert (
        binary_probability_squeezed.dim() == expected_dim
    ), f"Expected binary_probability to be 2D when not using batch processing, got {binary_probability_squeezed.dim()}D"
    assert (
        conditional_pred_squeezed.dim() == expected_dim
    ), f"Expected conditional_pred to be 2D when not using batch processing, got {conditional_pred_squeezed.dim()}D"

    # Return unsqueezed since subsequent metrics computation require 4D. Can squeeze elsewhere when necessary
    return {
        "x_dict": x_dict,
        "binary_probability": binary_probability,
        "conditional_pred": conditional_pred,
        "marginal_pred": marginal_pred,
    }


def generate_predictions(
    main_data: dict,
    reference_data: list[dict],
    model: nn.Module,
    device: torch.device | str | int,
    band_extractor: BaseBandExtractor,
    lossFn: TwoPartLoss,
) -> dict:
    """Generate model predictions for the main and reference images."""
    x_dict = {
        "crop_main": torch.Tensor(main_data["crop_arrays"][np.newaxis, ...]),
        "crop_before": torch.Tensor(reference_data[0]["crop_arrays"][np.newaxis, ...]),
        "crop_earlier": torch.Tensor(reference_data[1]["crop_arrays"][np.newaxis, ...]),
    }
    y = torch.zeros_like(x_dict["crop_main"][:, :1])
    x, y = band_extractor((x_dict, y))
    x = x.to(device)

    with torch.no_grad():
        pred_ = model(x)
    marginal_pred, binary_prob, conditional_pred, _ = lossFn.get_prediction_parts(pred_)

    return {
        "x_dict": x_dict,
        "binary_probability": binary_prob.squeeze().cpu(),
        "marginal_pred": marginal_pred.squeeze().cpu(),
        "conditional_pred": conditional_pred.squeeze().cpu(),
    }


####################################################
############# VISUALIZATION FUNCTIONS #############
####################################################


def plot_main_results(
    row: pd.Series,
    swir_ratio_main: xr.DataArray,
    rgb_main: xr.DataArray,
    preds: dict,
    basename_prefix: str,
    basename: str,
    binary_threshold: float,
    azure_cluster: bool,
    center_plumes: list,
) -> None:
    """Plot main results including SWIR ratio, predictions, and RGB."""
    fig = plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.title("SWIR ratio")
    plt.imshow(swir_ratio_main, interpolation="nearest", cmap="pink")
    grid16()
    plt.colorbar()

    plt.subplot(2, 2, 2)
    if len(center_plumes) > 0:
        biggest_plume = center_plumes[0]
        q = np.round(biggest_plume["properties"]["Q"], 2)
        q_low = np.round(biggest_plume["properties"]["Q_low"], 2)
        q_high = np.round(biggest_plume["properties"]["Q_high"], 2)
        plt.title(
            f"Expected Q: {q:.0f} kg/h ({q_low:.0f} - {q_high:.0f})\nMarg. Prediction, "
            f"Sum: {preds['marginal_pred'].sum():.1f}"
        )
    else:
        plt.title(f"Nothing detected in the center\nMarg. Prediction, Sum: {preds['marginal_pred'].sum():.1f}")
    plot_frac(preds["marginal_pred"])
    grid16()
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title(f"Binary prediction, Sum: {preds['binary_probability'].sum():.0f}")
    plt.imshow(preds["binary_probability"] * 10, vmin=0.0, vmax=10.0, cmap="pink_r", interpolation="nearest")
    grid16()
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title("RGB with source and prediction")
    plt.imshow(rgb_main / rgb_main.max(), interpolation="nearest")
    preds_mask = np.abs(preds["marginal_pred"]) > binary_threshold
    plt.imshow(np.ma.masked_where(preds_mask == 0, preds_mask), vmin=0.0, vmax=1.0, interpolation="nearest")
    grid16()

    # Add a dot in the center of the plot
    plt.scatter(rgb_main.shape[1] // 2, rgb_main.shape[0] // 2, color="green", marker="x")

    fig.suptitle(
        f"{row['site']} ({row['lat']}, {row['lon']})\n"
        f"Q: {row['quantification_kg_h']:.0f} kg/h ({row['source']})\n"
        f"{row['date']}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    if azure_cluster:
        mlflow.log_figure(fig, f"{basename_prefix}/{basename}.png")
        plt.close()
    else:
        plt.show()


def plot_rgb_history(  # noqa
    rgb_main: xr.DataArray,
    crop_before: xr.DataArray,
    crop_earlier: xr.DataArray,
    date_main: str,
    date_before: str,
    date_earlier: str,
    satellite_id: SatelliteID,
    basename_prefix: str,
    basename: str,
    azure_cluster: bool,
) -> None:
    """Plot RGB history of earlier, before and main chips."""
    rgb_main_max = rgb_main.max()

    f, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].set_title(f"Earlier = {date_earlier}", fontsize=16)
    rgb_image_earlier = get_rgb_from_xarray(crop_earlier, satellite_id)
    ax[0].imshow((rgb_image_earlier / rgb_main_max).clip(0, 1), interpolation="nearest")

    ax[1].set_title(f"Before  = {date_before}", fontsize=16)
    rgb_image_before = get_rgb_from_xarray(crop_before, satellite_id)
    ax[1].imshow((rgb_image_before / rgb_main_max).clip(0, 1), interpolation="nearest")

    ax[2].set_title(f"Main  = {date_main}", fontsize=16)
    ax[2].imshow((rgb_main / rgb_main_max).clip(0, 1), interpolation="nearest")
    plt.tight_layout()
    if azure_cluster:
        mlflow.log_figure(f, f"{basename_prefix}/{basename}_RGB_history.png")
        plt.close()
    else:
        plt.show()


def plot_ratio_history(
    swir_ratio_main: xr.DataArray,
    swir_ratio_before: xr.DataArray,
    swir_ratio_earlier: xr.DataArray,
    date_main: str,
    date_before: str,
    date_earlier: str,
    basename_prefix: str,
    basename: str,
    azure_cluster: bool,
) -> None:
    """Plot the B12/B11 ratio history of earlier, before and main chips."""
    fontsize = 15
    f, ax = plt.subplots(1, 3, figsize=(12, 4))

    swir_ratio_main_min = swir_ratio_main.min()
    swir_ratio_main_max = swir_ratio_main.max()

    ax[0].set_title(f"Ratio Earlier = {date_earlier}", fontsize=fontsize)
    im = ax[0].imshow(
        swir_ratio_earlier, vmin=swir_ratio_main_min, vmax=swir_ratio_main_max, interpolation="nearest", cmap="pink"
    )
    grid16(ax[0])
    plt.colorbar(im, ax=ax[0])
    ax[1].set_title(f"Ratio Before = {date_before}", fontsize=fontsize)
    im = ax[1].imshow(
        swir_ratio_before, vmin=swir_ratio_main_min, vmax=swir_ratio_main_max, interpolation="nearest", cmap="pink"
    )
    grid16(ax[1])
    plt.colorbar(im, ax=ax[1])

    ax[2].set_title(f"Ratio Main = {date_main}", fontsize=fontsize)
    im = ax[2].imshow(
        swir_ratio_main, vmin=swir_ratio_main_min, vmax=swir_ratio_main_max, interpolation="nearest", cmap="pink"
    )
    grid16(ax[2])
    plt.colorbar(im, ax=ax[2])
    plt.tight_layout()
    if azure_cluster:
        mlflow.log_figure(f, f"{basename_prefix}/{basename}_Ratio_history.png")
        plt.close()
    else:
        plt.show()


def plot_ratio_diff_history(
    swir_ratio_main: xr.DataArray,
    swir_ratio_before: xr.DataArray,
    swir_ratio_earlier: xr.DataArray,
    basename_prefix: str,
    basename: str,
    azure_cluster: bool,
    vmin_vmax: float = 0.03,
) -> None:
    """Plot the B12/B11 SWIR Difference (mean centered) between main and earlier/before/avg of both."""
    fontsize = 14

    ratio_diff_avg = swir_ratio_main - (swir_ratio_before + swir_ratio_earlier) / 2
    ratio_diff_before = swir_ratio_main - swir_ratio_before
    ratio_diff_earlier = swir_ratio_main - swir_ratio_earlier

    # Remove systematic shifts in ratio by subtracting mean
    ratio_diff_avg = ratio_diff_avg - np.nanmean(ratio_diff_avg)
    ratio_diff_before = ratio_diff_before - np.nanmean(ratio_diff_before)
    ratio_diff_earlier = ratio_diff_earlier - np.nanmean(ratio_diff_earlier)

    f, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].set_title(
        f"Ratio Diff (main - earlier)\nMin {ratio_diff_earlier.min():.3f}",
        fontsize=fontsize,
    )
    ax[0].imshow(ratio_diff_earlier, vmin=-vmin_vmax, vmax=vmin_vmax, cmap="RdBu", interpolation="none")
    grid16(ax[0])

    ax[1].set_title(
        f"Ratio Diff (main - before)\nMin {ratio_diff_before.min():.3f}",
        fontsize=fontsize,
    )
    ax[1].imshow(ratio_diff_before, vmin=-vmin_vmax, vmax=vmin_vmax, cmap="RdBu", interpolation="none")
    grid16(ax[1])

    ax[2].set_title(f"Ratio Diff (main - avg of both)\nMin {ratio_diff_avg.min():.3f}", fontsize=fontsize)
    im = ax[2].imshow(ratio_diff_avg, vmin=-vmin_vmax, vmax=vmin_vmax, cmap="RdBu", interpolation="none")
    grid16(ax[2])
    plt.colorbar(im, ax=ax[2])
    plt.tight_layout()
    if azure_cluster:
        mlflow.log_figure(f, f"{basename_prefix}/{basename}_Ratio_Diff_norm.png")
        plt.close()
    else:
        plt.show()


####################################################
################ HELPER FUNCTIONS ##################
####################################################


def is_duplicate_item(item: Sentinel2L1CItem, existing_items: list[Sentinel2L1CItem]) -> bool:
    """Check if an item is a duplicate based on ID without processing time."""
    item_id_base = "_".join(item.id.split("_")[:-1])
    for existing_item in existing_items:
        existing_id_base = "_".join(existing_item.id.split("_")[:-1])
        if item_id_base == existing_id_base:
            logger.info(f"Duplicate found: {item.id} matches {existing_item.id}")
            return True
    return False


def cloud_ratio(scl: np.ndarray) -> float:
    """Calculate the mean per pixel cloud ratio from the SCL band."""
    crop_mask = Sentinel2Item.get_mask_from_scmap(
        labels=SC_MASK_LABELS,
        classification_map=scl,
    )
    cloud_ratio = crop_mask.mean()
    return cloud_ratio


def _get_crop_from_cache(container_client: ContainerClient, blob_name: str) -> dict:
    """Download and load a cached crop from ABS."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "temp.npy"

        # Download blob to temporary file
        with open(temp_path, "wb") as temp_file:
            blob_data = container_client.download_blob(blob_name).readall()
            temp_file.write(blob_data)

        with open(temp_path, "rb") as fs:
            result = joblib.load(fs)

        return result


def _cache_crop_to_abs(crop: dict, blob_name: str, container_client: ContainerClient) -> None:
    """Cache a cropped array to ABS."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "temp.npy"

        # Save array to temporary file
        with open(temp_path, "wb") as fs:
            # protocol -1 recommended by xarray:
            # https://docs.xarray.dev/en/stable/user-guide/io.html#pickle
            joblib.dump(crop, fs, protocol=-1)

        # Upload to ABS cache
        with open(temp_path, "rb") as data:
            container_client.upload_blob(
                name=blob_name,
                data=data.read(),
                validate_content=True,
                overwrite=True,
            )


def predict_and_save_omnicloud(img: npt.NDArray, item: Sentinel2L1CItem, abs_client: BlobServiceClient) -> npt.NDArray:
    """Predict and upload OmniCloud 4 class (clear view/thick/thin clouds/cloud shadows) prediction to ABS."""
    from omnicloudmask import predict_from_array

    start = time.time()
    probs = predict_from_array(
        input_array=img[
            [
                S2_BANDS.index("B04"),
                S2_BANDS.index("B03"),
                S2_BANDS.index("B8A"),
            ]
        ],  # Red=B04, Green=B03 and NIR=B8A
        patch_size=1700,
        patch_overlap=300,
        batch_size=1,
        inference_device=torch.device("cuda"),
        export_confidence=True,
        softmax_output=True,
    )
    logger.info(f"Predicting with OmniCloud took {time.time() - start:.1f}s")
    # Save 4 classes probabilities scaled to 0-100 and as uint8 to save space
    # set nan values to 255, outside of normal 0-100 values
    probs[np.isnan(probs)] = OMNICLOUD_NODATA
    # Scale all probabilities to 0-100 --> round them
    probs[probs != OMNICLOUD_NODATA] = np.round(probs[probs != OMNICLOUD_NODATA] * 100, 0)
    # Use the much smaller dtype uint8
    probs = probs.astype(np.uint8)

    with rasterio.open(item.item.assets["B8A"].href) as ds:
        profile = ds.profile
    profile["nodata"] = OMNICLOUD_NODATA
    profile["dtype"] = "uint8"
    # 4 classes = clear view/thick/thin clouds/cloud shadows
    profile["count"] = 4

    omni_local_path = "OmniCloud.tif"
    with rasterio.open(omni_local_path, "w", **profile) as dst:
        dst.write(probs)

    item.transfer_omnicloud_to_abs(omni_local_path, abs_client)
    return probs
