"""Compute evaluation metrics for a trained model on our ground truth data."""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from affine import Affine
from azure.ai.ml import MLClient
from mypy_boto3_s3 import S3Client
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, calculate_default_transform, reproject
from torch.nn import Module
from tqdm import tqdm

from src.azure_wrap.blob_storage_sdk_v2 import download_from_blob
from src.azure_wrap.ml_client_utils import get_azureml_uri, make_acceptable_uri
from src.data.sentinel2 import (
    SceneClassificationLabel as SCLabel,
)
from src.inference.inference_functions import predict, prepare_model_input
from src.training.loss_functions import TwoPartLoss
from src.training.transformations import BaseBandExtractor
from src.utils.exceptions import InsufficientImageryException
from src.utils.parameters import MAIN_IDX, SatelliteID
from src.validation.metrics import Metrics


# # FIXME: implement metrics calculation for S2 using new chip selection
def compute_metrics_for_ground_truth_data(  # noqa: PLR0913 (too-many-arguments)
    ml_client: MLClient,
    s3_client: S3Client,
    ground_truth_dataset: str,
    model: Module,
    band_extractor: BaseBandExtractor,
    model_training_params: dict,
    crop_size: int,
    lossFn: TwoPartLoss,
    satellite_id: SatelliteID,
    device: int | torch.device,
) -> pd.DataFrame:
    """
    Compute metrics for each entry in the ground truth dataset and output results to a CSV.

    If l1c = True (default) use L1C data, otherwise use L2A data.
    """
    ground_truth_plumes = pd.read_csv(
        ground_truth_dataset, dtype={"lat": str, "lon": str}
    )  # to avoid any rounding in lat, lon

    sc_mask_labels = [
        SCLabel.NO_DATA,
        SCLabel.CLOUD_HIGH_PROBABILITY,
        SCLabel.CLOUD_MEDIUM_PROBABILITY,
    ]

    probability_threshold = model_training_params["probability_threshold"]

    # Initialize overall metrics
    overall_metrics = Metrics(lossFn.binary_threshold, probability_threshold, manual_target_mask=True)
    metrics_list = []

    for _, row in tqdm(ground_truth_plumes.iterrows(), total=len(ground_truth_plumes), desc="Ground truth plumes"):
        # Download mask from Azure and skip if missing
        try:
            mask_filepath = download_mask_from_azure(ml_client, row)
        except Exception as e:
            print(f"No mask for {row['lat']}, {row['lon']}, {row['date']}. Skipping. Error: {e}")
            continue

        # Prepare  model input data
        lat_float = float(row["lat"])
        lon_float = float(row["lon"])
        try:
            cropped_data, data_item = prepare_model_input(
                lat_float,
                lon_float,
                datetime.datetime.fromisoformat(row["date"]),
                crop_size,
                sc_mask_labels,
                satellite_id=satellite_id,
                ml_client=ml_client,
                s3_client=s3_client,
            )
        except InsufficientImageryException as err:
            # TODO: should we actually raise an error here?
            # TODO: at least this should use logging instead of print
            print(f"No valid image stack for model to run. Error: {err}")
            continue

        target_b12_crop_params = cropped_data[MAIN_IDX]["crop_params"]["B12"]
        target_crop_b12_metadata = cropped_data[MAIN_IDX]["tile_item"].get_raster_meta("B12")
        target_crop_crs = f"{cropped_data[MAIN_IDX]['tile_item'].crs}"

        # Reproject the mask to the target CRS if needed and validate
        mask_500x500_data, mask_500x500_transform = reproject_if_needed(mask_filepath, target_crop_crs)

        expected_ground_truth_ndim = 2
        assert (
            mask_500x500_data.squeeze().ndim == expected_ground_truth_ndim
        ), "Ground truth mask should be a 2D array, if not something has gone wrong."

        # Perform prediction
        pred = predict(
            model=model, device=device, band_extractor=band_extractor, recycled_item=data_item, lossFn=lossFn
        )

        # Crop ground truth mask to CV crop extent
        ground_truth_mask = get_ground_truth_mask_cv_crop(
            mask_500x500_data, target_crop_b12_metadata, target_b12_crop_params, mask_500x500_transform, crop_size
        )
        # Convert target to torch and ensure on same device as input for metrics calculation
        ground_truth_mask_tensor = torch.from_numpy(ground_truth_mask).to(device)

        # Update metrics
        metrics = Metrics(lossFn.binary_threshold, probability_threshold, manual_target_mask=True)
        metrics.update(
            pred["marginal_pred"],
            pred["binary_probability"],
            ground_truth_mask_tensor.reshape(1, 1, crop_size, crop_size),  # need to reshape to 4D
        )
        overall_metrics += metrics

        # Add the metrics dictionary to the list for CSV output
        row_metrics_dict = metrics.as_dict()
        row_metrics_dict.update({"lat": row["lat"], "lon": row["lon"], "date": row["date"]})
        metrics_list.append(row_metrics_dict)

    # Calculate the overall metrics and add to the list
    overall_metrics_dict = overall_metrics.as_dict()
    overall_metrics_dict.update({"lat": "overall", "lon": "overall", "date": "overall"})
    metrics_list.append(overall_metrics_dict)

    # Convert the list of metrics dictionaries to a DataFrame and log on MLFlow
    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df


####################################################
################ HELPER FUNCTIONS ##################
####################################################


def download_mask_from_azure(ml_client: MLClient, row: pd.Series) -> Path:
    """Return local ground truth mask filepath based on row data."""
    azure_uri_base = "data/ground_truth/ground_truth_masks/"
    filename = f"mask_{row['lat']}_{row['lon']}_{row['date']}.tif"
    azure_uri = make_acceptable_uri(str(get_azureml_uri(ml_client, f"{azure_uri_base}/{filename}")))

    local_dir = Path.home() / "localfiles" / "data" / "ground_truth_masks"
    local_dir = local_dir.expanduser()
    download_from_blob(azure_uri, local_dir, recursive=False)

    return local_dir / filename


def get_ground_truth_mask_cv_crop(
    mask_500x500_data: np.ndarray,
    target_crop_b12_metadata: dict,
    target_crop_params: dict,
    mask_500x500_transform: Affine,
    crop_size: int,
) -> np.ndarray:
    """Get the CV crop size ground truth mask from the original 500x500 binary mask."""
    affine_128x128 = Affine(
        target_crop_b12_metadata["transform"].a,
        target_crop_b12_metadata["transform"].b,
        target_crop_b12_metadata["bounds"].left
        + (target_crop_params["crop_start_x"]) * target_crop_b12_metadata["transform"].a,
        target_crop_b12_metadata["transform"].c,
        target_crop_b12_metadata["transform"].d,
        target_crop_b12_metadata["bounds"].top
        - (target_crop_params["crop_start_y"]) * target_crop_b12_metadata["transform"].a,
    )

    affine_500x500 = mask_500x500_transform

    # Calculate the pixel coordinates in the 500x500 array
    x_offset = (affine_128x128.c - affine_500x500.c) / affine_500x500.a
    y_offset = (affine_500x500.f - affine_128x128.f) / -affine_500x500.e

    # Crop the 500x500 ground truth mask to the CV crop extent
    ground_truth_crop = mask_500x500_data[
        int(y_offset) : int(y_offset) + crop_size, int(x_offset) : int(x_offset) + crop_size
    ]
    return ground_truth_crop


def reproject_if_needed(mask_filepath: Path, target_crs: str) -> tuple[np.ndarray, Affine]:
    """Reproject the ground truth mask to the target CV crop CRS if needed."""
    with rasterio.open(mask_filepath) as src:
        # Check if CRS match
        if src.crs != target_crs:
            print(f"Reprojecting mask from {src.crs} to {target_crs}")
            # Reproject the mask to the target CRS
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "crs": target_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                }
            )

            with MemoryFile() as memfile_out, memfile_out.open(**kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                )
                reprojected_mask_data = dst.read(1)
                reprojected_mask_transform = dst.transform

            return reprojected_mask_data, reprojected_mask_transform
        else:
            return src.read(1), src.transform
