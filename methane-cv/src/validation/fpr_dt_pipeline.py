"""FPR and Detection Threshold Pipeline.

```bash
python src/validation/fpr_dt_pipeline.py model_id=1226 experiment_name=<experiment name>
```
"""

import copy
import json
import logging
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import planetary_computer
import skimage
import torch
import torchvision
from azure.ai.ml import MLClient
from azureml.fsspec import AzureMachineLearningFileSystem
from lib.models.schemas import WatershedParameters
from lib.plume_masking import retrieval_mask_using_watershed_algo
from omegaconf import DictConfig
from pydantic import AnyUrl
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from skimage.morphology import erosion, footprint_rectangle
from torch import nn
from tqdm import tqdm
from urllib3 import Retry

from src.azure_wrap.blob_storage_sdk_v2 import DATASTORE_URI, download_from_blob
from src.azure_wrap.ml_client_utils import (
    create_ml_client_config,
    get_azureml_uri,
    initialize_ml_client,
    make_acceptable_uri,
)
from src.data.common.data_item import get_frac
from src.data.common.sim_plumes import randomly_position_sim_plume
from src.data.sentinel2 import (
    PLANETARY_COMPUTER_CATALOG_URL,
    PLANETARY_COMPUTER_COLLECTION_L2A,
)
from src.data.sentinel2_l1c import Sentinel2L1CItem
from src.plotting.plotting_functions import (
    CMAP,
    S2_LAND_COVER_CLASSIFICATIONS,
    get_band_ratio_from_tensor,
    get_rgb_from_tensor,
)
from src.training.transformations import BaseBandExtractor, UseOriginalB11B12, UseOriginalB11B12_X
from src.utils.parameters import S2_BANDS, SATELLITE_COLUMN_CONFIGS, SatelliteID
from src.utils.profiling import MEASUREMENTS, timer
from src.utils.radtran_utils import RadTranLookupTable
from src.utils.utils import (
    load_model_and_concatenator,
    setup_logging,
)

# Constants
GORRONO_PLUME_EMISSION_RATE: float = 1000.0  # kg/hr (used by Gorroño for sim plumes)
MGRS_DATE_DF_PATH: str = "src/data/tiles/s2/csv_files/2025_03_03_fpr_areas_122_from_h_m_p_valset.csv"
MODEL_NAME: str = "models:/torchgeo_pwr_unet"
REGIONS: list[str] = ["hassi", "permian", "marcellus"]
REGION_COLORS: dict[str, str] = {"hassi": "#1f77b4", "permian": "#ff7f0e", "marcellus": "#2ca02c"}
TARGET_FPRS: list[float] = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6]
MARKER_FLOOR_COMBINATIONS: list[tuple] = [
    (0.25, 0.1),
    (0.25, 0.15),
    (0.3, 0.15),
    (0.3, 0.2),
    (0.35, 0.1),
    (0.35, 0.175),
    (0.35, 0.225),
    (0.35, 0.275),
    (0.4, 0.2),
    (0.4, 0.25),
    (0.4, 0.3),
    (0.45, 0.225),
    (0.45, 0.275),
    (0.45, 0.325),
]
MARKER_THRESHOLDS_WITH_DIFF_FLOORS = sorted(list(set([k[0] for k in MARKER_FLOOR_COMBINATIONS])))
MARKER_THRESHOLDS_COLORS = {0.25: "blue", 0.3: "red", 0.35: "magenta", 0.4: "brown", 0.45: "green"}
MARKER_DISTANCE: int = 1

BAND_FRAC_OFFSET: float = 10.0

# Initialize logging and filesystem
logger: logging.Logger = setup_logging()
fs: AzureMachineLearningFileSystem = AzureMachineLearningFileSystem(DATASTORE_URI)

# Load MGRS tiles data
MGRS_DATE_DF: pd.DataFrame = pd.read_csv(MGRS_DATE_DF_PATH)
TILE_ID_COUNTS_BY_REGION: dict[str, pd.DataFrame] = {
    "hassi": MGRS_DATE_DF[MGRS_DATE_DF["region"] == "hassi"].shape[0],
    "permian": MGRS_DATE_DF[MGRS_DATE_DF["region"] == "permian"].shape[0],
    "marcellus": MGRS_DATE_DF[MGRS_DATE_DF["region"] == "marcellus"].shape[0],
}

# Configure input/output columns
columns_config: dict[str, dict] = copy.deepcopy(SATELLITE_COLUMN_CONFIGS[SatelliteID.S2])
if "orig_swir16" in columns_config:
    columns_config["orig_band_11"] = columns_config["orig_swir16"]
    del columns_config["orig_swir16"]
if "orig_swir22" in columns_config:
    columns_config["orig_band_12"] = columns_config["orig_swir22"]
    del columns_config["orig_swir22"]

target_column: str = "target"
target_config: dict = columns_config.pop(target_column)
target_shape: tuple[int, ...] = target_config["shape"]
target_dtype: torch.dtype = target_config["dtype"]

input_columns: list[str] = []
input_shapes: list[tuple[int, ...]] = []
input_dtypes: list[torch.dtype] = []
for col, config in columns_config.items():
    input_columns.append(col)
    input_shapes.append(config["shape"])
    input_dtypes.append(config["dtype"])

####################################################
################## MAIN FUNCTION ###################
####################################################


@hydra.main(version_base=None, config_path="config", config_name="fpr_dt_config")
def main(config: DictConfig) -> None:
    """Run the complete FPR-DT pipeline for all regions."""
    # Extract configuration parameters
    val_parquet_folder_path: str = config.val_parquet_folder_path
    ncrops: int = config.ncrops
    seed: int = config.seed
    regions: list[str] = REGIONS
    target_fprs: list[float] = TARGET_FPRS
    experiment_name: str | None = config.experiment_name
    model_id: str = str(config.model_id)
    satellite_id: SatelliteID = config.satellite_id
    crop_size: int = config.crop_size
    azure_cluster: bool = config.azure_cluster
    target_spatial_resolution: float = config.target_spatial_resolution
    hapi_data_path: AnyUrl = config.hapi_data_path

    # Calculate area metrics
    if satellite_id == SatelliteID.S2:
        crop_area_km2: float = (crop_size**2 * target_spatial_resolution**2) / 1e6
        tile_area_km2: float = (500 * 500 * 20 * 20) / 1e6
    else:
        raise ValueError(
            f"Currently only Sentinel-2 is supported. Got: {satellite_id}. "
            "Update the script to support other satellites."
        )

    if azure_cluster:
        params: dict[str, Any] = {
            "val_parquet_folder_path": val_parquet_folder_path,
            "MGRS_DATE_DF_PATH": MGRS_DATE_DF_PATH,
            "ncrops": ncrops,
            "seed": seed,
            "target_fprs": target_fprs,
            "experiment_name": experiment_name,
            "model_id": model_id,
            "satellite_id": satellite_id,
            "crop_size": crop_size,
            "hapi_data_path": hapi_data_path,
            "azure_cluster": azure_cluster,
            "regions": regions,
            "MARKER_FLOOR_COMBINATIONS": MARKER_FLOOR_COMBINATIONS,
            "MARKER_DISTANCE": MARKER_DISTANCE,
        }
        mlflow.log_params(params)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device={device}")

    if azure_cluster:
        create_ml_client_config()

    model_identifier: str = f"{MODEL_NAME}/{model_id}"
    model: nn.Module
    band_concatenator: BaseBandExtractor
    training_params: dict
    model, band_concatenator, training_params = load_model_and_concatenator(
        model_identifier, device=device, satellite_id=satellite_id
    )
    model.eval()
    binary_threshold: float = training_params["binary_threshold"]

    ml_client: MLClient = initialize_ml_client(azure_cluster)

    # Extra stuff so we get a flexible plot name
    if experiment_name is None:
        experiment_name = model_id

    plot_name = f"Model {model_id}" if experiment_name == model_id else f"Model {model_id}-{experiment_name}"

    exchange_b11_b12_with_original = UseOriginalB11B12()

    # Define transformation pipeline to reuse original B11/B12 of chips
    transform_orig = torchvision.transforms.Compose(
        [
            exchange_b11_b12_with_original,
            band_concatenator,
        ]
    )

    ########################################
    # False Positive Rate Calculation
    ########################################
    fpr_results: dict[str, dict] = compute_fpr_results(
        val_parquet_folder_path,
        regions,
        model,
        device,
        transform_orig,
        visualize=False,
    )

    thresholds: np.ndarray = np.linspace(0, 1, 42)
    plot_and_save_fpr_curves(
        fpr_results,
        thresholds,
        crop_area_km2,
        tile_area_km2,
        plot_name,
        f"fpr_results_all_regions_{model_id}.json",
        f"fp_px_sums_all_regions_{model_id}.json",
        azure_cluster,
    )

    fpr_to_likeli_threshs: dict[float, float] = compute_fpr_threshold(
        fpr_results,
        thresholds,
        target_fprs,
        crop_area_km2,
        tile_area_km2,
        azure_cluster,
    )

    ########################################
    # Detection Threshold Calculation
    ########################################
    # Load and crop Gorroño plumes
    raw_enhancements: list[np.ndarray] = load_gorrono_plumes(ml_client)
    enhancements_molperm2: list[np.ndarray] = crop_enhancements(raw_enhancements)

    dt_results, dt_results_thresholds = compute_dt_results(
        val_parquet_folder_path,
        regions,
        enhancements_molperm2,
        model,
        band_concatenator,
        binary_threshold,
        ncrops,
        crop_size,
        seed,
        fpr_to_likeli_threshs,
        hapi_data_path,
        device,
        visualize=False,
    )

    plot_detection_curves(dt_results, dt_results_thresholds, plot_name, azure_cluster)

    # Save results and calculate POD metrics
    save_det_prob_results(dt_results, azure_cluster)


####################################################
################# CORE FUNCTIONS ###################
####################################################


@timer(phase="compute_fpr_threshold", accumulator=MEASUREMENTS, verbose=True, logger=logger)
def compute_fpr_threshold(
    results: dict[str, Any],
    thresholds: np.ndarray,
    target_fprs: list[float],
    crop_area_km2: float,
    tile_area_km2: float,
    azure_cluster: bool,
) -> dict[float, float]:
    """Compute likelihood thresholds for target FPRs.

    Args:
        results (Dict[str, Any]): Dictionary containing FPR results per region.
        thresholds (np.ndarray): Array of threshold values to interpolate.
        target_fprs (List[float]): Target false positive rates (e.g., [1, 2, ..., 10]).
        crop_area_km2 (float): Area of a single crop in square kilometers.
        tile_area_km2 (float): Area of a 500x500 px tile in square kilometers.
        azure_cluster (bool): Flag indicating if running on Azure cluster.

    Returns
    -------
        Dict[float, float]: Mapping of target FPR to likelihood thresholds.
    """
    logger.info("=" * 130)
    logger.info("  Calculating likelihood threshold for target FPR")
    logger.info("=" * 130)

    # Initialize lists to store ccdf_counts_per_km2 for each region
    ccdf_counts_per_region: list[list[float]] = []
    # Loop over each region to calculate ccdf_counts_per_km2
    for region in ["hassi", "marcellus", "permian"]:
        threshold_data = results[region]
        max_probs = np.array(threshold_data["max_probs"][MARKER_FLOOR_COMBINATIONS[0]])
        total_area_km2 = threshold_data["total_crops"] * crop_area_km2

        # Calculate ccdf_counts_per_km2 for the current region
        ccdf_counts = [np.sum(max_probs >= threshold) / total_area_km2 for threshold in thresholds]
        ccdf_counts_per_region.append(ccdf_counts)

    # Average the ccdf_counts_per_km2 across the three regions
    ccdf_counts_per_km2: np.ndarray = np.mean(np.array(ccdf_counts_per_region), axis=0)
    # Convert to per-tile counts
    ccdf_counts_per_tile: list[float] = [count * tile_area_km2 for count in ccdf_counts_per_km2]

    # Interpolate to find threshold for target FPR
    likeli_threshs: dict[float, float] = {}
    for target_fpr in target_fprs:
        likeli_thresh = np.round(
            (
                np.interp(
                    target_fpr,
                    ccdf_counts_per_tile[::-1],
                    thresholds[::-1],
                )
                * 10  # Convert to 0-10 scale
            ).item(),
            3,
        )
        logger.info(f"Likelihood threshold for all regions at target FPR {target_fpr:.2f}: {likeli_thresh:.2f}")

        if azure_cluster:
            mlflow.log_metric(f"threshold_fpr_{target_fpr:02.2f}", likeli_thresh)
        likeli_threshs[target_fpr] = likeli_thresh

    return likeli_threshs


@timer(phase="compute_fpr_results", accumulator=MEASUREMENTS, verbose=True, logger=logger)
def compute_fpr_results(  # noqa
    val_parquet_folder_path: str,
    regions: list[str],
    model: nn.Module,
    device: torch.device | str,
    transform_orig: torchvision.transforms.Compose,
    visualize: bool = False,
) -> dict[str, dict[str, Any]]:
    """Compute false positive rates for all regions.

    Args:
        val_parquet_folder_path (str): Path to the validation Parquet folder.
        regions (List[str]): List of region names to process.
        model (nn.Module): Trained PyTorch model for inference.
        device (Union[torch.device, str]): Device to run model on (e.g., 'cpu' or 'cuda').
        transform_orig (torchvision.transforms.Compose): Transformation pipeline.
        visualize (bool, optional): Flag to enable visualization. Defaults to False.

    Returns
    -------
        results: dict[str, dict[str, Any]]: Results dictionary with max probabilities/area sums
            per region per marker/floor threshold
    """
    logger.info("=" * 130)
    logger.info("  Computing False Positive results for all regions")
    logger.info("=" * 130)
    results: dict[str, dict[str, Any]] = defaultdict(dict)
    for region in regions:
        results[region] = {"total_crops": 0, "max_probs": {}, "area_sums": {}}
        for marker_threshold, floor_threshold in MARKER_FLOOR_COMBINATIONS:
            results[region]["max_probs"][(marker_threshold, floor_threshold)] = []
            results[region]["area_sums"][(marker_threshold, floor_threshold)] = []

    for region in tqdm(regions, desc="regions", disable=None):
        region_df = MGRS_DATE_DF[MGRS_DATE_DF["region"] == region].reset_index(drop=True)

        for i, row in region_df.iterrows():
            logger.info(f"FPR {region} TILE ({i + 1:2}/{len(region_df)}): {row['mgrs']} - {row['date']}")

            parquet_path = f"{val_parquet_folder_path}/{row['mgrs']}_{row['date']}.parquet"
            try:
                # Read the full Parquet file into a Pandas DataFrame
                with fs.open(parquet_path, "rb") as f:
                    df_chips = pd.read_parquet(f)
            except Exception as e:
                # Fails if the dataframe does not exist. Sometimes happens e.g. L2C
                logger.error(f"Failed to read Parquet file {parquet_path}: {e}")
                continue
            for _, chip_row in df_chips.iterrows():
                sample = chip_row[input_columns]
                inputs: dict[str, torch.Tensor] = {}
                # Read in data (with potential synthetic plumes inserted)

                with warnings.catch_warnings():  # (ignore warning about non-writable buffer)
                    warnings.simplefilter("ignore", category=UserWarning)
                    for col, shape, dtype in zip(input_columns, input_shapes, input_dtypes, strict=True):
                        inputs[col] = torch.frombuffer(sample[col], dtype=dtype).reshape(shape).unsqueeze(0)
                    target = torch.rand(target_shape, dtype=target_dtype).unsqueeze(0)

                # Exchange B11/B12 with the original band values + Concatenate
                Xy = transform_orig((inputs, target))
                X = Xy[0].to(device)

                with torch.no_grad():
                    binary_prob_crop = torch.sigmoid(model(X)[0, 0]).float().cpu().numpy()

                results[region]["total_crops"] += 1
                max_binary = binary_prob_crop.max()
                # below eg 25% probability, we dont track results as our min marker_t is 25%
                if max_binary < MARKER_THRESHOLDS_WITH_DIFF_FLOORS[0]:
                    for marker_threshold, floor_threshold in MARKER_FLOOR_COMBINATIONS:
                        results[region]["area_sums"][(marker_threshold, floor_threshold)].append(0)
                        results[region]["max_probs"][(marker_threshold, floor_threshold)].extend([0.0])
                    continue

                for marker_threshold, floor_threshold in MARKER_FLOOR_COMBINATIONS:
                    watershed_params = WatershedParameters(
                        marker_distance=MARKER_DISTANCE,
                        watershed_floor_threshold=floor_threshold,
                        marker_threshold=marker_threshold,
                        closing_footprint_size=0,
                    )
                    plume_mask = retrieval_mask_using_watershed_algo(watershed_params, binary_prob_crop)
                    labeled_plume = skimage.measure.label(plume_mask, connectivity=2)
                    max_probs = skimage.measure.regionprops_table(
                        labeled_plume,
                        intensity_image=binary_prob_crop,
                        properties=["intensity_max", "area", "intensity_mean", "intensity_std"],
                    )
                    results[region]["area_sums"][(marker_threshold, floor_threshold)].append(sum(max_probs["area"]))
                    results[region]["max_probs"][(marker_threshold, floor_threshold)].extend(max_probs["intensity_max"])

                    if visualize and (
                        np.random.random() < 0.001  # visualize in 0.1% of cases or very high probs # noqa
                        or (len(max_probs["intensity_max"]) > 0 and max_probs["intensity_max"].max() > 0.95)  # noqa
                    ):
                        for col in max_probs:
                            max_probs[col] = max_probs[col].round(4).astype(np.float32)
                        f, ax = plt.subplots(1, 5, figsize=(25, 5))
                        ax = cast(np.ndarray, ax)  # for mypy
                        rgb_main = get_rgb_from_tensor(inputs["crop_main"], S2_BANDS, 0).cpu()
                        ax[0].imshow(
                            (rgb_main.numpy() / 0.35 * 255).clip(0, 255).astype(np.uint8),
                            vmin=0.0,
                            vmax=1.0,
                            interpolation="nearest",
                        )
                        ax[0].set_title("RGB", fontsize=15)

                        ratio_earlier = get_band_ratio_from_tensor(inputs["crop_earlier"], S2_BANDS, 0).cpu()
                        ratio_before = get_band_ratio_from_tensor(inputs["crop_before"], S2_BANDS, 0).cpu()
                        ratio_main = get_band_ratio_from_tensor(inputs["crop_main"], S2_BANDS, 0).cpu()
                        ax[1].imshow(ratio_main, interpolation="none")
                        ax[1].set_title("B12/B11", fontsize=15)
                        ratio_diff = ratio_main - (ratio_before + ratio_earlier) / 2
                        max_min = 0.04
                        ax[2].set_title(
                            f"Ratio Diff (main - reference)\nMin {ratio_diff.min():.3f}, Max {ratio_diff.max():.3f}, "
                            f"Mean {ratio_diff.mean():.3f}",
                            fontsize=15,
                        )
                        ax[2].imshow(ratio_diff, vmin=-max_min, vmax=max_min, cmap="RdBu", interpolation="none")
                        ax[3].imshow(binary_prob_crop, vmin=0, vmax=1, interpolation="none")
                        ax[3].set_title(
                            f"Binary max prob: {100 * binary_prob_crop.max():.1f}%\nFloor Threshold = "
                            f"{100 * floor_threshold:.1f}%",
                            fontsize=14,
                        )
                        ax[4].imshow(labeled_plume, interpolation="none")
                        ax[4].set_title(
                            f"Areas: {max_probs['area']}\nMax intensities: {max_probs['intensity_max']}\n"
                            f"Mean: {max_probs['intensity_mean']} +/- {max_probs['intensity_std']}",
                            fontsize=13,
                        )
                        plt.tight_layout()
                        plt.show()
    return results


@timer(phase="compute_dt_results", accumulator=MEASUREMENTS, verbose=True, logger=logger)
def compute_dt_results(  # noqa
    val_parquet_folder_path: str,
    regions: list[str],
    enhancements_molperm2: list[np.ndarray],
    model: nn.Module,
    band_concatenator: BaseBandExtractor,
    binary_threshold: float,
    ncrops: int,
    crop_size: int,
    seed: int,
    likeli_threshs: dict[float, float],
    hapi_data_path: AnyUrl,
    device: torch.device | str = "cuda",
    visualize: bool = False,
) -> tuple[dict[float, dict[str, dict]], dict[tuple, dict[str, dict]]]:
    """Compute detection thresholds for all regions.

    Args:
        val_parquet_folder_path (str): Path to validation Parquet folder.
        regions (List[str]): List of region names.
        enhancements_molperm2 (List[np.ndarray]): Preprocessed plume enhancements.
        model (nn.Module): Trained PyTorch model.
        band_concatenator (BaseBandExtractor): Band concatenation transform.
        binary_threshold (float): Frac threshold for methane True/False
        ncrops (int): Number of random crops to sample.
        crop_size (int): Size of each crop.
        seed (int): Random seed for reproducibility.
        likeli_threshs (Dict[float, float]): Likelihood thresholds for FPRs.
        hapi_data_path (AnyUrl): Path to HAPI data.
        device (Union[torch.device, str], optional): Device to use. Defaults to "cuda".
        visualize (bool, optional): Enable visualization. Defaults to False.

    Returns
    -------
    results: dict[float, dict[str, dict]] = {}: average Recall per FPR and region where Recall
        100% = Caught at least one pixel
    results_thresholds: dict[tuple[float, float], dict[str, dict]]: average Recall for different
        marker/floor thresholds and region where Recall is calculated on all pixels
    """
    logger.info("=" * 130)
    logger.info("  Computing and aggregating DT results for all regions")
    logger.info("=" * 130)
    use_orig_b11_b12 = UseOriginalB11B12_X()

    emission_rates = np.concatenate(
        (
            np.arange(100, 1000, 100),  # Fine resolution for small emissions
            np.arange(1000, 2000, 200),  # Fine resolution for small emissions
            np.arange(2000, 4000, 400),  # Medium resolution for mid-range
            np.arange(4000, 10000, 1000),  # Medium resolution for mid-range
            np.arange(10000, 25100, 5000),  # Coarse resolution for large emissions
        )
    )

    # Set seed for reproduciblility. This way we sample the same crops for the same inputs
    rng: np.random.Generator = np.random.default_rng(seed)

    region_detection_results: dict[float, dict[str, list[float]]] = {}
    for fpr in likeli_threshs:
        region_detection_results[fpr] = {}
        for region in regions:
            region_detection_results[fpr][region] = []

    avg_recall_for_emissions: dict[str, dict[tuple, list]] = {}
    for region in REGIONS:
        avg_recall_for_emissions[region] = {}
        for marker_threshold, floor_threshold in MARKER_FLOOR_COMBINATIONS:
            avg_recall_for_emissions[region][(marker_threshold, floor_threshold)] = []

    for region in regions:
        region_df = MGRS_DATE_DF[MGRS_DATE_DF["region"] == region].reset_index(drop=True)

        for i, row in region_df.iterrows():
            parquet_path = f"{val_parquet_folder_path}/{row['mgrs']}_{row['date']}.parquet"
            try:
                # Open the Parquet file as a file-like object
                with fs.open(parquet_path, "rb") as f:
                    # Read the Parquet file into a Pandas DataFrame
                    df_chips = pd.read_parquet(f)
            except Exception as e:
                logger.error(f"Failed to read Parquet file {parquet_path}: {e}")
                continue

            try:
                main_id = df_chips.loc[0, "main_and_reference_ids"][0]
                main_scene_sentinel2_item = get_sentinel_item_for_pystac_id(main_id)
            except Exception as e:
                logger.error(f"Failed to get Sentinel item for {main_id}: {e}")
                continue

            # Get RadTran params for the Tile ID
            instrument = main_scene_sentinel2_item.instrument
            radtran_lookup = RadTranLookupTable.from_params(
                instrument=instrument,
                solar_angle=main_scene_sentinel2_item.solar_angle,
                observation_angle=main_scene_sentinel2_item.observation_angle,
                hapi_data_path=hapi_data_path,
                min_ch4=0.0,
                max_ch4=20.0,
                spacing_resolution=40000,
                ref_band="B11",
                band="B12",
                full_sensor_name="Sentinel2",
            )

            idxs = rng.integers(len(df_chips), size=ncrops)
            for sample_count, idx in zip(range(ncrops), idxs, strict=False):
                logger.info(
                    f"DT {region}: TILE ({i + 1}/{len(region_df)}): {row['mgrs']} - {row['date']}: "
                    f"SAMPLE ({sample_count + 1}/{ncrops})"
                )
                sample = df_chips.loc[
                    idx, [*list(input_columns), target_column, "main_and_reference_ids", "exclusion_mask_plumes"]
                ]
                inputs: dict[str, torch.Tensor] = {}
                # Read in data (with potential synthetic plumes inserted)
                with warnings.catch_warnings():  # (ignore warning about non-writable buffer)
                    warnings.simplefilter("ignore", category=UserWarning)
                    for col, shape, dtype in zip(input_columns, input_shapes, input_dtypes, strict=True):
                        inputs[col] = torch.frombuffer(sample[col], dtype=dtype).reshape(shape).unsqueeze(0)
                    target = (
                        torch.frombuffer(sample[target_column], dtype=target_dtype).reshape(target_shape).unsqueeze(0)
                    )
                # Exchange B11/B12 with the original band values
                orig_input = use_orig_b11_b12(inputs)

                # Retrieve the exclusion mask for inserting plumes
                exclusion_mask_plumes = np.frombuffer(sample["exclusion_mask_plumes"], dtype=bool).reshape(
                    (1, crop_size, crop_size)
                )[0]

                # It is rather conservative, erode it a bit if it covers too much area
                use_eroded: bool = exclusion_mask_plumes.sum() > 0.3 * crop_size * crop_size
                eroded_img = None
                if use_eroded:
                    selem = footprint_rectangle((2, 2))
                    eroded_img = erosion(exclusion_mask_plumes, selem)
                    logger.info(
                        f"We have {100 * exclusion_mask_plumes.sum() / (crop_size * crop_size):.1f}% excluded px, "
                        f"eroded to {100 * eroded_img.sum() / (crop_size * crop_size):.1f}% excluded px"
                    )

                orig_swir16 = orig_input["crop_main"][0, S2_BANDS.index("B11")].float().numpy()
                orig_swir22 = orig_input["crop_main"][0, S2_BANDS.index("B12")].float().numpy()
                offset_swir16 = orig_swir16 + BAND_FRAC_OFFSET
                offset_swir22 = orig_swir22 + BAND_FRAC_OFFSET

                if visualize:
                    clouds = np.frombuffer(df_chips.loc[idx, "cloud_main"], dtype=bool).reshape(
                        (1, crop_size, crop_size)
                    )[0]
                    visualize_dt_input_data(
                        orig_input=orig_input,
                        orig_swir22=orig_swir22,
                        exclusion_mask_plumes=exclusion_mask_plumes,
                        eroded_img=eroded_img,
                        clouds=clouds,
                        use_eroded=use_eroded,
                    )

                positioned_enhancements_arr, plumes_inserted_idxs = generate_random_plume_enhancements(
                    orig_input["orig_band_12"][0, 0].numpy(),
                    eroded_img if use_eroded else exclusion_mask_plumes,  # type:ignore
                    enhancements_molperm2,
                    rng,
                )
                max_likelihood_curves: list[list[float]] = []
                recall_for_all_crops_plumes: dict[tuple, list] = {}
                for marker_threshold, floor_threshold in MARKER_FLOOR_COMBINATIONS:
                    recall_for_all_crops_plumes[(marker_threshold, floor_threshold)] = []
                for plume_count, enhancement in enumerate(positioned_enhancements_arr):
                    if len(plumes_inserted_idxs[plume_count]) == 0:
                        logger.warning(f"PLUME ({plume_count + 1} could not be inserted")
                        continue
                    max_likelihoods: list[float] = []
                    recall_for_all_emissions: dict[tuple, list] = {}
                    for marker_threshold, floor_threshold in MARKER_FLOOR_COMBINATIONS:
                        recall_for_all_emissions[(marker_threshold, floor_threshold)] = []
                    for emissions in emission_rates[:]:
                        new_input, target, sim_swir22, sim_swir16 = insert_plume(  # type: ignore
                            orig_input,
                            orig_swir16,
                            radtran_lookup,
                            offset_swir16,
                            offset_swir22,
                            enhancement,
                            emissions,
                        )
                        if abs(target.min()) < binary_threshold:
                            max_likelihoods.append(np.nan)
                            for marker_threshold, floor_threshold in MARKER_FLOOR_COMBINATIONS:
                                recall_for_all_emissions[(marker_threshold, floor_threshold)].append(np.nan)
                            continue

                        X = band_concatenator((new_input, torch.from_numpy(target)))[0].to(device)
                        with torch.no_grad():
                            binary_prob_crop = torch.sigmoid(model(X)[0, 0]).float().cpu().numpy()

                        plume_mask = np.abs(target) >= binary_threshold
                        max_likelihood_score = binary_prob_crop[plume_mask].max() * 10
                        max_likelihoods.append(max_likelihood_score)

                        for marker_threshold, floor_threshold in MARKER_FLOOR_COMBINATIONS:
                            if max_likelihood_score < marker_threshold:
                                # Nothing is detected
                                recall = 0.0
                            else:
                                watershed_params = WatershedParameters(
                                    marker_distance=MARKER_DISTANCE,
                                    watershed_floor_threshold=floor_threshold,
                                    marker_threshold=marker_threshold,
                                    closing_footprint_size=0,
                                )
                                plume_pred_mask = retrieval_mask_using_watershed_algo(
                                    watershed_params, binary_prob_crop
                                )
                                tps = (plume_pred_mask[plume_mask] == 1).sum()
                                fns = (plume_pred_mask[plume_mask] == 0).sum()
                                recall = tps / (tps + fns) if (tps + fns) > 0 else 0
                            recall_for_all_emissions[(marker_threshold, floor_threshold)].append(recall)
                            if visualize and max_likelihood_score < marker_threshold:
                                mean_likelihood_score = binary_prob_crop[plume_mask].mean() * 10
                                visualize_dt_prediction(
                                    sim_swir16=sim_swir16,
                                    sim_swir22=sim_swir22,
                                    binary_prob=binary_prob_crop,
                                    plume_pred_mask=plume_pred_mask,
                                    floor_threshold=floor_threshold,
                                    marker_threshold=marker_threshold,
                                    target=target,
                                    binary_threshold=binary_threshold,
                                    max_likelihood_score=max_likelihood_score,
                                    mean_likelihood_score=mean_likelihood_score,
                                )
                    max_likelihood_curves.append(max_likelihoods)
                    for marker_threshold, floor_threshold in MARKER_FLOOR_COMBINATIONS:
                        recall_for_all_crops_plumes[(marker_threshold, floor_threshold)].append(
                            recall_for_all_emissions[(marker_threshold, floor_threshold)]
                        )
                if len(max_likelihood_curves) > 0:
                    # stack to (nplumes=5, emissions)
                    max_likelihood_curves = np.stack(max_likelihood_curves).astype(np.float32)  # type:ignore
                    # For each emission rate, calculate the proportion of randomly positioned plumes
                    # that have likelihood scores above the threshold
                    for fpr, likeli_thresh in likeli_threshs.items():
                        detection_prob_for_thresh = np.nanmean(  # type: ignore
                            max_likelihood_curves > likeli_thresh,  # type:ignore
                            axis=0,
                            dtype=np.float32,  # type: ignore
                        )
                        region_detection_results[fpr][region].append(detection_prob_for_thresh.tolist())

                    for marker_threshold, floor_threshold in MARKER_FLOOR_COMBINATIONS:
                        # stack to (ncrops*nplumes=5, emissions)
                        recalls = np.stack(recall_for_all_crops_plumes[(marker_threshold, floor_threshold)]).astype(
                            np.float32
                        )  # type:ignore
                        # For each emission rate, calculate the average recall
                        means = np.nanmean(recalls, axis=0, dtype=np.float32)
                        # fill nans (All 5 plumes were "invisible") with recall 0
                        means = np.nan_to_num(means, nan=0.0)
                        avg_recall_for_emissions[region][(marker_threshold, floor_threshold)].append(means.tolist())

    # Save average Recall per FPR and region where Recall 100% = Caught at least one pixel
    results: dict[float, dict[str, dict]] = {}
    for fpr in likeli_threshs:
        results[fpr] = {}
        for region in regions:
            if region_detection_results[fpr][region]:
                results[fpr][region] = {
                    "emission_rates": emission_rates.tolist(),
                    "total_crops": TILE_ID_COUNTS_BY_REGION[region] * ncrops * 5 * len(emission_rates),
                    # Calculate average detection probabilities for each region over all tiles
                    "mean_pod": (np.mean(region_detection_results[fpr][region], axis=0) * 100).tolist(),
                    "std_pod": (np.std(region_detection_results[fpr][region], axis=0) * 100).tolist(),
                }
            else:
                logger.warning(f"No results for region: {region}")
                # If no results for this region, use zeros or some default value
                results[fpr][region] = {
                    "emission_rates": emission_rates.tolist(),
                    "total_crops": 0,
                    "mean_pod": [0.0] * len(emission_rates),
                    "std_pod": [0.0] * len(emission_rates),
                }

    # Save average Recall for different marker/floor thresholds and region where Recall is calculated on all pixels
    results_thresholds: dict[tuple, dict[str, dict]] = {}
    for marker_threshold, floor_threshold in MARKER_FLOOR_COMBINATIONS:
        results_thresholds[(marker_threshold, floor_threshold)] = {}
        for region in regions:
            if avg_recall_for_emissions[region][(marker_threshold, floor_threshold)]:
                results_thresholds[(marker_threshold, floor_threshold)][region] = {
                    "emission_rates": emission_rates.tolist(),
                    "total_crops": TILE_ID_COUNTS_BY_REGION[region] * ncrops * 5 * len(emission_rates),
                    # Calculate average detection probabilities for each region over all tiles
                    "mean_pod": (
                        np.mean(avg_recall_for_emissions[region][(marker_threshold, floor_threshold)], axis=0) * 100
                    ).tolist(),
                    "std_pod": (
                        np.std(avg_recall_for_emissions[region][(marker_threshold, floor_threshold)], axis=0) * 100
                    ).tolist(),
                }
            else:
                logger.warning(f"No results for region: {region}")
                # If no results for this region, use zeros or some default value
                results_thresholds[(marker_threshold, floor_threshold)][region] = {
                    "emission_rates": emission_rates.tolist(),
                    "total_crops": 0,
                    "mean_pod": [0.0] * len(emission_rates),
                    "std_pod": [0.0] * len(emission_rates),
                }
    return results, results_thresholds


def visualize_dt_input_data(
    orig_input: dict[str, torch.Tensor],
    orig_swir22: np.ndarray,
    exclusion_mask_plumes: np.ndarray,
    eroded_img: np.ndarray | None,
    clouds: np.ndarray,
    use_eroded: bool,
) -> None:
    """Visualize initial detection threshold sample data."""
    f, ax = plt.subplots(1, 5, figsize=(25, 5))
    ax = cast(np.ndarray, ax)  # for mypy

    rgb_main = get_rgb_from_tensor(orig_input["crop_main"], S2_BANDS, 0)
    ax[0].imshow(
        (rgb_main.numpy() / 0.35 * 255).clip(0, 255).astype(np.uint8),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax[1].imshow(orig_swir22, interpolation="none")
    ax[2].imshow(exclusion_mask_plumes)
    ax[2].set_title(f"{exclusion_mask_plumes.sum()=}")

    lcc_main = orig_input["crop_main"][0, S2_BANDS.index("SCL")].numpy()
    im = ax[4].imshow(lcc_main, cmap=CMAP, interpolation="nearest")
    cbar = f.colorbar(im, ax=ax, boundaries=np.arange(-0.5, 12, 1), ticks=range(12))
    cbar.set_label("Classification")
    im.set_clim(-0.5, 11.5)
    cbar.ax.set_yticklabels(S2_LAND_COVER_CLASSIFICATIONS, fontsize=15)

    if use_eroded:
        ax[3].imshow(eroded_img, interpolation="none")
        ax[3].set_title(f"{eroded_img.sum()=}")  # type:ignore
    else:
        ax[3].imshow(clouds, interpolation="none")
        ax[3].set_title("Clouds")
    plt.show()


def visualize_dt_prediction(  # noqa
    sim_swir16: np.ndarray,
    sim_swir22: np.ndarray,
    binary_prob: np.ndarray,
    plume_pred_mask: np.ndarray,
    marker_threshold: float,
    floor_threshold: float,
    target: torch.Tensor,
    binary_threshold: float,
    max_likelihood_score: float,
    mean_likelihood_score: float,
) -> None:
    """Visualize detection threshold prediction data."""
    f, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax = cast(np.ndarray, ax)  # for mypy
    sim_band_ratio = sim_swir22 / sim_swir16
    ax[0].imshow(sim_band_ratio, interpolation="none")
    ax[0].set_title(f"B12/B11 ratio Mean: {sim_band_ratio.mean():.4f}, Min: {sim_band_ratio.min():.4f}")
    ax[1].imshow(binary_prob, vmin=0, vmax=1, interpolation="none")
    ax[1].set_title(f"Binary Likelihood, Plume max: {max_likelihood_score:.2f}, mean: {mean_likelihood_score:.2f}")

    plume_mask = (np.abs(target) >= binary_threshold).astype(np.uint8)
    tps = ((plume_mask == 1) & (plume_pred_mask == 1)).sum()
    fns = ((plume_mask == 1) & (plume_pred_mask == 0)).sum()
    recall = tps / (tps + fns)
    ax[2].imshow(plume_pred_mask, vmin=0, vmax=1, interpolation="none")
    ax[2].set_title(
        f"Recall={100 * recall:.1f}%, TP={tps:.0f}, FN={fns:.0f}\nPred with "
        f"marker={marker_threshold:.2f}, floor={floor_threshold:.3f}"
    )

    gt_min_max = 0.005
    ax[3].imshow(target, vmin=-gt_min_max, vmax=gt_min_max, cmap="RdBu", interpolation="none")
    ax[3].set_title(
        f"Plume > 0.001 {plume_mask.sum():.0f} - Plume > 0.01 {np.sum(np.abs(target) >= binary_threshold * 10):.0f}"
        f" - Plume > 0.1 {np.sum(np.abs(target) >= binary_threshold * 100):.0f}"
    )
    plt.tight_layout()
    plt.show()


def insert_plume(
    orig_input: dict[str, torch.Tensor],
    orig_swir16: np.ndarray,
    radtran_lookup: RadTranLookupTable,
    offset_swir16: np.ndarray,
    offset_swir22: np.ndarray,
    enhancement: np.ndarray,
    emissions: float,
) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, np.ndarray]:
    """Insert a simulated plume into the input data.

    Args:
        orig_input (Dict[str, torch.Tensor]): Original input tensor dictionary.
        orig_swir16 (np.ndarray): Original SWIR16 band data.
        radtran_lookup (RadTranLookupTable): Radiative transfer lookup table.
        offset_swir16 (np.ndarray): Offset-adjusted SWIR16 band.
        offset_swir22 (np.ndarray): Offset-adjusted SWIR22 band.
        enhancement (np.ndarray): Methane enhancement data.
        emissions (float): Emission rate in kg/hr.

    Returns
    -------
        Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray, np.ndarray]: Updated input,
        target, simulated SWIR22, and SWIR16.

    Raises
    ------
        AssertionError: If shapes are incompatible.
    """
    new_input = copy.deepcopy(orig_input)
    scaled_enhancement = enhancement * emissions / GORRONO_PLUME_EMISSION_RATE
    assert orig_swir16.shape == scaled_enhancement.shape

    nB_band, nB_ref_band = radtran_lookup.lookup(scaled_enhancement)
    sim_swir16 = nB_ref_band * offset_swir16
    sim_swir22 = nB_band * offset_swir22

    # calculate FRAC from the simulated bands
    target = get_frac(swir16=sim_swir16, swir16_o=offset_swir16, swir22=sim_swir22, swir22_o=offset_swir22).astype(
        np.float32
    )
    # now remove the offset from the simulated bands and clip to zero (otherwise we're giving
    # the neural network information that wouldn't actually be present in a real image)
    np.clip(sim_swir16 - BAND_FRAC_OFFSET, a_min=0.0, a_max=None, out=sim_swir16)
    np.clip(sim_swir22 - BAND_FRAC_OFFSET, a_min=0.0, a_max=None, out=sim_swir22)
    # and also round to the nearest integer, again so the neural network can't use
    # non-integer values as a way to detect methane
    np.round(sim_swir16, decimals=0, out=sim_swir16)
    np.round(sim_swir22, decimals=0, out=sim_swir22)

    # Modify the main crop array's band 11 and 12
    new_input["crop_main"][0, S2_BANDS.index("B11")] = torch.from_numpy(sim_swir16.astype(np.int16))
    new_input["crop_main"][0, S2_BANDS.index("B12")] = torch.from_numpy(sim_swir22.astype(np.int16))

    return new_input, target, sim_swir22, sim_swir16


@timer(phase="plot_fpr_curves", accumulator=MEASUREMENTS, verbose=True, logger=logger)
def plot_and_save_fpr_curves(  # noqa: PLR0912, PLR0915 (too-many-arguments, too-many-branches)
    results: dict[str, Any],
    thresholds: np.ndarray,
    crop_area_km2: float,
    tile_area_km2: float,
    plot_name: str,
    results_path: str,
    results_fp_px_sums_path: str,
    azure_cluster: bool,
) -> None:
    """Generate and save FPR curves plot.

    Args:
        results (Dict[str, Any]): FPR results per region per marker, floor thresholds
        thresholds (np.ndarray): Threshold values for FPR calculation.
        crop_area_km2 (float): Area of a crop in square kilometers.
        tile_area_km2 (float): Area of a tile in square kilometers.
        plot_name (str): Name for the plot.
        results_path (str): Path to save JSON #FP plume results.
        results_fp_px_sums_path (str): Path to save JSON #FP px results.
        azure_cluster (bool): Flag for Azure cluster execution.
    """
    logger.info("=" * 130)
    logger.info("  Plotting FPR results to MLFlow")
    logger.info("=" * 130)

    for region in REGIONS:
        for marker_threshold, floor_threshold in MARKER_FLOOR_COMBINATIONS:
            if results[region]["total_crops"] > 0:
                nb_detections = len(results[region]["max_probs"][(marker_threshold, floor_threshold)])
                nb_fp_px = sum(results[region]["area_sums"][(marker_threshold, floor_threshold)])
                logger.info(
                    f"{region:9}: marker={marker_threshold:.3f}, floor={floor_threshold:.3f}: FP px Avg per "
                    f"128x128 crop ={nb_fp_px / results[region]['total_crops']:6.1f}"
                    f", Avg per 500x500 tile = {15.26 * nb_fp_px / results[region]['total_crops']:6.1f} ||| "
                    f"Plumes Avg per 128x128 crop={nb_detections / results[region]['total_crops']:5.2f}, "
                    f"Avg per 500x500 tile = {15.26 * nb_detections / results[region]['total_crops']:5.2f} "
                    f"({results[region]['total_crops']} crops)"
                )

    ###### FIRST PLOT: #FP PLUMES PER TILE VS. MARKER THRESHOLD
    fig = plt.figure(figsize=(10, 6))
    formatted_results: dict[str, dict] = {}

    for region, data in results.items():
        total_area_km2 = data["total_crops"] * crop_area_km2

        ccdf_counts_per_km2 = [
            np.sum(np.array(data["max_probs"][MARKER_FLOOR_COMBINATIONS[0]]) > threshold) / total_area_km2
            for threshold in thresholds
        ]
        ccdf_counts_per_tile = [count * tile_area_km2 for count in ccdf_counts_per_km2]

        formatted_results[region] = {
            "thresholds": thresholds.tolist(),
            "ccdf_counts_per_km2": ccdf_counts_per_km2,
            "ccdf_counts_per_tile": ccdf_counts_per_tile,
            "total_area_km2": total_area_km2,
            "total_crops": data["total_crops"],
        }

        plt.plot(
            thresholds * 10,
            ccdf_counts_per_tile,
            marker="o",
            linestyle="-",
            color=REGION_COLORS[region],
            label=f"{region} (n_crops={data['total_crops']}, n_tiles={TILE_ID_COUNTS_BY_REGION[region]})",
        )

    # Save results to file
    with open(results_path, "w") as f:
        json.dump(formatted_results, f, indent=2)
    if azure_cluster:
        mlflow.log_dict(dictionary=formatted_results, artifact_file=results_path)

    plt.xlabel("Marker Threshold * 10")
    plt.ylabel("# FP plumes per (500x500) Tile per Overpass")
    plt.title(f"False Positive Plume Rate - {plot_name}")
    plt.grid(True)
    plt.legend()
    if azure_cluster:
        plot_path = f"fpr_plume_curves_{plot_name}.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_figure(fig, plot_path)
    else:
        plt.show()

    ###### SECOND PLOT: FP PX PER TILE VS. MARKER/FLOOR THRESHOLDS
    formatted_fp_px_results: dict[str, dict] = {}
    crops_per_tile = tile_area_km2 / crop_area_km2
    fig, ax = plt.subplots(figsize=(10, 6))
    for region in REGIONS:
        marker_floor_t = []
        fp_px_per_tile = []
        for thresholds_, v in results[region]["area_sums"].items():
            marker_floor_t.append(thresholds_)
            # avg px per 128x128 crop * 15.26 crops inside a 500x500 tile
            fp_px_per_tile.append((sum(v) / results[region]["total_crops"]) * crops_per_tile)
        formatted_fp_px_results[region] = {
            "marker_floor_threshold": marker_floor_t,
            "fp_px_per_tile": fp_px_per_tile,
            "total_crops": data["total_crops"],
        }

        plt.plot(
            range(len(marker_floor_t)),
            fp_px_per_tile,
            marker="o",
            linestyle="-",
            label=f"{region} (n_crops={results[region]['total_crops']}, n_tiles={TILE_ID_COUNTS_BY_REGION[region]})",
        )
    plt.xticks(range(len(marker_floor_t)), [str(x_) for x_ in marker_floor_t], rotation=30, ha="right")
    for label in ax.get_xticklabels():
        mark_t = float(label.get_text()[1:].split(",")[0])
        label.set_color(MARKER_THRESHOLDS_COLORS[mark_t])
    plt.xlabel("(Marker, Floor) Thresholds")
    plt.ylabel("FP px per (500x500) Tile")
    plt.title(f"False Positive Px Curves - {plot_name}")
    plt.grid(True)
    plt.legend()
    plt.ylim(0)
    if azure_cluster:
        plot_path = f"fp_px_threshold_curves_{plot_name}.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_figure(fig, plot_path)
    else:
        plt.show()

    # Save results to file
    with open(results_fp_px_sums_path, "w") as f:
        json.dump(formatted_fp_px_results, f, indent=2)
    if azure_cluster:
        mlflow.log_dict(dictionary=formatted_fp_px_results, artifact_file=results_fp_px_sums_path)


@timer(phase="plot_detection_curves", accumulator=MEASUREMENTS, verbose=True, logger=logger)
def plot_detection_curves(
    results: dict[int, dict[str, dict]], results_thresholds: dict, plot_name: str, azure_cluster: bool
) -> None:
    """Generate and save detection probability curves plot = Mean Recalls at FPR=X over emission rates.

    Args:
        results (Dict[int, Dict[str, dict]]): Detection results per FPR and region.
        plot_name (str): Name for the plot.
        azure_cluster (bool): Flag for Azure cluster execution.
    """
    logger.info("=" * 130)
    logger.info("  Plotting DT results to MLFlow")
    logger.info("=" * 130)

    #### FIRST PLOTS: For each FPR=X, plot mRecall (at least one px correctly predicted) per emission rate
    for fpr, region_data in results.items():
        fig = plt.figure(figsize=(10, 6))
        for region, data in region_data.items():
            probs = np.array(data["mean_pod"])
            emission_rates = np.array(data["emission_rates"])

            plt.plot(
                emission_rates,
                probs,
                marker="o",
                linestyle="-",
                color=REGION_COLORS[region],
                label=f"{region} (n_crops={data['total_crops']}, n_tiles={TILE_ID_COUNTS_BY_REGION[region]})",
            )

        plt.xlabel("Emission Rate (kg/hr)")
        plt.ylabel("Mean Probability of Detection")
        plt.ylim(0, 101)
        plt.xlim(100, 25200)
        plt.xscale("log")
        plt.yticks(range(0, 100, 10))
        plt.title(f"FPR={fpr:.2f} Mean Detection Probability Curves - {plot_name}")
        plt.grid(True)
        plt.legend(loc="upper left")

        if azure_cluster:
            plot_path = f"fpr_{fpr:02.2f}_Mean_detection_curves_{plot_name}.png"
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_figure(fig, f"FPR_{fpr:02.2f}/{plot_path}")
        else:
            plt.show()

    #### SECOND PLOTS: For each Marker Thresh, plot real px Recalls of various floor threshold per region per emission
    for marker_t in MARKER_THRESHOLDS_WITH_DIFF_FLOORS:
        fig = plt.figure(figsize=(10, 6))
        for region in REGIONS:
            for (marker_threshold, floor_threshold), marker_data in results_thresholds.items():
                if marker_threshold != marker_t:
                    continue
                data = marker_data[region]
                probs = np.array(data["mean_pod"])
                emission_rates = np.array(data["emission_rates"])

                plt.plot(
                    emission_rates,
                    probs,
                    marker="o",
                    linestyle="-",
                    label=f"{region}, Floor={floor_threshold:.3f}",
                )

            plt.xlabel("Emission Rate (kg/hr)")
            plt.ylabel("Mean Probability of Detection")
            plt.ylim(0, 101)
            plt.xlim(100, 25200)
            plt.xscale("log")
            plt.yticks(range(0, 100, 10))
            plt.title(f"Marker Threshold {marker_t:.2f}: Mean pixelwise Recall Curves")
            plt.grid(True)
            plt.legend(loc="upper left")

        if azure_cluster:
            plot_path = f"Marker_{marker_t:.2f}_Floor_{floor_threshold:.3f}_px_recall_curves_{plot_name}.png"
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_figure(fig, f"Marker_{marker_t:.2f}/{plot_path}")
        else:
            plt.show()

    # # FIXME: FIGURE OUT WHAT CONFIDENCE INTERVAL WE WANT TO SHOW HERE
    ###############################################################################################################
    # BERTs comment: can the title be reworded to be explicit about what's been calculated and what's being shown?
    # e.g. "Mean of mean of mean of xxx +/- 95% C.I."
    # Instead of calculating Confidence Intervals here, I would suggest just using the Standard Error = std / sqrt(n),
    # as we're more interested in reporting the uncertainty of the mean point estimate than estimating a range for the
    # true population.  Also avoids some of the complications with Confidence Intervals like: interpretability,
    # corrections for multiple comparisons, and 95% confidence being an arbitrary choice.
    ###############################################################################################################
    # # Show mean + confidence intervals over all regions in a plot
    # confidence = 0.95
    # for fpr, region_data in results.items():
    #     fig = plt.figure(figsize=(10, 6))
    #     for region, data in region_data.items():
    #         mean_probs = np.array(data["mean_pod"])
    #         std_probs = np.array(data["std_pod"])  # std of the avg detection over all tiles
    #         emission_rates = np.array(data["emission_rates"])

    #         # Calculate confidence interval (95% CI using t-distribution)
    #         n = TILE_ID_COUNTS_BY_REGION[region]  # sample size = nb of MGRS tiles
    #         t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)  # critical t-value
    #         ci_half_width = t_crit * std_probs / np.sqrt(n)

    #         # Plot mean line
    #         plt.plot(
    #             emission_rates,
    #             mean_probs,
    #             marker="o",
    #             linestyle="-",
    #             linewidth=2.0,
    #             color=REGION_COLORS[region],
    #             label=f"{region} (n_crops={data['total_crops']}, n_tiles={n}",
    #         )
    #         # Plot bottom CI line
    #         plt.plot(
    #             emission_rates,
    #             mean_probs - ci_half_width,
    #             linestyle="--",
    #             linewidth=1.0,
    #             color=REGION_COLORS[region],
    #         )
    #         # Plot top CI line
    #         plt.plot(
    #             emission_rates,
    #             mean_probs + ci_half_width,
    #             linestyle="--",
    #             linewidth=1.0,
    #             color=REGION_COLORS[region],
    #         )

    #     plt.xlabel("Emission Rate (kg/hr)")
    #     plt.ylabel("Mean +/- Std Probability of Detection")
    #     plt.ylim(0, 101)
    #     plt.xlim(100, 25200)
    #     plt.xscale("log")
    #     plt.yticks(range(0, 100, 10))
    #     plt.title(f"FPR={fpr} Mean +/- Std PODs - {plot_name}")
    #     plt.grid(True, alpha=0.3)
    #     plt.legend(loc="upper left")
    #     if azure_cluster:
    #         plot_path = f"fpr_{fpr:02}_CI_{plot_name}.png"
    #         plt.savefig(plot_path)
    #         plt.close()
    #         mlflow.log_figure(fig, f"FPR_{fpr:02}/{plot_path}")
    #     else:
    #         plt.show()


def save_pixelwise_recall_results(
    results: dict[tuple, dict[str, dict]], azure_cluster: bool
) -> dict[str, dict[str, dict]]:
    """Save pixelwise recall calculate POD metrics.

    Args:
        results (dict[float, dict[str, dict]]): Avg. pixel Recall for diff. marker/floor thresholds and regions.
        azure_cluster (bool): Flag for Azure cluster execution.

    Returns
    -------
        dict[str, dict[str, dict]]: POD metrics per marker/floor threshold and region.
    """
    logger.info("=" * 130)
    logger.info("  Saving DT results to MLFlow")
    logger.info("=" * 130)

    pixel_recall_pod_results: dict[str, dict[str, dict]] = {}
    for (marker_threshold, floor_threshold), region_data in results.items():
        marker_floor_str = f"{marker_threshold:.2f}_{floor_threshold:.3f}"
        pixel_recall_pod_results[marker_floor_str] = {}
        for region, data in region_data.items():
            emission_rates = np.array(data["emission_rates"])
            pixel_recall_pod_results[marker_floor_str][region] = {}
            # Mean over all tiles of avg pods over all chips/sims per emission rates ()
            probabilties = np.array(data["mean_pod"])

            # Calculate POD metrics
            pixel_recall_pod_results[marker_floor_str][region] = {
                f"POD_{prob}": int(np.round(np.interp(prob, probabilties, emission_rates), 0)) for prob in [10, 50, 90]
            }
            min_prob, min_emission = get_min_nonzero_detection_prob(emission_rates, probabilties)
            # Due to interpolation, sometimes we get a lower pod10 than the min, fix that
            pixel_recall_pod_results[marker_floor_str][region]["POD_10"] = max(
                pixel_recall_pod_results[marker_floor_str][region]["POD_10"], min_emission
            )
            pixel_recall_pod_results[marker_floor_str][region].update({"min_emission": min_emission})

        # Save raw results
        dt_region_probs = {
            # Emission rates are identical for all regions, so we just take them from the first region
            "emission_rates": data["emission_rates"],
            **{f"{region}_mean_pod": data["mean_pod"] for region, data in region_data.items()},
            **{f"{region}_std_pod": data["std_pod"] for region, data in region_data.items()},
            **{f"{region}_total_crops": data["total_crops"] for region, data in region_data.items()},
        }

        results_path = f"Marker_{marker_threshold:.3f}_Floor_{floor_threshold:.3f}_px_PODs.json"
        with open(results_path, "w") as f:
            json.dump(dt_region_probs, f, indent=2)
        if azure_cluster:
            mlflow.log_dict(
                dictionary=dt_region_probs,
                artifact_file=f"Marker_{marker_threshold:.2f}/{results_path}",
            )

    # Save POD results
    pod_results_path = "Pixelwise_PODs_10_50_90_min.json"
    with open(pod_results_path, "w") as f:
        json.dump(pixel_recall_pod_results, f, indent=2)
    if azure_cluster:
        mlflow.log_dict(dictionary=pixel_recall_pod_results, artifact_file=pod_results_path)  # type: ignore

    return pixel_recall_pod_results


def save_det_prob_results(results: dict[float, dict[str, dict]], azure_cluster: bool) -> None:
    """Save detection probability results and calculate POD metrics.

    Args:
        results (Dict[float, Dict[str, dict]]): Detection results per FPR and region.
        azure_cluster (bool): Flag for Azure cluster execution.
    """
    logger.info("=" * 130)
    logger.info("  Saving DT results to MLFlow")
    logger.info("=" * 130)

    pod_results: dict[float, dict[str, dict]] = {}
    for fpr, region_data in results.items():
        pod_results[fpr] = {}
        for region, data in region_data.items():
            emission_rates = np.array(data["emission_rates"])
            pod_results[fpr][region] = {}
            # Mean over all tiles of avg pods over all chips/sims per emission rates ()
            probabilties = np.array(data["mean_pod"])

            # Calculate POD metrics
            pod_results[fpr][region] = {
                f"POD_{prob}": int(np.round(np.interp(prob, probabilties, emission_rates), 0)) for prob in [10, 50, 90]
            }
            min_prob, min_emission = get_min_nonzero_detection_prob(emission_rates, probabilties)
            # Due to interpolation, sometimes we get a lower pod10 than the min, fix that
            pod_results[fpr][region]["POD_10"] = max(pod_results[fpr][region]["POD_10"], min_emission)
            pod_results[fpr][region].update({"min_emission": min_emission})

        # Save raw results
        dt_region_probs = {
            # Emission rates are identical for all regions, so we just take them from the first region
            "emission_rates": data["emission_rates"],
            **{f"{region}_mean_pod": data["mean_pod"] for region, data in region_data.items()},
            **{f"{region}_std_pod": data["std_pod"] for region, data in region_data.items()},
            **{f"{region}_total_crops": data["total_crops"] for region, data in region_data.items()},
        }

        results_path = f"fpr_{fpr:02.2f}_PODs.json"
        with open(results_path, "w") as f:
            json.dump(dt_region_probs, f, indent=2)
        if azure_cluster:
            mlflow.log_dict(dictionary=dt_region_probs, artifact_file=f"FPR_{fpr:02.2f}/{results_path}")

    # Save POD results
    pod_results_path = "PODs_10_50_90_min.json"
    with open(pod_results_path, "w") as f:
        json.dump(pod_results, f, indent=2)
    if azure_cluster:
        mlflow.log_dict(dictionary=pod_results, artifact_file=pod_results_path)  # type: ignore


####################################################
################ HELPER FUNCTIONS ##################
####################################################


def generate_random_plume_enhancements(
    tile_band: np.ndarray,
    exclusion_mask_plumes: np.ndarray,
    enhancements_molperm2: list[np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[list[int]]]:
    """Generate randomly positioned and rotated plume enhancements.

    Args:
        tile_band (np.ndarray): Base tile band data.
        exclusion_mask_plumes (np.ndarray): Mask for excluded plume areas.
        enhancements_molperm2 (List[np.ndarray]): List of plume enhancements.
        rng (np.random.Generator): Random number generator.

    Returns
    -------
        Tuple[np.ndarray, List[np.ndarray]]: Stacked enhancements and insertion indices.
    """
    positioned_enhancements: list[np.ndarray] = []
    plumes_inserted_idxs: list[list[int]] = []
    for enhancement in enhancements_molperm2:
        (
            methane_enhancement_molperm2,
            methane_enhancement_mask,
            plumes_inserted_idxs_,
        ) = randomly_position_sim_plume(
            sim_plumes=[
                (enhancement, enhancement >= 0),
            ],
            tile_band=tile_band,
            exclusion_mask_plumes=exclusion_mask_plumes,
            rng=rng,
            random_rotate=True,
            allow_overlap=True,
        )
        positioned_enhancements.append(methane_enhancement_molperm2)
        plumes_inserted_idxs.append(plumes_inserted_idxs_)
    return np.stack(positioned_enhancements), plumes_inserted_idxs


def get_sentinel_item_for_pystac_id(pystac_id: str) -> Sentinel2L1CItem:
    """Get the sentinel2 item with the given pystac id."""
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None)
    stac_api_io = StacApiIO(max_retries=retry)
    catalog = Client.open(
        PLANETARY_COMPUTER_CATALOG_URL,
        modifier=planetary_computer.sign_inplace,
        stac_io=stac_api_io,
    )

    search = catalog.search(collections=[PLANETARY_COMPUTER_COLLECTION_L2A], query={"id": {"eq": pystac_id}})
    item = search.item_collection()
    assert len(item) == 1
    pystac_item = next(iter(item))
    return Sentinel2L1CItem(pystac_item)


def load_gorrono_plumes(ml_client: MLClient) -> list[np.ndarray]:
    """Download Gorroño plumes to a temporary directory, convert them to mol/m² and return their paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        gorrono_plumes_uri = make_acceptable_uri(
            str(get_azureml_uri(ml_client, "orbio-data/methane_enhancements_molpercm2"))
        )
        download_from_blob(gorrono_plumes_uri, temp_path, recursive=True)

        # We need to convert the enhancements from mol/cm² to mol/m² as the radtran functions expect mol/m²
        raw_enhancements = [np.load(temp_path / f"{i}/methane_enhancement.npy") * 1e4 for i in range(5)]

    return raw_enhancements


def crop_enhancements(raw_enhancements: list[np.ndarray], min_molm2: float = 1e-4) -> list[np.ndarray]:
    """Crop raw methane enhancements based on the minimum threshold."""

    def crop(padded_enhancement: np.ndarray, min_molm2: float) -> np.ndarray:
        """Crop the input array to the smallest subarray containing values greater than the threshold."""
        any_axis1 = np.any(padded_enhancement > min_molm2, axis=0)
        any_axis0 = np.any(padded_enhancement > min_molm2, axis=1)
        cropped_enhancement = padded_enhancement[any_axis0, :]
        cropped_enhancement = cropped_enhancement[:, any_axis1]
        return cropped_enhancement

    return [crop(raw, min_molm2) for raw in raw_enhancements]


def get_min_nonzero_detection_prob(
    emission_rates: np.ndarray, detection_prob: np.ndarray, threshold: float = 0.01
) -> tuple[float | None, float | None]:
    """Get the minimum non-zero detection probability and its corresponding emission rate."""
    valid_indices = np.where(detection_prob > threshold)[0]
    if not valid_indices.size:
        return None, None

    min_prob_idx = valid_indices[np.argmin(detection_prob[valid_indices])]
    return float(detection_prob[min_prob_idx]), int(emission_rates[min_prob_idx])


if __name__ == "__main__":
    import os

    # This makes MLFLOW not hang 2mins when trying to log artifacts
    os.environ["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "10"
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "10"

    main()

    df = pd.DataFrame([m.as_dict() for m in MEASUREMENTS])
    df.to_csv("timer_results.csv", index=False)
    mlflow.log_artifact("timer_results.csv")

    logging.info("FPR-DT pipeline completed successfully")
