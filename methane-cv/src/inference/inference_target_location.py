"""
Script to generate images of methane detections using a neural network at a particular latitude, longitude and date.

This script loads the model and exports the outputs using mlflow, and is expected to be run on Azure ML Studio.
"""

import argparse
import logging
import os
from datetime import datetime, timedelta

import geojson
import geopandas as gpd
import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import requests  # type: ignore
import torch
import xarray as xr
from affine import Affine
from azure.ai.ml import MLClient
from dateutil.parser import parse as parse_datetime  # type: ignore
from lib.models.schemas import WatershedParameters
from lib.plume_masking import retrieval_mask_using_watershed_algo
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mypy_boto3_s3 import S3Client
from rasterio import CRS
from skimage.measure import label, regionprops
from torch import nn

from src.azure_wrap.ml_client_utils import (
    create_ml_client_config,
    initialize_blob_service_client,
)
from src.data.common.utils import tensor_to_dataarray
from src.data.landsat_data import LandsatGranuleAccess
from src.data.sentinel2 import Sentinel2Item
from src.inference.inference_functions import (
    crop_main_data,
    crop_main_data_landsat,
    crop_reference_data,
    crop_reference_data_landsat,
    fetch_landsat_items_for_point,
    fetch_sentinel2_items_for_point,
    generate_predictions,
    plot_main_results,
    plot_ratio_diff_history,
    plot_ratio_history,
    plot_rgb_history,
)
from src.plotting import plotting_functions as plotting
from src.plotting.plotting_functions import (
    get_rgb_from_xarray,
    get_swir_ratio_from_xarray,
)
from src.training.loss_functions import TwoPartLoss
from src.training.transformations import (
    BaseBandExtractor,
)
from src.utils.exceptions import (
    InsufficientTemporalImageryException,
    MissingImageException,
)
from src.utils.parameters import (
    LANDSAT_BANDS,
    LANDSAT_HAPI_DATA_PATH,
    REQUIRED_NUM_PREVIOUS_SNAPSHOTS,
    S2_BANDS,
    S2_HAPI_DATA_PATH,
    SatelliteID,
)
from src.utils.quantification_utils import (
    calc_effective_wind_speed,
    calc_L_IME,
    calc_Q_IME,
    calc_u_eff,
    calc_wind_direction,
    calc_wind_error,
    get_plume_source_location_v2,
)
from src.utils.radtran_utils import RadTranLookupTable
from src.utils.utils import initialize_clients, load_model_and_concatenator

WIND_DATA_BASE_URL_TPL = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y{:04}/M{:02}/D{:02}"
WIND_ENTRY_NAME_TPL = "GEOS.fp.asm.{}.{:04}{:02}{:02}_{:02}00.V01.nc4"
WIND_PRODUCT = "inst3_2d_asm_Nx"

BACKGROUND_METHANE1 = 0.01

# TODO: find better solution than 0.5 for error
WIND_ERROR = 0.5

logger = logging.getLogger(__name__)

#######################################
######## Main function and loop #######
#######################################


def main(  # noqa PLR0913 (too-many-arguments)
    satellite_id: SatelliteID,
    lat: float,
    lon: float,
    start_date: datetime,
    end_date: datetime,
    crop_size: int,
    model_id: str,
    ml_client: MLClient,
    s3_client: S3Client,
    azure_cluster: bool,
) -> None:
    """Run trained model on specified lat, lon, date range using data."""
    model_id = str(model_id)

    if satellite_id == SatelliteID.S2:
        model_name = "models:/torchgeo_pwr_unet"
    elif satellite_id == SatelliteID.LANDSAT:
        model_name = "models:/landsat"
    else:
        raise ValueError(f"Unsupported satellite ID: {satellite_id}")

    model_identifier = f"{model_name}/{model_id}"
    if azure_cluster:
        mlflow.log_param("lat", lat)
        mlflow.log_param("lon", lon)
        mlflow.log_param("start_date", start_date)
        mlflow.log_param("end_date", end_date)
        mlflow.log_param("model_identifier", model_identifier)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device={device}")

    model, band_concatenator, train_params = load_model_and_concatenator(model_identifier, "cpu", satellite_id)
    model = model.to(device)
    model.eval()

    lossFn = TwoPartLoss(train_params["binary_threshold"], train_params["MSE_multiplier"])

    # Construct DataFrame to predict for every day between start_date and end_date
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.date().isoformat())
        current_date += timedelta(days=1)

    ground_truth_df = pd.DataFrame(
        {
            "lat": [lat] * len(dates),
            "lon": [lon] * len(dates),
            "date": dates,
        }
    )
    ground_truth_df["site"] = "Manual Input"
    ground_truth_df["quantification_kg_h"] = 0
    ground_truth_df["source"] = "Manual Input"

    if satellite_id == SatelliteID.S2:
        preds, plumes = plot_ground_truth_plots_S2(
            ground_truth_df,
            model,
            device,
            crop_size,
            binary_threshold=train_params["binary_threshold"],
            band_extractor=band_concatenator,
            lossFn=lossFn,
            ml_client=ml_client,
            s3_client=s3_client,
            azure_cluster=azure_cluster,
            visualize=True,
        )
    elif satellite_id == SatelliteID.LANDSAT:
        preds, plumes = plot_ground_truth_plots_landsat(
            ground_truth_df,
            model,
            device,
            crop_size,
            binary_threshold=train_params["binary_threshold"],
            band_extractor=band_concatenator,
            lossFn=lossFn,
            ml_client=ml_client,
            s3_client=s3_client,
            azure_cluster=azure_cluster,
            visualize=True,
        )
    else:
        raise ValueError(f"Unsupported satellite ID: {satellite_id}")

    # save predictions as tifs
    for pred in preds:
        save_retrieval_as_tif(
            pred["marginal_retrieval"],
            pred["conditional_retrieval"],
            pred["binary_probability"],
            pred["crop_x"],
            pred["crop_y"],
            crop_size,
            pred,
            str(pred["datetime"].date().isoformat()),
            azure_cluster,
        )

    df_preds = get_predictions_table(preds, plumes)
    df_preds.to_csv("df_detections.csv", index=False)
    if azure_cluster:
        mlflow.log_artifact("df_detections.csv")
        os.remove("df_detections.csv")

    # Save plumes as geojson
    start_date_iso = start_date.date().isoformat()
    end_date_iso = end_date.date().isoformat()
    filename = f"""{lat}_{lon}_from_{start_date_iso}_to_{end_date_iso}_plumepoints.geojson"""
    feature_collection = geojson.FeatureCollection(plumes)
    # Remove empty lists
    feature_collection["features"] = [
        feature for sublist in feature_collection["features"] if sublist for feature in sublist
    ]
    try:
        gpd.GeoDataFrame.from_features(feature_collection).to_file(filename, driver="GeoJSON")
        if azure_cluster:
            mlflow.log_artifact(filename)
            os.remove(filename)
    except Exception as e:
        logger.error(f"Failed to save geojson file: {e}")

    # Plot time series of detections/non-detections
    plot_time_series(
        df_preds,
        lat,
        lon,
        f"{lat}_{lon}_from_{start_date_iso}_to_{end_date_iso}.png",
        azure_cluster,
    )


#####################################
#### Obtain and export retrievals ###
#####################################


def get_predictions_table(predictions: list[dict], plumes_list: list) -> pd.DataFrame:
    """Create a dataframe with the predictions."""
    dates_arr = [pred["datetime"] for pred in predictions]
    min_frac = [pred["marginal_pred"].min().item() for pred in predictions]
    max_prob = [pred["binary_probability"].max().item() for pred in predictions]
    max_retrieval = [pred["marginal_retrieval"].max() for pred in predictions]

    quantifications = []
    quantifications_low = []
    quantifications_high = []
    for plumes in plumes_list:
        if len(plumes) > 0:
            props = plumes[0]["properties"]
            quantifications.append(np.round(props["Q"], 2))
            quantifications_low.append(np.round(props["Q_low"], 2))
            quantifications_high.append(np.round(props["Q_high"], 2))
        else:
            quantifications.append(0)
            quantifications_low.append(0)
            quantifications_high.append(0)
    df_preds = pd.DataFrame(
        {
            "time": dates_arr,
            "min_frac": min_frac,
            "max_prob": max_prob,
            "max_retrieval": max_retrieval,
            "quantifications": quantifications,
            "quantifications_low": quantifications_low,
            "quantifications_high": quantifications_high,
        }
    )
    return df_preds


def add_retrieval_to_pred(
    pred: dict, granule_item: LandsatGranuleAccess | Sentinel2Item, lookup_table: RadTranLookupTable | None = None
) -> dict:
    """Add retrieval to prediction dictionary."""
    for key in pred:
        if isinstance(pred[key], np.ndarray | torch.Tensor):
            pred[key] = pred[key].squeeze()  # should convert to 2D

    if isinstance(granule_item, LandsatGranuleAccess):
        hapi_data_path = LANDSAT_HAPI_DATA_PATH
    elif isinstance(granule_item, Sentinel2Item):
        hapi_data_path = S2_HAPI_DATA_PATH

    if lookup_table is None:
        lookup_table = RadTranLookupTable.from_params(
            instrument=granule_item.instrument,
            solar_angle=granule_item.solar_angle,
            observation_angle=granule_item.observation_angle,
            hapi_data_path=hapi_data_path,
            min_ch4=0.0,
            max_ch4=200.0,  # this value was selected based on the common value ranges of the sim plume datasets
            spacing_resolution=40000,
            ref_band=granule_item.swir16_band_name,
            band=granule_item.swir22_band_name,
            full_sensor_name=granule_item.sensor_name,
        )
    conditional_retrieval = lookup_table.reverse_lookup(pred["conditional_pred"].numpy())
    marginal_retrieval = conditional_retrieval * pred["binary_probability"].numpy()

    # Add retrievals to pred dict
    pred["conditional_retrieval"] = conditional_retrieval
    pred["marginal_retrieval"] = marginal_retrieval

    # Assert all arrays/tensors are 2D
    for key in pred:
        if isinstance(pred[key], np.ndarray | torch.Tensor):
            expected_dim = 2
            assert pred[key].ndim == expected_dim, "All arrays/tensors should be 2D"
    return pred


def save_retrieval_as_tif(
    marginal_retrieval: npt.NDArray,
    conditional_retrieval: npt.NDArray,
    binary_probability: npt.NDArray,
    crop_x: int,
    crop_y: int,
    crop_size: int,
    preds: dict,
    date: str,
    azure_cluster: bool,
) -> None:
    """Export retrieval as geotiff."""
    window = rasterio.windows.Window(crop_x, crop_y, crop_size, crop_size)
    if "s2_href" in preds:
        with rasterio.open(preds["s2_href"]) as src:
            metadata = src.meta.copy()
            new_transform = rasterio.windows.transform(window, src.transform)

    else:
        metadata = preds["meta"].copy()
        new_transform = rasterio.windows.transform(window, metadata["transform"])

    metadata.update(
        {
            "height": crop_size,
            "width": crop_size,
            "count": 3,  # number of bands
            "dtype": marginal_retrieval.dtype,  # could think about uint8
        }
    )

    metadata.update({"transform": new_transform})

    tif_output_path = f"{date}.tif"
    with rasterio.open(tif_output_path, "w", **metadata) as dst:
        # dst.update_tags(**src.tags()) # TODO: can we remove this?
        dst.update_tags(datetime=date)
        dst.write(marginal_retrieval, 1)
        dst.write(conditional_retrieval, 2)
        dst.write(binary_probability, 3)
    if azure_cluster:
        mlflow.log_artifact(tif_output_path, artifact_path="retrieval_tifs")
        os.remove(tif_output_path)


#################################
### Mask and Quantify Plumes ####
#################################


def quantify_retrieval(
    retrieval_array: npt.NDArray,
    retrieval_transform: Affine,
    retrieval_crs: CRS,
    retrieval_binary_probability: torch.Tensor,
    sensing_time: str,
    floor_t: float = 0.075,
    marker_t: float = 0.1,
    spatial_resolution: float = 20.0,
) -> list[geojson.Feature] | None:
    """
    Mask the retrieval arary using a threshold and quantify the masked plumes.

    Fields
    ------
    retrieval_array: numpy array with retrieval values
    retrieval_transform: retrieval tiff transfrom
    retrieval_crs: CRS of retrieval tile
    mask: binary mask indicating which pixels are plumes (=1)
    sensing_time: time (str) the sentinel-2 tile was captured
    """
    # Fetch wind tile for date
    wind_ds = download_wind_data_from_GEOS_FP(sensing_time)
    if wind_ds is None:
        return None

    watershed_segmentation_params = WatershedParameters(
        marker_distance=1,
        watershed_floor_threshold=floor_t,
        marker_threshold=marker_t,
        closing_footprint_size=0,
    )
    mask = retrieval_mask_using_watershed_algo(
        watershed_segmentation_params,
        retrieval_binary_probability.numpy()
        if isinstance(retrieval_binary_probability, torch.Tensor)
        else retrieval_binary_probability,
    )

    labeled_mask = label(mask, connectivity=2)  # Label connected regions

    plumes = []
    for region in regionprops(labeled_mask):
        # Get plume bounding box
        min_row, min_col, max_row, max_col = region.bbox

        # Get plume mask and array
        plume_mask = (labeled_mask[min_row:max_row, min_col:max_col] == region.label).astype(np.uint8)
        plume_data = retrieval_array[min_row:max_row, min_col:max_col] * plume_mask
        plume_prob_array = retrieval_binary_probability[min_row:max_row, min_col:max_col] * plume_mask
        # The likelihood score for the plume will be the maximum of the binary probability mask within
        # the mask of the plume. We then multiply it by 10. Why 10? The likelihood score shouldn't be
        # interpreted as a probability. Multiplying it by 10 makes it less tempting for people to do so.
        plume_likelihood = np.nanmax(plume_prob_array) * 10

        # Get plume geographical meta data
        plume_transform = retrieval_transform * Affine.translation(min_col, min_row)

        # Get plume source location
        latitude, longitude = get_plume_source_location_v2(plume_data, retrieval_crs, plume_transform)

        # Get wind data
        u_eff, u_eff_high, u_eff_low, wind_direction, u_wind_component = get_wind_speed_for_plume(
            wind_ds, latitude, longitude
        )

        # Quantify plume
        plume_area, L, IME = calc_L_IME(plume_data, spatial_resolution)
        emission_rate = calc_Q_IME(u_eff, L, IME)
        emission_rate_low = calc_Q_IME(u_eff_low, L, IME)
        emission_rate_high = calc_Q_IME(u_eff_high, L, IME)

        feature = geojson.Feature(
            geometry=geojson.Point((longitude, latitude)),
            properties={
                "Q": emission_rate,
                "Q_low": emission_rate_low,
                "Q_high": emission_rate_high,
                "Q_uncertainty_type": None,
                "wind_direction": wind_direction,
                "u10_wind_speed": u_wind_component,
                "u_eff_wind_speed": u_eff,
                "wind_speed_eff_upper": u_eff_high,
                "wind_speed_eff_lower": u_eff_low,
                "wind_speed_uncertainty_type": None,
                "latitude": latitude,
                "longitude": longitude,
                "IME": IME,
                "length": L,
                "area": plume_area,
                "plume_likelihood_score": plume_likelihood,
                "date": sensing_time.split("T")[0],
                "bbox": region.bbox,
            },
        )
        plumes.append(feature)
    return plumes


def quantify_plume_for_lat_lon(
    lat: float,
    lon: float,
    retrieval_array: npt.NDArray,
    retrieval_binary_probability: npt.NDArray,
    sensing_time: str,
    crop_size: int,
    floor_t: float = 0.075,
    marker_t: float = 0.1,
) -> dict[str, float] | None:
    """Quantify plume for a given lat, lon, date."""
    # Fetch wind tile for date
    wind_ds = download_wind_data_from_GEOS_FP(sensing_time)
    if wind_ds is None:
        logger.info(f"No wind data found for {sensing_time}")
        return None

    # Get wind data
    u_eff, u_eff_high, u_eff_low, wind_direction, u_wind_component = get_wind_speed_for_plume(wind_ds, lat, lon)

    # Get retrieval mask
    watershed_segmentation_params = WatershedParameters(
        marker_distance=1,
        watershed_floor_threshold=floor_t,
        marker_threshold=marker_t,
        closing_footprint_size=0,
    )
    mask = retrieval_mask_using_watershed_algo(watershed_segmentation_params, retrieval_binary_probability)

    labeled_mask = label(mask, connectivity=2)  # Label connected regions

    def is_plume_in_the_center_of_crop(
        plume_bbox: tuple[int, int, int, int], crop_size: int, center_buffer: int = 10
    ) -> bool:
        """Check if plume is in the center of the crop, centre buffer is 10 pixels."""
        crop_centre_x = crop_size // 2
        crop_centre_y = crop_size // 2

        crop_centre_array = np.zeros((crop_size, crop_size))
        crop_centre_array[
            crop_centre_x - center_buffer : crop_centre_x + center_buffer,
            crop_centre_y - center_buffer : crop_centre_y + center_buffer,
        ] = 1

        plume_bbox_array = np.zeros((crop_size, crop_size))
        plume_bbox_array[plume_bbox[0] : plume_bbox[2], plume_bbox[1] : plume_bbox[3]] = 1

        intersects = np.logical_and(crop_centre_array, plume_bbox_array).any()

        return intersects

    plume_data = None
    for region in regionprops(labeled_mask):
        if is_plume_in_the_center_of_crop(region.bbox, crop_size):
            min_row, min_col, max_row, max_col = region.bbox

            # Get plume mask and array
            plume_mask = (labeled_mask[min_row:max_row, min_col:max_col] == region.label).astype(np.uint8)
            plume_data = retrieval_array[min_row:max_row, min_col:max_col] * plume_mask
            break

    if plume_data is None:
        logger.info(f"No plume found in center of crop for {sensing_time}")
        return {
            "emission_rate": 0.0,
            "emission_rate_low": 0.0,
            "emission_rate_high": 0.0,
        }

    _, L, IME = calc_L_IME(plume_data, spatial_res_m=60)
    emission_rate = calc_Q_IME(u_eff, L, IME)
    emission_rate_low = calc_Q_IME(u_eff_low, L, IME)
    emission_rate_high = calc_Q_IME(u_eff_high, L, IME)

    logger.info(
        f"Plume found in center: Q: {emission_rate:4.0f}"
        f"kg/h ({emission_rate_low:4.0f} kg/hr - {emission_rate_high:4.0f} kg/hr)"
    )

    return {
        "emission_rate": emission_rate,
        "emission_rate_low": emission_rate_low,
        "emission_rate_high": emission_rate_high,
    }


#################################
########## Wind Data ############
#################################


def download_wind_data_from_GEOS_FP(sensing_time_str: str) -> xr.Dataset | None:
    """Download wind data from GEOS_FP for a given date."""
    sensing_time = parse_datetime(sensing_time_str)

    base_url = WIND_DATA_BASE_URL_TPL.format(sensing_time.year, sensing_time.month, sensing_time.day)
    entry_path = WIND_ENTRY_NAME_TPL.format(
        WIND_PRODUCT,
        sensing_time.year,
        sensing_time.month,
        sensing_time.day,
        sensing_time.hour,
    )

    # TODO: We should use the implementation as used in get_wind_vectors() from sbr_2025/utils/quantification.py
    # This does not return the closest 3h data
    url = f"{base_url}/{entry_path}"  # URL for the GEOS-FP data file

    print(f"{url=}")
    response = requests.get(url)  # Download the file

    tmp_file = "/tmp/geos_fp_data.nc4"  # Temporary file location
    try:
        # Save the file to the temporary location
        with open(tmp_file, "wb") as f:
            f.write(response.content)

        # Open the NetCDF file using xarray
        ds = xr.open_dataset(tmp_file)
        os.remove(tmp_file)
        return ds

    except Exception as e:
        print(f"Failed to download wind data. Reason: {e}")
        return None


def get_wind_vectors_from_wind_tile(wind_ds: xr.Dataset, plume_lat: float, plume_lon: float) -> tuple[float, float]:
    """Extract the U and V components from the wind dataset for the specific lat and lon."""
    u_wind_component = wind_ds["U10M"].sel(lat=plume_lat, lon=plume_lon, method="nearest").values.item()
    v_wind_component = wind_ds["V10M"].sel(lat=plume_lat, lon=plume_lon, method="nearest").values.item()

    return u_wind_component, v_wind_component


def get_wind_speed_for_plume(
    wind_ds: xr.Dataset,
    plume_source_lat: float,
    plume_source_lon: float,
) -> tuple[float, float, float, float, float]:
    """Get the wind vector for the plume source location and calculate the effective wind speeds."""
    u_wind_component, v_wind_component = get_wind_vectors_from_wind_tile(wind_ds, plume_source_lat, plume_source_lon)

    wind_speed = calc_effective_wind_speed(u_wind_component, v_wind_component)
    wind_direction = calc_wind_direction(u_wind_component, v_wind_component)

    # Calculate the wind error margin
    wind_low, wind_high = calc_wind_error(wind_speed, wind_error=WIND_ERROR)

    u_eff, u_eff_high, u_eff_low = calc_u_eff(wind_speed, wind_low, wind_high)

    return u_eff, u_eff_high, u_eff_low, wind_direction, u_wind_component


#################################
########## Plotting #############
#################################


def plot_ground_truth_plots_S2(  # noqa: PLR0913, PLR0915 (too many arguments, statements)
    ground_truth_df: pd.DataFrame,
    model: nn.Module,
    device: int | torch.device,
    crop_size: int,
    binary_threshold: float,
    band_extractor: BaseBandExtractor,
    lossFn: TwoPartLoss,
    ml_client: MLClient,
    s3_client: S3Client,
    required_num_previous_snapshots: int = REQUIRED_NUM_PREVIOUS_SNAPSHOTS,
    azure_cluster: bool = False,
    visualize: bool = True,
    sbr_notebook: bool = False,
) -> tuple[list, list]:
    """Plot the band ratio, predicted frac and RGB for a list of known methane sites."""
    abs_client = initialize_blob_service_client(ml_client)
    satellite_id = SatelliteID.S2

    preds_list, selected_plumes = [], []
    for _, row in ground_truth_df.iterrows():
        lat, lon = row[["lat", "lon"]]
        logger.info(f"PLOT FOR {row['site']} on {row['date']} ({lat}, {lon}) - {row['source']}")
        logger.info("=" * 120)
        query_datetime = datetime.fromisoformat(row["date"])
        try:
            items = fetch_sentinel2_items_for_point(lat, lon, query_datetime, crop_size, sbr_notebook)

            main_data = crop_main_data(items, abs_client, s3_client, lat, lon, crop_size)
            main_item = main_data["tile_item"]
            logger.info(f"Main tile for {query_datetime}: USE {main_item.id}")
        except (InsufficientTemporalImageryException, MissingImageException) as e:
            logger.info(f"{e}. Skipping")
            logger.info("=" * 120)
            logger.info("=" * 120)
            continue

        reference_data = crop_reference_data(
            items, main_data, abs_client, s3_client, lat, lon, crop_size, required_num_previous_snapshots
        )
        preds = generate_predictions(main_data, reference_data, model, device, band_extractor, lossFn)

        # Get Retrieval for the biggest plume in the center
        preds = add_retrieval_to_pred(preds, main_item)
        timestamp = main_item.time
        preds["datetime"] = timestamp
        preds["s2_href"] = main_item.item.assets["B12"].href

        crop_x = main_data["crop_params"]["B12"]["crop_start_x"]
        crop_y = main_data["crop_params"]["B12"]["crop_start_y"]
        preds["crop_x"] = crop_x
        preds["crop_y"] = crop_y
        preds_list.append(preds)

        with rasterio.open(main_item.item.assets["B12"].href) as src:
            crop_crs = src.crs
            window = rasterio.windows.Window(crop_x, crop_y, crop_size, crop_size)
            crop_transform = rasterio.windows.transform(window, src.transform)

        # Quantify plumes in retrieval
        plume_list = quantify_retrieval(
            preds["conditional_retrieval"],
            crop_transform,
            crop_crs,
            preds["binary_probability"],
            str(timestamp.date().isoformat()),
            floor_t=0.075,
            marker_t=0.1,
        )

        def intersects_center(
            min_row: int, min_col: int, max_row: int, max_col: int, img_size: int = 128, buffer: int = 3
        ) -> bool:
            """Check intersection of bbox with buffered center of an image."""
            # Calculate center of the image
            center_row = img_size / 2
            center_col = img_size / 2

            # Expand the check to include a buffer zone around the center
            return (
                min_row <= center_row + buffer
                and max_row >= center_row - buffer
                and min_col <= center_col + buffer
                and max_col >= center_col - buffer
            )

        if plume_list:
            # Get Retrieval for the biggest plume in the center
            plume_list = sorted(plume_list, key=lambda x: x["properties"]["Q"])[::-1]
            center_plumes = []
            for plume in plume_list:
                min_row, min_col, max_row, max_col = plume["properties"]["bbox"]
                intersects = intersects_center(min_row, min_col, max_row, max_col)
                if intersects:
                    max_likeli = np.round(plume["properties"]["plume_likelihood_score"], 2)
                    q = np.round(plume["properties"]["Q"], 2)
                    q_low = np.round(plume["properties"]["Q_low"], 2)
                    q_high = np.round(plume["properties"]["Q_high"], 2)
                    logger.info(
                        f"Plume found in center: Max Likeli {max_likeli:5.2f}, Q: {q:4.0f} kg/h "
                        f"({q_low:4.0f} - {q_high:4.0f})"
                    )
                    center_plumes.append(plume)
        else:
            center_plumes = []
        selected_plumes.append(center_plumes)

        if visualize:
            # TODO: Now that we have quantifications here, we could also plot marginal and conditional
            # RETRIEVAL instead of frac. Look at src.inference.inference_target_location.plot_prediction
            basename_prefix = row["site"].replace("/", "_")
            basename = f"{basename_prefix:15}_{row['lat']:.4f}_{row['lon']:.4f}_{row['date']}"

            # [:-4] skips the omnicloud bands
            crop_main = tensor_to_dataarray(preds["x_dict"]["crop_main"][0][:-4], S2_BANDS)
            crop_before = tensor_to_dataarray(preds["x_dict"]["crop_before"][0][:-4], S2_BANDS)
            crop_earlier = tensor_to_dataarray(preds["x_dict"]["crop_earlier"][0][:-4], S2_BANDS)

            swir_ratio_main = get_swir_ratio_from_xarray(crop_main, satellite_id)
            swir_ratio_before = get_swir_ratio_from_xarray(crop_before, satellite_id)
            swir_ratio_earlier = get_swir_ratio_from_xarray(crop_earlier, satellite_id)

            rgb_main = get_rgb_from_xarray(crop_main, satellite_id)

            ids = [main_data["tile_item"].id, reference_data[0]["tile_item"].id, reference_data[1]["tile_item"].id]
            date_earlier = ids[2].split("_")[2]
            date_earlier = f"{date_earlier[:4]}-{date_earlier[4:6]}-{date_earlier[6:8]}"
            date_before = ids[1].split("_")[2]
            date_before = f"{date_before[:4]}-{date_before[4:6]}-{date_before[6:8]}"
            date_main = ids[0].split("_")[2]
            date_main = f"{date_main[:4]}-{date_main[4:6]}-{date_main[6:8]}"

            plot_main_results(
                row,
                swir_ratio_main,
                rgb_main,
                preds,
                basename_prefix,
                basename,
                binary_threshold,
                azure_cluster,
                center_plumes,
            )
            plot_rgb_history(
                rgb_main,
                crop_before,
                crop_earlier,
                date_main,
                date_before,
                date_earlier,
                satellite_id,
                basename_prefix,
                basename,
                azure_cluster,
            )
            plot_ratio_history(
                swir_ratio_main,
                swir_ratio_before,
                swir_ratio_earlier,
                date_main,
                date_before,
                date_earlier,
                basename_prefix,
                basename,
                azure_cluster,
            )
            plot_ratio_diff_history(
                swir_ratio_main, swir_ratio_before, swir_ratio_earlier, basename_prefix, basename, azure_cluster
            )
        logger.info("=" * 120)
        logger.info("=" * 120)
    return preds_list, selected_plumes


def plot_ground_truth_plots_landsat(  # noqa: PLR0913, PLR0915 (too many arguments, statements)
    ground_truth_df: pd.DataFrame,
    model: nn.Module,
    device: int | torch.device,
    crop_size: int,
    binary_threshold: float,
    band_extractor: BaseBandExtractor,
    lossFn: TwoPartLoss,
    ml_client: MLClient,
    s3_client: S3Client,
    azure_cluster: bool = False,
    visualize: bool = True,
) -> tuple[list, list]:
    """Plot the band ratio, predicted frac and RGB for a list of known methane sites."""
    satellite_id = SatelliteID.LANDSAT

    preds_list, selected_plumes = [], []
    count = 0
    for _, row in ground_truth_df.iterrows():
        # We get a fresh ABS client for every 10th row, as this can take a while
        # and the token can expire halfway through the process.
        if count % 10 == 0:
            abs_client = initialize_blob_service_client(ml_client)
        count += 1
        lat, lon = row[["lat", "lon"]]
        logger.info(f"PLOT FOR {row['site']} on {row['date']} ({lat}, {lon}) - {row['source']}")
        logger.info("=" * 120)
        query_datetime = datetime.fromisoformat(row["date"])
        try:
            items = fetch_landsat_items_for_point(lat, lon, query_datetime)

            main_data = crop_main_data_landsat(items, abs_client, s3_client, lat, lon, crop_size)
            main_item = main_data["tile_item"]
            logger.info(f"Main tile for {query_datetime}: USE {main_item.id}")
        except (InsufficientTemporalImageryException, MissingImageException) as e:
            logger.info(f"{e}. Skipping")
            logger.info("=" * 120)
            logger.info("=" * 120)
            continue

        reference_data = crop_reference_data_landsat(items, main_data, abs_client, s3_client, lat, lon, crop_size)
        preds = generate_predictions(main_data, reference_data, model, device, band_extractor, lossFn)

        # Get Retrieval for the biggest plume in the center
        try:
            preds = add_retrieval_to_pred(preds, main_item)
        except Exception as e:
            logger.info(f"{e}. Skipping")
            logger.info("=" * 120)
            logger.info("=" * 120)
            continue
        timestamp = main_item.time
        preds["datetime"] = timestamp
        preds["meta"] = main_item.get_raster_meta("swir22", abs_client=abs_client)

        crop_x = main_data["crop_params"]["swir22"]["crop_start_x"]
        crop_y = main_data["crop_params"]["swir22"]["crop_start_y"]
        preds["crop_x"] = crop_x
        preds["crop_y"] = crop_y
        preds_list.append(preds)

        crop_crs = preds["meta"]["crs"]
        window = rasterio.windows.Window(crop_x, crop_y, crop_size, crop_size)
        crop_transform = rasterio.windows.transform(window, preds["meta"]["transform"])

        # Quantify plumes in retrieval
        plume_list = quantify_retrieval(
            preds["conditional_retrieval"],
            crop_transform,
            crop_crs,
            preds["binary_probability"],
            str(timestamp.date().isoformat()),
            floor_t=0.075,
            marker_t=0.1,
        )

        def intersects_center(
            min_row: int, min_col: int, max_row: int, max_col: int, img_size: int = 128, buffer: int = 3
        ) -> bool:
            """Check intersection of bbox with buffered center of an image."""
            # Calculate center of the image
            center_row = img_size / 2
            center_col = img_size / 2

            # Expand the check to include a buffer zone around the center
            return (
                min_row <= center_row + buffer
                and max_row >= center_row - buffer
                and min_col <= center_col + buffer
                and max_col >= center_col - buffer
            )

        if plume_list:
            # Get Retrieval for the biggest plume in the center
            plume_list = sorted(plume_list, key=lambda x: x["properties"]["Q"])[::-1]
            center_plumes = []
            for plume in plume_list:
                min_row, min_col, max_row, max_col = plume["properties"]["bbox"]
                intersects = intersects_center(min_row, min_col, max_row, max_col)
                if intersects:
                    max_likeli = np.round(plume["properties"]["plume_likelihood_score"], 2)
                    q = np.round(plume["properties"]["Q"], 2)
                    q_low = np.round(plume["properties"]["Q_low"], 2)
                    q_high = np.round(plume["properties"]["Q_high"], 2)
                    logger.info(
                        f"Plume found in center: Max Likeli {max_likeli:5.2f}, Q: {q:4.0f} kg/h "
                        f"({q_low:4.0f} - {q_high:4.0f})"
                    )
                    center_plumes.append(plume)
        else:
            center_plumes = []
        selected_plumes.append(center_plumes)

        if visualize:
            # TODO: Now that we have quantifications here, we could also plot marginal and conditional
            # RETRIEVAL instead of frac. Look at src.inference.inference_target_location.plot_prediction
            basename_prefix = row["site"].replace("/", "_")
            basename = f"{basename_prefix:15}_{row['lat']:.4f}_{row['lon']:.4f}_{row['date']}"

            crop_main = tensor_to_dataarray(preds["x_dict"]["crop_main"][0], LANDSAT_BANDS)
            crop_before = tensor_to_dataarray(preds["x_dict"]["crop_before"][0], LANDSAT_BANDS)
            crop_earlier = tensor_to_dataarray(preds["x_dict"]["crop_earlier"][0], LANDSAT_BANDS)

            swir_ratio_main = get_swir_ratio_from_xarray(crop_main, satellite_id)
            swir_ratio_before = get_swir_ratio_from_xarray(crop_before, satellite_id)
            swir_ratio_earlier = get_swir_ratio_from_xarray(crop_earlier, satellite_id)

            rgb_main = get_rgb_from_xarray(crop_main, satellite_id)

            ids = [main_data["tile_item"].id, reference_data[0]["tile_item"].id, reference_data[1]["tile_item"].id]
            date_earlier = LandsatGranuleAccess.parse_landsat_tile_id(ids[2])["acquisition_date"]
            date_earlier = f"{date_earlier[:4]}-{date_earlier[4:6]}-{date_earlier[6:8]}"
            date_before = LandsatGranuleAccess.parse_landsat_tile_id(ids[1])["acquisition_date"]
            date_before = f"{date_before[:4]}-{date_before[4:6]}-{date_before[6:8]}"
            date_main = LandsatGranuleAccess.parse_landsat_tile_id(ids[0])["acquisition_date"]
            date_main = f"{date_main[:4]}-{date_main[4:6]}-{date_main[6:8]}"

            plot_main_results(
                row,
                swir_ratio_main,
                rgb_main,
                preds,
                basename_prefix,
                basename,
                binary_threshold,
                azure_cluster,
                center_plumes,
            )
            plot_rgb_history(
                rgb_main,
                crop_before,
                crop_earlier,
                date_main,
                date_before,
                date_earlier,
                satellite_id,
                basename_prefix,
                basename,
                azure_cluster,
            )
            plot_ratio_history(
                swir_ratio_main,
                swir_ratio_before,
                swir_ratio_earlier,
                date_main,
                date_before,
                date_earlier,
                basename_prefix,
                basename,
                azure_cluster,
            )
            plot_ratio_diff_history(
                swir_ratio_main, swir_ratio_before, swir_ratio_earlier, basename_prefix, basename, azure_cluster
            )
        logger.info("=" * 120)
        logger.info("=" * 120)
    abs_client.close()  # close this client, ready for a new one to be created for the next row
    return preds_list, selected_plumes


def plot_prediction(pred_dict: dict, title: str) -> None:
    """Plot the RGB image and the predicted probability and marginal retrieval."""
    # Assert these are 2D, then continue with plotting
    marginal_retrieval_squeezed = pred_dict["marginal_retrieval"].squeeze()
    binary_probability_squeezed = pred_dict["binary_probability"].squeeze()
    conditional_retrieval_squeezed = pred_dict["conditional_retrieval"].squeeze()

    expected_dim = 2
    assert len(marginal_retrieval_squeezed.shape) == expected_dim, (
        f"Expected marginal_retrieval to be 2D, but got {len(marginal_retrieval_squeezed.shape)}D. "
        "Was add_retrieval_to_pred ran on the input dict before?"
    )
    assert len(binary_probability_squeezed.shape) == expected_dim, (
        f"Expected binary_probability to be 2D, but got {len(binary_probability_squeezed.shape)}D. "
        "Was add_retrieval_to_pred ran on the input dict before?"
    )
    assert len(conditional_retrieval_squeezed.shape) == expected_dim, (
        f"Expected conditional_pred to be 2D, but got {len(conditional_retrieval_squeezed.shape)}D. "
        "Was add_retrieval_to_pred ran on the input dict before?"
    )

    plt.subplot(2, 2, 1)
    rgb_image = plotting.get_rgb_from_tensor(pred_dict["x_dict"]["crop_main"], S2_BANDS, batch_idx=0)
    plt.imshow(rgb_image / rgb_image.max())
    prob = pred_dict["binary_probability"]
    vmax = max(np.abs(marginal_retrieval_squeezed).max(), 0.1)
    plt.imshow(
        conditional_retrieval_squeezed,
        vmin=0,
        vmax=vmax,
        cmap="viridis",
        alpha=prob,
    )
    plotting.grid16()
    plt.title(title)
    plt.colorbar(label="mol/m²")

    plt.subplot(2, 2, 2)
    plt.title("Likelihood score of methane presence (0-10)")
    plt.imshow(prob * 10, norm=LogNorm(vmin=0.1, vmax=10.0), cmap="pink_r")
    plotting.grid16()
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.imshow(
        marginal_retrieval_squeezed,
        norm=LogNorm(vmin=0.01, vmax=vmax),
        cmap="viridis",
    )
    plotting.grid16()
    plt.colorbar(label="mol/m²")
    plt.title("Marginal retrieval")

    plt.subplot(2, 2, 4)
    rgb_image = plotting.get_rgb_from_tensor(pred_dict["x_dict"]["crop_main"], S2_BANDS, batch_idx=0)
    plt.imshow(rgb_image / rgb_image.max())
    mask = np.where(marginal_retrieval_squeezed > BACKGROUND_METHANE1, 1.0, np.nan)
    plt.imshow(
        mask,
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    plotting.grid16()
    plt.title("Detection mask")
    plt.colorbar()


def plot_time_series(df_preds: pd.DataFrame, lat: float, lon: float, filename: str, azure_cluster: bool) -> None:
    """Plot time series of predicted retrieval values."""
    fig = plt.figure(figsize=(12, 6))
    # Calculate error bars (distance from mean to low/high)
    errors_low = [q - ql for q, ql in zip(df_preds["quantifications"], df_preds["quantifications_low"], strict=False)]
    errors_high = [qh - q for qh, q in zip(df_preds["quantifications_high"], df_preds["quantifications"], strict=False)]

    # Create the plot
    plt.errorbar(
        df_preds["time"],
        df_preds["quantifications"],
        yerr=[errors_low, errors_high],
        fmt="o",
        capsize=5,
        capthick=2,
        ecolor="red",
        markerfacecolor="blue",
        markeredgecolor="blue",
        label="Quantifications",
    )

    plt.ylabel("kg/h")
    plt.title(f"Methane Detections for ({lat}, {lon})")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    if azure_cluster:
        mlflow.log_figure(fig, filename)
    else:
        plt.show()


#################################
###### Handling inputs ##########
#################################


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    def parse_date(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%d")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--satellite",
        choices=SatelliteID.list(),
        type=SatelliteID,
        required=True,
        help="Satellite data source to use",
    )
    parser.add_argument("--crop_size", type=int, default=128, help="Size of the crop to use")
    parser.add_argument("--lat", required=True, type=float)
    parser.add_argument("--lon", required=True, type=float)
    parser.add_argument(
        "--start_date",
        help="The starting date of the analysis in YYYY-MM-DD format.",
        required=True,
        type=parse_date,
    )
    parser.add_argument(
        "--end_date",
        help="The end date of the analysis in YYYY-MM-DD format.",
        required=True,
        type=parse_date,
    )
    parser.add_argument(
        "--model_id",
        help="The model version to use, for example '70' Assumes its models:/torchgeo_pwr_unet/70.",
        required=True,
        type=str,
    )
    parser.add_argument("--azure_cluster", action="store_true", help="Is this running on an azure cluster?")

    return parser.parse_args()


if __name__ == "__main__":
    import argparse

    #  lat, lon = 31.6585, 5.9053  # Hassi
    #  start_time = datetime(2019, 10, 1)
    #  end_time = datetime(2019, 11, 1)
    #  model_id = "70"
    args = parse_args()

    # If running on Azure, create config.json
    if args.azure_cluster:
        create_ml_client_config()

    ml_client, _, _, _, s3_client = initialize_clients(args.azure_cluster)

    main(
        satellite_id=args.satellite,
        lat=args.lat,
        lon=args.lon,
        start_date=args.start_date,
        end_date=args.end_date,
        crop_size=args.crop_size,
        model_id=args.model_id,
        ml_client=ml_client,
        s3_client=s3_client,
        azure_cluster=args.azure_cluster,
    )
