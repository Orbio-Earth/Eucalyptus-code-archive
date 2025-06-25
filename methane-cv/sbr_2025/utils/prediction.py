"""Prediction functions."""

from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import torch
from lib.models.schemas import WatershedParameters
from lib.plume_masking import retrieval_mask_using_watershed_algo
from rasterio import open as rio_open
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.transform import Affine, array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from skimage.measure import label
from torch import nn

from sbr_2025.utils import convert_mol_m2_to_ppb, update_transform_for_crop
from src.data.landsat_data import LandsatGranuleAccess
from src.data.sentinel2 import Sentinel2Item
from src.inference.inference_functions import generate_predictions
from src.inference.inference_target_location import add_retrieval_to_pred
from src.training.loss_functions import TwoPartLoss
from src.training.transformations import BaseBandExtractor
from src.utils import PROJECT_ROOT
from src.utils.parameters import LANDSAT_HAPI_DATA_PATH, S2_HAPI_DATA_PATH
from src.utils.radtran_utils import RadTranLookupTable


@dataclass
class TileInfo:
    """Holds the different tile properties needed for reporting SBRs."""

    instrument_name: str
    date_analysis: str
    observation_date: str
    observation_timestamp: str
    start_time: str
    end_time: str
    imaging_mode: str
    off_nadir_angle: float
    viewing_azimuth: float
    solar_zenith: float
    solar_azimuth: float
    orbit_state: str | None = None
    gas: str = "methane"

    def asdict(self) -> dict[str, Any]:
        """Convert the TileInfo object to a dictionary."""
        return asdict(self)


@dataclass
class PlumeInfo:
    """Holds the different plume properties needed for reporting SBRs."""

    latitude: float
    longitude: float
    length: float
    IME: float
    Q: float  # emission rate
    Q_low: float  # emission rate lower bound
    Q_high: float  # emission rate upper bound
    Q_uncertainty_type: str | None
    u10_wind_speed: float
    u_eff_wind_speed: float
    wind_speed_eff_lower: float
    wind_speed_eff_upper: float
    wind_speed_uncertainty_type: str
    wind_direction: float
    area: float
    plume_likelihood_score: float
    date: str
    bbox: tuple[float, float, float, float]

    def asdict(self) -> dict[str, Any]:
        """Convert the PlumeInfo object to a dictionary."""
        return asdict(self)


@dataclass
class Prediction:
    """Holds the different prediction parts and derived results."""

    x_dict: np.ndarray
    binary_probability: np.ndarray
    marginal: np.ndarray
    conditional: np.ndarray
    mask: np.ndarray
    marginal_retrieval: np.ndarray | None = None
    conditional_retrieval: np.ndarray | None = None
    masked_conditional_retrieval: np.ndarray | None = None
    crop_x: int | None = None
    crop_y: int | None = None
    main_item: Any | None = None
    reference_items: list[Any] | None = None

    def asdict(self) -> dict[str, Any]:
        """Convert the Prediction object to a dictionary."""
        return asdict(self)


def predict_for_all_pairs(
    main_data: dict,
    reference_data: list[dict],
    model: nn.Module,
    device: torch.device,
    band_concatenator: BaseBandExtractor,
    lossFn: TwoPartLoss,
    watershed_params: WatershedParameters,
    skip_retrieval: bool = True,
) -> dict[tuple[datetime, datetime], Prediction]:
    """Predict for all pairs of reference data.

    Args:
        main_data: Dictionary containing the main tile data.
        reference_data: List of dictionaries containing reference tile data.
        model: PyTorch model to use for for predictions.
        device: Device to run the model on (e.g., "cpu" or "cuda").
        band_concatenator: Transform for concatenating bands.
        lossFn: Loss function for extracting preds parts.
        watershed_params: Parameters for watershed segmentation.
        skip_retrieval: If True, skips retrieval step and lookup table creation.

    Returns
    -------
        Dictionary mapping date pairs to prediction dictionaries.

    Raises
    ------
        ValueError: If the granule item type is unsupported.
    """
    if skip_retrieval:
        lookup_table = None
    else:
        granule_item = main_data["tile_item"]
        if isinstance(granule_item, LandsatGranuleAccess):
            hapi_data_path = LANDSAT_HAPI_DATA_PATH
        elif isinstance(granule_item, Sentinel2Item):
            hapi_data_path = S2_HAPI_DATA_PATH
        else:
            raise ValueError(f"Unsupported granule access type: {type(granule_item)}")
        lookup_table = RadTranLookupTable.from_params(
            instrument=granule_item.instrument,
            solar_angle=granule_item.solar_angle,
            observation_angle=granule_item.observation_angle,
            hapi_data_path=hapi_data_path,
            min_ch4=0.0,
            max_ch4=21.0,  # this value was selected based on the common value ranges of the sim plume datasets
            spacing_resolution=40000,
            ref_band=granule_item.swir16_band_name,
            band=granule_item.swir22_band_name,
            full_sensor_name=granule_item.sensor_name,
        )

    # Predict and store predictions for all date pairs
    predictions: dict[tuple[datetime, datetime], Prediction] = {}
    for before_idx, earlier_idx in product(range(len(reference_data)), repeat=2):
        data_before = reference_data[before_idx]
        data_item_before = data_before["tile_item"]
        data_earlier = reference_data[earlier_idx]
        data_item_earlier = data_earlier["tile_item"]

        date_before = data_item_before.time  # .date().isoformat()
        date_earlier = data_item_earlier.time  # .date().isoformat()

        predictions[(date_before, date_earlier)] = predict(
            main_data,
            [data_before, data_earlier],
            watershed_params,
            model,
            device,
            band_concatenator,
            lossFn,
            lookup_table,
            create_lookup_table=False,
        )
    return predictions


def predict(
    main_data: dict,
    reference_data: list[dict],
    watershed_params: WatershedParameters,
    model: nn.Module,
    device: torch.device | str | int,
    band_concatenator: BaseBandExtractor,
    lossFn: TwoPartLoss,
    lookup_table: RadTranLookupTable | None = None,
    create_lookup_table: bool = False,
) -> Prediction:
    """Generate predictions for a given data pair.

    Args:
        main_data: Dictionary containing the main tile data.
        reference_data: List of dictionaries containing reference tile data.
        model: PyTorch model to use for for predictions.
        device: Device to run the model on (e.g., "cpu" or "cuda").
        band_concatenator: Transform for concatenating bands.
        lossFn: Loss function for extracting preds parts.
        watershed_params: Parameters for watershed segmentation.
        lookup_table: Optional pre-loaded lookup table for faster retrieval.
        create_lookup_table: If True, creates a lookup table inside add_retrieval_to_pred().

    Returns
    -------
        Dictionary containing prediction arrays and input data.
    """
    preds = generate_predictions(main_data, reference_data, model, device, band_concatenator, lossFn)
    # Add retrievals if lookup table is provided or creation is requested
    if create_lookup_table or lookup_table is not None:
        preds = add_retrieval_to_pred(preds, main_data["tile_item"], lookup_table)

    preds["conditional_pred"] = preds["conditional_pred"].numpy()
    preds["binary_probability"] = preds["binary_probability"].numpy()
    preds["marginal_pred"] = preds["marginal_pred"].numpy()

    # Apply watershed segmentation to generate mask
    preds["predicted_mask"] = retrieval_mask_using_watershed_algo(watershed_params, preds["binary_probability"])

    if "conditional_retrieval" in preds:
        preds["marginal_retrieval"] = preds["conditional_retrieval"] * preds["binary_probability"]
        preds["masked_conditional_retrieval"] = np.ma.masked_where(
            preds["predicted_mask"] == 0, preds["conditional_retrieval"]
        )
    else:
        preds["marginal_retrieval"] = None
        preds["conditional_retrieval"] = None
        preds["masked_conditional_retrieval"] = None

    return Prediction(
        binary_probability=preds["binary_probability"],
        conditional=preds["conditional_pred"],
        marginal=preds["marginal_pred"],
        conditional_retrieval=preds["conditional_retrieval"],
        marginal_retrieval=preds["marginal_retrieval"],
        masked_conditional_retrieval=preds["masked_conditional_retrieval"],
        mask=preds["predicted_mask"],
        x_dict=preds["x_dict"],
        main_item=main_data["tile_item"],
        reference_items=[x["tile_item"] for x in reference_data],
    )


def get_center_buffer(arr: np.ndarray, buffer: int = 3) -> np.ndarray:
    """Extract a subarray from a 2D NumPy array of size (buffer*2 + 1, buffer*2 + 1).

    Example: if buffer=1 then the 3x3 center of 9 values will be returned.
    """
    center_row = arr.shape[0] // 2
    center_col = arr.shape[1] // 2

    return arr[center_row - buffer : center_row + buffer + 1, center_col - buffer : center_col + buffer + 1]


def calculate_rescaled_retrieval(
    labeled_mask: npt.NDArray, binary_probability: npt.NDArray, conditional_retrieval: npt.NDArray
) -> npt.NDArray:
    """
    Calculate rescaled retrieval by normalizing binary probability within each plume.

    Args:
        labeled_mask: Labeled mask of plumes (integer array)
        binary_probability: Binary probability map
        conditional_retrieval: Conditional retrieval values

    Returns
    -------
        numpy.ndarray: Rescaled retrieval map
    """
    # Create empty rescaled retrieval map
    rescaled_retrieval = np.zeros_like(conditional_retrieval)

    # Assert that labeled_mask contains only non-negative integers
    assert np.issubdtype(labeled_mask.dtype, np.integer), "labeled_mask must be an integer array"
    assert np.all(labeled_mask >= 0), "labeled_mask must contain only non-negative values"

    # Process each plume separately
    for i in range(1, labeled_mask.max() + 1):
        plume_mask = labeled_mask == i

        # Skip if plume is empty
        if not np.any(plume_mask):
            continue

        # Get binary probability values for this plume
        binary_prob_plume = binary_probability[plume_mask]

        # Normalize binary probability within the plume
        conditional_prob_plume = binary_prob_plume / binary_prob_plume.max()

        # Scale conditional retrieval by normalized probability
        rescaled_retrieval[plume_mask] = conditional_retrieval[plume_mask] * conditional_prob_plume

    return rescaled_retrieval


def predict_retrieval(  # noqa: PLR0913
    main_data: dict,
    reference_chips: list[dict],
    watershed_params: WatershedParameters,
    model: nn.Module,
    device: torch.device,
    band_concatenator: BaseBandExtractor,
    lossFn: TwoPartLoss,
    lookup_table: RadTranLookupTable,
    max_distance_pixels: int,
    pixel_width: int,
    wind_speed: float,
) -> tuple[float, float, float, Prediction, npt.NDArray, npt.NDArray]:
    """Predict and return center retrieval properties."""
    from sbr_2025.utils.quantification import calculate_major_axis_quantification, find_central_plume

    """Predict and return center retrieval properties."""
    preds = predict(
        main_data,
        reference_chips,
        watershed_params,
        model,
        device,
        band_concatenator,
        lossFn,
        lookup_table,
        create_lookup_table=False,
    )
    # Find the central plume
    labeled_mask = label(preds.mask)
    try:
        nearest_plume_label = find_central_plume(labeled_mask, max_distance_pixels, pixel_width)
    except Exception:
        return -1.0, -1.0, -1.0, preds, np.zeros_like(preds.binary_probability), np.zeros_like(preds.binary_probability)
    center_plume_mask = labeled_mask == nearest_plume_label

    # Calculate rescaled retrieval
    max_binary_probability = np.max(preds.binary_probability[center_plume_mask])  # type:ignore
    rescaled_retrieval = preds.conditional_retrieval * preds.binary_probability / max_binary_probability

    # Major axis method
    L, IME, emission_rate = calculate_major_axis_quantification(
        center_plume_mask,
        rescaled_retrieval,
        pixel_width,
        wind_speed=wind_speed,
    )
    return L, IME, emission_rate, preds, rescaled_retrieval, center_plume_mask


def export_predition_and_mask_to_geotiff(
    unmasked_retrieval_mol_m2: np.ndarray,
    binary_mask: np.ndarray,
    main_item_transform: Affine,
    crop_start_x: int,
    crop_start_y: int,
    main_item_crs: CRS,
    observation_date: str,
    satellite_name: str,
    output_root: str = "./data/submission_geotiffs/phase1_submission",
) -> None:
    """Export the prediction to a GeoTIFF file in submission format."""
    # Get the current date in the format required by the submission
    data_generation_date = datetime.now().strftime("%m%d%Y")

    # Construct the file names
    unmasked_retrieval_name = f"{data_generation_date}_{observation_date}_{satellite_name}_OrbioEarth_Enhancement.tif"
    binary_mask_name = f"{data_generation_date}_{observation_date}_{satellite_name}_OrbioEarth_Mask.tif"

    # Create output directory if it doesn't exist
    output_path = Path(output_root) / satellite_name

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Export has to be in WGS84
    dst_crs = CRS.from_epsg(4326)

    crop_transform = update_transform_for_crop(main_item_transform, crop_start_x, crop_start_y)

    height, width = unmasked_retrieval_mol_m2.shape

    # Calculate bounds and new transform
    bounds = array_bounds(height, width, crop_transform)
    dst_transform, width, height = calculate_default_transform(
        main_item_crs, dst_crs, width, height, left=bounds[0], bottom=bounds[1], right=bounds[2], top=bounds[3]
    )

    # Convert retrieval to ppb as this is what the submission format requires
    unmasked_retrieval_ppb = convert_mol_m2_to_ppb(unmasked_retrieval_mol_m2)

    # Load submission boundary and reproject to WGS84
    boundary_gdf = gpd.read_file(
        PROJECT_ROOT / "sbr_2025" / "notebooks" / "data" / "submission_boundary_2kmx2km.geojson"
    )
    boundary_gdf = boundary_gdf.to_crs(dst_crs)

    # Get geometry in GeoJSON-like dict format
    geometry = [feature.__geo_interface__ for feature in boundary_gdf.geometry]

    file_array_pairs = zip(
        [unmasked_retrieval_name, binary_mask_name],
        [unmasked_retrieval_ppb, binary_mask],
        ["float32", "uint8"],
        strict=False,
    )

    for filename, array, dtype in file_array_pairs:
        # Prepare destination array
        dst_array = np.empty((height, width), dtype=array.dtype)

        # Perform reprojection
        reproject(
            source=array,
            destination=dst_array,
            src_transform=crop_transform,
            src_crs=main_item_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )

        try:
            with (
                MemoryFile() as memfile,
                memfile.open(
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=1,
                    dtype=dst_array.dtype,
                    crs=dst_crs,
                    transform=dst_transform,
                ) as tmp_ds,
            ):
                tmp_ds.write(dst_array, 1)

                masked_array, masked_transform = mask(
                    tmp_ds, shapes=geometry, all_touched=False, filled=True, nodata=0, crop=True
                )

                # Save the masked result to disk (only write once here)
                with rio_open(
                    output_path / filename,
                    "w",
                    driver="GTiff",
                    height=masked_array.shape[1],
                    width=masked_array.shape[2],
                    count=1,
                    dtype=dtype,
                    crs=dst_crs,
                    transform=masked_transform,
                ) as dst:
                    dst.write(masked_array[0], 1)
            print(f"Successfully exported {filename} to {output_path / filename}")
        except Exception as e:
            print(f"Error exporting {filename}: {e}")


def get_reference_data(reference_data: list, target_date: datetime, max_days_difference: int = 30) -> list[dict]:
    """Return reference data within a certain number of days from the target date."""
    reference_data_ = []
    for reference in reference_data:
        ref_date = reference[
            "tile_item"
        ].time  # datetime.strptime(reference["tile_item"].item.datetime.date().isoformat(), "%Y-%m-%d")
        time_difference = abs(target_date - ref_date)
        # Check if the difference in days is within the threshold
        # timedelta.days gives the difference purely in days (ignoring hours etc.)
        if time_difference.days <= max_days_difference and time_difference.days > 0:
            reference_data_.append(reference)
    return reference_data_


def get_reference_data_before(reference_data: list, target_date: datetime, max_days_difference: int = 30) -> list[dict]:
    """Return reference data within a certain number of days BEFORE the target date."""
    reference_data_ = []
    for reference in reference_data:
        ref_date = reference[
            "tile_item"
        ].time  # datetime.strptime(reference["tile_item"].item.datetime.date().isoformat(), "%Y-%m-%d")
        time_difference = target_date - ref_date
        # Check if the difference in days is within the threshold
        # timedelta.days gives the difference purely in days (ignoring hours etc.)
        if time_difference.days <= max_days_difference and time_difference.days > 0:
            reference_data_.append(reference)
    return reference_data_
