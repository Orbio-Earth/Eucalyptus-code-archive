"""
Methane quantification utilities.

Includes different methods for choosing the lengthscale.
These are tested and compared in the SBR_masking.ipynb notebook.
"""

from datetime import datetime
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio.features
import torch
import xarray as xr
from azure.storage.blob import BlobServiceClient
from lib.models.schemas import WatershedParameters
from lib.plume_quantification import _generate_geos_wind_data_url, _wind_velocities
from rasterio.crs import CRS
from shapely.geometry import MultiPolygon, Polygon
from torch import nn

from sbr_2025.utils import select_reference_tiles_from_dates_str
from sbr_2025.utils.prediction import export_predition_and_mask_to_geotiff
from src.training.loss_functions import TwoPartLoss
from src.training.transformations import BaseBandExtractor
from src.utils import PROJECT_ROOT
from src.utils.parameters import SatelliteID
from src.utils.quantification_utils import calc_effective_wind_speed, calc_Q_IME, calc_wind_direction
from src.utils.radtran_utils import RadTranLookupTable


def calculate_sqrtA_quantification(
    plume_mask: npt.NDArray, retrieval: npt.NDArray, pixel_width: float, wind_speed: float
) -> tuple[float, float, float]:
    """
    Calculate methane quantification using the square root of area method.

    This is the method we use in production, but without the buggy effective wind speed step.

    Args:
        plume_mask: Binary mask of the plume
        retrieval: Conditional or marginal retrieval values (mol/m2)
        pixel_width: Width of a pixel in meters
        wind_speed: Wind speed in m/s

    Returns
    -------
        tuple: (plume_length, IME, quantification)
    """
    pixel_area = pixel_width**2
    # Assert that plume_mask is a binary mask (contains only 0s and 1s)
    assert np.array_equal(
        plume_mask, plume_mask.astype(bool)
    ), "plume_mask must be a binary mask containing only 0s and 1s"

    # Calculate IME
    IME = retrieval[plume_mask].sum() * pixel_area

    # Calculate plume length (in m) using sqrt of area
    L = np.sqrt(np.sum(plume_mask)) * pixel_width

    # Calculate quantification
    quantification = calc_Q_IME(wind_speed, L, IME)

    return L, IME, quantification


def calculate_major_axis_quantification(
    plume_mask: npt.NDArray, retrieval: npt.NDArray, pixel_width: float, wind_speed: float
) -> tuple[float, float, float]:
    """
    Calculate methane quantification using the major axis length method.

    Args:
        plume_mask: Binary mask of the plume
        retrieval: Conditional or marginal retrieval values (mol/m2)
        pixel_width: Width of a pixel in meters
        wind_speed: Wind speed in m/s

    Returns
    -------
        tuple: (major_axis_length, IME, quantification)
    """
    pixel_area = pixel_width**2
    # Assert that plume_mask is a binary mask (contains only 0s and 1s)
    assert np.array_equal(
        plume_mask, plume_mask.astype(bool)
    ), "plume_mask must be a binary mask containing only 0s and 1s"

    # Calculate IME
    IME = retrieval[plume_mask].sum() * pixel_area

    # Get plume length using major axis
    contours = rasterio.features.shapes(plume_mask.astype(np.uint8), mask=plume_mask)
    polygons = [Polygon([(j[0], j[1]) for j in i[0]["coordinates"][0]]) for i in contours]

    plume_polygon = MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]

    min_rect = plume_polygon.minimum_rotated_rectangle
    coords = np.array(min_rect.exterior.coords)
    sides = np.array([np.linalg.norm(coords[i] - coords[i - 1]) for i in range(1, len(coords))])
    major_axis_length = np.max(sides) * pixel_width

    # Calculate quantification
    major_axis_quantification = calc_Q_IME(wind_speed, major_axis_length, IME)

    return major_axis_length, IME, major_axis_quantification


def calculate_circle_distance_quantification(
    plume_mask: npt.NDArray,
    retrieval: npt.NDArray,
    pixel_width: float,
    wind_speed: float,
    min_distance_pixels: float = 2,
    max_distance_pixels: float = 10,
) -> tuple[float, float, float]:
    """
    Calculate methane quantification using the circle distance method.

    Args:
        predicted_mask: Binary mask of the plume
        retrieval: Retrieval values (conditional, marginal, or rescaled)
        pixel_width: Width of a pixel in meters
        wind_speed: Wind speed in m/s
        min_distance_pixels: Minimum distance in pixels to consider
        max_distance_pixels: Maximum distance in pixels to consider

    Returns
    -------
        tuple: (optimal_distance, IME_at_optimal, quantification)
    """
    pixel_area = pixel_width**2
    # Assert that plume_mask is a binary mask (contains only 0s and 1s)
    assert np.array_equal(
        plume_mask, plume_mask.astype(bool)
    ), "plume_mask must be a binary mask containing only 0s and 1s"

    # Get dimensions of the raster
    height, width = retrieval.shape

    # Calculate center coordinates
    center_y = height // 2
    center_x = width // 2

    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:height, :width]

    # Calculate Euclidean distance from each pixel to center (in pixels)
    distance_to_center = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

    # Flatten arrays to 1D for sorting
    flat_distance = distance_to_center[plume_mask].flatten()
    flat_retrieval = retrieval[plume_mask].flatten()

    # Sort retrieval values by distance from center
    sorted_indices = np.argsort(flat_distance)
    sorted_distances = flat_distance[sorted_indices] * pixel_width  # Convert to meters
    sorted_retrieval = flat_retrieval[sorted_indices]

    # Calculate cumulative sum of retrieval values
    cumulative_retrieval = np.cumsum(sorted_retrieval) * pixel_area

    # Calculate quantification at different distances
    # Only consider distances between 2-10 pixels (or user-defined max)
    min_distance = min_distance_pixels * pixel_width
    max_distance = max_distance_pixels * pixel_width

    mask_in_range = (sorted_distances >= min_distance) & (sorted_distances <= max_distance)
    distances_filtered = sorted_distances[mask_in_range]
    cumulative_filtered = cumulative_retrieval[mask_in_range]

    if len(distances_filtered) == 0:
        raise ValueError("No distances in range found")

    # Calculate quantification at each distance
    quantification_by_distance = [
        calc_Q_IME(wind_speed, d, c) for (d, c) in zip(distances_filtered, cumulative_filtered, strict=False)
    ]

    # Find the maximum quantification value and corresponding distance
    max_idx = np.argmax(quantification_by_distance)
    optimal_distance = distances_filtered[max_idx]
    optimal_IME = cumulative_filtered[max_idx]
    max_quantification = quantification_by_distance[max_idx]

    return optimal_distance, optimal_IME, max_quantification


def find_central_plume(plume_labels: np.typing.NDArray, max_distance_pixels: float, pixel_width: float) -> int:
    """
    Find the plume closest to the center of the image.

    Args:
        plume_labels: Labeled mask of plumes
        max_distance_pixels: Maximum allowed distance in pixels from the center

    Returns
    -------
        int: Label of the central plume
    """
    # Find the plume closest to the center
    height, width = plume_labels.shape
    center_y, center_x = height // 2, width // 2
    y_coords, x_coords = np.ogrid[:height, :width]
    distance_to_center = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) * pixel_width

    # Mask pixels beyond max distance
    distance_mask = (distance_to_center / pixel_width) <= max_distance_pixels
    combined_mask = np.logical_and(plume_labels != 0, distance_mask)
    masked_distance = np.ma.masked_where(~combined_mask, distance_to_center)

    if masked_distance.mask.all():
        raise ValueError(f"No plumes found within {max_distance_pixels} pixels from the center.")

    nearest_plume_pixel = np.unravel_index(np.argmin(masked_distance), plume_labels.shape)
    nearest_plume_label = plume_labels[nearest_plume_pixel]

    return nearest_plume_label


def download_era5_wind(overpass_dt: datetime, lat: float, lon: float, out_file: str) -> str | None:
    """
    Download ERA5 wind data for a given datetime and location.

    Args:
        overpass_dt: Datetime of the observation (with format YYYY-MM-DD HH:MM:SS)
        lat: Latitude
        lon: Longitude
        out_file: Path to save the ERA5 wind data

    Returns
    -------
        Path to the saved ERA5 wind data or None if download fails
    """
    try:
        import cdsapi
    except ImportError as e:
        raise ImportError(
            "cdsapi package is required for ERA5 wind data. Install it with: `pip install cdsapi` "
            "and then you may have to do some manual authentication on your account"
        ) from e

    c = cdsapi.Client()

    try:
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "year": str(overpass_dt.year),
                "month": f"{overpass_dt.month:02d}",
                "day": f"{overpass_dt.day:02d}",
                "time": [f"{overpass_dt.hour:02d}:00"],
                "format": "netcdf",
                "area": [lat + 0.25, lon - 0.25, lat - 0.25, lon + 0.25],  # N, W, S, E
            },
            out_file,
        )
        return out_file
    except Exception as e:
        print(f"❌ ERA5 download failed for {overpass_dt.date()}: {e}")
        return None


def extract_era5_wind(nc_file: str, lat: float, lon: float) -> tuple[float, float] | None:
    """
    Extract wind vectors from ERA5 NetCDF file for a given latitude and longitude.

    Args:
        nc_file: Path to the ERA5 NetCDF file
        lat: Latitude
        lon: Longitude

    Returns
    -------
        Tuple of (ux, uy) wind components in m/s. Returns (None, None) if data not found.
    """
    try:
        ds = xr.open_dataset(nc_file)
        ds_point = ds.sel(latitude=lat, longitude=lon, method="nearest")

        ux = ds_point["u10"].values.item()
        uy = ds_point["v10"].values.item()
        return float(ux), float(uy)

    except Exception as e:
        print(f"❌ Error reading ERA5 NetCDF: {e}")
        return None


def get_wind_vectors(
    dt: datetime,
    lat: float,
    lon: float,
    source: Literal["GEOS", "ERA5"],
) -> tuple[float, float] | tuple[None, None]:
    """
    Get wind vectors (ux, uy) for a given datetime and location.

    Note: This function requires the cdsapi package and proper authentication.
    To authenticate:
    1. Sign up at https://cds.climate.copernicus.eu/
    2. Install cdsapi: pip install cdsapi
    3. Create ~/.cdsapirc file with your API key:
        url: https://cds.climate.copernicus.eu/api/v2
        key: <your-api-key>

    Args:
        dt: Datetime of the observation
        lat: Latitude
        lon: Longitude
        source: Wind data source ("GEOS" or "ERA5")

    Returns
    -------
        Tuple of (ux, uy) wind components in m/s. Returns (None, None) if data not found.
    """
    if source == "GEOS":
        # Query GEOS-FP
        try:
            url = _generate_geos_wind_data_url(dt)
            ux, uy = _wind_velocities(url, lat, lon)
            return float(ux), float(uy)
        except Exception as e:
            print(f"❌ Error generating GEOS-FP URL: {e}")
            return None, None

    else:  # ERA5
        # Query ERA5
        outfile = f"era5_wind_{dt.strftime('%Y%m%d_%H')}.nc"
        nc_path = download_era5_wind(dt, lat, lon, outfile)

        if nc_path:
            wind = extract_era5_wind(nc_path, lat, lon)
            if wind:
                return wind

    return None, None


def get_wind_components(wind_data: pd.DataFrame, sensing_time: datetime) -> dict[str, tuple[float, float]]:
    """Get wind components from wind data DataFrame for a specific sensing time.

    Args:
        wind_data: DataFrame containing wind data
        sensing_time: Sensing time string in format 'YYYY-MM-DDThh:mm:ss+0000'

    Returns
    -------
        Dictionary with wind components for each source (geos, era5, gt)
    """
    # Convert sensing_time to datetime for comparison
    wind_data["sensing_time"] = pd.to_datetime(wind_data["sensing_time"])

    target_dt = sensing_time

    # Find row where hours and minutes match
    mask = (
        (wind_data["sensing_time"].dt.year == target_dt.year)
        & (wind_data["sensing_time"].dt.month == target_dt.month)
        & (wind_data["sensing_time"].dt.day == target_dt.day)
        & (wind_data["sensing_time"].dt.hour == target_dt.hour)
        & (wind_data["sensing_time"].dt.minute == target_dt.minute)
    )
    if not mask.any():
        raise ValueError(f"No matching wind data found for sensing time {sensing_time}")

    if mask.sum() > 1:
        raise ValueError(f"Multiple matches found for sensing time {sensing_time}")

    # Get the matching row
    row = wind_data.loc[mask].iloc[0]

    # Extract wind components
    wind_components = {
        "geos": (float(row["geos_ux"]), float(row["geos_uy"])),
        "era5": (float(row["era5_ux"]), float(row["era5_uy"])),
        "gt": (float(row["gt_ux"]), float(row["gt_uy"])),
    }
    return wind_components


def quantification_interval(
    Q_central: float,
    wind_speed: float,
    Q_ensemble: npt.ArrayLike | None = None,
    model_mdALE: float = 0.1083,
    quantification_MdALE: float = 0.13,
    wind_MdALE: float = 0.32,
) -> dict[str, tuple[float, float]]:
    """
    Calculate the 95% confidence interval for the quantification and wind speed.

    This function estimates the uncertainty by combining three primary sources of error,
    assuming they are independent and log-normally distributed:
    1.  **Quantification Method Error:** Intrinsic error of the chosen quantification
        algorithm (e.g., major axis, sqrt(A), circle). Estimated using controlled
        simulations like the Gorroño plumes.
    2.  **Wind Speed Error:** Uncertainty in the wind speed data used for quantification.
        Estimated by comparing meteorological models (e.g., GEOS-FP, ERA5) to
        ground truth measurements.
    3.  **Model/Reference Variability:** Uncertainty arising from the choice of specific
        reference images and potentially slight variations in model predictions.
        Estimated from the spread of an ensemble of quantifications performed
        using different reference images but the *same* wind speed.

    The errors are combined in log space, assuming normality, and a 2-sigma
    interval (approximating 95% confidence) is calculated and transformed back
    to the original scale.
    Args:
        Q_central: The central quantification value (e.g., the median or mean of an
                   ensemble, or a single best estimate) in kg/h.
        Q_ensemble: An array-like collection of quantification values (kg/h)
        wind_speed: The central wind speed value used for quantification (m/s).
        Q_ensemble, Optional, if empty, the default model_mdALE will be used:
                    An array-like collection of quantification values (kg/h)
                    representing the spread due to model/reference choices.
                    Crucially, these *must* all be calculated using the *same*
                    wind speed data as Q_central.
        wind_speed: The central wind speed value used for quantification (m/s).
        model_mdALE: The Median Absolute Logarithmic Error (MdALE) representing the
                    uncertainty in the model/reference variability. The default of
                    0.1083 is based on an example with the following emission rates:
                    [477, 474, 451, 466, 430, 466, 501, 540, 502, 769, 739, 724, 641,
                    572, 481, 435, 494, 485, 508, 826, 434, 686, 426, 581, 553, 589, 558]
        quantification_MdALE: The Median Absolute Logarithmic Error (MdALE)
                              representing the intrinsic error of the quantification
                              method itself. The default of 0.13 is based on the
                              major axis method applied to Gorroño plumes (known
                              enhancement and wind speed).
                              MdALE = median(|ln(Q_pred) - ln(Q_true)|).
        wind_MdALE: The Median Absolute Logarithmic Error (MdALE) representing the
                    uncertainty in the wind speed estimate. The default of 0.32
                    corresponds to GEOS-FP wind speed estimates compared to ground
                    truth during SBR Phase 0. ERA5 had an estimated
                    MdALE of 0.45.

    Returns
    -------
        dict[str, tuple[float, float]]: A dictionary containing:
            - "quantification": (lower_Q, upper_Q) bounds for quantification (kg/h).
            - "wind": (lower_wind, upper_wind) bounds for wind speed (m/s).

    Notes
    -----
        - The factor 1.4826 is used to convert Median Absolute Deviation (MAD)
          to the standard deviation (sigma) under the assumption
          of a normal distribution.
        - We model the errors as log normal, as the retrievals are positive and skewed,
          and so we can model the errors as relative and still add them in quadrature.
        - The total variance in log space (sigma_log^2) is calculated as the sum
          of the variances from the three error sources:
          sigma_log^2 = (sigma_quant_method)^2 + (sigma_wind)^2 + (sigma_model)^2
        - The 95% CI is then [Q_central * exp(-2*sigma_log), Q_central * exp(2*sigma_log)].
    """
    # Convert Q_ensemble to a array, default to empty array if None
    Q_ensemble = np.array([]) if Q_ensemble is None else np.asarray(Q_ensemble)

    # The 1.4826 factor converts Median Absolute Deviation (MAD)
    # to the standard deviation (sigma) under the assumption
    # of a normal distribution.
    # to the standard deviation (sigma) under the assumption of a normal distribution.
    mad_to_sigma = 1.4826
    # 1.96 sigma corresponds to the 95% confidence interval for a normal distribution.
    z_score_95 = 1.96

    # Calculate wind speed interval
    log_wind_speed = np.log(wind_speed)
    wind_log_sigma = mad_to_sigma * wind_MdALE
    wind_low = np.exp(log_wind_speed - z_score_95 * wind_log_sigma)
    wind_high = np.exp(log_wind_speed + z_score_95 * wind_log_sigma)

    # Calculate quantification interval components
    quantification_log_sigma = mad_to_sigma * quantification_MdALE

    if len(Q_ensemble) > 0:
        # Compute the standard deviation of the quantification
        # in the ensemble, on a log scale, as an estimate of the model uncertainty.
        # Use median absolute deviation from the median for robustness
        log_Q_ensemble = np.log(np.asarray(Q_ensemble))
        log_Q_ensemble = np.log(Q_ensemble)
        log_Q_central = np.log(Q_central)
        # Combine central value with ensemble for MAD calculation
        log_Q_all = np.append(log_Q_ensemble, log_Q_central)
        print(f"Model/Reference Variability is calculated from {len(log_Q_all)} ensemble emission rates.")
        median_log_Q = np.median(log_Q_all)
        print(f"{median_log_Q=:.3f}")
        model_mdALE = float(np.median(np.abs(log_Q_all - median_log_Q)))
        print(f"{model_mdALE=:.3f}")
        model_mdALE = float(np.median(np.abs(log_Q_all - median_log_Q)))
    else:
        print(
            f"Model/Reference Variability is calculated using the default "
            f"Median Absolute Logarithmic Error {model_mdALE:.3f}."
        )

    model_log_sigma = mad_to_sigma * model_mdALE
    print(f"{quantification_log_sigma=:.3f}")
    print(f"{wind_log_sigma=:.3f}")
    print(f"{model_log_sigma=:.3f}")

    # We're modelling the error as normally distributed in logarithmic scale
    # So the total variance is the sum of the variances of the three sources of error
    total_log_sigma = np.sqrt(quantification_log_sigma**2 + wind_log_sigma**2 + model_log_sigma**2)

    # Calculate quantification 95% CI bounds
    quantification_lower = Q_central * np.exp(-z_score_95 * total_log_sigma)
    quantification_upper = Q_central * np.exp(z_score_95 * total_log_sigma)

    return {
        "quantification": (quantification_lower, quantification_upper),
        "wind": (wind_low, wind_high),
    }


def sbr_form_outputs(  # noqa: PLR0913, PLR0915, PLR0912
    decision_dict: dict,
    wind_data: pd.DataFrame,
    model_ids: str,
    models: list[nn.Module],
    device: torch.device,
    band_concatenators: list[BaseBandExtractor],
    main_data: dict,
    reference_data: list[dict],
    lossFn: TwoPartLoss,
    lookup_table: RadTranLookupTable,
    max_distance_pixels: int,
    pixel_width: int,
    satellite_id: SatelliteID,
    abs_client: BlobServiceClient,
) -> str:
    """Print the SBR form outputs and export the submission .tif files."""
    from sbr_2025.utils.prediction import predict_retrieval

    main_item = main_data["tile_item"]
    target_datetime = main_item.time

    if "wind_source" not in decision_dict:
        raise ValueError("decision_dict needs to contain the key 'wind_source'")
    if "watershed_marker_t" not in decision_dict:
        raise ValueError("decision_dict needs to contain the key 'watershed_marker_t'")
    if "watershed_floor_t" not in decision_dict:
        raise ValueError("decision_dict needs to contain the key 'watershed_floor_t'")

    watershed_params = WatershedParameters(
        marker_distance=1,
        marker_threshold=decision_dict["watershed_marker_t"],
        watershed_floor_threshold=decision_dict["watershed_floor_t"],
        closing_footprint_size=0,
    )

    wind_components = get_wind_components(wind_data, target_datetime)

    u_wind_component, v_wind_component = wind_components["geos"]
    wind_speed_geos = calc_effective_wind_speed(u_wind_component, v_wind_component)
    wind_direction_geos = calc_wind_direction(u_wind_component, v_wind_component)

    u_wind_component, v_wind_component = wind_components["era5"]
    wind_speed_era5 = calc_effective_wind_speed(u_wind_component, v_wind_component)
    wind_direction_era5 = calc_wind_direction(u_wind_component, v_wind_component)

    # Wind speed and direction
    if decision_dict["wind_source"] == "era5":
        wind_speed = wind_speed_era5
        wind_direction = wind_direction_era5
    elif decision_dict["wind_source"] == "geos":
        wind_speed = wind_speed_geos
        wind_direction = wind_direction_geos
    elif decision_dict["wind_source"] == "era5_geos_mean":
        wind_speed = (wind_speed_geos + wind_speed_era5) / 2
        wind_direction = wind_direction_era5
        print("[WARN] using ERA5 for wind direction when wind speed is averaged.")
    else:
        raise ValueError(f"Unknown 'wind_source'={decision_dict['wind_source']}")

    # PlumeLength (L), IME, EmissionRate from 'selected_retrieval'
    model_id = decision_dict["selected_retrieval"]["model_id"]
    model_idx = model_ids.index(model_id)

    before_date = decision_dict["selected_retrieval"]["date_before"]
    earlier_date = decision_dict["selected_retrieval"]["date_earlier"]
    reference_chips = select_reference_tiles_from_dates_str(
        reference_data, before_date=before_date, earlier_date=earlier_date
    )
    L, IME, emission_rate, preds, rescaled_retrieval, center_plume_mask = predict_retrieval(
        main_data,
        reference_chips,
        watershed_params,
        models[model_idx],
        device,
        band_concatenators[model_idx],
        lossFn,
        lookup_table,
        max_distance_pixels,
        pixel_width,
        wind_speed,
    )

    # EmissionRateUpper, EmissionRateLower, EmissionRateUncertaintyType from quantification_interval using
    # multiple emission rates from 'emission_ensemble_selections'
    emission_rates = []
    for emission_ensemble_selection in decision_dict["emission_ensemble_selections"]:
        model_id = emission_ensemble_selection["model_id"]
        model_idx = model_ids.index(model_id)

        before_date = emission_ensemble_selection["date_before"]
        earlier_date = emission_ensemble_selection["date_earlier"]
        reference_chips = select_reference_tiles_from_dates_str(
            reference_data, before_date=before_date, earlier_date=earlier_date
        )
        _, _, emission_rate_, _, _, _ = predict_retrieval(
            main_data,
            reference_chips,
            watershed_params,
            models[model_idx],
            device,
            band_concatenators[model_idx],
            lossFn,
            lookup_table,
            max_distance_pixels,
            pixel_width,
            wind_speed,
        )
        if emission_rate_ != -1:
            emission_rates.append(emission_rate_)
    print(f"Emission Rates from all ensemble selections: {[int(k) for k in emission_rates]}")

    # Also, WindSpeedUpper, WindSpeedLower from quantification_interval
    intervals = quantification_interval(
        Q_central=emission_rate,
        Q_ensemble=emission_rates,
        wind_speed=wind_speed,
        quantification_MdALE=0.13,
        wind_MdALE=0.32,
    )
    quantification_lower = intervals["quantification"][0]
    quantification_upper = intervals["quantification"][1]
    wind_low = intervals["wind"][0]
    wind_high = intervals["wind"][1]
    # convert wind direction from a ±180° format (relative to North) to a 0-360° format (meteorological standard)
    wind_direction_0_360 = (wind_direction + 360) % 360

    column_names = [
        "PlumeLength",
        "IME",
        "EmissionRate",
        "EmissionRateUpper",
        "EmissionRateLower",
        "EmissionRateUncertaintyType",
        "U10WindSpeed",
        "UeffWindSpeed",
        "WindSpeedUpper",
        "WindSpeedLower",
        "WindSpeedUncertaintyType",
        "WindDirection",
    ]
    row_values = [
        L,
        IME,
        emission_rate,
        quantification_upper,
        quantification_lower,
        "2 sigma",
        wind_speed,
        wind_speed,
        wind_high,
        wind_low,
        "2 sigma",
        wind_direction_0_360,
        decision_dict["note"],
    ]

    for col, value in zip(column_names, row_values, strict=False):
        print(f"{col:30}: {value}")

    print("#" * 100)
    print("#" * 100)
    if L == -1:
        print("Nothing detected. Here we assume we don't need to fill in the SBR form for this date.")
        csv_row = ",".join("" for val in row_values) + "\n"
    else:
        print(
            "Copy the following ',' separated line into the 'SBR 2025 - phase 1 v1' sheets into the row your current "
            "date is and into the column 'PlumeLength'. Then click on the clipboard symbol that appears and select "
            "'Split text to columns'"
        )
        csv_row = ",".join(str(val) for val in row_values) + "\n"
        print(csv_row)

    print("#" * 100)
    print("Exporting .tifs")

    if satellite_id == SatelliteID.S2:
        main_item_transform = main_item.get_raster_meta("B12")["transform"]
        crop_start_x = main_data["crop_params"]["B12"]["crop_start_x"]
        crop_start_y = main_data["crop_params"]["B12"]["crop_start_y"]
        satellite_name = main_item.instrument_name.replace("-", "")
    elif satellite_id == SatelliteID.LANDSAT:
        main_item_transform = main_item.get_raster_meta("blue", abs_client)["transform"]
        crop_start_x = main_data["crop_params"]["blue"]["crop_start_x"]
        crop_start_y = main_data["crop_params"]["blue"]["crop_start_y"]
        satellite_name = main_item.instrument_name.replace("_", "")

    export_predition_and_mask_to_geotiff(
        unmasked_retrieval_mol_m2=preds.marginal_retrieval,  # type: ignore
        binary_mask=center_plume_mask.astype(np.uint8),
        main_item_transform=main_item_transform,
        crop_start_x=crop_start_x,
        crop_start_y=crop_start_y,
        main_item_crs=CRS.from_string(main_item.crs),
        observation_date=target_datetime.date().strftime("%m%d%Y"),
        satellite_name=satellite_name,
        output_root=str(PROJECT_ROOT / "sbr_2025" / "notebooks" / "data" / "submission_geotiffs" / "phase1_submission"),
    )
    return csv_row
