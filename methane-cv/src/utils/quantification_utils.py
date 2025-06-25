"""Methane plume quantification functions."""

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import Affine


def calc_L_IME(data: np.ndarray, spatial_res_m: float = 20.0) -> tuple[float, float, float]:
    """
    Calculate plume area, effective plume size (L), and integrated methane enhancement (IME).

    Parameters
    ----------
    data : np.ndarray
        2D array of methane retrieval values in mol/m².
    spatial_res_m : float
        Side length of each pixel in meters. Default is 20.0 m for Sentinel 2.

    Returns
    -------
    area : float
        Total area of plume pixels in m².
    L : float
        Effective plume size in meters, defined as sqrt(area).
    IME : float
        Integrated Methane Enhancement in moles (summed retrieval * pixel area).
    """
    # Area of pixel
    # FIXME: Default value for the spatial resolution is 20.0 m, which is specific to Sentinel 2.
    pixel_area: float = spatial_res_m * spatial_res_m

    # Calculate area of plume pixels in m² (non-zero pixels)
    area: float = np.count_nonzero(data > 0) * pixel_area

    # Effective plume size (L)
    L: float = np.sqrt(area)

    # Integrated Methane Enhancement (IME)
    IME: float = np.sum(data[data > 0]) * pixel_area
    return area, L, IME


def calc_Q_IME(u_eff: float, L: float, IME: float) -> float:
    """
    Estimate the methane emission rate (Q) from effective wind speed (u_eff), effective plume size (L), and IME.

    This function implements:
    Q = (u_eff / L) x IME x 57.75286,
    where 57.75286 is the conversion factor from mol/s to kg/hr for CH₄, given:
        1 mol CH₄/s = 16.04246 g/s ≈ 0.01604 kg/s = 57.75 kg/hr.

    Parameters
    ----------
    u_eff : float
        Effective wind speed in m/s.
    L : float
        Effective plume size in meters.
    IME : float
        Integrated Methane Enhancement in moles.

    Returns
    -------
    float
        Methane emission rate, in kg/hr.
    """
    return (u_eff / L) * IME * 57.75286


def get_plume_source_location_v2(
    plume_retrieval: np.ndarray, source_crs: CRS, source_transform: Affine
) -> tuple[float, float]:
    """
    Calculate the latitude and longitude coordinates of the maximum methane retrieval pixel.

    This function:
    1. Identifies the pixel with the maximum value in `plume_retrieval`.
    2. Converts that pixel's row/column indices to real-world coordinates in `source_crs`.
    3. Transforms those coordinates into geographic coordinates (EPSG:4326).

    Parameters
    ----------
    plume_retrieval : np.ndarray
        A 2D array of methane retrieval or enhancement data. Must have two dimensions.
    source_crs : rasterio.crs.CRS
        The coordinate reference system of the `plume_retrieval`.
    source_transform : rasterio.transform.Affine
        The affine transform for converting the `plume_retrieval` array indices to real-world coordinates.

    Returns
    -------
    latitude : float
        Latitude.
    longitude : float
        Longitude.
    """
    assert len(plume_retrieval.shape) == 2, "Plume retrieval must have 2 dimesions for source location calculation"  # noqa: PLR2004 (magic-number-comparison)

    # Calculate the grid coordinates of the maximum retrieval pixel
    source_index = np.unravel_index(plume_retrieval.argmax(), plume_retrieval.shape)

    # Convert the grid (array) coordinates to real world coordinates in the source CRS
    transformer = rasterio.transform.AffineTransformer(source_transform)
    source_x_coord, source_y_coord = transformer.xy(source_index[0], source_index[1])

    # Transform real world coordinates in source CRS to latitude and longitude in CRS: 4326
    coordinate_transformer = Transformer.from_crs(source_crs.to_epsg(), 4326)
    latitude, longitude = coordinate_transformer.transform(source_x_coord, source_y_coord)

    return latitude, longitude


def calc_effective_wind_speed(u10m: float, v10m: float) -> float:
    """
    Compute the total wind speed in m/s from the zonal (u) and meridional (v) components at 10 m height.

    Calculated as the magnitude of the 2D wind vector: sqrt(u10m^2 + v10m^2).

    Parameters
    ----------
    u10m : float
        Zonal wind component at 10 m, in m/s. Positive values indicate eastward flow.
    v10m : float
        Meridional wind component at 10 m, in m/s. Positive values indicate northward flow.

    Returns
    -------
    float
        The wind speed at 10 m height, in m/s.
    """
    return np.sqrt(np.square(u10m) + np.square(v10m))


def calc_u_eff(wind_speed: float, wind_low: float, wind_high: float) -> tuple[float, float, float]:
    """
    Calculate effective wind speed and its uncertainty bounds using Varon et al. (2021) fitted parameters.

    The effective wind speed, u_eff, is modeled as:
        u_eff = alpha * ln(wind_speed + epsilon) + beta
    where alpha=0.33 and beta=0.45 are empirically derived for Sentinel 2 based on LES Simulations. A minimum
    effective wind speed of 0.01 m/s is enforced. This function also calculates the upper and lower effective
    wind speed by applying the same formula to `wind_high` and `wind_low`, respectively.

    Parameters
    ----------
    wind_speed : float
        Central estimate of wind speed in m/s.
    wind_low : float
        Lower bound of wind speed in m/s.
    wind_high : float
        Upper bound of wind speed in m/s.

    Returns
    -------
        tuple[float, float, float]: Effective wind speed, upper bound, and lower bound in m/s

    References
    ----------
        Varon 2021: https://amt.copernicus.org/articles/14/2771/2021/
        Discussion on gitlab: https://git.orbio.earth/orbio/orbio/-/issues/869
    """
    # Model parameters from Varon 2021
    alpha, beta = 0.33, 0.45

    # Small constant to avoid log(0)
    epsilon = 1e-16

    # Calculate effective wind speeds
    u_eff = max(alpha * np.log(wind_speed + epsilon) + beta, 0.01)
    u_eff_low = max(alpha * np.log(wind_low + epsilon) + beta, 0.009)
    u_eff_high = max(alpha * np.log(wind_high + epsilon) + beta, 0.011)

    return u_eff, u_eff_high, u_eff_low


def calc_wind_direction(u10m: float, v10m: float) -> float:
    """
    Compute wind direction in degrees from the zonal (u) and meridional (v) wind components at 10 m height.

    Parameters
    ----------
    u10m : float
        Zonal wind component at 10 m, in m/s.
    v10m : float
        Meridional wind component at 10 m, in m/s.

    Returns
    -------
    float
        Wind direction in degrees, measured from north.
    """
    return float(np.degrees(np.arctan2(u10m, v10m)))


def calc_wind_error(wind_speed: float, wind_error: float) -> tuple[float, float]:
    """
    Calculate lower and upper bounds of the wind speed given a fractional error.

    Parameters
    ----------
    wind_speed : float
        Central estimate of wind speed in m/s.
    wind_error : float
        Fractional error (e.g., 0.1 for ±10% error).

    Returns
    -------
    wind_low : float
        Wind speed lower bound in m/s (wind_speed * (1 - wind_error)).
    wind_high : float
        Wind speed upper bound in m/s (wind_speed * (1 + wind_error)).
    """
    wind_low: float = wind_speed * (1 - wind_error)
    wind_high: float = wind_speed * (1 + wind_error)
    return wind_low, wind_high
