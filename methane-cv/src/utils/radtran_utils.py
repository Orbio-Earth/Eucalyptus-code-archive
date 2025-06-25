"""
Radtran utilities for S2 and EMIT.

Currently, the functions are implemented separately for each satellite.
The goal is to develop sensor-agnostic functions and migrate them to the main radtran library.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pydantic import AnyUrl
from radtran.filter_functions import SpectralFilter, generate_global_filter
from radtran.lib import get_gamma
from radtran.lib import radtran as generate_lookup_tables
from radtran.utils.constants import m2_to_cm2, mol_to_N
from tqdm import tqdm
from xarray_einstats.stats import logsumexp

from src.utils import PROJECT_ROOT
from src.utils.utils import setup_logging

logger = setup_logging()

####################################################
################# S2 FUNCTIONS #####################
####################################################


class RadTranLookupTable:
    """
    Class for the radtran lookup table, to encapsulate and simplify its use for various applications.

    This class precomputes normalized brightness values over a range of methane concentrations
    to facilitate lookup operations for the reference band and target band.
    """

    def __init__(
        self,
        min_ch4: float,
        max_ch4: float,
        spacing_resolution: int,
        nB_grid_band: np.ndarray,
        nB_grid_ref_band: np.ndarray,
    ):
        self.min_ch4 = min_ch4
        self.max_ch4 = max_ch4
        self.spacing_resolution = spacing_resolution
        self.nB_grid_band = nB_grid_band
        self.nB_grid_ref_band = nB_grid_ref_band
        self.retrieval_grid = np.linspace(min_ch4, max_ch4, spacing_resolution)
        self.frac_grid = nB_grid_band / nB_grid_ref_band - 1

    @classmethod
    @lru_cache
    def from_params(  # noqa: PLR0913 (too-many-arguments)
        cls,
        instrument: str,
        solar_angle: float,
        observation_angle: float,
        hapi_data_path: AnyUrl,
        min_ch4: float,
        max_ch4: float,
        spacing_resolution: int,
        ref_band: str,
        band: str,
        full_sensor_name: str,
    ) -> RadTranLookupTable:
        """Create a RadTranLookupTable instance using instrument-specific parameters and precomputed radtran data."""
        instrument_full_name = cls._get_full_instrument_name(instrument, full_sensor_name)

        gamma = get_gamma(solar_angle, observation_angle)

        global_filter_ref_band = cls._prepare_global_filter_function(
            hapi_data_path,
            instrument_full_name,
            ref_band,
            gamma,
            T=300,
            p=1.013,
            aux_data_dir=f"{PROJECT_ROOT}/src/data/ancillary",
        )
        global_filter_band = cls._prepare_global_filter_function(
            hapi_data_path,
            instrument_full_name,
            band,
            gamma,
            T=300,
            p=1.013,
            aux_data_dir=f"{PROJECT_ROOT}/src/data/ancillary",
        )

        _, nB_grid_ref_band = generate_lookup_tables(
            pressure=1.013,
            temperature=300,
            solarangle=solar_angle,
            obsangle=observation_angle,
            filter_function=global_filter_ref_band,
            instrument=instrument_full_name,
            band=ref_band,
            hapi_data_prefix=hapi_data_path,
            min_ch4=min_ch4,
            max_ch4=max_ch4,
            spacing_resolution=spacing_resolution,
        )

        _, nB_grid_band = generate_lookup_tables(
            pressure=1.013,
            temperature=300,
            solarangle=solar_angle,
            obsangle=observation_angle,
            filter_function=global_filter_band,
            instrument=instrument_full_name,
            band=band,
            hapi_data_prefix=hapi_data_path,
            min_ch4=min_ch4,
            max_ch4=max_ch4,
            spacing_resolution=spacing_resolution,
        )

        return cls(min_ch4, max_ch4, spacing_resolution, nB_grid_band, nB_grid_ref_band)

    @staticmethod
    def _get_full_instrument_name(instrument: str, full_sensor_name: str) -> str:
        return f"{full_sensor_name}{instrument}"

    @staticmethod
    def _prepare_global_filter_function(
        hapi_data_prefix: AnyUrl,
        instrument: str,
        band: str,
        gamma: float,
        T: float = 300,
        p: float = 1.013,
        aux_data_dir: str = "data/aux_data",
    ) -> SpectralFilter:
        """Prepare global filter function."""
        global_filter = generate_global_filter(
            temperature=T,
            pressure=p,
            instrument=instrument,
            hapi_data_prefix=hapi_data_prefix,
            aux_data_dir=aux_data_dir,
            band=band,
            gamma=gamma,
        )
        return global_filter

    def lookup(self, retrieval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Map methane retrieval values to corresponding reflectance values for bands ref_band and band.

        Parameters
        ----------
        retrieval : np.ndarray
            Methane retrieval values [mol/m²].

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Corresponding reflectance values for bands ref_band and band (unitless).
        """
        dims: tuple = retrieval.shape
        retrieval_1d = retrieval.reshape(-1)
        assert (
            np.max(retrieval) <= self.max_ch4
        ), f"Retrieval values are too small  {np.max(retrieval)} < {self.max_ch4}"
        index = np.round((retrieval_1d / self.max_ch4) * self.spacing_resolution, 0).astype(int)
        # Gorroño plumes have elements smaller than zero, which are mapped to indices smaller than zero
        # we correct those here
        index[np.where(index < 0)] = 0
        nB_band: np.ndarray = np.clip(self.nB_grid_band[index], 0.0, 1.0).reshape(dims)
        nB_ref_band: np.ndarray = np.clip(self.nB_grid_ref_band[index], 0.0, 1.0).reshape(dims)
        return nB_band, nB_ref_band

    def reverse_lookup(self, frac: np.ndarray) -> np.ndarray:
        """
        Map fractional reflectance values (frac) back to methane concentrations.

        Parameters
        ----------
        frac: Fractional reflectance values (unitless).

        Returns
        -------
        np.ndarray: Methane concentration values [mol/m²].
        """
        if frac.min() >= self.frac_grid.min():
            logger.warning(f"Frac is lower than the minimum of our grid: {frac.min()=}, {self.frac_grid.min()=}")
        frac = np.clip(frac, self.frac_grid.min(), self.frac_grid.max())

        # Note: [::-1] is there because the 'x' input is expected by numpy to
        # be in ascending order. We have the retrieval in ascending order
        # which corresponds to a frac in descending order, so we need to reverse them.
        retrieval = np.interp(frac, self.frac_grid[::-1], self.retrieval_grid[::-1])
        return retrieval


####################################################
################# EMIT FUNCTIONS ###################
####################################################


def precompute_log_norm_brightness(
    sensor_band_parameters: xr.Dataset,
    CH4_absorption: xr.DataArray,
    gamma_concentration: npt.NDArray,
) -> xr.DataArray:
    """
    Precompute logarithmic normalized brightness values for methane absorption, using the Beer-Lambert law.

    Parameters
    ----------
    sensor_band_parameters : xr.DataArray
        Sensor specifications including:
        - 'fwhm': Full Width at Half Maximum for each band
        - 'wavelengths': Central wavelength for each band
    CH4_absorption : xr.DataArray
        Methane absorption cross-sections with wavelength coordinates
        in the same units as used in sensor_band_parameters.
        Units: cm/#
    gamma_concentration : npt.NDArray
        Sequence of concentration values to compute absorption for.
        Units: mol/m2 (passed as array of floats)

    Returns
    -------
    xr.DataArray
        Logarithmic normalized brightness values for each band and concentration.
        Dimensions: (gamma_concentration, bands)

    Notes
    -----
    The calculation uses the following steps:
    1. Creates Gaussian SRF using sensor parameters
    2. Applies Beer-Lambert law with numerical stability using logsumexp
    3. Normalizes by the sum of the spectral response function

    The final result represents log(I/I₀) where:
    - I is the intensity after absorption
    - I₀ is the initial intensity

    Uses mol_to_N and m2_to_cm2 constants imported from external module.
    """
    # Create wavelength DataArray
    wavelength_da = CH4_absorption.wavelength

    # Calculate Gaussian spectral response function parameters
    sigma2 = sensor_band_parameters["fwhm"] ** 2 / (8 * np.log(2))
    all_bands_gaussian_srf = np.exp(-((wavelength_da - sensor_band_parameters["wavelengths"]) ** 2) / (2 * sigma2))

    # Create gamma concentration dimension
    gamma_concentration = pd.Index(gamma_concentration, name="gamma_concentration")

    # Implement the Beer-Lambert law in a numerically stable manner
    lse = [
        logsumexp(-gc * mol_to_N / m2_to_cm2 * CH4_absorption, dims=["wavelength"], b=all_bands_gaussian_srf)
        for gc in tqdm(gamma_concentration, desc="Precomputing log-normalized brightness")
    ]
    lse = xr.concat(lse, dim=gamma_concentration)

    # Normalize by spectral response function
    logdenom = np.log(all_bands_gaussian_srf.sum(dim="wavelength"))
    log_normalized_brightness = lse - logdenom

    return log_normalized_brightness


def interpolate_normalized_brightness(
    gamma_concentration: np.ndarray,
    log_nB_table_for_band: xr.DataArray,
) -> np.ndarray:
    """
    Interpolate normalized brightness values for methane pixels using a lookup table.

    Parameters
    ----------
    gamma_concentration : np.ndarray
        Array of gamma * concentration values for pixels with methane present,
        where concentration is in mol/m².
    log_nB_table_for_band : xr.DataArray
        Log-normalized brightness values indexed by gamma_concentration for a specific band.

    Returns
    -------
    np.ndarray
        Interpolated normalized brightness values for the input pixels.
    """
    absorption = -np.expm1(log_nB_table_for_band)  # type: ignore
    return -np.expm1(
        np.interp(
            np.log(gamma_concentration),
            np.log(absorption.gamma_concentration),  # type: ignore
            np.log(absorption),
        )
    )


def compute_normalized_brightness(
    gamma: xr.DataArray,
    concentration: xr.DataArray,
    log_nB_table: xr.DataArray,
) -> xr.DataArray:
    """
    Interpolate normalized brightness values for each band based on methane concentration.

    For pixels containing methane (where gamma * concentration > minimum threshold),
    this function interpolates band-specific normalized brightness values using a
    pre-computed lookup table. For all other pixels, the normalized brightness is 1.0
    (indicating no absorption).

    Bands with negligible methane absorption (maximum absorption < 1e-5) are skipped
    and their normalized brightness values remain at 1.0.

    Parameters
    ----------
    gamma: xr.DataArray
        The gamma value for each pixel. Gamma is the path length multiplier
        for light traveling through the atmosphere, calculated as:
        1/cos(theta_sun) + 1/cos(theta_sensor)
        where theta_sun is the solar zenith angle
        and theta_sensor is the sensor zenith angle.
    concentration : xr.DataArray
        Methane concentration values in mol/m². Must have same spatial dimensions
        as gamma. Values > 0 indicate presence of methane.
    log_nB_table : xr.DataArray
        Pre-computed lookup table of log-normalized brightness values. Has dimensions
        'bands' and 'gamma_concentration', where gamma_concentration is the product
        of gamma and methane concentration in mol/m².

    Returns
    -------
    xr.DataArray
        The normalized brightness values for each band and pixel. Has the same spatial
        dimensions as the input arrays plus a 'bands' dimension. Values are between
        0 and 1, where:
        - 1.0 indicates no absorption (either no methane or band insensitive to methane)
        - < 1.0 indicates absorption, with smaller values meaning stronger absorption
    """
    MIN_ABSORPTION_THRESHOLD = 1e-5

    # Identify pixels with methane
    gamma_concentration = gamma * concentration
    methane_pixels = gamma_concentration > MIN_ABSORPTION_THRESHOLD
    methane_gamma_concentrations = gamma_concentration.values[methane_pixels]

    # Initialize an array to store normalized brightness
    nB = xr.DataArray(
        np.ones((*concentration.shape, len(log_nB_table.bands)), dtype=float),
        dims=[*concentration.dims, "bands"],
        coords={**concentration.coords, "bands": log_nB_table.bands},
    )

    if methane_gamma_concentrations.size == 0:
        logger.warning(f"No pixels meet the threshold for detectable methane: {MIN_ABSORPTION_THRESHOLD}")
        return nB

    # Process each band separately
    for band in tqdm(log_nB_table.bands.values, desc="Computing normalized brightness"):
        # Skip bands with negligible methane absorption
        if -np.expm1(log_nB_table.sel(bands=band).min()) < MIN_ABSORPTION_THRESHOLD:
            continue

        nB_methane_pixels = interpolate_normalized_brightness(
            methane_gamma_concentrations, log_nB_table.sel(bands=band)
        )

        # Sanity checks
        assert nB_methane_pixels.min() > 0
        assert nB_methane_pixels.max() <= 1

        # Create a temporary array to hold the full raster for this band
        nB_raster = np.ones(concentration.shape)
        nB_raster[methane_pixels] = nB_methane_pixels

        # Assign the raster to the band in the output DataArray
        nB.loc[dict(bands=band)] = nB_raster

    logger.info(f"Biggest reduction factor: {nB.min().item():.4f}")
    return nB
