"""Main library functions for the radtran package."""

from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import AnyUrl

from radtran.filter_functions import SpectralFilter, get_absorption_cross_section_vector
from radtran.utils.constants import (
    MAX_OBSERVER_ANGLE,
    MAX_SOLAR_ANGLE,
    MIN_OBSERVER_ANGLE,
    MIN_SOLAR_ANGLE,
    m2_to_cm2,
    mol_to_N,
)
from radtran.utils.data import Instrument, Species

# Using a TypeVar allows us to tell the type checker that the
# input and output are the same type, which helps mypy to
# avoid having a hissy fit.
T = TypeVar("T", bound=ArrayLike)


def get_gamma(solarangle: T, obsangle: T) -> T:
    """
    Calculate the gamma scaling factor.

    This function calculates the gamma scaling factor which takes under
    account the fact that incoming beam and outgoing beam is not exactly
    vertical.

    Args:
        solarangle: Angle between the sun and the normal to the surface
                    [Degrees]
        obsangle: Angle between the observer-satellite and the normal to
                    the surface [Degrees]

    Returns
    -------
        gamma: Scaling factor

    Raises
    ------
        ValueError: If solarangle or obsangle is not a
        number between 0 and 90
    """
    if not np.all((solarangle >= MIN_SOLAR_ANGLE) & (solarangle <= MAX_SOLAR_ANGLE)):  # type: ignore[operator]
        raise ValueError(
            f"Solar angle must be between {MIN_SOLAR_ANGLE!r} and {MAX_SOLAR_ANGLE!r}. Yours is {solarangle!r}"
        )  #!r decodes bytes
    if not np.all((obsangle >= MIN_OBSERVER_ANGLE) & (obsangle <= MAX_OBSERVER_ANGLE)):  # type: ignore[operator]
        raise ValueError(
            f"Observer angle must be between {MIN_OBSERVER_ANGLE!r} and {MAX_OBSERVER_ANGLE!r}. Yours is {obsangle!r}")

    gamma = 1 / np.cos(np.deg2rad(solarangle)) + 1 / np.cos(np.deg2rad(obsangle))
    return gamma


def get_normalized_brightness(
    concentration_of_species: NDArray,
    gamma: float,
    absorption_cross_section: SpectralFilter,
    filter_function: SpectralFilter,
) -> NDArray:
    """
    Calculate the normalized brightness due to species in the atmosphere.

    This function accounts for all factors affecting the transmittance of
    light, such as other species and solar radiance (grouped into the global
    filter function).

    Args:
        concentration_of_species (np.ndarray):
            The concentration of the species in the atmosphere.
            Units: mol/m².
        gamma (float):
            A factor compensating for the angle of the incoming and outgoing
            radiation relative to the ground normal.
            Units: unitless.
        absorption_cross_section (SpectralFilter):
            The absorption cross section as a function of wavelength for the
            species, representing the magnitude of absorption caused by a
            specific concentration in the atmosphere.
            Units: cm².
        filter_function (SpectralFilter):
            The filter as a function of wavelength for the portion of the
            electromagnetic spectrum corresponding to the band's spectral
            range.
            Units: unitless.

    Returns
    -------
        np.ndarray:
            The portion of light transmitted through the atmosphere,
            accounting for the concentration of the species.
            Units: unitless.
    """
    filter_function._check_wavelength_compatibility(absorption_cross_section)

    nB: NDArray = np.sum(
        filter_function.response[np.newaxis, :]
        * np.exp(
            -gamma
            * concentration_of_species[:, np.newaxis]
            * mol_to_N
            / m2_to_cm2
            * absorption_cross_section.response[np.newaxis, :]
        ),
        axis=1,
    )
    assert nB.shape == concentration_of_species.shape, (
        "The normalized brightness should have the same shape as the "
        "concentration of species, instead got "
        f"{nB.shape} and {concentration_of_species.shape}"
    )
    return nB


def radtran(  # noqa: PLR0913 (too-many-arguments)
    pressure: float,
    temperature: float,
    solarangle: float,
    obsangle: float,
    filter_function: SpectralFilter,
    instrument: Instrument,
    band: str,
    hapi_data_prefix: AnyUrl,
    min_ch4: float,
    max_ch4: float,
    spacing_resolution: int = 10000,
) -> tuple[NDArray, NDArray]:
    """
    Calculate the normalized brightness of a band given methane concentrations.

    This function calculates the normalized brightness of a band given the
    presence of anomalous methane for a range of concentrations of methane at
    specific temperature and pressure. It is calculated through Beer-Lambert
    law weighing the absorption cross section of methane with the global filter
    function.

    Args:
        pressure: Pressure at which the methane is present [mbar]
        temperature: Temperature at which the methane is present [K]
        solarangle: Angle between the sun and the normal to the surface
                    [Degrees]
        obsangle: Angle between the observer-satellite and the normal to
                    the surface [Degrees]
        filter_function: Global filter function
        instrument: ['Sentinel2A', 'Sentinel2B']
        band: ['B11', 'B12']
        hapi_data_prefix: Object storage prefix where the HAPI data is stored
        min_ch4: Minimum concentration of methane to be considered [mol/m2]
        max_ch4: Maximum concentration of methane to be considered [mol/m2]
        spacing_resolution: Number of points to consider between min_ch4 and
        max_ch4

    Returns
    -------
        delta_CH4: np.ndarray
            Concentration of methane at which the normalized brightness is
            calculated
            Units: mol/m2
        nB: np.ndarray
            Normalized brightness at the concentration of methane
            Units: unitless
    """
    gamma = get_gamma(solarangle, obsangle)
    abs_cross_section_vector_CH4 = get_absorption_cross_section_vector(
        temperature, pressure, instrument, band, Species.CH4, hapi_data_prefix
    )

    delta_CH4: NDArray = np.linspace(min_ch4, max_ch4, spacing_resolution)
    nB: NDArray = get_normalized_brightness(delta_CH4, gamma, abs_cross_section_vector_CH4, filter_function)

    return delta_CH4, nB


def retrieve_methane(frac_tile: NDArray, frac_table: NDArray, methane_table: NDArray) -> NDArray:
    """
    Calculate the tile methane concentration from the fractions-methane lookup table.

    This function calculates the methane concentration from the
    fractions-methane lookup table. The fractions-methane lookup table is a
    table that contains the fractions of methane in the atmosphere and the
    corresponding methane concentrations. This is parsed as two separate arrays.
    The function then interpolates the methane concentration for the given tile of
    frac.
    concentrations. This is parsed as two separate arrays. The function then
    interpolates the methane concentration for the given tile of frac.

    Args:
        frac_tile: Tile with calculated frac
        frac_table: Frac lookup table
        methane_table: Methane concentrations lookup table [mol/m2]

    Returns
    -------
        methane_tile: Methane concentrations of the tile [mol/m2]
    """
    # Note: [::-1] is there because the 'x' input is expected by numpy to
    # be in ascending order. We have the retrieval in ascending order
    # which corresponds to a frac in descending order,
    # so we need to reverse them.
    assert np.all(frac_table.min() <= frac_tile.min())

    if frac_table[0] > frac_table[-1]:
        frac_table = frac_table[::-1]
        methane_table = methane_table[::-1]

    retrieval_tile = np.interp(frac_tile, frac_table, methane_table)
    return retrieval_tile
