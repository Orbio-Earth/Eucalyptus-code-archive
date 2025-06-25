"""Filter functions which quantify the relative spectrum of light reaching the sensor."""

from typing import Union

import numpy as np
from numpy.typing import NDArray
from pydantic import AnyUrl, BaseModel
from scipy.interpolate import RegularGridInterpolator as rgi

from radtran.utils.constants import ch4_concentration, co2_concentration, h2o_concentration, m2_to_cm2, mol_to_N
from radtran.utils.data import HapiDataFiles, Instrument, Species, load_hapi_file


class SpectralFilter(BaseModel):
    """A spectral filter that represents the wavelength-dependent response of various components.

    This model can represent different types of spectral filters including:
    - Sensor spectral response functions
    - Solar irradiance spectra
    - Absorbing species filter functions
    - Unit vectors
    - Global filter functions

    Attributes
    ----------
        wavelength: Array of wavelength values
        response: Array of response values (can be sensor response, solar radiance,
                 filter function, or unit vector depending on usage)
    """

    wavelength: NDArray
    response: NDArray
    model_config = {"arbitrary_types_allowed": True}

    def _check_wavelength_compatibility(self, other: "SpectralFilter") -> None:
        """Check if two filters have compatible wavelengths."""
        if not np.array_equal(self.wavelength, other.wavelength):
            raise ValueError("Wavelengths of filters must match for arithmetic operations")

    def __mul__(self, other: Union["SpectralFilter", float, int]) -> "SpectralFilter":
        """Multiply two filters or scale a filter by a number."""
        if isinstance(other, float | int):
            return SpectralFilter(wavelength=self.wavelength, response=self.response * other)

        self._check_wavelength_compatibility(other)
        return SpectralFilter(wavelength=self.wavelength, response=self.response * other.response)

    def __pow__(self, exponent: float) -> "SpectralFilter":
        """Raise filter response to a power."""
        return SpectralFilter(wavelength=self.wavelength, response=self.response**exponent)

    def __truediv__(self, other: Union["SpectralFilter", float, int]) -> "SpectralFilter":
        """Divide two filters or scale a filter by a number."""
        if isinstance(other, float | int):
            return SpectralFilter(wavelength=self.wavelength, response=self.response / other)

        self._check_wavelength_compatibility(other)
        return SpectralFilter(wavelength=self.wavelength, response=self.response / other.response)

    def mean(self) -> float:
        """Calculate mean of the response."""
        return float(np.mean(self.response))

    def sum(self) -> float:
        """Calculate mean of the response."""
        return float(np.sum(self.response))

    def __repr__(self) -> str:
        """Readable representation of the filter."""
        return (
            f"SpectralFilter(wavelength_range=[{self.wavelength.min():.2f}, {self.wavelength.max():.2f}], "
            f"response_range=[{self.response.min():.3f}, {self.response.max():.3f}])"
        )


def get_sensor_spectral_response_function(
    hapi_data_prefix: AnyUrl, instrument: Instrument, band: str
) -> SpectralFilter:
    """
    Get the sensor spectral response function for a given instrument and band.

    This function retrieves the sensor spectral response function for a given
    instrument and band.

    Args:
        hapi_data_prefix [str]: Object storage prefix where the HAPI data is stored
        aux_data_dir [str]: Path to the directory where the auxiliary data is stored
        instrument [str]: ['Sentinel2A', 'Sentinel2B']
        band [str]: ['B11', 'B12']

    Returns
    -------
        SpectralFilter: A SpectralFilter object containing:
            wavelength (np.ndarray): Wavelengths of the band
            response (np.ndarray): Sensor spectral response function at the wavelengths
    """
    wavelength = load_hapi_file(HapiDataFiles.WAVELENGTH, hapi_data_prefix, instrument, band)
    sensor_srf = load_hapi_file(HapiDataFiles.SENSOR_SRF, hapi_data_prefix, instrument, band)
    wavesensor = sensor_srf[0]
    srf = sensor_srf[1]
    sensor_srf = np.interp(wavelength, wavesensor, srf)

    return SpectralFilter(wavelength=wavelength, response=sensor_srf)


def generate_solar_filter(
    hapi_data_prefix: AnyUrl, aux_data_dir: str, instrument: Instrument, band: str
) -> SpectralFilter:
    """
    Calculate the solar filter function for a given instrument and band.

    This function generates the solar filter function for a given instrument
    and band. It outputs the wavelengths and the solar radiance at the
    associated wavelengths.

    Args:
        hapi_data_prefix [AnyUrl]: Object storage prefix where the HAPI data is
                            stored
        aux_data_dir [str]: Path to the directory where the auxiliary data is
                            stored
        instrument [str]: ['Sentinel2A', 'Sentinel2B']
        band [str]: ['B11', 'B12']

    Returns
    -------
        SpectralFilter: A pydantic BaseModel containing:
            wavelength (np.ndarray): Wavelengths of the band
            solar_radiance (np.ndarray): Solar radiance at the wavelengths [W/m2-sr]
    """
    wavelength = load_hapi_file(HapiDataFiles.WAVELENGTH, hapi_data_prefix, instrument, band)

    # Get solar spectrum
    solarspec = np.load(f"{aux_data_dir}/solar_spectrum.npy")
    wavesolar = solarspec[0]
    radiancesolar = solarspec[1]
    solarradiance = np.interp(wavelength, wavesolar, radiancesolar)

    return SpectralFilter(wavelength=wavelength, response=solarradiance)


def get_absorption_cross_section_vector(
    temperature: float, pressure: float, instrument: Instrument, band: str, species: Species, hapi_data_prefix: AnyUrl
) -> SpectralFilter:
    """
    Get the absrption cross section vector for a given species.

    This function retrieves the absorption cross section of a given species
    at a given temperature and pressure. This is done for a specific instrument
    and band.

    Args:
        - temperature [float] : Temperature at which one would like to retrieve
                            the {species} absorption cross section [K]
        - pressure [float] : Pressure at which one would like to retrieve
                            the {species} absorption cross section [mbar]
        - instrument [Instruemnt] : ['Sentinel2A', 'Sentinel2B']
        - band [str]       : ['B12', 'B11']
        - species [Species]    : ['CH4', 'CO2', 'H2O']
        - hapi_data_prefix [AnyUrl] : object storage prefix where the HAPI data is
                                stored

    Returns
    -------
        SpectralFilter: A pydantic BaseModel containing:
            wavelength (np.ndarray): Wavelengths of the band
            response (np.ndarray): Absorption cross section vector for wavelengths of band ['cm2/#'],
                                where # is number of molecules

    Raises
    ------
        - ValueError: If the input values are not valid
    """
    abs_load = load_hapi_file(HapiDataFiles.ABS, hapi_data_prefix, instrument, band, species=species)
    wavelength = load_hapi_file(HapiDataFiles.WAVELENGTH, hapi_data_prefix, instrument, band)
    press_load = load_hapi_file(HapiDataFiles.PRESS, hapi_data_prefix, instrument, band)
    temp_load = load_hapi_file(HapiDataFiles.TEMP, hapi_data_prefix, instrument, band)

    abs_int_fun = rgi((press_load, temp_load, wavelength), abs_load)

    num_of_wavelengths = wavelength.shape[0]

    temps_arr = np.zeros((num_of_wavelengths,)).reshape(num_of_wavelengths, 1) + temperature
    press_arr = np.zeros((num_of_wavelengths,)).reshape(num_of_wavelengths, 1) + pressure
    wavelengths_arr = wavelength.reshape(num_of_wavelengths, 1)

    requested_vector = np.append(np.append(press_arr, temps_arr, axis=1), wavelengths_arr, axis=1)

    return SpectralFilter(wavelength=wavelength, response=abs_int_fun(requested_vector))


def generate_absorbing_species_filter(
    temperature: float,
    pressure: float,
    instrument: Instrument,
    hapi_data_prefix: AnyUrl,
    band: str,
    species: Species,
    concentration: float,
) -> "SpectralFilter":
    """
    Get the optical depth of a species given its concentration.

    Function to calculate optical depth optical depth of a {species} given
    the {concentration} of the {species}. In contrary to normal Varon
    function it is doing it assuming that all species concentration is
    absorbing using absorbption cross-section associated with the same
    temperature T, and p. This assumption is not true, however we argue
    that there is much larger sources of error such as albedo function
    of which we dont know the dependance on the wavelengths.

    Args:
        - targheight [float] : Target elevation above sea level [km]
        - obsheight  [float] : Altitude of satellite instrument in [km]
                            For Sentinel 2A, 2B it is 100 km
        - instrument [Instrument]   : Sensing instrument
        - hapi_data_prefix [AnyUrl]: Object store prefix to hapi data
        - band               : ['B11' or 'B12']
        - species    [Species]   : ['CH4', 'CO2', 'H2O']
        - temperature [float] : Temperature at which the absorbption
                            cross-section is evaluated [K]
        - pressure   [float] : pressure at which the absorbption
                            cross section is evaluated [mbar]
        - concentration [float] : Amount of the species in
                            the atmosphere [mol/m2]

    Returns
    -------
        SpectralFilter: A pydantic BaseModel containing:
            wavelength (np.ndarray): Wavelengths of the band
            response (np.ndarray): Relative amount of light transmitted by the species
    """
    cross_section = get_absorption_cross_section_vector(
        temperature, pressure, instrument, band, species, hapi_data_prefix
    )

    mlog_filterf = cross_section * (concentration * (mol_to_N / m2_to_cm2))

    return SpectralFilter(wavelength=mlog_filterf.wavelength, response=np.exp(-mlog_filterf.response))


def generate_unit_filter(hapi_data_prefix: AnyUrl, instrument: Instrument, band: str) -> "SpectralFilter":
    """
    Generate a unit filter function.

    This function generates a unit filter function for a given instrument
    and band. It outputs the wavelengths and the unit filter function at the
    associated wavelengths.

    Args:
        - hapi_data_prefix [AnyUrl] : Object storage prefix where the HAPI data is stored
        - instrument [Insturment] : Sensing instrument
        - band [str] : ['B11', 'B12']

    Returns
    -------
        SpectralFilter: A SpectralFilter object containing:
            wavelength: Wavelengths of the band
            response: Unit filter function

    """
    wavelength = load_hapi_file(HapiDataFiles.WAVELENGTH, hapi_data_prefix, instrument, band)

    unit_vector = wavelength / wavelength

    return SpectralFilter(wavelength=wavelength, response=unit_vector)


def generate_global_filter(
    temperature: float,
    pressure: float,
    instrument: Instrument,
    hapi_data_prefix: AnyUrl,
    aux_data_dir: str,
    band: str,
    gamma: float,
) -> SpectralFilter:
    """
    Generate the global filter function.

    This function generates the global filter function by multiplying the
    sensor spectral response function with the solar filter function and the
    absorption filter function.

    Args:
        - temperature [float] : Temperature at which the absorbption cross section is evaluated [K]
        - pressure [float] : Pressure at which the absorbption cross section is evaluated [mbar]
        - instrument [Instrument] : ['Sentinel2A', 'Sentinel2B']
        - hapi_data_prefix [AnyUrl] : Object storage prefix where the HAPI data is stored
        - aux_data_dir [str] : Path to the directory where the auxiliary data is stored
        - band [str] : ['B11', 'B12']
        - gamma [float] : Gamma scaling factor

    Returns
    -------
        SpectralFilter: The global filter function containing wavelength and response arrays
    """
    """Generate the global filter function."""
    solar_filter = generate_solar_filter(hapi_data_prefix, aux_data_dir, instrument, band)
    sensor_srf = get_sensor_spectral_response_function(hapi_data_prefix, instrument, band)

    h2o_filter = generate_absorbing_species_filter(
        temperature, pressure, instrument, hapi_data_prefix, band, Species.H2O, h2o_concentration
    )
    co2_filter = generate_absorbing_species_filter(
        temperature, pressure, instrument, hapi_data_prefix, band, Species.CO2, co2_concentration
    )
    ch4_filter = generate_absorbing_species_filter(
        temperature, pressure, instrument, hapi_data_prefix, band, Species.CH4, ch4_concentration
    )

    absorption_filter = h2o_filter * co2_filter * ch4_filter

    global_filter = sensor_srf * solar_filter * (absorption_filter**gamma)
    return global_filter / global_filter.sum()
