"""Utilities for working with data files.

Utilities for working with data files, in particular HAPI data.
"""

import io
import os
from enum import Enum

import fsspec
import numpy as np
from numpy.typing import NDArray
from pydantic import AnyUrl

from radtran.utils.logging_utils import logger


class HapiDataFiles(Enum):
    """HAPI data file types.

    Attributes
    ----------
        ABS: [TODO:attribute]
        PRESS: [TODO:attribute]
        SENSOR_SRF: [TODO:attribute]
        TEMP: [TODO:attribute]
        WAVELENGTH: [TODO:attribute]
    """

    # FIXME: what are these abbreviations short for? Write out in full
    ABS = "ABS"
    PRESS = "PRESS"
    SENSOR_SRF = "SENSOR_SRF"
    TEMP = "TEMP"
    WAVELENGTH = "WAVELENGTH"


class Instrument(str, Enum):
    """Instrument types."""

    SENTINEL2A = "Sentinel2A"
    SENTINEL2B = "Sentinel2B"
    LANDSAT8 = "Landsat8"
    LANDSAT9 = "Landsat9"
    EMIT = "EMIT"

    def __str__(self) -> str:
        """Convert to string representation."""
        return self.value


class Species(str, Enum):
    """Species of gas."""

    CH4 = "CH4"
    CO2 = "CO2"
    # FIXME: why is H2O with a zero?
    H2O = "H20"

    def __str__(self) -> str:
        """Convert to string representation."""
        return self.value


# Mappings for HAPI data files and their basename formats.
hapi_data_basenames = {
    HapiDataFiles.ABS: "abs_{species}_hapi_{instrument}_band_{band}.npy",
    HapiDataFiles.PRESS: "abs_press_hapi_{instrument}_band_{band}.npy",
    HapiDataFiles.SENSOR_SRF: "SRF_{instrument}_{band}.npy",
    HapiDataFiles.TEMP: "abs_temp_hapi_{instrument}_band_{band}.npy",
    HapiDataFiles.WAVELENGTH: "abs_wave_hapi_{instrument}_band_{band}.npy",
}


def load_hapi_file(
    hapi_file: HapiDataFiles,
    hapi_data_prefix: AnyUrl,
    instrument: Instrument,
    band: str,
    *,
    species: Species | None = None,
) -> NDArray:
    """Load a HAPI data file.

    Args:
        hapi_file (HapiDataFiles): The HAPI file to load.
        hapi_data_prefix (AnyUrl): The prefix of the HAPI data.
        instrument (Instrument): The instrument name.
        band (str): The band name.
        species (Species): The species name.

    Returns
    -------
        NDArray: The numpy array.
    """
    uri = _get_hapi_data_uri(hapi_file, hapi_data_prefix, instrument, band, species=species)
    return _load_numpy(uri)


def _load_numpy(uri: AnyUrl, *, cache_file: bool = False) -> NDArray:
    """Load a numpy array from a URI.

    Args:
        uri (AnyUrl): The URI to load the numpy array from.
        cache_file (bool, optional): Whether to cache the file in a local directory.
            Defaults to False.

    Returns
    -------
        NDArray: The numpy array.
    """
    cache_dir = "/tmp/radtran"

    uristr = str(uri)
    if cache_file:
        if uristr[:10] == "azureml://":
            # fsspec's caching seems to be incompatible with azureml URIs which is
            # likely an issue in the underlying azureml-fsspec package.
            # see https://git.orbio.earth/orbio/orbio/-/merge_requests/837#note_73984
            logger.warning("Caching not supported for azureml URIs.")
            cache_file = False
        else:
            uristr = f"filecache::{uristr}"
            os.makedirs(cache_dir, exist_ok=True)

    # FIXME: put more thought into cache_storage location
    # TODO: do we need to create this directory first?
    kwargs = dict(filecache={"cache_storage": cache_dir}) if cache_file else dict()
    with fsspec.open(uristr, **kwargs) as fs:
        assert isinstance(fs, io.IOBase)
        data = np.load(fs)
    return data


def _get_hapi_data_uri(
    hapi_file: HapiDataFiles,
    hapi_data_prefix: AnyUrl,
    instrument: Instrument,
    band: str,
    *,
    species: Species | None = None,
) -> AnyUrl:
    """Get the URI a HAPI data numpy file.

    Args:
        instrument (Instrument): The instrument name.
        band (str): The band name.
        species (Species): The species name.

    Returns
    -------
        AnyUrl: The URI for the HAPI data.
    """
    if hapi_file == HapiDataFiles.ABS and species is None:
        raise ValueError("species is required to build the URI to the abs HAPI file.")

    uri_suffix = hapi_data_basenames[hapi_file].format(instrument=instrument, band=band, species=species)
    return AnyUrl(str(hapi_data_prefix).rstrip("/") + f"/{instrument}/{band}/{uri_suffix}")
