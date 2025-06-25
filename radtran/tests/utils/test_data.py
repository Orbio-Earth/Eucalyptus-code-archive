"""Tests for data utilities."""

from contextlib import AbstractContextManager
from contextlib import nullcontext as no_exception

import pytest
from pydantic import AnyUrl

from radtran.utils.data import HapiDataFiles, Instrument, Species, _get_hapi_data_uri


@pytest.mark.parametrize(
    "hapi_file,filestore_prefix,instrument,band,species,expected_context,expected_uri",
    [
        # Successful ABS URL building
        (
            HapiDataFiles.ABS,
            AnyUrl("s3://bucket/prefix/"),
            Instrument.SENTINEL2A,
            "B11",
            Species.CH4,
            no_exception(),
            AnyUrl("s3://bucket/prefix/Sentinel2A/B11/abs_CH4_hapi_Sentinel2A_band_B11.npy"),
        ),
        # ABS file retrieval should raise an error if species is not passed
        (
            HapiDataFiles.ABS,
            AnyUrl("s3://bucket/prefix/"),
            Instrument.SENTINEL2A,
            "B11",
            None,
            pytest.raises(ValueError),
            # In this case we should not get to the return type
            AnyUrl("s3://bucket/prefix/Sentinel2A/B11/INVALID"),
        ),
        # Successful PRESS URL building
        (
            HapiDataFiles.PRESS,
            AnyUrl("azureml://store/prefix/"),
            Instrument.LANDSAT8,
            "swir16",
            None,
            no_exception(),
            AnyUrl("azureml://store/prefix/Landsat8/swir16/abs_press_hapi_Landsat8_band_swir16.npy"),
        ),
        # Successful SENSOR_SRF URL building
        (
            HapiDataFiles.SENSOR_SRF,
            AnyUrl("s3://bucket/prefix/"),
            Instrument.EMIT,
            "VSWIR",
            None,
            no_exception(),
            AnyUrl("s3://bucket/prefix/EMIT/VSWIR/SRF_EMIT_VSWIR.npy"),
        ),
        # Successful TEMP URL building
        (
            HapiDataFiles.TEMP,
            AnyUrl("azureml://store/prefix/"),
            Instrument.SENTINEL2A,
            "B11",
            None,
            no_exception(),
            AnyUrl("azureml://store/prefix/Sentinel2A/B11/abs_temp_hapi_Sentinel2A_band_B11.npy"),
        ),
        # Successful WAVELENGTH URL building
        (
            HapiDataFiles.WAVELENGTH,
            AnyUrl("s3://bucket/prefix/"),
            Instrument.SENTINEL2A,
            "B11",
            None,
            no_exception(),
            AnyUrl("s3://bucket/prefix/Sentinel2A/B11/abs_wave_hapi_Sentinel2A_band_B11.npy"),
        ),
        # Successful WAVELENGTH URL building despite passing of superfluous species parameter.
        # This should succeed for all hapid data file types save for ABS.
        (
            HapiDataFiles.WAVELENGTH,
            AnyUrl("azureml://store/prefix/"),
            Instrument.SENTINEL2A,
            "B11",
            Species.H2O,
            no_exception(),
            AnyUrl("azureml://store/prefix/Sentinel2A/B11/abs_wave_hapi_Sentinel2A_band_B11.npy"),
        ),
    ],
)
def test_get_hapi_data_uri(
    hapi_file: HapiDataFiles,
    filestore_prefix: AnyUrl,
    instrument: Instrument,
    band: str,
    species: Species | None,
    expected_context: AbstractContextManager,
    expected_uri: AnyUrl,
) -> None:
    with expected_context:
        uri = _get_hapi_data_uri(hapi_file, filestore_prefix, instrument, band, species=species)
        assert uri == expected_uri
