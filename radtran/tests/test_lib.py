# ruff: noqa: D102, D205, D101

"""Unit tests for the radtran library."""

import unittest

import numpy as np
from numpy.typing import NDArray
from pydantic import AnyUrl

from radtran.filter_functions import SpectralFilter, generate_unit_filter
from radtran.lib import (
    get_gamma,
    get_normalized_brightness,
    radtran,
    retrieve_methane,
)
from radtran.utils.data import Instrument


class TestRadtran(unittest.TestCase):
    def test_get_gamma(self) -> None:
        """
        Gamma function is calculated as:
        gamma = (1 / np.cos(solarangle * (2 * np.pi / 360)) +
                 1 / np.cos(obsangle * (2 * np.pi / 360))).

        for solarangle=observerangle=0, gamma=2.0
        for solarangle=observerangle=45, gamma=2*sqrt(2)
        """
        self.assertAlmostEqual(get_gamma(0, 0), 2.0)
        self.assertAlmostEqual(get_gamma(45, 45), 2.8284271247461903)

    def test_get_gamma_raises_value_error(self) -> None:
        """
        Gamma function should raise ValueError for solarangle or obsangle
        outside the range [MIN_SOLAR_ANGLE, MAX_SOLAR_ANGLE] or
        [MIN_OBSERVER_ANGLE, MAX_OBSERVER_ANGLE], where the angles are
        given in degrees and set in radtran.utils.constants.py.
        """
        with self.assertRaises(ValueError):
            get_gamma(-1, 45)
        with self.assertRaises(ValueError):
            get_gamma(45, 91)

    def test_get_normalized_brightness_output_values(self) -> None:
        concentration_of_species = np.array([1.0, 2.0])
        gamma = 1.0
        wavelengths: NDArray = np.array([1000.0, 2000.0])
        absorption_cross_section = SpectralFilter(wavelength=wavelengths, response=np.array([1e-21, 2e-21]))
        filter_function = SpectralFilter(wavelength=wavelengths, response=np.array([1, 1]))
        result = get_normalized_brightness(concentration_of_species, gamma, absorption_cross_section, filter_function)
        expected = np.array([1.82808382, 1.67245928])
        np.testing.assert_almost_equal(result, expected)

    def test_radtran_output_shape(self) -> None:
        pressure = 1.013
        temperature = 288.15
        solarangle = 30
        obsangle = 45
        hapi_data_prefix = AnyUrl("s3://orbio-scratch/radtran/hapi/v1.1.1.0")
        instrument = Instrument.SENTINEL2A
        band = "B11"
        min_ch4 = 0.0
        max_ch4 = 1.0
        spacing_resolution = 10

        filter_function = generate_unit_filter(hapi_data_prefix, instrument, band)

        delta_CH4, nB = radtran(
            pressure,
            temperature,
            solarangle,
            obsangle,
            filter_function,
            instrument,
            band,
            hapi_data_prefix,
            min_ch4,
            max_ch4,
            spacing_resolution,
        )
        self.assertEqual(len(delta_CH4), spacing_resolution)
        self.assertEqual(len(nB), spacing_resolution)

    def test_radtran_output_values(self) -> None:
        pressure = 1.013
        temperature = 288.15
        solarangle = 30
        obsangle = 45
        hapi_data_prefix = AnyUrl("s3://orbio-scratch/radtran/hapi/v1.1.1.0")
        instrument = Instrument.SENTINEL2A
        band = "B11"
        min_ch4 = 0.0
        max_ch4 = 1.0
        spacing_resolution = 10

        filter_function = generate_unit_filter(hapi_data_prefix, instrument, band)

        delta_CH4, nB = radtran(
            pressure,
            temperature,
            solarangle,
            obsangle,
            filter_function,
            instrument,
            band,
            hapi_data_prefix,
            min_ch4,
            max_ch4,
            spacing_resolution,
        )
        np.testing.assert_almost_equal(nB[1], 7263.681289958593, decimal=10)
        np.testing.assert_almost_equal(delta_CH4[1], 0.1111111111111111, decimal=10)

    def test_retrieve_methane(self) -> None:
        frac_tile = np.array([0.1, 0.2, 0.3])
        frac_table = np.array([0.1, 0.2, 0.3])
        methane_table = np.array([1.0, 2.0, 3.0])
        result = retrieve_methane(frac_tile, frac_table, methane_table)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_almost_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
