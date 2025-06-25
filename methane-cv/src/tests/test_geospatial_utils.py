# ruff: noqa: PLR2004
"""Tests for geospatial utility functions.

This module contains tests for the geospatial utility functions used in the methane-cv
project, including coordinate transformations and raster cropping operations.
"""

import numpy as np
import pyproj
import pytest
import rasterio
import shapely

from src.utils.geospatial import get_crop_params, reproject_geometry, reproject_latlon


def test_reproject_geometry() -> None:
    """Test reprojecting a simple point geometry between coordinate systems."""
    # Create a point in WGS84
    point = shapely.geometry.Point(-122.4194, 37.7749)  # San Francisco coordinates

    # Define input and output CRS
    input_crs = pyproj.CRS("EPSG:4326")  # WGS84
    output_crs = pyproj.CRS("EPSG:3857")  # Web Mercator

    # Reproject the point
    transformed_point = reproject_geometry(point, input_crs, output_crs)

    # Check that we got a Point back
    assert isinstance(transformed_point, shapely.geometry.Point)

    # Check that coordinates were transformed (rough check)
    assert transformed_point.x != point.x
    assert transformed_point.y != point.y

    # Transform back should give original coordinates (within tolerance)
    back_transformed = reproject_geometry(transformed_point, output_crs, input_crs)
    assert np.isclose(back_transformed.x, point.x, rtol=1e-7)
    assert np.isclose(back_transformed.y, point.y, rtol=1e-7)


def test_reproject_latlon() -> None:
    """Test reprojecting lat/lon coordinates to UTM."""
    lat, lon = 37.7749, -122.4194  # San Francisco
    output_crs = pyproj.CRS("EPSG:32610")  # UTM Zone 10N

    # Transform coordinates
    x, y = reproject_latlon(lat, lon, output_crs)
    print("San Francisco UTM:", x, y)

    # Check that we got different coordinates
    assert x != lon
    assert y != lat

    # Values should be in UTM range (meters) for San Francisco
    assert 540_000 < x < 560_000  # SF easting range in UTM 10N
    assert 4_170_000 < y < 4_190_000  # SF northing range in UTM 10N


def test_get_crop_params() -> None:
    """Test calculation of crop parameters for a point in a raster."""
    # Test inputs
    lat, lon = 37.7749, -122.4194
    out_size = 100
    shape = (1000, 1000)  # height, width
    crs = "EPSG:32610"  # UTM Zone 10N
    # Transform adjusted to cover San Francisco area
    transform = rasterio.Affine(
        30,
        0,
        540000,  # a, b, c
        0,
        -30,
        4190000,
    )  # d, e, f

    # Get crop parameters
    crop_params = get_crop_params(lat=lat, lon=lon, out_size=out_size, shape=shape, crs=crs, transform=transform)

    # Check that we got all expected keys
    expected_keys = {"crop_start_x", "crop_start_y", "crop_height", "crop_width", "out_height", "out_width"}
    assert set(crop_params.keys()) == expected_keys
    print(crop_params)

    # Check output dimensions match requested size
    assert crop_params["out_height"] == out_size
    assert crop_params["out_width"] == out_size

    # Check crop coordinates are within image bounds
    assert 0 <= crop_params["crop_start_x"] < shape[1]
    assert 0 <= crop_params["crop_start_y"] < shape[0]
    assert crop_params["crop_start_x"] + crop_params["crop_width"] <= shape[1]
    assert crop_params["crop_start_y"] + crop_params["crop_height"] <= shape[0]


def test_get_crop_params_with_resolution() -> None:
    """Test crop parameters with explicit input/output resolutions."""
    lat, lon = 37.7749, -122.4194
    out_size = 100
    shape = (1000, 1000)
    crs = "EPSG:32610"
    # Using same transform as above for consistency
    transform = rasterio.Affine(
        30,
        0,
        540000,  # a, b, c
        0,
        -30,
        4190000,
    )  # d, e, f

    # Test with different input/output resolutions
    crop_params = get_crop_params(
        lat=lat, lon=lon, out_size=out_size, shape=shape, crs=crs, transform=transform, out_res=10.0
    )

    # Crop size should be 3x the output size due to resolution difference
    assert crop_params["crop_height"] == int(out_size / 3)
    assert crop_params["crop_width"] == int(out_size / 3)
    assert crop_params["out_height"] == out_size
    assert crop_params["out_width"] == out_size


def test_get_crop_params_out_of_bounds() -> None:
    """Test that ValueError is raised when coordinates are outside raster bounds."""
    # Use San Francisco coordinates
    lat, lon = 37.7749, -122.4194
    out_size = 100
    shape = (1000, 1000)
    crs = "EPSG:32610"  # UTM Zone 10N

    # Test point too far east (moved ~100km east)
    transform_east = rasterio.Affine(
        30,
        0,
        640000,  # a, b, c (moved x origin east)
        0,
        -30,
        4190000,
    )  # d, e, f
    with pytest.raises(ValueError, match=".*fall outside raster bounds.*"):
        get_crop_params(lat=lat, lon=lon, out_size=out_size, shape=shape, crs=crs, transform=transform_east)

    # Test point too far west (moved ~100km west)
    transform_west = rasterio.Affine(
        30,
        0,
        440000,  # a, b, c (moved x origin west)
        0,
        -30,
        4190000,
    )  # d, e, f
    with pytest.raises(ValueError, match=".*fall outside raster bounds.*"):
        get_crop_params(lat=lat, lon=lon, out_size=out_size, shape=shape, crs=crs, transform=transform_west)

    # Test point too far north (moved ~100km north)
    transform_north = rasterio.Affine(
        30,
        0,
        540000,  # a, b, c
        0,
        -30,
        4290000,
    )  # d, e, f (moved y origin north)
    with pytest.raises(ValueError, match=".*fall outside raster bounds.*"):
        get_crop_params(lat=lat, lon=lon, out_size=out_size, shape=shape, crs=crs, transform=transform_north)

    # Test point too far south (moved ~100km south)
    transform_south = rasterio.Affine(
        30,
        0,
        540000,  # a, b, c
        0,
        -30,
        4090000,
    )  # d, e, f (moved y origin south)
    with pytest.raises(ValueError, match=".*fall outside raster bounds.*"):
        get_crop_params(lat=lat, lon=lon, out_size=out_size, shape=shape, crs=crs, transform=transform_south)
