"""Utility functions for geospatial operations."""

import pyproj
import rasterio
import shapely


def reproject_geometry(
    geom: shapely.geometry.base.BaseGeometry, input_crs: pyproj.CRS, output_crs: pyproj.CRS
) -> shapely.geometry.base.BaseGeometry:
    """Reproject a geometry from one CRS to another.

    Args:
        geom: The geometry to reproject
        input_crs: The input CRS
        output_crs: The output CRS

    Returns
    -------
        The reprojected geometry
    """
    # Create the coordinate transformation
    project = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True).transform

    # Apply the transformation to the geometry
    return shapely.ops.transform(project, geom)


def reproject_latlon(lat: float, lon: float, output_crs: pyproj.CRS) -> tuple[float, float]:
    """Convert latitude/longitude coordinates to a different coordinate reference system.

    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        output_crs: The target coordinate reference system

    Returns
    -------
        tuple[float, float]: The transformed coordinates (x, y) in the output CRS
    """
    # Create a Point geometry from lat/lon
    point = shapely.geometry.Point(lon, lat)

    # Reproject the point from WGS84 (EPSG:4326) to the target CRS
    transformed_point = reproject_geometry(point, pyproj.CRS("EPSG:4326"), output_crs)

    # Return the transformed coordinates
    return (transformed_point.x, transformed_point.y)


def get_crop_params(
    lat: float,
    lon: float,
    out_size: int,
    shape: tuple[int, int],  # (height, width) of the raster
    crs: str,
    transform: rasterio.Affine,
    out_res: float | None = None,
) -> dict:
    """
    Calculate crop parameters for a given lat/lon point in a raster.

    Parameters
    ----------
    lat : float
        Latitude of the point
    lon : float
        Longitude of the point
    out_size : int
        Desired output size in pixels
    shape : tuple[int, int]
        (height, width) of the raster
    crs : str
        Coordinate reference system of the raster
    transform : rasterio.Affine
        Affine transform for the input raster
    out_res : float | None
        Desired output resolution in meters. If None, uses input resolution.

    Returns
    -------
    dict
        Crop parameters including start coordinates, height, width, and output dimensions

    Raises
    ------
    ValueError
        If the point falls outside the raster bounds
    """
    # Convert lat/lon to raster CRS
    x, y = reproject_latlon(lat, lon, pyproj.CRS(crs))

    # Convert x,y to pixel coordinates using the transform
    px, py = ~transform * (x, y)

    # Check if point is within raster bounds
    height, width = shape
    if not (0 <= px < width and 0 <= py < height):
        raise ValueError(
            f"Point ({lat}, {lon}) projects to pixel coordinates ({px:.1f}, {py:.1f}) "
            f"which fall outside raster bounds (width={width}, height={height})"
        )

    # Calculate resolutions if not provided
    in_res = abs(transform.a)  # Use pixel size from transform
    if out_res is None:
        out_res = in_res

    # Scale factors for resolution differences
    scale = out_res / in_res
    crop_size = int(out_size * scale)
    half_size = crop_size // 2

    # Calculate crop bounds ensuring they're within raster dimensions
    crop_start_x = int(max(0, min(px - half_size, width - crop_size)))
    crop_start_y = int(max(0, min(py - half_size, height - crop_size)))

    return {
        "crop_start_x": crop_start_x,
        "crop_start_y": crop_start_y,
        "crop_height": crop_size,
        "crop_width": crop_size,
        "out_height": out_size,
        "out_width": out_size,
    }
