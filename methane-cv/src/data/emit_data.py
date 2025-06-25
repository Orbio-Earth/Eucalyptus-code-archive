"""
Module for accessing EMIT data products.

This module provides classes and utilities for working with EMIT data, including:
- EmitTileUris: Handles URI construction and parsing of EMIT tile identifiers
- EMITL2AMaskLabel: Enumeration of mask labels in EMIT L2A products
- EmitGranuleAccess: Main class for accessing and processing EMIT data products

These classes are used during data generation, training, and inference.

The module supports accessing various EMIT data products including:
- L1B Radiance data
- L1B Observation data
- L2A Mask data

"""

import datetime
import io
import logging
import re
from dataclasses import dataclass
from enum import Enum, unique

import earthaccess
import requests
import xarray as xr
from shapely.geometry import Point, Polygon

from src.data.granule_access import BaseGranuleAccess


@dataclass
class EmitTileUris:
    """
    Handles URI construction for EMIT data products.

    Attributes
    ----------
        tile_id: The EMIT tile identifier (e.g., 'EMIT_L1B_RAD_001_20240127T195840_2402713_006')
        sensor: Always "EMIT"
        product_level: L1B or L2A
        product_type: RAD, OBS, etc.
        version: The product version (e.g., '001')
        acquisition_date: The acquisition date (YYYYMMDD)
        acquisition_time: The acquisition time (HHMMSS)
        orbit_id: The orbit identifier
        scene_id: The scene identifier
    """

    tile_id: str
    sensor: str
    product_level: str
    product_type: str
    version: str
    acquisition_date: str
    acquisition_time: str
    orbit_id: str
    scene_id: str

    @classmethod
    def from_tile_id(cls, tile_id: str) -> "EmitTileUris":
        """Create an EmitTileUris instance by parsing a tile ID."""
        components = cls.parse_emit_id(tile_id)
        return cls(
            tile_id=tile_id,
            sensor=components["sensor"],
            product_level=components["product_level"],
            product_type=components["product_type"],
            version=components["version"],
            acquisition_date=components["acquisition_date"],
            acquisition_time=components["acquisition_time"],
            orbit_id=components["orbit_id"],
            scene_id=components["scene_id"],
        )

    @staticmethod
    def parse_emit_id(emit_id: str) -> dict:
        """
        Parse an EMIT Level 1 or Level 2A filename and extract its components.

        Following naming conventions here: https://lpdaac.usgs.gov/data/get-started-data/collection-overview/missions/emit-overview/#emit-naming-conventions

        Example filename/id: EMIT_L1B_RAD_001_20240127T195840_2402713_006

        """
        pattern = (
            r"EMIT_(L[12][A-Z])_([A-Z]+)_(\d{3})_"  # Product Level, Type, Version
            r"(\d{8})T(\d{6})_"  # Acquisition Date and Time
            r"(\d{7})_(\d{3})"  # Orbit ID, Scene ID
        )

        match = re.match(pattern, emit_id)
        if not match:
            raise ValueError("Invalid EMIT ID format")

        return {
            "sensor": "EMIT",
            "product_level": match.group(1),
            "product_type": match.group(2),
            "version": match.group(3),
            "acquisition_date": match.group(4),  # Keep as YYYYMMDD
            "acquisition_time": match.group(5),  # Keep as HHMMSS
            "orbit_id": match.group(6),
            "scene_id": match.group(7),
        }

    @property
    def datetime_str(self) -> str:
        """Get the datetime string used in URIs."""
        return f"{self.acquisition_date}T{self.acquisition_time}"

    @property
    def uris(self) -> dict[str, str]:
        """Get the URIs for this tile's data products."""
        base_url = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected"

        # Build file identifiers
        rad_id = f"EMIT_L1B_RAD_{self.version}_{self.datetime_str}_{self.orbit_id}_{self.scene_id}"
        obs_id = f"EMIT_L1B_OBS_{self.version}_{self.datetime_str}_{self.orbit_id}_{self.scene_id}"
        l2_rfl_id = f"EMIT_L2A_RFL_{self.version}_{self.datetime_str}_{self.orbit_id}_{self.scene_id}"
        mask_id = f"EMIT_L2A_MASK_{self.version}_{self.datetime_str}_{self.orbit_id}_{self.scene_id}"

        return {
            "rad": f"{base_url}/EMITL1BRAD.001/{rad_id}/{rad_id}.nc",
            "obs": f"{base_url}/EMITL1BRAD.001/{rad_id}/{obs_id}.nc",
            "mask": f"{base_url}/EMITL2ARFL.001/{l2_rfl_id}/{mask_id}.nc",
        }


@unique
class EMITL2AMaskLabel(Enum):
    """
    Labels in the EMIT L2A Mask file.

    Source: https://lpdaac.usgs.gov/products/emitl2arflv001/

    NOTE: we've dropped these bands since they are not boolean:
        5 = AOD550 (Aerosol Optical Depth)
        6 = Water vapor (in g/cm2)

    """

    CLOUD = 0
    CIRRUS_CLOUD = 1
    WATER = 2
    SPACECRAFT = 3  # what is this haha
    DILATED_CLOUD = 4
    AGGREGATE = 7  # composite mask - all flags combined


class EmitGranuleAccess(BaseGranuleAccess):
    """
    Class encapsulating an EMIT tile.

    Additional convenience methods to get metadata and band data.
    """

    def __init__(self, emit_id: str):
        """Initialize EmitGranuleAccess with EMIT tile ID.

        Args:
            emit_id: The EMIT tile identifier
            ml_client: MLClient instance for Azure authentication
        """
        self.emit_id = emit_id
        self.tile_uris = EmitTileUris.from_tile_id(emit_id)
        try:
            self._session = earthaccess.get_requests_https_session()
        except AttributeError as e:
            if "'NoneType' object has no attribute 'get_requests_session'" in str(e):
                raise RuntimeError(
                    "Authentication failed. Please call earthaccess_login() first to "
                    "authenticate with NASA Earthdata Login."
                ) from e
            raise

    @property
    def id(self) -> str:
        """PySTAC ID of the item."""
        return self.emit_id

    @property
    def instrument(self) -> str:
        """Instrument name."""
        return "EMIT"

    @property
    def instrument_name(self) -> str:
        """Instrument name."""
        return "EMIT"

    @property
    def time(self) -> datetime.datetime:
        """Acquisition timestamp."""
        return datetime.datetime.strptime(
            f"{self.tile_uris.acquisition_date}T{self.tile_uris.acquisition_time}",
            "%Y%m%dT%H%M%S",
        )

    @property
    def datetime_(self) -> datetime.datetime:
        """Get the acquisition timestamp."""
        return datetime.datetime.strptime(
            f"{self.tile_uris.acquisition_date}T{self.tile_uris.acquisition_time}",
            "%Y%m%dT%H%M%S",
        )

    @property
    def timestamp(self) -> str:
        """Get the acquisition timestamp."""
        return self.datetime_.time().isoformat(timespec="seconds")

    @property
    def date(self) -> str:
        """Get the acquisition timestamp."""
        return self.datetime_.date().isoformat()

    @property
    def acquisition_start_time(self) -> str:
        """Get the start time of the acquisition."""
        raise NotImplementedError

    @property
    def acquisition_end_time(self) -> str:
        """Get the end time of the acquisition."""
        raise NotImplementedError

    @property
    def imaging_mode(self) -> str:
        """Get the imaging mode."""
        raise NotImplementedError

    @property
    def off_nadir_angle(self) -> float:
        """Get the off-nadir angle."""
        raise NotImplementedError

    @property
    def viewing_azimuth(self) -> float:
        """Get the viewing azimuth."""
        raise NotImplementedError

    @property
    def solar_zenith(self) -> float:
        """Get the solar zenith."""
        raise NotImplementedError

    @property
    def solar_azimuth(self) -> float:
        """Get the solar azimuth."""
        raise NotImplementedError

    @property
    def orbit_state(self) -> str:
        """Get the orbit state (ascending/descending)."""
        raise NotImplementedError

    def _get_dataset(self, product_type: str, group: str | None = None) -> xr.Dataset:
        """Get an xarray Dataset for a specific product type and group."""
        try:
            response = self._session.get(self.tile_uris.uris[product_type])
            response.raise_for_status()
            ds = xr.open_dataset(io.BytesIO(response.content), group=group)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:  # noqa: PLR2004 (magic-number-comparison)
                raise RuntimeError(
                    "Authentication failed. Please check your NASA Earthdata Login "
                    "credentials and try earthaccess_login() again."
                ) from e
            raise
        return ds

    def get_radiance(self) -> xr.DataArray:
        """Get radiance data.

        Returns
        -------
            xr.DataArray: Radiance data with dimensions (bands, y, x).
        """
        rad_ds = self._get_dataset("rad")
        radiance = rad_ds.radiance
        # Copy dataset attributes to maintain CRS and other metadata
        radiance.attrs.update(rad_ds.attrs)
        return radiance

    def get_mask(self) -> xr.DataArray:
        """Get mask data for specified labels."""
        mask_ds = self._get_dataset("mask")  # No group needed for mask
        mask = mask_ds.mask
        mask.attrs.update(mask_ds.attrs)
        return mask

    def get_sensor_band_parameters(self) -> xr.Dataset:
        """Get sensor band parameters."""
        # TODO: this currently will redownload the entire radiance dataset
        # We could think of a way to download it only once.
        sensor_band_parameters = self._get_dataset("rad", group="sensor_band_parameters")
        return sensor_band_parameters

    def get_glt(self) -> xr.Dataset:
        """Get GLT data."""
        glt_ds = self._get_dataset("rad", group="location")
        return glt_ds

    def get_obs(self) -> xr.DataArray:
        """Get observation data containing geometric parameters.

        Returns
        -------
            xr.DataArray: Observation data with dimensions (bands, y, x) containing:
                - path_length: meters - distance between sensor and ground
                - sensor_azimuth: degrees (0-360) clockwise from N
                - sensor_zenith: degrees (0-90) from zenith
                - solar_azimuth: degrees (0-360) clockwise from N
                - solar_zenith: degrees (0-90) from zenith
                - phase: degrees between sensor and sun vectors in principal plane
                - slope: degrees - local surface slope from DEM
                - aspect: degrees (0-360) clockwise from N
                - cosine_i: unitless (0-1) - local illumination factor
                - utc_time: fractional hours since UTC midnight for mid-line pixels
                - earth_sun_dist: AU - distance between Earth and sun
        """
        obs_ds = self._get_dataset("obs")
        # Add band names to observation dataset
        # Note: descriptive labels are available in the netcdf files,
        # but they're not convenient as variables names. Here they are for reference:
        # (thank you to Tim for pointing this out!)
        # > obs_dt = xr.open_datatree(f"{obs_granule_id}.nc")
        # > obs_dt["sensor_band_parameters"].observation_bands.values
        # array(['Path length (sensor-to-ground in meters)',
        #     'To-sensor azimuth (0 to 360 degrees CW from N)',
        #     'To-sensor zenith (0 to 90 degrees from zenith)',
        #     'To-sun azimuth (0 to 360 degrees CW from N)',
        #     'To-sun zenith (0 to 90 degrees from zenith)',
        #     'Solar phase (degrees between to-sensor and to-sun vectors in principal plane)',
        #     'Slope (local surface slope as derived from DEM in degrees)',
        #     'Aspect (local surface aspect 0 to 360 degrees clockwise from N)',
        #     'Cosine(i) (apparent local illumination factor based on DEM slope and aspect and to sun vector)',
        #     'UTC Time (decimal hours for mid-line pixels)',
        #     'Earth-sun distance (AU)'], dtype=object)

        obs_ds["obs"] = obs_ds["obs"].assign_coords(
            bands=[
                "path_length",
                "sensor_azimuth",
                "sensor_zenith",
                "solar_azimuth",
                "solar_zenith",
                "phase",
                "slope",
                "aspect",
                "cosine_i",
                "utc_time",
                "earth_sun_dist",
            ]
        )
        obs = obs_ds.obs
        obs.attrs.update(obs_ds.attrs)
        return obs


def query_emit_catalog(
    lat: float,
    lon: float,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> list[str]:
    """
    Query the EMIT catalog for scenes that cover a given lat/lon point within a time range.

    When multiple scenes from the same date cover the query point, this function selects
    the scene where the point is furthest from the scene boundaries. This helps ensure
    the point is well within the scene's coverage area rather than near its edges.

    Args:
        lat: Latitude of point of interest
        lon: Longitude of point of interest
        start_time: Start datetime to search from
        end_time: End datetime to search until

    Returns
    -------
        List of EMIT scene IDs that cover the point, with one scene selected per date
        when multiple options exist.
    """
    # Query EMIT catalog through earthaccess for tiles containing the point
    results = earthaccess.search_data(
        short_name="EMITL1BRAD",  # L1B Radiance product
        point=(lon, lat),  # Query tiles intersecting this point
        temporal=(start_time, end_time),
        count=100,
    )

    # Helper function to calculate minimum distance from point to polygon boundary
    def distance_to_boundary(polygon: Polygon, point_lon: float, point_lat: float) -> float:
        point = Point(point_lon, point_lat)
        assert polygon.contains(point), "Point is not within the polygon"
        return point.distance(polygon.boundary)

    def create_polygon_from_granule(
        granule: earthaccess.results.DataGranule,
    ) -> Polygon:
        points = granule["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"][
            "Points"
        ]
        coords = [(p["Longitude"], p["Latitude"]) for p in points]
        return Polygon(coords)

    # Group results by date and select granule with maximum distance to boundary for each date
    date_groups: dict[datetime.date, list[earthaccess.results.DataGranule]] = {}
    for r in results:
        date_str = r["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        date = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").date()

        if date not in date_groups:
            date_groups[date] = []
        date_groups[date].append(r)

    emit_ids = []
    for granules in date_groups.values():
        # Find granule with maximum distance to boundary
        best_granule = max(
            granules,
            key=lambda x: distance_to_boundary(create_polygon_from_granule(x), lon, lat),
        )
        if len(granules) > 1:
            logging.info(
                "Choosing %s out of %s options",
                best_granule["meta"]["native-id"],
                [g["meta"]["native-id"] for g in granules],
            )
        emit_ids.append(best_granule["meta"]["native-id"])

    return emit_ids
