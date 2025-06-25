"""PySTAC item wrapper class to get Sentinel L2A data conveniently from Planetary Computer."""

import datetime
import urllib
import warnings
import xml.etree.ElementTree as ET
from enum import Enum, unique
from typing import Any

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac
import rasterio
import shapely
import tqdm
from numpy import typing as npt
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry

from src.data.granule_access import BaseGranuleAccess
from src.utils.geospatial import get_crop_params

PLANETARY_COMPUTER_CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
PLANETARY_COMPUTER_COLLECTION_L2A = "sentinel-2-l2a"

COPERNICUS_CATALOG_URL = "https://stac.dataspace.copernicus.eu/v1/"
COPERNICUS_COLLECTION_L1C = "sentinel-2-l1c"


BAND_RESOLUTIONS = {
    "AOT": 10980,
    "B01": 1830,
    "B02": 10980,
    "B03": 10980,
    "B04": 10980,
    "B05": 5490,
    "B06": 5490,
    "B07": 5490,
    "B08": 10980,
    "B09": 1830,
    "B11": 5490,
    "B12": 5490,
    "B8A": 5490,
    "SCL": 5490,
    "WVP": 10980,
}


@unique
class SceneClassificationLabel(Enum):
    """
    Labels in the scene classification (band SCL). The values are the same as the values in the band data.

    See https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm-overview
    for details.
    """

    NO_DATA = 0
    SATURATED_OR_DEFECTIVE = 1
    CAST_SHADOWS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    SNOW_OR_ICE = 11


class Sentinel2Item(BaseGranuleAccess):
    """
    Class encapsulating a Sentinel 2 tile of type pystac.Item.

    Additional convenience methods to get metadata and band data.
    """

    def __init__(self, pystac_item: pystac.Item):
        self.item = pystac_item
        self._granule_metadata = None

        self.load_metadata()

    def load_metadata(self) -> None:
        """Load metadata."""
        metadata_url = self.item.assets["granule-metadata"].href
        with urllib.request.urlopen(metadata_url) as response:
            self._granule_metadata = response.read()

    @classmethod
    def from_id(cls, item_id: str) -> "Sentinel2Item":
        """Retrieve Sentinel L2A item by its PySTAC ID."""
        retry = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None)
        stac_api_io = StacApiIO(max_retries=retry)

        catalog = Client.open(
            PLANETARY_COMPUTER_CATALOG_URL,
            modifier=planetary_computer.sign_inplace,
            stac_io=stac_api_io,
        )
        search_result = list(catalog.search(collections=[PLANETARY_COMPUTER_COLLECTION_L2A], ids=[item_id]).items())

        try:
            item = search_result[0]
        except IndexError:
            raise ValueError(f"Sentinel 2 item not found with id: {item_id}") from None
        return cls(item)

    @property
    def id(self) -> str:
        """PySTAC ID of the item."""
        return self.item.id

    @property
    def instrument(self) -> str:
        """Instrument id."""
        return self.item.properties["platform"][-1]

    @property
    def instrument_name(self) -> str:
        """Instrument name."""
        return self.item.properties["platform"]

    @property
    def solar_angle(self) -> float:
        """Mean solar zenith angle."""
        return self.item.properties["s2:mean_solar_zenith"]

    @property
    def datetime_(self) -> datetime.datetime:
        """Acquisition datetime."""
        return datetime.datetime.strptime(self.item.properties["datetime"], "%Y-%m-%dT%H:%M:%S.%f%z")

    @property
    def timestamp(self) -> str:
        """Get the acquisition timestamp."""
        return self.datetime_.time().isoformat(timespec="seconds")

    @property
    def date(self) -> str:
        """Get the acquisition date."""
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
    def imaging_mode(self) -> str | None:
        """Get the imaging mode."""
        return None

    @property
    def off_nadir_angle(self) -> float:
        """Get the off-nadir angle."""
        return self.observation_angle

    @property
    def viewing_azimuth(self) -> float:
        """Get the viewing azimuth."""
        granule_metadata_root = ET.fromstring(self._granule_metadata)  # type: ignore
        viewing_azimuth = float(
            granule_metadata_root.find(".//Mean_Viewing_Incidence_Angle_List/*[@bandId='12']/AZIMUTH_ANGLE").text  # type: ignore
        )
        return viewing_azimuth

    @property
    def solar_zenith(self) -> float:
        """Mean solar zenith angle."""
        return self.solar_angle

    @property
    def solar_azimuth(self) -> float:
        """Mean solar azimuth angle."""
        return self.item.properties["s2:mean_solar_azimuth"]

    @property
    def orbit_state(self) -> str:
        """Get the orbit state (ascending/descending)."""
        return self.item.properties["sat:orbit_state"]

    @property
    def crs(self) -> str:
        """Coordinate reference system."""
        return self.item.properties["proj:code"]

    @property
    def observation_angle(self) -> float:
        """Mean observation zenith angle."""
        granule_metadata_root = ET.fromstring(self._granule_metadata)  # type: ignore
        observation_angle = float(
            granule_metadata_root.find(".//Mean_Viewing_Incidence_Angle_List/*[@bandId='12']/ZENITH_ANGLE").text  # type: ignore
        )
        return observation_angle

    @property
    def time(self) -> datetime.datetime:
        """Acquisition time."""
        format = "%Y-%m-%dT%H:%M:%S.%f%z"
        dt = datetime.datetime.strptime(self.item.properties["datetime"], format)
        return dt

    @property
    def swir16_band_name(self) -> str:
        """Reference methane absorption band."""
        return "B11"

    @property
    def swir22_band_name(self) -> str:
        """Main methane absorption band."""
        return "B12"

    @property
    def sensor_name(self) -> str:
        """Sensor name."""
        return "Sentinel2"

    def get_raster_meta(self, band: str) -> dict:
        """Get raster metadata for a band."""
        with rasterio.open(self.item.assets[band].href) as ds:
            transform = ds.transform
            bounds = ds.bounds
            shape = (ds.height, ds.width)

        return {"transform": transform, "bounds": bounds, "shape": shape}

    def get_band(
        self,
        band: str,
        out_height: int | None = None,
        out_width: int | None = None,
        harmonize_if_needed: bool = True,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Download an entire band."""
        with rasterio.open(self.item.assets[band].href) as ds:
            if out_height is None:
                out_height = ds.height
            if out_width is None:
                out_width = ds.width
            band_data = ds.read(
                out_shape=(
                    1,
                    out_height,
                    out_width,
                )
            )
        if harmonize_if_needed:
            band_data = self.harmonize_to_old(band, band_data)
        return band_data

    def get_bands(
        self,
        bands: list[str],
        out_height: int,
        out_width: int,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Download multiple bands."""
        result = np.concatenate([self.get_band(b, out_height, out_width, **kwargs) for b in bands], axis=0)

        return result

    def get_band_crop(  # (too-many-arguments)
        self,
        band: str,
        crop_start_x: int,
        crop_start_y: int,
        crop_height: int,
        crop_width: int,
        out_height: int | None = None,
        out_width: int | None = None,
        harmonize_if_needed: bool = True,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Download a crop of a band."""
        window = rasterio.windows.Window(
            crop_start_x,
            crop_start_y,
            crop_width,
            crop_height,
        )
        out_height = out_height if out_height is not None else crop_height
        out_width = out_width if out_width is not None else crop_width
        with rasterio.open(self.item.assets[band].href) as ds:
            crop = ds.read(
                out_shape=(
                    1,
                    out_height,
                    out_width,
                ),
                window=window,
            )
        if harmonize_if_needed:
            crop = self.harmonize_to_old(band, crop)
        return crop

    def harmonize_to_old(self, band: str, data: npt.NDArray) -> npt.NDArray:
        """
        Harmonize new Sentinel-2 data to the old baseline.

        Based on https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change
        Either returns the input array if no harmonization is needed,
        or returns a new array.
        """
        cutoff = datetime.datetime(2022, 1, 25, tzinfo=datetime.timezone.utc)
        bands_to_process = [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B10",
            "B11",
            "B12",
        ]
        needs_harmonization = self.time >= cutoff and band in bands_to_process
        if not needs_harmonization:
            return data

        offset = 1000
        # To harmonize the post-2022 data to match the pre-2022 data,
        # we need to subtract the offset and clip to zero.
        # But the band values as stored as unsigned integers,
        # so implementing it as `(data - offset).clip(min=0)`
        # results in integer underflow.
        # See https://git.orbio.earth/orbio/methane-cv/-/issues/5
        # for details.
        harmonized = data.clip(min=offset) - offset
        return harmonized

    def get_mask(
        self,
        labels: list[SceneClassificationLabel],
        out_height: int,
        out_width: int,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Generate a mask by combining classifications from the scene classification band based on specified labels."""
        scene_classification_band = "SCL"

        classification_map = self.get_band(scene_classification_band, out_height, out_width, **kwargs)
        result = np.zeros((out_height, out_width), dtype=np.bool_)
        for label in labels:
            result = np.logical_or(result, classification_map == label.value)
        return result

    def get_mask_crop(
        self,
        labels: list[SceneClassificationLabel],
        crop_start_x: int,
        crop_start_y: int,
        crop_height: int,
        crop_width: int,
        out_height: int | None = None,
        out_width: int | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Generate a scene classification mask for a specified crop area based on given labels."""
        scene_classification_band = "SCL"
        if out_height is None:
            out_height = crop_height
        if out_width is None:
            out_width = crop_width

        classification_map = self.get_band_crop(
            scene_classification_band,
            crop_start_x,
            crop_start_y,
            crop_height,
            crop_width,
            out_height,
            out_width,
            **kwargs,
        ).squeeze()
        result = np.zeros((out_height, out_width), dtype=bool)
        for label in labels:
            result = np.logical_or(result, classification_map == label.value)
        return result

    @staticmethod
    def get_mask_from_scmap(labels: list[SceneClassificationLabel], classification_map: npt.NDArray) -> npt.NDArray:
        """Generate a mask from a classification map by combining classifications for specified labels."""
        result = np.zeros((classification_map.shape[0], classification_map.shape[1]), dtype=bool)
        for label in labels:
            result = np.logical_or(result, classification_map == label.value)
        return result


def query_sentinel2_catalog_for_tile(
    tile_id: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    cloud_cover_range: tuple[float, float] | None = None,
) -> list[pystac.Item]:
    """Search for Sentinel 2 items from a specific (mgrs) tile via the Planetary Computer STAC API.

    Arguments:
        tile_id: MGRS tile ID (e.g. '06VVN')
        start_time: Start datetime to search from
        end_time: End datetime to search until
        cloud_cover_range: Optional tuple specifying (lower limit, upper limit) of cloud coverage as decimal

    NOTE: this function is not used by the SBR notebooks, so it is only meant to work with the Planetary
    Computer STAC catalog.
    """
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None)
    stac_api_io = StacApiIO(max_retries=retry)

    catalog = Client.open(
        PLANETARY_COMPUTER_CATALOG_URL,
        modifier=planetary_computer.sign_inplace,
        stac_io=stac_api_io,
    )
    query: dict[str, Any] = {
        "collections": [PLANETARY_COMPUTER_COLLECTION_L2A],
        "query": {"s2:mgrs_tile": {"eq": tile_id}},
        "datetime": f"{start_time.isoformat()}/{end_time.isoformat()}",
    }

    if cloud_cover_range is not None:
        lower_limit, upper_limit = cloud_cover_range
        query["query"]["eo:cloud_cover"] = {
            "gte": lower_limit * 100,
            "lt": upper_limit * 100,
        }

    search = catalog.search(**query)
    items = search.item_collection()
    return list(items)


def pick_single_MGRS(
    item_collection_as_dict: dict, bbox: tuple[float, float, float, float], sbr_notebook: bool = False
) -> str:
    """Pick the MGRS tile that has the largest intersection with the bounding box."""
    query_gdf = gpd.GeoDataFrame.from_features(item_collection_as_dict, crs="epsg:4326")
    crop_bbox = shapely.geometry.box(*bbox)
    with warnings.catch_warnings():  # (ignore warning about CRS)
        warnings.simplefilter("ignore")
        if sbr_notebook:
            intersection_area_sums = query_gdf.groupby("grid:code").apply(
                lambda df: df.intersection(crop_bbox).area.sum()
            )
        else:
            intersection_area_sums = query_gdf.groupby("s2:mgrs_tile").apply(
                lambda df: df.intersection(crop_bbox).area.sum()
            )
    imax = intersection_area_sums.argmax()
    best_MGRS = intersection_area_sums.index[imax]
    if sbr_notebook:
        query_gdf["S2A_B"] = query_gdf["platform"]  # ex: ' "sentinel-2b"'
    else:
        query_gdf["S2A_B"] = query_gdf["s2:granule_id"].apply(lambda x: x.split("_")[0])  # ex: ' S2B'
    return best_MGRS


def query_sentinel2_catalog_for_point(
    lat: float,
    lon: float,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    crop_size: int,
    cloud_cover_range: tuple[float, float] | None = None,
    sbr_notebook: bool = False,
) -> list[pystac.Item]:
    """Query the Sentinel-2 catalog for items within a bounding box defined by latitude, longitude, and time range.

    Arguments:
        lat: Latitude of the point
        lon: Longitude of the point
        start_time: Start datetime to search from
        end_time: End datetime to search until
        crop_size: Size of the crop area in pixels
        cloud_cover_range: Optional tuple specifying (lower limit, upper limit) of cloud coverage as decimal
    """
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None)
    stac_api_io = StacApiIO(max_retries=retry)
    if sbr_notebook:
        catalog = Client.open(
            COPERNICUS_CATALOG_URL,
            stac_io=stac_api_io,
        )
        collection = COPERNICUS_COLLECTION_L1C
    else:
        catalog = Client.open(
            PLANETARY_COMPUTER_CATALOG_URL,
            modifier=planetary_computer.sign_inplace,
            stac_io=stac_api_io,
        )
        collection = PLANETARY_COMPUTER_COLLECTION_L2A

    query: dict[str, Any] = {
        "collections": [collection],
        "datetime": f"{start_time.isoformat()}/{end_time.isoformat()}",
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
    }

    km2deg = 1.0 / 111  # rough conversion factor from kilometers to degrees of latitude
    # At 20m resolution, we want to search a 128x128
    # pixel bounding box around the target point
    halfwidth = 20 * crop_size / 1000 * km2deg
    bbox = (lon - halfwidth, lat - halfwidth, lon + halfwidth, lat + halfwidth)
    # (note: this ignores the fact that a degree of longitude is a smaller
    #  distance, as it doesn't hugely matter)

    if cloud_cover_range is not None:
        lower_limit, upper_limit = cloud_cover_range
        query["query"] = {"eo:cloud_cover": {"gte": lower_limit * 100, "lt": upper_limit * 100}}

    search = catalog.search(**query)
    items = list(search.items())

    if len(items) == 0:
        return []

    best_MGRS = pick_single_MGRS(search.item_collection_as_dict(), bbox, sbr_notebook)

    if sbr_notebook:
        items = [item for item in items if item.properties["grid:code"] == best_MGRS]
    else:
        items = [item for item in items if item.properties["s2:mgrs_tile"] == best_MGRS]

    items = sorted(items, key=lambda item: item.properties["datetime"])
    return items


def load_cropped_s2_items(
    items: list[Sentinel2Item],
    bands: list[str],
    lat: float,
    lon: float,
    image_size: int,
    show_progress: bool = False,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Load cropped Sentinel-2 items for a given list of bands, latitude, longitude, and image size."""
    item_iter = tqdm.tqdm(items) if show_progress else items
    cropped_data = []
    for item in item_iter:
        band_arrs = []
        crop_param_dict = {}
        # wrap the STAC item in our own Sentinel2Item
        for band in bands:
            # TODO: we should remove SCL?
            item_meta = item.get_raster_meta(band if band not in ["OmniCloud", "SCL"] else "B12")
            crop_params = get_crop_params(
                lat,
                lon,
                image_size,
                item_meta["shape"],
                item.crs,
                item_meta["transform"],
                out_res=20,  # Standard resolution for bands 11 & 12
            )
            # then get those pixels
            # (Note: this opens the file twice, which is a bit wasteful)
            cropped_band = item.get_band_crop(band, **crop_params, **kwargs)
            band_arrs.append(cropped_band)
            crop_param_dict[band] = crop_params
        # We then concatenate the individual bands into a 13-band numpy array
        cropped_data.append(
            {
                "crop_arrays": np.concatenate(band_arrs, axis=0),
                "crop_params": crop_param_dict,
                "tile_item": item,
            }
        )
    return cropped_data
