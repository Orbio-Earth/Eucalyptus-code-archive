"""PySTAC item wrapper class to get Landsat data."""

from __future__ import annotations

import datetime
import math
import os
import re
import tempfile
from collections import Counter
from enum import Enum, unique
from pathlib import Path
from typing import Any

import azure.core.exceptions
import numpy as np
import pystac
import rasterio
import tqdm
from azure.storage.blob import BlobServiceClient, ContainerClient
from mypy_boto3_s3 import S3Client
from numpy import typing as npt
from pydantic import BaseModel, ConfigDict
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry

from src.azure_wrap.ml_client_utils import download_blob_directly
from src.data.granule_access import BaseGranuleAccess
from src.utils.geospatial import get_crop_params
from src.utils.utils import setup_logging

logger = setup_logging()

CATALOG_URL = "https://landsatlook.usgs.gov/stac-server"
# NOTE: this is for Landsat Collection 2, Level 1 data. Level 1 has 3 Tiers: RT, T1 (L1TP), T2 (L1GT/L1GS). Landsat data
# is first uploaded as RT (within 12hrs of aqcuistion) then later updated to either T1 or T2. Typically, RT is available
# 4-6 hrs after acquistion, so we may only encounter it if we are processing data in real time.
COLLECTION2_LANDSAT_L1 = "landsat-c2l1"

LANDSAT_STAC_ASSET_MAP = {
    "coastal": "B1",
    "blue": "B2",
    "green": "B3",
    "red": "B4",
    "nir08": "B5",
    "swir16": "B6",
    "swir22": "B7",
    "cirrus": "B9",
    "lwir11": "B10",
    "lwir12": "B11",
    "qa_pixel": "QA_PIXEL",
    "qa_radsat": "QA_RADSAT",
    "MTL.json": "MTL.json",
}
LANDSAT_STAC_REVERSE_MAP = dict((val, key) for key, val in LANDSAT_STAC_ASSET_MAP.items())


class LandsatImageAttributes(BaseModel):
    """Attributes of a Landsat image found in the MTL.json file."""

    model_config = ConfigDict(extra="allow")

    SPACECRAFT_ID: str | None = None
    SENSOR_ID: str | None = None
    WRS_TYPE: int | None = None
    WRS_PATH: int | None = None
    WRS_ROW: int | None = None
    NADIR_OFFNADIR: str | None = None
    TARGET_WRS_PATH: int | None = None
    TARGET_WRS_ROW: int | None = None
    DATE_ACQUIRED: str | None = None
    SCENE_CENTER_TIME: str | None = None
    STATION_ID: str | None = None
    CLOUD_COVER: float | None = None
    CLOUD_COVER_LAND: float | None = None
    IMAGE_QUALITY_OLI: int | None = None
    IMAGE_QUALITY_TIRS: int | None = None
    SATURATION_BAND_1: str | None = None
    SATURATION_BAND_2: str | None = None
    SATURATION_BAND_3: str | None = None
    SATURATION_BAND_4: str | None = None
    SATURATION_BAND_5: str | None = None
    SATURATION_BAND_6: str | None = None
    SATURATION_BAND_7: str | None = None
    SATURATION_BAND_8: str | None = None
    SATURATION_BAND_9: str | None = None
    ROLL_ANGLE: float | None = None
    SUN_AZIMUTH: float | None = None
    SUN_ELEVATION: float | None = None
    EARTH_SUN_DISTANCE: float | None = None
    SENSOR_MODE: str | None = None


class LandsatRadiometricRescaling(BaseModel):
    """Radiometric rescaling coefficients found in the MTL.json file."""

    RADIANCE_MULT_BAND_1: float
    RADIANCE_ADD_BAND_1: float
    RADIANCE_MULT_BAND_2: float
    RADIANCE_ADD_BAND_2: float
    RADIANCE_MULT_BAND_3: float
    RADIANCE_ADD_BAND_3: float
    RADIANCE_MULT_BAND_4: float
    RADIANCE_ADD_BAND_4: float
    RADIANCE_MULT_BAND_5: float
    RADIANCE_ADD_BAND_5: float
    RADIANCE_MULT_BAND_6: float
    RADIANCE_ADD_BAND_6: float
    RADIANCE_MULT_BAND_7: float
    RADIANCE_ADD_BAND_7: float
    RADIANCE_MULT_BAND_8: float
    RADIANCE_ADD_BAND_8: float
    RADIANCE_MULT_BAND_9: float
    RADIANCE_ADD_BAND_9: float
    RADIANCE_MULT_BAND_10: float
    RADIANCE_ADD_BAND_10: float
    RADIANCE_MULT_BAND_11: float
    RADIANCE_ADD_BAND_11: float
    REFLECTANCE_MULT_BAND_1: float
    REFLECTANCE_ADD_BAND_1: float
    REFLECTANCE_MULT_BAND_2: float
    REFLECTANCE_ADD_BAND_2: float
    REFLECTANCE_MULT_BAND_3: float
    REFLECTANCE_ADD_BAND_3: float
    REFLECTANCE_MULT_BAND_4: float
    REFLECTANCE_ADD_BAND_4: float
    REFLECTANCE_MULT_BAND_5: float
    REFLECTANCE_ADD_BAND_5: float
    REFLECTANCE_MULT_BAND_6: float
    REFLECTANCE_ADD_BAND_6: float
    REFLECTANCE_MULT_BAND_7: float
    REFLECTANCE_ADD_BAND_7: float
    REFLECTANCE_MULT_BAND_8: float
    REFLECTANCE_ADD_BAND_8: float
    REFLECTANCE_MULT_BAND_9: float
    REFLECTANCE_ADD_BAND_9: float


class LandsatThermalConstants(BaseModel):
    """Thermal constants found in the MTL.json file."""

    K1_CONSTANT_BAND_10: float
    K2_CONSTANT_BAND_10: float
    K1_CONSTANT_BAND_11: float
    K2_CONSTANT_BAND_11: float


class LandsatImageMetadata(BaseModel):
    """Constants from the MTL.json file."""

    IMAGE_ATTRIBUTES: LandsatImageAttributes
    LEVEL1_RADIOMETRIC_RESCALING: LandsatRadiometricRescaling
    LEVEL1_THERMAL_CONSTANTS: LandsatThermalConstants


class LandsatImageMetadataFile(BaseModel):
    """Metadata from the MTL.json file."""

    LANDSAT_METADATA_FILE: LandsatImageMetadata


@unique
class LandsatQAValues(Enum):
    """
    Bit values in the QA_PIXEL band for Landsat Collection 2.

    Each value represents specific bits being set in the 16-bit QA band.
    See https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands
    """

    # Single bit flags
    FILL = 1 << 0  # 0b0000000000000001
    DILATED_CLOUD = 1 << 1  # 0b0000000000000010
    CIRRUS = 1 << 2  # 0b0000000000000100
    CLOUD = 1 << 3  # 0b0000000000001000
    CLOUD_SHADOW = 1 << 4  # 0b0000000000010000
    SNOW = 1 << 5  # 0b0000000000100000
    CLEAR = 1 << 6  # 0b0000000001000000
    WATER = 1 << 7  # 0b0000000010000000

    # Cloud confidence (bits 8-9)
    CLOUD_CONFIDENCE_LOW = 1 << 8  # 0b0000000100000000
    CLOUD_CONFIDENCE_MEDIUM = 2 << 8  # 0b0000001000000000
    CLOUD_CONFIDENCE_HIGH = 3 << 8  # 0b0000001100000000

    # Cloud shadow confidence (bits 10-11)
    CLOUD_SHADOW_CONFIDENCE_LOW = 1 << 10  # 0b0000010000000000
    CLOUD_SHADOW_CONFIDENCE_MEDIUM = 2 << 10  # 0b0000100000000000
    CLOUD_SHADOW_CONFIDENCE_HIGH = 3 << 10  # 0b0000110000000000

    # Snow/Ice confidence (bits 12-13)
    SNOW_ICE_CONFIDENCE_LOW = 1 << 12  # 0b0001000000000000
    SNOW_ICE_CONFIDENCE_MEDIUM = 2 << 12  # 0b0010000000000000
    SNOW_ICE_CONFIDENCE_HIGH = 3 << 12  # 0b0011000000000000

    # Cirrus confidence (bits 14-15)
    CIRRUS_CONFIDENCE_LOW = 1 << 14  # 0b0100000000000000
    CIRRUS_CONFIDENCE_MEDIUM = 2 << 14  # 0b1000000000000000
    CIRRUS_CONFIDENCE_HIGH = 3 << 14  # 0b1100000000000000


class LandsatGranuleAccess(BaseGranuleAccess):
    """
    Class encapsulating a Landsat tile of type pystac.Item.

    Additional convenience methods to get metadata and band data.
    """

    def __init__(self, pystac_item: pystac.Item):
        self.item = pystac_item
        self._granule_metadata: LandsatImageMetadataFile | None = None
        self._granule_metadata_url = self.item.assets["MTL.json"].to_dict()["alternate"]["s3"]["href"]
        self.l1_abs_bucket = "l1c-data"

    def load_metadata(self, s3_client: S3Client, abs_client: BlobServiceClient) -> None:
        """Load metadata from S3.

        This makes a request to S3 to retrieve the metadata file.
        The request cost is charged to the requester.
        """
        self.prefetch_l1(s3_client, abs_client)
        self._granule_metadata = self.get_mtl_data(abs_client=abs_client)

    @classmethod
    def from_id(cls, item_id: str) -> LandsatGranuleAccess:
        """Retrieve Landsat item by its PySTAC ID."""
        catalog = Client.open(
            CATALOG_URL,
        )
        search_result = list(catalog.search(collections=[COLLECTION2_LANDSAT_L1], ids=[item_id]).items())
        try:
            item = search_result[0]
        except IndexError:
            raise ValueError(f"Landsat item not found with id: {item_id}") from None
        return cls(item)

    @property
    def id(self) -> str:
        """PySTAC ID of the item."""
        return self.item.id

    @property
    def instrument(self) -> str:
        """Instrument ID."""
        # FIXME: we are matching what we do with Sentinel and how the Radtran lib is implemented.
        # In Radtran, the instrument name is constructed as `f"{full_sensor_name}{instrument}"`
        # (see src.utils.radtran_utils import RadTranLookupTable._get_full_instrument_name). So
        # in Sentinel, instrument is just the letter (i.e "A") and we have a separate method to
        # get the full sensor name "Sentinel2", which has to match the enums used in
        # radtran.radtran.utils.data import Instrument. So for Landsat we do the same: although
        # platform would return something like `LANDSAT_8`, we just take the last index `8` and
        # use that to construct the instrument name "Landsat8" which matches the enum in
        # radtran.radtran.utils.data import Instrument. I think the ideal solution would be to
        # match the radtran enums with the pystac item properties, but this should be addressed in
        # a separate MR and update all radtran files accordingly.
        return self.item.properties["platform"][-1]

    @property
    def instrument_name(self) -> str:
        """Instrument name."""
        return self.item.properties["platform"]

    @property
    def sensor_name(self) -> str:
        """Sensor name."""
        return "Landsat"

    @property
    def solar_angle(self) -> float:
        """Mean solar zenith angle."""
        # As indicated here https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product
        # the solar zenith angle is equal 90 - sun_elevation
        return 90 - self.item.properties["view:sun_elevation"]

    @property
    def crs(self) -> str:
        """Coordinate reference system."""
        return self.item.properties["proj:code"]

    @property
    def observation_angle(self) -> float:
        """Mean observation zenith angle."""
        return self.item.properties["view:off_nadir"]

    @property
    # NOTE: should rename this to datetime_ but unclear how much of a refactor this would be
    def time(self) -> datetime.datetime:
        """Acquisition time."""
        format = "%Y-%m-%dT%H:%M:%S.%f%z"
        dt = datetime.datetime.strptime(self.item.properties["datetime"], format)
        return dt

    @property
    def datetime_(self) -> datetime.datetime:
        """Get the acquisition timestamp."""
        raise NotImplementedError

    @property
    def timestamp(self) -> str:
        """Get the acquisition timestamp."""
        return self.time.time().isoformat(timespec="seconds")

    @property
    def date(self) -> str:
        """Get the acquisition date."""
        return self.time.date().isoformat()

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
        if not self._granule_metadata:
            raise ValueError("To access metadata, first run `load_metadata(s3_client: S3Client)`.")
        return self._granule_metadata.LANDSAT_METADATA_FILE.IMAGE_ATTRIBUTES.SENSOR_MODE

    @property
    def off_nadir_angle(self) -> float:
        """Get the off-nadir angle."""
        return self.item.properties["view:off_nadir"]

    @property
    def viewing_azimuth(self) -> float | None:
        """Get the viewing azimuth."""
        if not self._granule_metadata:
            raise ValueError("To access metadata, first run `load_metadata(s3_client: S3Client)`.")
        return self._granule_metadata.LANDSAT_METADATA_FILE.IMAGE_ATTRIBUTES.SUN_AZIMUTH

    @property
    def solar_zenith(self) -> float:
        """Get the solar zenith."""
        return self.item.properties["view:sun_elevation"]

    @property
    def solar_azimuth(self) -> float:
        """Get the solar azimuth."""
        return self.item.properties["view:sun_azimuth"]

    @property
    def orbit_state(self) -> str:
        """Get the orbit state (ascending/descending)."""
        raise NotImplementedError

    @property
    def swir16_band_name(self) -> str:
        """Reference methane absorption band."""
        return "swir16"

    @property
    def swir22_band_name(self) -> str:
        """Main methane absorption band."""
        return "swir22"

    def get_band_s3_uri(self, band: str) -> str:
        """Retrieve the href for a specific band."""
        asset = self.item.assets[band]
        s3_uri = asset.extra_fields["alternate"]["s3"]["href"]
        return s3_uri

    def get_band_abs_path(self, band: str) -> str:
        """Retrieve the href for a specific band."""
        _, prefix, file_name = self.s3_components_for_band(band)
        abs_path = prefix + "/" + file_name
        return abs_path

    # FIXME: get_raster_meta was so simple for sentinel2 since the pystac item had the ABS path. For Landsat, we only
    # have the S3 path in the pystac item. We have three options: 1. figure out how to pass in S3 credentials and use
    # the rasterio AWS session auth. 2. download the file to a temp location and pass in the local ABS path (like we do
    # for the other methods i.e get_band). 3. figure out how to stream in our file from ABS like whats done with the S2
    # ABS path. Option 3 is best, but requires some changes to the code which is planned for another MR
    # https://git.orbio.earth/orbio/orbio/-/merge_requests/1136
    def get_raster_meta(self, band: str, abs_client: BlobServiceClient) -> dict:
        """Get raster metadata for a band."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            download_blob_directly(
                blob_name=self.get_band_abs_path(band),
                local_download_filepath=temp_path / "tmp.tif",
                blob_service_client=abs_client,
                container_name=self.l1_abs_bucket,
            )

            with rasterio.open(temp_path / "tmp.tif") as ds:
                metadata = ds.meta

        return metadata

    def get_raster_as_tmp(self, band: str, abs_client: BlobServiceClient) -> None:
        """Download band as tmp.tif."""
        download_blob_directly(
            blob_name=self.get_band_abs_path(band),
            local_download_filepath=Path("tmp.tif"),
            blob_service_client=abs_client,
            container_name=self.l1_abs_bucket,
        )

    def get_mtl_data(
        self,
        **kwargs: Any,
    ) -> LandsatImageMetadataFile:
        """Download and parse the Landsat MTL.json metadata file.."""
        abs_client = kwargs.get("abs_client")
        if abs_client is None:
            raise ValueError("abs_client is required")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            download_blob_directly(
                blob_name=self.get_band_abs_path("MTL.json"),
                local_download_filepath=temp_path / "tmp.json",
                blob_service_client=abs_client,
                container_name=self.l1_abs_bucket,
            )

            with open(temp_path / "tmp.json", "rb") as f:
                mtl_json = f.read().decode("utf-8")

            mtl_data = LandsatImageMetadataFile.model_validate_json(mtl_json)

        return mtl_data

    def get_band(
        self,
        band: str,
        out_height: int | None = None,
        out_width: int | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Download an entire band."""
        abs_client = kwargs.get("abs_client")
        if abs_client is None:
            raise ValueError("abs_client is required")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            download_blob_directly(
                blob_name=self.get_band_abs_path(band),
                local_download_filepath=temp_path / "tmp.tif",
                blob_service_client=abs_client,
                container_name=self.l1_abs_bucket,
            )

            with rasterio.open(temp_path / "tmp.tif") as ds:
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
        return band_data

    def get_bands(
        self,
        bands: list[str],
        out_height: int | None = None,
        out_width: int | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Download multiple bands."""
        assert "MTL.json" not in bands
        result = np.concatenate([self.get_band(b, out_height, out_width, **kwargs) for b in bands], axis=0)

        return result

    def get_band_crop(
        self,
        band: str,
        crop_start_x: int,
        crop_start_y: int,
        crop_height: int,
        crop_width: int,
        out_height: int | None = None,
        out_width: int | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Download a crop of a band."""
        abs_client = kwargs.get("abs_client")
        if abs_client is None:
            raise ValueError("abs_client is required")

        _, l1_prefix, _ = self.s3_components_for_band(band)
        # Create cache path that includes crop parameters
        cache_blob_name = (
            f"cached_crops/{l1_prefix}/{band}_{crop_start_x}_{crop_start_y}"
            f"_{crop_height}_{crop_width}_{out_height}_{out_width}.npy"
        )

        container_client = abs_client.get_container_client(self.l1_abs_bucket)

        try:
            # Try to download from cache
            crop = self._get_crop_from_cache(container_client, cache_blob_name)
        except azure.core.exceptions.ResourceNotFoundError:
            # If not in cache, download full band, crop it, and cache the crop
            window = rasterio.windows.Window(
                crop_start_x,
                crop_start_y,
                crop_width,
                crop_height,
            )
            out_height = out_height if out_height is not None else crop_height
            out_width = out_width if out_width is not None else crop_width

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                download_blob_directly(
                    blob_name=self.get_band_abs_path(band),
                    local_download_filepath=temp_path / "tmp.tif",
                    blob_service_client=abs_client,
                    container_name=self.l1_abs_bucket,
                )

                with rasterio.open(temp_path / "tmp.tif") as ds:
                    crop = ds.read(
                        out_shape=(
                            1,
                            out_height,
                            out_width,
                        ),
                        window=window,
                    )

            # Cache the crop
            self._cache_crop_to_abs(crop, cache_blob_name, container_client)

        return crop

    def _get_crop_from_cache(self, container_client: ContainerClient, blob_name: str) -> npt.NDArray:
        """Download and load a cached crop from ABS."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "temp.npy"

            # Download blob to temporary file
            with open(temp_path, "wb") as temp_file:
                blob_data = container_client.download_blob(blob_name).readall()
                temp_file.write(blob_data)

            # Load numpy array
            return np.load(temp_path)

    def _cache_crop_to_abs(self, crop: npt.NDArray, blob_name: str, container_client: ContainerClient) -> None:
        """Cache a cropped array to ABS."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "temp.npy"

            # Save array to temporary file
            np.save(temp_path, crop)

            # Upload to ABS cache
            with open(temp_path, "rb") as data:
                container_client.upload_blob(
                    name=blob_name,
                    data=data.read(),
                    validate_content=True,
                    overwrite=True,
                )

    @staticmethod
    def get_mask_from_scmap(labels: list[LandsatQAValues], classification_map: npt.NDArray) -> npt.NDArray:
        """Alias for get_mask_from_qa_pixel for compatibility with Sentinel2Item."""
        return LandsatGranuleAccess.get_mask_from_qa_pixel(labels, qa_array=classification_map)

    @staticmethod
    def get_mask_from_qa_pixel(labels: list[LandsatQAValues], qa_array: npt.NDArray) -> npt.NDArray:
        """
        Create a boolean mask from QA values.

        Args:
            labels: Sequence of QA values to check for
            qa_array: The QA_PIXEL band array

        Returns
        -------
            Boolean mask where True indicates the bits match any of the QA values
        """
        mask = np.zeros_like(qa_array, dtype=bool)
        for qa_value in labels:
            mask |= (qa_array & qa_value.value) == qa_value.value
        return mask

    def get_mask(
        self,
        labels: list[LandsatQAValues],
        out_height: int | None = None,
        out_width: int | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Generate a mask by combining QA values from the QA_PIXEL band based on specified labels.

        Args:
            labels: List of QA values to check for
            out_height: Output height of the mask
            out_width: Output width of the mask
            **kwargs: Additional arguments passed to get_band

        Returns
        -------
            Boolean mask where True indicates any of the specified QA values are present
        """
        qa_band = "qa_pixel"
        qa_array = self.get_band(qa_band, out_height, out_width, **kwargs)
        return self.get_mask_from_qa_pixel(labels, qa_array[0])  # Remove the channel dimension

    def get_mask_crop(
        self,
        labels: list[LandsatQAValues],
        crop_start_x: int,
        crop_start_y: int,
        crop_height: int,
        crop_width: int,
        out_height: int | None = None,
        out_width: int | None = None,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Generate a QA mask for a specified crop area based on given QA values.

        Args:
            labels: List of QA values to check for
            crop_start_x: Starting x coordinate of the crop
            crop_start_y: Starting y coordinate of the crop
            crop_height: Height of the crop
            crop_width: Width of the crop
            out_height: Optional output height (defaults to crop_height)
            out_width: Optional output width (defaults to crop_width)
            **kwargs: Additional arguments passed to get_band_crop

        Returns
        -------
            Boolean mask where True indicates any of the specified QA values are present
        """
        qa_band = "qa_pixel"
        if out_height is None:
            out_height = crop_height
        if out_width is None:
            out_width = crop_width

        qa_array = self.get_band_crop(
            qa_band,
            crop_start_x,
            crop_start_y,
            crop_height,
            crop_width,
            out_height,
            out_width,
            **kwargs,
        )
        return self.get_mask_from_qa_pixel(labels, qa_array[0])  # Remove the channel dimension

    def _decompose_s3_uri(self, s3_uri: str) -> tuple[str, str, str]:
        """
        Decompose an S3 URI into its components.

        Example S3 URI:
        s3://usgs-landsat/collection02/level-1/standard/oli-tirs/2025/223/128/LC09_L1GT_223128_20250210_20250210_02_T2/LC09_L1GT_223128_20250210_20250210_02_T2_B4.TIF

        Returns
        -------
            tuple[str, str, str]: A tuple containing:
                - bucket_name: The S3 bucket name (e.g., 'usgs-landsat')
                - prefix: The path without bucket and filename (e.g., 'collection02/level-1/...')
                - file_name: The file name (e.g., 'LC09_L1GT_223128_20250210_20250210_02_T2_B4.TIF')
        """
        # Remove 's3://' prefix
        s3_uri = s3_uri.split("s3://", 1)[1]
        # Split into bucket name and remaining path
        bucket_name, path = s3_uri.split("/", 1)
        # Split path into prefix and filename
        prefix, file_name = path.rsplit("/", 1)
        return bucket_name, prefix, file_name

    def s3_components_for_band(self, band: str) -> tuple[str, str, str]:
        """
        Get the S3 bucket, prefix, and filename components for a specific band.

        Args:
            band: The band name (e.g., 'red', 'nir08', 'qa_pixel')

        Returns
        -------
            tuple[str, str, str]: A tuple containing:
                - bucket_name: The S3 bucket name (e.g., 'usgs-landsat')
                - prefix: The path without bucket and filename
                - file_name: The band's file name
        """
        s3_uri = self.get_band_s3_uri(band)
        return self._decompose_s3_uri(s3_uri)

    # FIXME: for landsat its prefetch_l1, S2 is prefetch_l1c
    def prefetch_l1(
        self, s3_client: S3Client, abs_client: BlobServiceClient, bands_to_transfer: list[str] | None = None
    ) -> None:
        """
        Check if Landsat data is on ABS. If not, transfer it from S3.

        This method ensures that all necessary Landsat L1 bands are available on Azure Blob Storage
        before attempting to access them, minimizing repeated S3 egress costs.

        Args:
            s3_client: AWS S3 client for downloading from USGS public bucket
            abs_client: Azure Blob Storage client for uploading to our storage
        """
        container_client = abs_client.get_container_client(self.l1_abs_bucket)
        assert container_client.exists()
        self.transfer_l1_to_abs(s3_client, container_client, bands_to_transfer)

    def return_bands_on_abs(self, container_client: ContainerClient) -> list[str]:
        """Which bands are already on ABS?."""
        _, l1_prefix, _ = self.s3_components_for_band("qa_pixel")  # we can use any band to get the prefix
        list_blobs = [k.name for k in container_client.list_blobs(name_starts_with=l1_prefix)]
        return list_blobs

    def is_band_on_abs(self, band: str, existing_bands: list[str]) -> bool:
        """Check if specific band exists on ABS."""
        _, l1_prefix, filename = self.s3_components_for_band(band)
        return f"{l1_prefix}/{filename}" in existing_bands

    def transfer_l1_to_abs(
        self, s3_client: S3Client, container_client: ContainerClient, bands_to_transfer: list[str] | None = None
    ) -> None:
        """
        Transfer Landsat data from S3 to ABS.

        Downloads each band from USGS S3 bucket and uploads to our Azure Blob Storage.
        File structure on ABS will mirror the S3 structure:
        collection02/level-1/standard/oli-tirs/YYYY/PPP/RRR/SCENE_ID/SCENE_ID_BN.TIF

        Args:
            s3_client: AWS S3 client for downloading from USGS public bucket
            container_client: Azure Blob Storage container client for uploading
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Get list of existing bands
            bands_existing = [k.split("/")[-1] for k in self.return_bands_on_abs(container_client)]

            # Use specified bands or all bands
            bands_to_process = bands_to_transfer if bands_to_transfer is not None else LANDSAT_STAC_ASSET_MAP

            for band in bands_to_process:
                s3_bucket, prefix, file_name = self.s3_components_for_band(band)

                # Skip if band already exists
                if file_name in bands_existing:
                    continue

                # Recreate directory structure
                os.makedirs(temp_path / prefix, exist_ok=True)
                source_file = f"{prefix}/{file_name}"
                temp_file = temp_path / prefix / file_name

                # Download from S3 (requires requester pays)
                s3_client.download_file(
                    Bucket=s3_bucket,
                    Key=source_file,
                    Filename=str(temp_file),
                    ExtraArgs={"RequestPayer": "requester"},
                )

                # Upload to ABS maintaining the same path structure
                with open(temp_file, "rb") as data:
                    container_client.upload_blob(
                        name=source_file,
                        data=data.read(),
                        validate_content=True,
                        overwrite=True,
                    )

                print(f"Successfully transferred {source_file} to {self.l1_abs_bucket}")

    # NOTE: taken from data_product/satellite_data_product/landsat/utils/download_utils.py
    @staticmethod
    def parse_landsat_tile_id(tile_id: str) -> dict[str, str]:
        """Parse the Landsat tile ID and extract its components."""
        pattern = r"^(L[CEOTM])(\d{2})_(L1[GTPS]{2})_(\d{3})(\d{3})_(\d{8})_(\d{8})_(\d{2})_(RT|T1|T2)$"
        match = re.match(pattern, tile_id)

        if not match:
            raise ValueError(f"Invalid Landsat tile ID format: {tile_id}")

        return {
            "satellite": match.group(1),  # L
            "sensor": match.group(2),  # X (e.g., "08")
            "processing_level": match.group(3),  # LLL (e.g., "L1TP")
            "wrs_path": match.group(4),  # PPP (e.g., "030")
            "wrs_row": match.group(5),  # RRR (e.g., "038")
            "acquisition_date": match.group(6),  # YYYYMMDD (e.g., "20220101")
            "processing_date": match.group(7),  # yyyymmdd (e.g., "20220106")
            "collection_number": match.group(8),  # CC (e.g., "02")
            "collection_category": match.group(9),  # TX (e.g., "T1")
        }

    @staticmethod
    def convert_band_dn_to_reflectance_values(
        band_data: npt.NDArray, solar_zenith_angle: float, reflectance_mult_factor: float, reflectance_add_factor: float
    ) -> npt.NDArray:
        """
        Landsat 7-9 pixel values are stored as DN, we need to convert them to TOA reflectance.

        See https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product for conversion details.
        """
        assert band_data.dtype == np.uint16, "Band data is the raw data from the Landsat product and must be uint16."

        # Keep track of original nodata pixels
        nodata_mask_dn = band_data == 0

        # Convert DN (digital number) values to TOA reflectance
        band_data = reflectance_mult_factor * band_data + reflectance_add_factor

        sun_angle_correction_factor = math.cos(
            math.radians(solar_zenith_angle)
        )  # Convert solar_zenith_angle to radians

        # Apply sun angle correction
        band_data /= sun_angle_correction_factor

        # The neural network expects inputs on a 0-1 scale (reflectance), but we scale to 0-10000
        # range for uint16 storage and save space. Later during training we will normalise by / 10000.
        band_data *= 10000

        # Set any negative reflectance values as 1. We use 1 instead of 0 because 0 is used to indicate nodata
        # NOTE: reflectance values can be above 10000, in training we will normalise using a bigger value
        band_data[band_data <= 0] = 1.0

        # Avoid overflow
        UINT16_MAX = 65535
        band_data[band_data >= UINT16_MAX] = UINT16_MAX

        # Restore original nodata pixels to 0
        band_data[nodata_mask_dn] = 0

        return np.round(band_data, 0).astype(np.uint16)

    @staticmethod
    def convert_thermal_band_dn_to_brightness_values(
        band_data: npt.NDArray,
        radiance_mult_scaling_factor: float,
        radiance_add_scaling_factor: float,
        k1_thermal_constant: float,
        k2_thermal_constant: float,
    ) -> npt.NDArray:
        """
        Convert thermal bands (lwir11, lwir12) from DN to brightness temperature values (Kelvin).

        See https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product for conversion details.
        """
        assert band_data.dtype == np.uint16, "Band data is the raw data from the Landsat product and must be uint16."

        # Keep track of original nodata pixels
        nodata_mask_dn = band_data == 0

        # Convert DN (digital number) values to TOA radiance in place
        band_data = radiance_mult_scaling_factor * band_data + radiance_add_scaling_factor

        # Convert TOA radiance to brightness temperature
        band_data = k2_thermal_constant / np.log((k1_thermal_constant / band_data) + 1)

        # toa_brightness is in Kelvin. We * 10 for uint16 storage. It is rare Kelvin values exceed 400, so *10 still
        # provides a reasonable range in uint16.
        band_data *= 10

        # Avoid overflow
        UINT16_MAX = 65535
        band_data[band_data >= UINT16_MAX] = UINT16_MAX

        # Restore original nodata pixels to 0
        band_data[nodata_mask_dn] = 0

        assert np.all((band_data > 0) | nodata_mask_dn), "Brightness values should be > 0 unless nodata."

        return np.round(band_data, 0).astype(np.uint16)


############################################################
################ HELPER FUNCTIONS TO QUERY #################
############################################################


def query_landsat_catalog_for_tile(
    wrs_path: str,
    wrs_row: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    cloud_cover_range: tuple[float, float] | None = None,
) -> list[pystac.Item]:
    """Search for Landsat Collection 2, Level 1 items for a specific WRS path/row tile.

    Arguments:
        wrs_path: WRS path of tile; like "111"
        wrs_row: WRS row of tile; like "222"
        start_time: Start datetime to search from
        end_time: End datetime to search until
        cloud_cover_range: Optional tuple specifying (lower limit, upper limit) of cloud coverage as decimal

    NOTE: In time series query, the main filter is making sure we use same WRS path/row for all items
    """
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None)
    stac_api_io = StacApiIO(max_retries=retry)

    catalog = Client.open(
        CATALOG_URL,
        modifier=None,
        stac_io=stac_api_io,
    )
    query: dict[str, Any] = {
        "collections": [COLLECTION2_LANDSAT_L1],
        "query": {
            "platform": {"in": ["LANDSAT_8", "LANDSAT_9"]},
            "landsat:wrs_path": {"eq": wrs_path},
            "landsat:wrs_row": {"eq": wrs_row},
        },
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


def query_landsat_catalog_for_point(
    lat: float,
    lon: float,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    cloud_cover_range: tuple[float, float] | None = None,
) -> list[pystac.Item]:
    """Query the Landsat catalog for items within a bounding box defined by latitude, longitude, and time range.

    Arguments:
        lat: Latitude of the point
        lon: Longitude of the point
        start_time: Start datetime to search from
        end_time: End datetime to search until
        cloud_cover_range: Optional tuple specifying (lower limit, upper limit) of cloud coverage as decimal
    """
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None)
    stac_api_io = StacApiIO(max_retries=retry)

    catalog = Client.open(CATALOG_URL, modifier=None, stac_io=stac_api_io)

    query: dict[str, Any] = {
        "collections": [COLLECTION2_LANDSAT_L1],
        "datetime": f"{start_time.isoformat()}/{end_time.isoformat()}",
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "query": {
            "platform": {"in": ["LANDSAT_8", "LANDSAT_9"]},
            "view:sun_elevation": {"gte": 0},  # Filter out night images that have a negative sun_elevation
        },
    }

    if cloud_cover_range is not None:
        lower_limit, upper_limit = cloud_cover_range
        query["query"] = {"eo:cloud_cover": {"gte": lower_limit * 100, "lt": upper_limit * 100}}

    search = catalog.search(**query)
    items = list(search.items())

    if len(items) == 0:
        return []

    items = sorted(items, key=lambda item: item.properties["datetime"])

    # Ensure that they all have the same CRS
    crs_list = [item.properties["proj:code"] for item in items]
    if len(set(crs_list)) > 1:
        most_common_crs = Counter(crs_list).most_common(1)[0][0]
        logger.info(f"Multiple CRS found in items: {set(crs_list)}. Will use most common CRS {most_common_crs}.")
        items = [item for item in items if item.properties["proj:code"] == most_common_crs]

    return items


def get_aligned_cropped_landsat_band(
    reference_item: LandsatGranuleAccess,
    reference_meta: dict[str, Any],
    target_meta: dict[str, Any],
    target_crop_params: dict[str, Any],
    band: str,
    **kwargs: Any,
) -> npt.NDArray:
    """Get a cropped band from a reference Landsat image aligned to a target image's coordinates.

    The reference image is not always aligned with the target image.
    We need to calculate the crop coordinates for the reference image
    based on the target image crop coordinates and the target image transform.

    Args:
        reference_item: The reference Landsat granule to crop from
        reference_meta: Metadata dictionary for the reference image
        target_meta: Metadata dictionary for the target image to align to
        target_crop_params: Dictionary containing crop parameters (crop_start_x, crop_start_y, crop_height, crop_width)
        band: Name of the band to crop
        **kwargs: Additional keyword arguments passed to get_band_crop()

    Returns
    -------
        np.ndarray: Cropped band data aligned to target coordinates. Returns zeros if crop coordinates are invalid.
    """
    if reference_meta["crs"] != target_meta["crs"]:
        raise ValueError(
            f"CRS mismatch: {reference_meta['crs']} != {target_meta['crs']}. In "
            "load_cropped_landsat_items we expect to use the same WRS tile and therefore have the "
            "same CRS."
        )
    ref_crop_x, ref_crop_y = (~reference_meta["transform"]) * (
        target_meta["transform"] * (target_crop_params["crop_start_x"], target_crop_params["crop_start_y"])
    )
    # Handling edge cases:
    # 1. When transformed coordinates (ref_crop_x, ref_crop_y) are beyond reference bounds. Current testing
    #    shows that this gives RasterioIOError
    # 2. When crop extends beyond the target image but the reference coordinates are still within bounds.
    #    Current testing shows this gives a 0 array, which is good.
    # 3. When transformed coordinates (ref_crop_x, ref_crop_y) are negative.
    #    Current testing shows this gives a 0 array, which is good.
    try:
        cropped_band = reference_item.get_band_crop(
            band,
            ref_crop_x,
            ref_crop_y,
            target_crop_params["crop_height"],
            target_crop_params["crop_width"],
            **kwargs,
        )
    except rasterio.errors.RasterioIOError as e:
        if "Access window out of range in RasterIO()" in str(e):
            # Create array of zeros with same dimensions as requested crop
            cropped_band = np.zeros((1, target_crop_params["crop_height"], target_crop_params["crop_width"]))
        else:
            raise e
    return cropped_band


def load_cropped_landsat_items(
    items: list[LandsatGranuleAccess],
    bands: list[str],
    lat: float,
    lon: float,
    image_size: int,
    abs_client: BlobServiceClient,
    show_progress: bool = False,
    skip_returning_main_id: bool = False,
    item_meta_dict: dict = {},  # noqa
) -> list[dict[str, Any]]:
    """
    Load cropped Landsat items for a given list of bands, latitude, longitude, and image size.

    Ensure that the items are aligned to the target item (idx 0).
    """
    # Verify idx0 is most recent item or is the only item
    if len(items) > 1:
        target_time = items[0].time
        other_times = [item.time for item in items[1:]]
        assert all(
            target_time >= t for t in other_times
        ), f"Target item (index 0) must be the most recent. Target time: {target_time}, other times: {other_times}"

    cropped_data = []
    target_item = items[0]
    if len(items) > 1 and skip_returning_main_id:
        items = items[1:]
    item_iter = tqdm.tqdm(items) if show_progress else items

    # Get target metadata and crop parameters once
    if target_item.id in item_meta_dict:
        target_meta = item_meta_dict[target_item.id]
    else:
        target_meta = target_item.get_raster_meta(bands[0], abs_client=abs_client)
    target_crop_params = get_crop_params(
        lat,
        lon,
        image_size,
        (target_meta["width"], target_meta["height"]),
        target_item.crs,
        target_meta["transform"],
        out_res=30,
    )

    for item in item_iter:
        band_arrs = []
        crop_param_dict = {}

        # Get MTL data for radiometric conversion
        mtl_data = item.get_mtl_data(abs_client=abs_client)

        # Get the Landsat metadata from the coastal band (arbitrary choice). We can get away with this
        # as all Landsat bands that we use have the same resolution and spatial extent.
        if item.id in item_meta_dict:
            item_meta = item_meta_dict[item.id]
        else:
            item_meta = item.get_raster_meta("coastal", abs_client=abs_client)
        for band in bands:
            cropped_band = get_aligned_cropped_landsat_band(
                item, item_meta, target_meta, target_crop_params, band, abs_client=abs_client
            )
            cropped_band = convert_band_values(band, cropped_band, mtl_data, item.solar_angle)

            band_arrs.append(cropped_band)
            crop_param_dict[band] = target_crop_params

        cropped_data.append(
            {
                "crop_arrays": np.concatenate(band_arrs, axis=0),
                "crop_params": crop_param_dict,
                "tile_item": item,
            }
        )

    return cropped_data


def convert_band_values(
    band: str, band_data: npt.NDArray, mtl_data: LandsatImageMetadataFile, solar_angle: float
) -> npt.NDArray:
    """Convert band DN values to reflectance or brightness temperature."""
    if band == "qa_pixel":
        return band_data

    band_num = LANDSAT_STAC_ASSET_MAP[band].replace("B", "")
    rescaling = mtl_data.LANDSAT_METADATA_FILE.LEVEL1_RADIOMETRIC_RESCALING

    if band in ["lwir11", "lwir12"]:
        # Convert thermal bands
        thermal_constants = mtl_data.LANDSAT_METADATA_FILE.LEVEL1_THERMAL_CONSTANTS
        constants = {
            "lwir11": (
                rescaling.RADIANCE_MULT_BAND_10,
                rescaling.RADIANCE_ADD_BAND_10,
                thermal_constants.K1_CONSTANT_BAND_10,
                thermal_constants.K2_CONSTANT_BAND_10,
            ),
            "lwir12": (
                rescaling.RADIANCE_MULT_BAND_11,
                rescaling.RADIANCE_ADD_BAND_11,
                thermal_constants.K1_CONSTANT_BAND_11,
                thermal_constants.K2_CONSTANT_BAND_11,
            ),
        }
        mult, add, k1, k2 = constants[band]
        return LandsatGranuleAccess.convert_thermal_band_dn_to_brightness_values(band_data, mult, add, k1, k2)
    else:
        # Convert reflectance bands
        mult = getattr(rescaling, f"REFLECTANCE_MULT_BAND_{band_num}")
        add = getattr(rescaling, f"REFLECTANCE_ADD_BAND_{band_num}")
        return LandsatGranuleAccess.convert_band_dn_to_reflectance_values(band_data, solar_angle, mult, add)
