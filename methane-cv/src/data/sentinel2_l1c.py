"""PySTAC item wrapper class to get Sentinel L1C data from AWS."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, ClassVar

import azure.core.exceptions
import numpy as np
import numpy.typing as npt
import pystac
import rasterio
from azure.storage.blob import BlobServiceClient, ContainerClient
from mypy_boto3_s3 import S3Client
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry

from src.azure_wrap.ml_client_utils import download_blob_directly
from src.data.sentinel2 import (
    COPERNICUS_CATALOG_URL,
    COPERNICUS_COLLECTION_L1C,
    SceneClassificationLabel,
    Sentinel2Item,
)
from src.utils.utils import setup_logging

logger = setup_logging()


class Sentinel2L1CItem_Copernicus(Sentinel2Item):
    """
    PySTAC item wrapper that gets Sentinel 2 L1C data from S3 instead of L2A from ABS.

    This class contains all the functionality to deal with the fact that we want to use
    L1C data for training, but only L2A data is available on the Microsoft Planetary Computer.
    L1C data is available directly from S3, but egress fees are expensive.
    So what we want to do is download the needed files once, and save them on
    an Azure Blob Storage (ABS) container. Any subsequent data generation jobs that use
    the same tiles will go straight to ABS.
    """

    # a few fixed parameters that don't need to ever change
    l1c_s3_bucket = r"sentinel-s2-l1c"
    l1c_abs_bucket = r"l1c-data"
    l1c_files: ClassVar[dict[str, str]] = {
        "B01": "B01.jp2",
        "B02": "B02.jp2",
        "B03": "B03.jp2",
        "B04": "B04.jp2",
        "B05": "B05.jp2",
        "B06": "B06.jp2",
        "B07": "B07.jp2",
        "B08": "B08.jp2",
        "B09": "B09.jp2",
        "B10": "B10.jp2",
        "B11": "B11.jp2",
        "B12": "B12.jp2",
        "B8A": "B8A.jp2",
        # "granule_metadata": "metadata.xml"
    }

    def __init__(self, pystac_item: pystac.Item):
        self.item = pystac_item

    @classmethod
    def from_id(cls, item_id: str) -> Sentinel2L1CItem_Copernicus:
        """Retrieve Sentinel L2A item by its PySTAC ID."""
        retry = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None)
        stac_api_io = StacApiIO(max_retries=retry)

        catalog = Client.open(
            COPERNICUS_CATALOG_URL,
            # modifier=planetary_computer.sign_inplace,
            stac_io=stac_api_io,
        )
        search_result = list(catalog.search(collections=[COPERNICUS_COLLECTION_L1C], ids=[item_id]).items())

        try:
            item = search_result[0]
        except IndexError:
            raise ValueError(f"Sentinel 2 item not found with id: {item_id}") from None
        return cls(item)

    @property
    def imaging_mode(self) -> str:
        """Get the imaging mode."""
        return "L1C"

    @classmethod
    def from_l2a(cls, s2item: Sentinel2Item) -> Sentinel2L1CItem_Copernicus:
        """Convert Sentinel2Item from L2A to L1C."""
        print(f"****** {type(s2item.item.id)}***** \n")
        return cls.from_id(s2item.item.id)

    def load_metadata(self) -> None:
        """No need to load metadata, it's already in the item properties."""
        # prefix = self.l1c_prefix()
        # filename = self.l1c_files["granule_metadata"]
        # blob_name = f"{prefix}/{filename}"
        # with tempfile.TemporaryDirectory() as temp_dir:
        #     temp_path = Path(temp_dir) / filename
        #     download_blob_directly(
        #         blob_name=blob_name,
        #         local_download_filepath=temp_path,
        #         blob_service_client=abs_client, # may need to be passed in
        #         container_name=self.l1c_abs_bucket,
        #     )

        #     with open(temp_path, "r") as f:
        #         self._granule_metadata = f.read()
        pass

    @property
    def solar_angle(self) -> float:
        """Mean solar zenith angle."""
        return 90.0 - self.item.properties["view:sun_elevation"]

    @property
    def solar_azimuth(self) -> float:
        """Mean solar azimuth angle."""
        return self.item.properties["view:sun_azimuth"]

    @property
    def observation_angle(self) -> float:
        """Mean observation zenith angle."""
        return self.item.properties["view:incidence_angle"]

    @property
    def viewing_azimuth(self) -> float:
        """Mean viewing azimuth angle."""
        return self.item.properties["view:azimuth"]

    @property
    def crs(self) -> str:
        """CRS of the item."""
        return self.item.assets["B01"].extra_fields["proj:code"]

    @property
    def instrument(self) -> str:
        """Instrument id."""
        return self.instrument_name[-1]

    def get_transform(self, band: str) -> rasterio.Affine:
        """Get the transform for a band."""
        return rasterio.Affine(*self.item.assets[band].extra_fields["proj:transform"])

    def get_shape(self, band: str) -> tuple[int, int]:
        """Get the shape for a band."""
        return tuple(self.item.assets[band].extra_fields["proj:shape"])

    def get_bounds(self, band: str) -> rasterio.coords.BoundingBox:
        """Get the bounds for a band."""
        return rasterio.coords.BoundingBox(*self.item.assets[band].extra_fields["proj:bbox"])

    def get_raster_meta(self, band: str) -> dict:
        """Get raster metadata for a band."""
        return {"transform": self.get_transform(band), "bounds": self.get_bounds(band), "shape": self.get_shape(band)}

    @property
    def instrument_name(self) -> str:
        """Instrument name."""
        # The planetary computer has the platform name in the format "Sentinel-2A"
        # but the L1C data has the platform name in the format "sentinel-2a"
        return self.item.properties["platform"].title()

    def l1c_prefix(self) -> str:
        """
        Obtain the URI prefix for the directory on S3 that contains the same scene as this item, but L1C instead of L2A.

        e.g. tiles/13/U/GQ/2021/1/3/0.
        """
        b01_href = self.item.assets[
            "B01"
        ].href  # e.g. 's3://eodata/Sentinel-2/MSI/L1C/2025/01/23/S2C_MSIL1C_20250123T180701_N0511_R041_T12SVB_20250123T194708.SAFE/GRANULE/L1C_T12SVB_A002013_20250123T180943/IMG_DATA/T12SVB_20250123T180701_B01.jp2'

        # Extract the tile ID (e.g., "12SVB") from the href
        tile_match = re.search(r"_T([A-Z0-9]{5})_", b01_href)
        if not tile_match:
            raise ValueError(f"Could not extract tile ID from href: {b01_href}")
        tile_id = tile_match.group(1)

        # Extract the date (e.g., "2025/01/23") from the href
        date_match = re.search(r"/L1C[^/]*/(\d{4})/(\d{2})/(\d{2})/", b01_href)
        if not date_match:
            raise ValueError(f"Could not extract date from href: {b01_href}")
        year = date_match.group(1)
        month = date_match.group(2).lstrip("0")
        day = date_match.group(3).lstrip("0")
        tile_id_parts = tile_id[:2], tile_id[2:3], tile_id[3:]
        prefix_parts = [*tile_id_parts, year, month, day]

        if self.item.id == "S2A_MSIL1C_20250329T181751_N0511_R084_T12SVB_20250330T010410":
            print(f"HACK: adding /1/ to the prefix for item: {self.item.id}")
            prefix_parts = ["tiles", *prefix_parts, "1"]
            print(f"{prefix_parts=}")
        else:
            prefix_parts = ["tiles", *prefix_parts, "0"]

        return "/".join(prefix_parts)

    def prefetch_l1c(self, s3_client: S3Client, abs_client: BlobServiceClient) -> None:
        """Check if the L1C data is already available on ABS. If yes, do nothing. If no, upload it."""
        # A blob client is a client for the overall Azure Blob Storage service.
        # When we zoom in on a particular container ("bucket" in AWS parlance) we get a ContainerClient.
        container_client = abs_client.get_container_client(self.l1c_abs_bucket)
        # the container must exist
        assert container_client.exists()
        self.transfer_l1c_to_abs(s3_client, container_client)

    def return_files_on_abs(self, container_client: ContainerClient) -> list[str]:
        """Which files are already on ABS?."""
        l1c_prefix = self.l1c_prefix()
        list_blobs = [k.name for k in list(container_client.list_blobs(name_starts_with=l1c_prefix))]
        return list_blobs

    def transfer_l1c_to_abs(self, s3_client: S3Client, container_client: ContainerClient) -> None:
        """
        Transfer L1C data from the S3 bucket to our ABS container.

        We transfer the JPEG2000 file for each band, under the same prefix and file name.
        """
        prefix = self.l1c_prefix()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Recreate the directory structure
            os.makedirs(temp_path / prefix)
            files_existing = [k.split("/")[-1] for k in self.return_files_on_abs(container_client)]
            for filename in self.l1c_files.values():
                if filename in files_existing:
                    continue
                if filename.endswith("MTD_TL.xml"):
                    # Skip the metadata file
                    continue
                source_file = f"{prefix}/{filename}"
                logger.info(f"Transferring {source_file} to {self.l1c_abs_bucket}")
                temp_file = temp_path / source_file

                # Download the file from the source bucket to a local file
                s3_client.download_file(
                    Bucket=self.l1c_s3_bucket,
                    Key=source_file,
                    Filename=str(temp_file),
                    ExtraArgs={"RequestPayer": "requester"},
                )

                # Upload the local file to Azure Blob Storage
                with open(temp_file, "rb") as data:
                    container_client.upload_blob(
                        name=source_file,
                        data=data.read(),
                        validate_content=True,
                        overwrite=True,
                    )

                logger.info(f"Successfully transferred {source_file} to {self.l1c_abs_bucket}")

    def check_omnicloud_on_abs(self, abs_client: BlobServiceClient) -> bool:
        """Does this ID already have OmniCloud data on ABS?"""  # noqa
        container_client = abs_client.get_container_client(self.l1c_abs_bucket)
        l1c_prefix = self.l1c_prefix()
        cloud_blobs = list(container_client.list_blobs(name_starts_with=l1c_prefix + "/OmniCloud.tif"))
        return len(cloud_blobs) > 0

    def transfer_omnicloud_to_abs(self, omni_local_path: str, abs_client: BlobServiceClient) -> None:
        """Upload OmniCloud thick clouds/thin clouds/cloud shadows prediction to our ABS container."""
        container_client = abs_client.get_container_client(self.l1c_abs_bucket)
        blob_name = self.l1c_prefix() + "/OmniCloud.tif"
        with open(omni_local_path, "rb") as data:
            container_client.upload_blob(
                name=blob_name,
                data=data.read(),
                validate_content=True,
                overwrite=True,
            )
        logger.info(f"Successfully transferred OmniCloud probs to {self.l1c_abs_bucket}/{blob_name}")

    def get_omnicloud(
        self,
        out_height: int | None = None,
        out_width: int | None = None,
        abs_client: BlobServiceClient | None = None,
    ) -> npt.NDArray:
        """
        Obtain a particular Sentinel 2 band as a numpy array.

        This function assumes that the L1C files have already been transferred from S3 to ABS.
        Otherwise, prefetch_l1c needs to be called first, which makes the data available.

        Parameters
        ----------
        out_height, out_width: integers
            the desired image size
            Note: GDAL will automatically resize if this is different from the original.
        """
        prefix = self.l1c_prefix()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            download_blob_directly(
                blob_name=f"{prefix}/OmniCloud.tif",
                local_download_filepath=temp_path / "tmp.tif",
                blob_service_client=abs_client,
                container_name=self.l1c_abs_bucket,
            )
            with rasterio.open(temp_path / "tmp.tif") as ds:
                if out_height is None:
                    out_height = ds.height
                if out_width is None:
                    out_width = ds.width
                probs = ds.read(
                    out_shape=(
                        4,
                        out_height,
                        out_width,
                    )
                )
        return probs

    def get_raster_as_tmp(self, band: str, abs_client: BlobServiceClient) -> None:
        """Download band as tmp.tif."""
        prefix = self.l1c_prefix()
        suffix = "jp2"
        blob_name = f"{prefix}/{band}.{suffix}"
        download_blob_directly(
            blob_name=blob_name,
            local_download_filepath=Path(f"tmp.{suffix}"),
            blob_service_client=abs_client,
            container_name=self.l1c_abs_bucket,
        )

    def _download_and_read_band(
        self,
        band: str,
        prefix: str,
        abs_client: BlobServiceClient,
        *,
        window: rasterio.windows.Window | None = None,
        out_height: int | None = None,
        out_width: int | None = None,
    ) -> npt.NDArray:
        """Download and read a band from Azure Blob Storage.

        Parameters
        ----------
        band : str
            The band to download (e.g. "B01")
        prefix : str
            The S3/Azure prefix path
        abs_client : BlobServiceClient
            Azure Blob Storage client
        window : rasterio.windows.Window, optional
            Window for cropping, by default None
        out_height : int, optional
            Desired output height, by default None
        out_width : int, optional
            Desired output width, by default None

        Returns
        -------
        npt.NDArray
            The band data as a numpy array
        """
        if band == "SCL":
            return np.ones((1, out_height, out_width), dtype=np.uint8) * SceneClassificationLabel.UNCLASSIFIED.value

        with tempfile.TemporaryDirectory() as temp_dir:
            suffix = "jp2" if band != "OmniCloud" else "tif"
            temp_path = Path(temp_dir) / f"tmp.{suffix}"
            blob_name = f"{prefix}/{band}.{suffix}" if band != "OmniCloud" else f"{prefix}/{band}.{suffix}"
            download_blob_directly(
                blob_name=blob_name,
                local_download_filepath=temp_path,
                blob_service_client=abs_client,
                container_name=self.l1c_abs_bucket,
            )

            with rasterio.open(temp_path) as ds:
                if out_height is None:
                    out_height = ds.height if window is None else window.height
                if out_width is None:
                    out_width = ds.width if window is None else window.width

                nb_bands = 1 if band != "OmniCloud" else 4
                band_data = ds.read(
                    out_shape=(nb_bands, out_height, out_width),
                    window=window,
                )

        return band_data

    def get_band(
        self,
        band: str,
        out_height: int | None = None,
        out_width: int | None = None,
        harmonize_if_needed: bool = True,
        **kwargs: Any,
    ) -> npt.NDArray:
        """
        Obtain a particular Sentinel 2 band as a numpy array.

        This function assumes that the L1C files have already been transferred from S3 to ABS.
        Otherwise, prefetch_l1c needs to be called first, which makes the data available.

        Parameters
        ----------
        band: str
            name of the band, like "B01"
        out_height, out_width: integers
            the desired image size
            Note: GDAL will automatically resize if this is different from the original.
        harmonize_if_needed: bool
            Remove the fixed offset and clip to zero from post-2022 data
            so it is harmonized to pre-2022 data.
        **kwargs: Any
            Additional keyword arguments. Expected:
            - abs_client: BlobServiceClient
                A client for the Azure Blob Service. This is used for downloading the files
                before reading them with rasterio.
        """
        abs_client = kwargs.get("abs_client")
        if abs_client is None:
            raise ValueError("abs_client is required")

        if band == "SCL":
            return np.ones((1, out_height, out_width), dtype=np.uint8) * SceneClassificationLabel.UNCLASSIFIED.value
            # # for the cloud mask, use L2A
            # return super().get_band(band, out_height, out_width, harmonize_if_needed)

        # otherwise, find the file on ABS
        prefix = self.l1c_prefix()

        band_data = self._download_and_read_band(
            band=band,
            prefix=prefix,
            abs_client=abs_client,
            out_height=out_height,
            out_width=out_width,
        )

        if harmonize_if_needed:
            band_data = self.harmonize_to_old(band, band_data)
        return band_data

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
        """
        Obtain a particular crop of a Sentinel 2 band as a numpy array.

        This function assumes that the L1C files have already been transferred from S3 to ABS.
        Otherwise, prefetch_l1c needs to be called first, which makes the data available.

        Parameters
        ----------
        band: str
            name of the band, like "B01"
        crop_start_x, crop_start_y: int
            starting coordinates for cropping
        crop_height, crop_width: int
            dimensions of the crop
        out_height, out_width: integers
            the desired output size
        harmonize_if_needed: bool
            Remove the fixed offset and clip to zero from post-2022 data
            so it is harmonized to pre-2022 data.
        abs_client: BlobServiceClient
            A client for the Azure Blob Service. This is used for downloading the files
            before reading them with rasterio.
        """
        abs_client = kwargs.get("abs_client")
        if abs_client is None:
            raise ValueError("abs_client is required")

        if band == "SCL":
            # for the cloud mask, use L2A
            return np.ones((1, out_height, out_width), dtype=np.uint8) * SceneClassificationLabel.UNCLASSIFIED.value
            # return super().get_band_crop(
            #     band,
            #     crop_start_x,
            #     crop_start_y,
            #     crop_height,
            #     crop_width,
            #     out_height,
            #     out_width,
            #     harmonize_if_needed,
            # )

        # Create cache path that includes crop parameters
        cache_blob_name = (
            f"cached_crops/{self.l1c_prefix()}/{band}_{crop_start_x}_{crop_start_y}"
            f"_{crop_height}_{crop_width}_{out_height}_{out_width}.npy"
        )

        # find the file on ABS
        prefix = self.l1c_prefix()
        # Check if cropped band exists in cache
        container_client = abs_client.get_container_client(self.l1c_abs_bucket)

        try:
            # Try to download from cache
            crop = self._get_crop_from_cache(container_client, cache_blob_name)
        except azure.core.exceptions.ResourceNotFoundError:
            logger.info(f"Crop is not saved on ABS. {cache_blob_name} --> Creating and saving it.")
            window = rasterio.windows.Window(
                crop_start_x,
                crop_start_y,
                crop_width,
                crop_height,
            )
            out_height = out_height if out_height is not None else crop_height
            out_width = out_width if out_width is not None else crop_width

            # Get the crop using existing method
            crop = self._download_and_read_band(
                band=band,
                prefix=prefix,
                abs_client=abs_client,
                window=window,
                out_height=out_height,
                out_width=out_width,
            )

            # Cache the crop
            self._cache_crop_to_abs(crop, cache_blob_name, container_client)

        if harmonize_if_needed:
            crop = self.harmonize_to_old(band, crop)
        return crop

    def _get_crop_from_cache(self, container_client: ContainerClient, blob_name: str) -> npt.NDArray:
        """Download and load a cached crop from ABS."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "temp.npy"

            # Download blob to temporary file
            with open(temp_path, "wb") as temp_file:
                blob_data = container_client.download_blob(blob_name).readall()
                temp_file.write(blob_data)
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


class Sentinel2L1CItem(Sentinel2Item):
    """
    PySTAC item wrapper that gets Sentinel 2 L1C data from S3 instead of L2A from ABS.

    This class contains all the functionality to deal with the fact that we want to use
    L1C data for training, but only L2A data is available on the Microsoft Planetary Computer.
    L1C data is available directly from S3, but egress fees are expensive.
    So what we want to do is download the needed files once, and save them on
    an Azure Blob Storage (ABS) container. Any subsequent data generation jobs that use
    the same tiles will go straight to ABS.
    """

    # regular expression to extract the path to the L2A tiles,
    # which we are going to use to find the L1C tiles on S3.
    # They follow the same convention, except on S3 the month and day
    # of the overpass do not have starting zeros.
    re_directory = re.compile(
        r"^https://sentinel2l2a01\.blob\.core\.windows\.net/sentinel2-l2/"
        r"([\d]+)/([A-Z])/([A-Z][A-Z])/([\d]+)/0?([1-9]\d?)/0?([1-9]\d?)"
        r"/S2.*"
    )
    # a few fixed parameters that don't need to ever change
    l1c_s3_bucket = r"sentinel-s2-l1c"
    l1c_abs_bucket = r"l1c-data"
    l1c_bands: ClassVar[list[str]] = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B09",
        "B10",
        "B11",
        "B12",
        "B8A",
    ]

    @property
    def imaging_mode(self) -> str:
        """Get the imaging mode."""
        return "L1C"

    @classmethod
    def from_l2a(cls, s2item: Sentinel2Item) -> Sentinel2L1CItem:
        """Convert Sentinel2Item from L2A to L1C."""
        return cls(s2item.item)

    def l1c_prefix(self) -> str:
        """
        Obtain the URI prefix for the directory on S3 that contains the same scene as this item, but L1C instead of L2A.

        e.g. tiles/13/U/GQ/2021/1/3/0.
        """
        scl_href = self.item.assets["SCL"].href
        # use regular expression to extract the subdirectory for a scene,
        # e.g. 13/U/GQ/2021/1/3/
        match = self.re_directory.fullmatch(scl_href)
        if not match:
            raise ValueError(f"Regular expression did not match. Invalid L2A path format: {scl_href}")
        prefix_parts = list(match.groups())
        # Convert the first element of prefix_parts to an integer and then back to a string to remove any leading zeros
        prefix_parts[0] = str(int(prefix_parts[0]))
        # We add the tiles/ prefix here.
        # After the date, there is a /0/. If multiple processing baselines exist,
        # there can also be a /1/. We don't want to deal with this complexity,
        # so we will just add a /0/.
        # TODO: this is a hack to get the S2 SBRs working. Update with better logic.
        if self.item.id == "S2A_MSIL2A_20250329T181751_R084_T12SVB_20250330T030313":
            print(f"HACK: adding /1/ to the prefix for item: {self.item.id}")
            prefix_parts = ["tiles", *prefix_parts, "1"]
            print(f"{prefix_parts=}")
        else:
            prefix_parts = ["tiles", *prefix_parts, "0"]
        prefix = "/".join(prefix_parts)
        return prefix

    def prefetch_l1c(self, s3_client: S3Client, abs_client: BlobServiceClient) -> None:
        """Check if the L1C data is already available on ABS. If yes, do nothing. If no, upload it."""
        # A blob client is a client for the overall Azure Blob Storage service.
        # When we zoom in on a particular container ("bucket" in AWS parlance) we get a ContainerClient.
        container_client = abs_client.get_container_client(self.l1c_abs_bucket)
        # the container must exist
        assert container_client.exists()
        self.transfer_l1c_to_abs(s3_client, container_client)

    def return_bands_on_abs(self, container_client: ContainerClient) -> list[str]:
        """Which bands are already on ABS?."""
        l1c_prefix = self.l1c_prefix()
        list_blobs = [k.name for k in list(container_client.list_blobs(name_starts_with=l1c_prefix + "/B"))]
        return list_blobs

    def transfer_l1c_to_abs(self, s3_client: S3Client, container_client: ContainerClient) -> None:
        """
        Transfer L1C data from the S3 bucket to our ABS container.

        We transfer the JPEG2000 file for each band, under the same prefix and file name.
        """
        prefix = self.l1c_prefix()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Recreate the directory structure
            os.makedirs(temp_path / prefix)
            bands_existing = [k.split("/")[-1] for k in self.return_bands_on_abs(container_client)]
            for band in self.l1c_bands:
                if f"{band}.jp2" in bands_existing:
                    continue
                source_file = f"{prefix}/{band}.jp2"
                temp_file = temp_path / source_file

                # Download the file from the source bucket to a local file
                s3_client.download_file(
                    Bucket=self.l1c_s3_bucket,
                    Key=source_file,
                    Filename=str(temp_file),
                    ExtraArgs={"RequestPayer": "requester"},
                )

                # Upload the local file to Azure Blob Storage
                with open(temp_file, "rb") as data:
                    container_client.upload_blob(
                        name=source_file,
                        data=data.read(),
                        validate_content=True,
                        overwrite=True,
                    )

                logger.info(f"Successfully transferred {source_file} to {self.l1c_abs_bucket}")

    def check_omnicloud_on_abs(self, abs_client: BlobServiceClient) -> bool:
        """Does this ID already have OmniCloud data on ABS?"""  # noqa
        container_client = abs_client.get_container_client(self.l1c_abs_bucket)
        l1c_prefix = self.l1c_prefix()
        cloud_blobs = list(container_client.list_blobs(name_starts_with=l1c_prefix + "/OmniCloud.tif"))
        return len(cloud_blobs) > 0

    def transfer_omnicloud_to_abs(self, omni_local_path: str, abs_client: BlobServiceClient) -> None:
        """Upload OmniCloud thick clouds/thin clouds/cloud shadows prediction to our ABS container."""
        container_client = abs_client.get_container_client(self.l1c_abs_bucket)
        blob_name = self.l1c_prefix() + "/OmniCloud.tif"
        with open(omni_local_path, "rb") as data:
            container_client.upload_blob(
                name=blob_name,
                data=data.read(),
                validate_content=True,
                overwrite=True,
            )
        logger.info(f"Successfully transferred OmniCloud probs to {self.l1c_abs_bucket}/{blob_name}")

    def get_omnicloud(
        self,
        out_height: int | None = None,
        out_width: int | None = None,
        abs_client: BlobServiceClient | None = None,
    ) -> npt.NDArray:
        """
        Obtain a particular Sentinel 2 band as a numpy array.

        This function assumes that the L1C files have already been transferred from S3 to ABS.
        Otherwise, prefetch_l1c needs to be called first, which makes the data available.

        Parameters
        ----------
        out_height, out_width: integers
            the desired image size
            Note: GDAL will automatically resize if this is different from the original.
        """
        prefix = self.l1c_prefix()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            download_blob_directly(
                blob_name=f"{prefix}/OmniCloud.tif",
                local_download_filepath=temp_path / "tmp.tif",
                blob_service_client=abs_client,
                container_name=self.l1c_abs_bucket,
            )
            with rasterio.open(temp_path / "tmp.tif") as ds:
                if out_height is None:
                    out_height = ds.height
                if out_width is None:
                    out_width = ds.width
                probs = ds.read(
                    out_shape=(
                        4,
                        out_height,
                        out_width,
                    )
                )
        return probs

    def get_raster_as_tmp(self, band: str, abs_client: BlobServiceClient) -> None:
        """Download band as tmp.tif."""
        prefix = self.l1c_prefix()
        suffix = "jp2"
        blob_name = f"{prefix}/{band}.{suffix}"
        download_blob_directly(
            blob_name=blob_name,
            local_download_filepath=Path(f"tmp.{suffix}"),
            blob_service_client=abs_client,
            container_name=self.l1c_abs_bucket,
        )

    def _download_and_read_band(
        self,
        band: str,
        prefix: str,
        abs_client: BlobServiceClient,
        *,
        window: rasterio.windows.Window | None = None,
        out_height: int | None = None,
        out_width: int | None = None,
    ) -> npt.NDArray:
        """Download and read a band from Azure Blob Storage.

        Parameters
        ----------
        band : str
            The band to download (e.g. "B01")
        prefix : str
            The S3/Azure prefix path
        abs_client : BlobServiceClient
            Azure Blob Storage client
        window : rasterio.windows.Window, optional
            Window for cropping, by default None
        out_height : int, optional
            Desired output height, by default None
        out_width : int, optional
            Desired output width, by default None

        Returns
        -------
        npt.NDArray
            The band data as a numpy array
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            suffix = "jp2" if band != "OmniCloud" else "tif"
            temp_path = Path(temp_dir) / f"tmp.{suffix}"
            blob_name = f"{prefix}/{band}.{suffix}" if band != "OmniCloud" else f"{prefix}/{band}.{suffix}"
            download_blob_directly(
                blob_name=blob_name,
                local_download_filepath=temp_path,
                blob_service_client=abs_client,
                container_name=self.l1c_abs_bucket,
            )

            with rasterio.open(temp_path) as ds:
                if out_height is None:
                    out_height = ds.height if window is None else window.height
                if out_width is None:
                    out_width = ds.width if window is None else window.width

                nb_bands = 1 if band != "OmniCloud" else 4
                band_data = ds.read(
                    out_shape=(nb_bands, out_height, out_width),
                    window=window,
                )

        return band_data

    def get_band(
        self,
        band: str,
        out_height: int | None = None,
        out_width: int | None = None,
        harmonize_if_needed: bool = True,
        **kwargs: Any,
    ) -> npt.NDArray:
        """
        Obtain a particular Sentinel 2 band as a numpy array.

        This function assumes that the L1C files have already been transferred from S3 to ABS.
        Otherwise, prefetch_l1c needs to be called first, which makes the data available.

        Parameters
        ----------
        band: str
            name of the band, like "B01"
        out_height, out_width: integers
            the desired image size
            Note: GDAL will automatically resize if this is different from the original.
        harmonize_if_needed: bool
            Remove the fixed offset and clip to zero from post-2022 data
            so it is harmonized to pre-2022 data.
        **kwargs: Any
            Additional keyword arguments. Expected:
            - abs_client: BlobServiceClient
                A client for the Azure Blob Service. This is used for downloading the files
                before reading them with rasterio.
        """
        abs_client = kwargs.get("abs_client")
        if abs_client is None:
            raise ValueError("abs_client is required")

        if band == "SCL":
            # for the cloud mask, use L2A (hosted on ABS, not S3)
            return super().get_band(band, out_height, out_width, harmonize_if_needed)

        # otherwise, find the file on ABS
        prefix = self.l1c_prefix()

        band_data = self._download_and_read_band(
            band=band,
            prefix=prefix,
            abs_client=abs_client,
            out_height=out_height,
            out_width=out_width,
        )

        if harmonize_if_needed:
            band_data = self.harmonize_to_old(band, band_data)
        return band_data

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
        """
        Obtain a particular crop of a Sentinel 2 band as a numpy array.

        This function assumes that the L1C files have already been transferred from S3 to ABS.
        Otherwise, prefetch_l1c needs to be called first, which makes the data available.

        Parameters
        ----------
        band: str
            name of the band, like "B01"
        crop_start_x, crop_start_y: int
            starting coordinates for cropping
        crop_height, crop_width: int
            dimensions of the crop
        out_height, out_width: integers
            the desired output size
        harmonize_if_needed: bool
            Remove the fixed offset and clip to zero from post-2022 data
            so it is harmonized to pre-2022 data.
        abs_client: BlobServiceClient
            A client for the Azure Blob Service. This is used for downloading the files
            before reading them with rasterio.
        """
        abs_client = kwargs.get("abs_client")
        if abs_client is None:
            raise ValueError("abs_client is required")

        if band == "SCL":
            # for the cloud mask, use L2A
            return super().get_band_crop(
                band,
                crop_start_x,
                crop_start_y,
                crop_height,
                crop_width,
                out_height,
                out_width,
                harmonize_if_needed,
            )

        # Create cache path that includes crop parameters
        cache_blob_name = (
            f"cached_crops/{self.l1c_prefix()}/{band}_{crop_start_x}_{crop_start_y}"
            f"_{crop_height}_{crop_width}_{out_height}_{out_width}.npy"
        )

        # find the file on ABS
        prefix = self.l1c_prefix()
        # Check if cropped band exists in cache
        container_client = abs_client.get_container_client(self.l1c_abs_bucket)

        try:
            # Try to download from cache
            crop = self._get_crop_from_cache(container_client, cache_blob_name)
        except azure.core.exceptions.ResourceNotFoundError:
            logger.info(f"Crop is not saved on ABS. {cache_blob_name} --> Creating and saving it.")
            window = rasterio.windows.Window(
                crop_start_x,
                crop_start_y,
                crop_width,
                crop_height,
            )
            out_height = out_height if out_height is not None else crop_height
            out_width = out_width if out_width is not None else crop_width

            # Get the crop using existing method
            crop = self._download_and_read_band(
                band=band,
                prefix=prefix,
                abs_client=abs_client,
                window=window,
                out_height=out_height,
                out_width=out_width,
            )

            # Cache the crop
            self._cache_crop_to_abs(crop, cache_blob_name, container_client)

        if harmonize_if_needed:
            crop = self.harmonize_to_old(band, crop)
        return crop

    def _get_crop_from_cache(self, container_client: ContainerClient, blob_name: str) -> npt.NDArray:
        """Download and load a cached crop from ABS."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "temp.npy"

            # Download blob to temporary file
            with open(temp_path, "wb") as temp_file:
                blob_data = container_client.download_blob(blob_name).readall()
                temp_file.write(blob_data)
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
