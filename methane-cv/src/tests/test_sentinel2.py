"""Test the Sentinel-2 data classes."""

import datetime
from pathlib import Path

import numpy as np
import pystac
import pytest
from azure.ai.ml import MLClient
from azure.storage.blob import BlobServiceClient
from matplotlib import pyplot as plt
from mypy_boto3_s3 import S3Client
from PIL import Image

from src.azure_wrap.ml_client_utils import initialize_blob_service_client
from src.data.sentinel2 import Sentinel2Item, query_sentinel2_catalog_for_tile
from src.data.sentinel2_l1c import Sentinel2L1CItem
from src.utils.utils import initialize_s3_client

#######################
### SETUP FUNCTIONS ###
#######################


@pytest.fixture(scope="module")
def local_out_dir_sentinel2(local_out_dir: Path) -> Path:
    """Local directory to store the sentinel2 outputs."""
    out_dir = local_out_dir / "test_sentinel2"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture(scope="module")
def s3_client(ml_client: MLClient) -> S3Client:
    """Initialize and return an S3 client."""
    return initialize_s3_client(ml_client)


@pytest.fixture(scope="module")
def abs_client(ml_client: MLClient) -> BlobServiceClient:
    """Initialize and return a Blob Service client."""
    return initialize_blob_service_client(ml_client)


######################
### TEST CLASSES ###
######################


class BaseSentinel2Test:
    """Base class for Sentinel-2 tests containing shared test logic."""

    item_class: Sentinel2Item | Sentinel2L1CItem
    level: str
    tile_id: str

    def test_init(self) -> None:
        """Test initialization of Sentinel-2 item."""
        item = self.item_class.from_id(self.tile_id)
        assert item.id == self.tile_id
        assert item.instrument == "B"
        assert np.isclose(item.observation_angle, 5.97847887319446)
        assert np.isclose(item.solar_angle, 56.8885029762379)
        assert item.crs == "EPSG:32632"
        assert item.time == datetime.datetime(2022, 2, 28, 10, 28, 49, 24000, tzinfo=datetime.timezone.utc)

    def prepare_item(self, item: Sentinel2Item, s3_client: S3Client, abs_client: BlobServiceClient) -> Sentinel2Item:
        """Provide a hook for subclasses to prepare item before use."""
        return item

    def test_get_band(self, local_out_dir_sentinel2: Path, s3_client: S3Client, abs_client: BlobServiceClient) -> None:
        """Test getting a single band."""
        item = self.item_class.from_id(self.tile_id)
        item = self.prepare_item(item, s3_client, abs_client)
        band11 = item.get_band("B11", out_height=400, out_width=400, abs_client=abs_client)
        img_arr = (band11.squeeze() * (255 / 5000)).astype(np.uint8)
        img = Image.fromarray(img_arr)
        img.save(str(local_out_dir_sentinel2 / f"test_get_band_11_PIL_{self.level}.png"))

    def test_get_bands(self, s3_client: S3Client, abs_client: BlobServiceClient) -> None:
        """Test getting multiple bands."""
        item = self.item_class.from_id(self.tile_id)
        item = self.prepare_item(item, s3_client, abs_client)
        arr = item.get_bands(["B11", "B12"], out_height=100, out_width=100, abs_client=abs_client)
        assert arr.shape == (2, 100, 100)

    def test_harmonization(
        self, local_out_dir_sentinel2: Path, s3_client: S3Client, abs_client: BlobServiceClient
    ) -> None:
        """Test harmonization by comparing harmonized and original data."""
        item = self.item_class.from_id("S2A_MSIL2A_20230116T021341_R060_T50KPC_20230116T112621")
        item = self.prepare_item(item, s3_client, abs_client)
        arr = item.get_bands(bands=["B04", "B03", "B02"], out_height=500, out_width=500, abs_client=abs_client)
        img_arr = arr.transpose((1, 2, 0)) / 10000
        plt.imsave(str(local_out_dir_sentinel2 / f"test_harmonization_{self.level}.png"), img_arr)

    def test_harmonization_histograms(
        self, local_out_dir_sentinel2: Path, s3_client: S3Client, abs_client: BlobServiceClient
    ) -> None:
        """Test harmonization by comparing histograms of harmonized and original data."""
        item = self.item_class.from_id("S2A_MSIL2A_20230116T021341_R060_T50KPC_20230116T112621")
        item = self.prepare_item(item, s3_client, abs_client)

        # Get harmonized band
        band11_crop = item.get_band_crop("B11", 0, 0, 256, 256, harmonize_if_needed=True, abs_client=abs_client)
        counts, bins = np.histogram(band11_crop, bins=100)
        plt.figure()
        plt.stairs(counts, bins)
        plt.savefig(local_out_dir_sentinel2 / f"test_get_band_11_hist_{self.level}.png")
        plt.close()

        # Get original band
        orig_swir16_crop = item.get_band_crop("B11", 0, 0, 256, 256, harmonize_if_needed=False, abs_client=abs_client)
        counts, bins = np.histogram(orig_swir16_crop, bins=100)
        plt.figure()
        plt.stairs(counts, bins)
        plt.savefig(local_out_dir_sentinel2 / f"test_get_band_11_hist_orig_{self.level}.png")
        plt.close()

    def test_get_band_crop(
        self, local_out_dir_sentinel2: Path, s3_client: S3Client, abs_client: BlobServiceClient
    ) -> None:
        """Test cropping a band."""
        item = self.item_class.from_id(self.tile_id)
        item = self.prepare_item(item, s3_client, abs_client)
        band11 = item.get_band_crop("B11", 100, 300, 4000, 2000, 400, 200, abs_client=abs_client)
        assert band11.shape == (1, 400, 200)
        img_arr = (band11.squeeze() * (255 / 5000)).astype(np.uint8)
        img = Image.fromarray(img_arr)
        img.save(str(local_out_dir_sentinel2 / f"test_get_band_crop_{self.level}.png"))


class TestSentinel2L2A(BaseSentinel2Test):
    """Tests for Sentinel-2 L2A data."""

    item_class = Sentinel2Item
    level = "L2A"
    tile_id = "S2B_MSIL2A_20220228T102849_R108_T32TMT_20220303T082201"


class TestSentinel2L1C(BaseSentinel2Test):
    """Tests for Sentinel-2 L1C data."""

    item_class = Sentinel2L1CItem
    level = "L1C"
    tile_id = "S2B_MSIL2A_20220228T102849_R108_T32TMT_20220303T082201"

    def prepare_item(
        self, item: Sentinel2L1CItem, s3_client: S3Client | None, abs_client: BlobServiceClient | None
    ) -> Sentinel2L1CItem:
        """Prefetch L1C data before use."""
        if s3_client and abs_client:
            item.prefetch_l1c(s3_client, abs_client)
        return item


def test_search_by_tile_id() -> None:
    tile_id = "06VVN"
    start_time = datetime.datetime.fromisoformat("2019-06-01")
    end_time = datetime.datetime.fromisoformat("2019-07-01")
    search_result = query_sentinel2_catalog_for_tile(tile_id, start_time, end_time)
    assert len(search_result) == 23  # noqa: PLR2004 (magic number)
    assert isinstance(search_result[0], pystac.Item)

    search_result_2 = query_sentinel2_catalog_for_tile(tile_id, start_time, end_time, (0.0, 0.3))
    assert len(search_result_2) == 9  # noqa: PLR2004 (magic number)

    start_time_alt = datetime.datetime.fromisoformat("2020-01-01")
    end_time_alt = datetime.datetime.fromisoformat("2021-01-01")
    search_result_3 = query_sentinel2_catalog_for_tile(tile_id, start_time_alt, end_time_alt)
    assert len(search_result_3) > 1
