"""Landsat data generation pipeline."""

import datetime
import itertools
import json
import math
import time
import traceback
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
from azure.storage.blob import BlobServiceClient
from matplotlib.axes import Axes
from mypy_boto3_s3 import S3Client
from pydantic import BaseModel
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from shapely import ops
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
from skimage.morphology import dilation, square

from src.data.common.data_item import BasePlumesDataItem, MultiTemporalPlumesDataItem
from src.data.common.sim_plumes import PlumeType
from src.data.generation.base import BaseDataGeneration, DataGenerationConfig
from src.data.landsat_data import (
    LANDSAT_STAC_ASSET_MAP,
    LandsatGranuleAccess,
    LandsatImageMetadataFile,
    LandsatQAValues,
    query_landsat_catalog_for_tile,
)
from src.plotting.plotting_functions import LANDSAT_QA_COLORS_DICT
from src.utils import PROJECT_ROOT
from src.utils.parameters import SatelliteID
from src.utils.profiling import MEASUREMENTS, timer
from src.utils.utils import setup_logging

MAX_MAIN_NODATA_PERCENTAGE = 80
OMNICLOUD_NODATA = 255
EARLY_STOP_CLEAR_REFERENCE_IMGS = 3

logger = setup_logging()


class LandsatChipSelection(BaseModel):
    """Helper class for data generation."""

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    crop_bands: list[npt.NDArray[np.int16]]
    crop_clouds: list[npt.NDArray[np.bool_]]
    crop_cloud_shadows: list[npt.NDArray[np.bool_]]
    main_and_reference_items: list[LandsatGranuleAccess]
    bands: list[str]
    crop_x: int
    crop_y: int
    main_id: str
    dilation_square: npt.NDArray[np.uint8]
    visualize: bool

    def prepare_main_chip(self) -> bool:
        """
        - Return True early (to skip) if the main chip has less than 20% valid px
        - Sets the main bands to self.main_crop_band
        - Sets the main cloud mask to self.main_clouds
        - Sets the main cloud shadow mask to self.main_cloud_shadows
        - Creates self.exclusion_mask_plumes = the exclusion mask where we are not inserting plumes
        - If self.visualize: Visualizes RGB main chip + clouds/cloud shadows + exclusion mask
        """  # noqa
        h, w = self.crop_clouds[0].shape[:2]
        self.main_crop_band = self.crop_bands[0].copy()
        self.main_qa = self.main_crop_band[-1, :]

        # Get nodata mask from QA band. At this point, FILL in QA represents both:
        # 1. Original FILL values from QA band
        # 2. Pixels where any reflectance/brightness band was 0 (set in LandsatTile.hide_nodata_px)
        nodata_mask = LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], self.main_qa)

        # Calculate quality percentages
        main_nodata_perc = 100 * float(nodata_mask.sum()) / (h * w)
        main_cloud_ratio = 100 * float(self.crop_clouds[0].sum()) / (h * w)
        main_cloud_shadow_ratio = 100 * float(self.crop_cloud_shadows[0].sum()) / (h * w)

        if main_nodata_perc > MAX_MAIN_NODATA_PERCENTAGE:
            if self.visualize:
                logger.info(
                    f"MAIN      {self.crop_x=}, {self.crop_y=} with {main_cloud_ratio:5.1f}% clouds, "
                    f"{main_cloud_shadow_ratio:5.1f}% cloud shadows , {main_nodata_perc:.1f}% nodata --> DONT USE"
                )
            return True

        self.main_crop_band = self.crop_bands[0].copy()

        # Create the mask where we will not insert plumes:
        # - clouds (from QA band cloud flags)
        # - cloud shadows (from QA band cloud shadow flags)
        # - nodata (from QA band FILL flag and 0 reflectance values in all bands except QA)
        # - water (from QA band WATER flag)
        self.main_clouds = self.crop_clouds[0].copy()
        self.main_cloud_shadows = self.crop_cloud_shadows[0].copy()
        self.exclusion_mask_plumes = (
            (self.main_clouds == 1)
            | (self.main_cloud_shadows == 1)
            # NOTE: using nodata mask is a little redundant, since the cloud arrays should already be 0 in nodata
            # regions due to hide_nodata_px() but lets be explicit here for robustness
            | nodata_mask
            | LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.WATER], self.main_qa)
        ).astype(np.uint8)

        self.exclusion_mask_plumes = dilation(self.exclusion_mask_plumes, self.dilation_square)

        self.rgb_image_main: npt.NDArray | None = None
        if self.visualize:
            self.visualize_main_chip(h, w, main_nodata_perc, main_cloud_ratio, main_cloud_shadow_ratio)
        return False

    def visualize_main_chip(
        self,
        h: int,
        w: int,
        main_nodata_perc: float,
        main_cloud_ratio: float,
        main_cloud_shadow_ratio: float,
    ) -> None:
        """Visualizes the RGB main chip, clouds, cloud shadows, and exclusion mask."""
        date = LandsatGranuleAccess.parse_landsat_tile_id(self.main_id)["acquisition_date"]
        date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        self.rgb_image_main = (
            np.transpose(
                self.main_crop_band[
                    [
                        self.bands.index("red"),
                        self.bands.index("green"),
                        self.bands.index("blue"),
                    ],
                    :,
                ],
                (1, 2, 0),
            )
            / 10000
        )
        norm_rgb = 0.4
        self.rgb_image_main = (255 * (self.rgb_image_main / norm_rgb)).clip(0, 255).astype(np.uint8)

        f, ax = plt.subplots(1, 4, figsize=(24, 6))
        ax = cast(np.ndarray, ax)  # for mypy

        # Plot RGB image
        ax[0].imshow(self.rgb_image_main, interpolation="nearest")
        ax[0].set_title(f"Main RGB, {main_nodata_perc:.1f}% nodata\n{date}", fontsize=25)

        # Plot RGB with clouds and cloud shadows overlayed
        ax[1].imshow(self.rgb_image_main, interpolation="nearest")
        ax[1].imshow(
            np.ma.masked_where(self.main_clouds != 1, self.main_clouds),
            cmap="spring",
            interpolation="none",
            alpha=0.5,
        )
        ax[1].imshow(
            np.ma.masked_where(self.main_cloud_shadows != 1, self.main_cloud_shadows),
            cmap="autumn",
            interpolation="none",
            alpha=0.5,
        )
        ax[1].set_title(
            f"Clouds: {main_cloud_ratio:.2f}%\n Cloud shadows: {main_cloud_shadow_ratio:.1f}%",
            fontsize=25,
        )

        # Plot RGB with exclusion mask overlayed
        ax[2].imshow(self.rgb_image_main, interpolation="nearest")
        ax[2].imshow(
            np.ma.masked_where(self.exclusion_mask_plumes != 1, self.exclusion_mask_plumes),
            cmap="spring",
            interpolation="none",
            alpha=0.5,
        )
        exclusion_perc = 100 * self.exclusion_mask_plumes.sum() / (h * w)
        ax[2].set_title(f"Exclusion for plumes: {exclusion_perc:.1f}%", fontsize=25)

        # Plot QA band
        plot_qa_band(self.main_qa, ax=ax[3], add_legend=False, title="QA_PIXEL", title_fontsize=20)

        plt.tight_layout()
        plt.show()

        logger.info(
            f"MAIN      {self.crop_x=}, {self.crop_y=} with {main_cloud_ratio:5.1f}% clouds, "
            f"{main_cloud_shadow_ratio:5.1f}% cloud shadows, {main_nodata_perc:.1f}% nodata "
            f"--> USE      ({self.main_id})"
        )

    def find_reference_chips(self, reference_crop_cloud_shadow_max: float) -> bool:  # noqa
        """Find the best two reference chips for the current main chip.

        Returns False if we have not found two reference chips. Returns True if we have.
        Sets chip data of MAIN, BEFORE and REFERENCE into the following lists:
        - self.chip_bands
        - self.chip_clouds
        - self.chip_cloud_shadows
        - self.chip_items
        - self.chip_ids
        """
        self.main_item = self.main_and_reference_items[0]
        self.reference_items = self.main_and_reference_items[1:]

        self.chip_items = [self.main_item]
        self.chip_bands = [self.main_crop_band]
        self.chip_clouds = [self.main_clouds]
        self.chip_cloud_shadows = [self.main_cloud_shadows]
        self.chip_ids = [self.main_id]
        self.reference_indices = []
        succeeded = False

        # Find clear view reference images
        if self.visualize:
            f, ax = plt.subplots(2, len(self.main_and_reference_items) - 1, figsize=(35, 6))
            ax = cast(np.ndarray, ax)  # for mypy
        for idx in range(1, len(self.main_and_reference_items)):
            id_ = self.main_and_reference_items[idx].id
            date = LandsatGranuleAccess.parse_landsat_tile_id(id_)["acquisition_date"]
            date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"

            crop_cloud = self.crop_clouds[idx].copy()
            crop_cloud_shadow = self.crop_cloud_shadows[idx].copy()
            qa_ = self.crop_bands[idx][-1]

            # Compute valid pixels (ignore nodata in main chip)
            valid_pixels_main = ~(LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], self.main_qa))
            valid_pixels_ref = ~(LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], qa_))

            # number of px where the main crop and the reference crop have data
            valid_and_overlapping_px = (valid_pixels_main & valid_pixels_ref).sum()

            # number of px where the main crop has data
            main_valid_px = float(valid_pixels_main.sum())

            # Of the px where the main crop has data, how much nodata % has the reference crop?
            nodata_mask_ref = ~valid_pixels_ref & valid_pixels_main
            nodata_perc = 100 * nodata_mask_ref.sum() / (main_valid_px + 0.01)

            # Of the px where both main and reference crop have data, how much % is cloudy?
            cloud_ratio = 100 * float(crop_cloud.sum()) / (valid_and_overlapping_px + 0.01)

            # Of the px where both main and reference crop have data, how much % is cloud shadows?
            cloud_shadow_ratio = 100 * float(crop_cloud_shadow.sum()) / (valid_and_overlapping_px + 0.01)

            if cloud_ratio + cloud_shadow_ratio + nodata_perc < 100 * reference_crop_cloud_shadow_max:
                if self.visualize:
                    logger.info(
                        f"REFERENCE {date}, {self.crop_x=}, {self.crop_y=} with {cloud_ratio:5.1f}% clouds, "
                        f"{cloud_shadow_ratio:5.1f}% cloud shadows, {nodata_perc:6.1f}% nodata"
                        f" --> USE      ({id_})"
                    )
                self.reference_indices.append(idx)
                title_string = f"USE\n{date}"
            else:
                if self.visualize:
                    logger.info(
                        f"REFERENCE {date}, {self.crop_x=}, {self.crop_y=} with {cloud_ratio:5.1f}% clouds, "
                        f"{cloud_shadow_ratio:5.1f}% cloud shadows, {nodata_perc:6.1f}% nodata"
                        f" --> DONT USE ({id_})"
                    )
                title_string = f"DONT USE\n{date}"

            if self.visualize:
                self._visualize_reference_chips(  # type: ignore
                    ax, idx, title_string, crop_cloud, crop_cloud_shadow, cloud_ratio, cloud_shadow_ratio, nodata_perc
                )

            def is_ax_empty(ax: Axes) -> bool:  # type: ignore
                return not (ax.lines or ax.collections or ax.patches or ax.images)

            if len(self.reference_indices) == 2:  # We have found the 2 reference chips #noqa
                succeeded = True
                for idx_ in self.reference_indices:
                    self.chip_bands.append(self.crop_bands[idx_])
                    self.chip_clouds.append(self.crop_clouds[idx_])
                    self.chip_cloud_shadows.append(self.crop_cloud_shadows[idx_])
                    self.chip_items.append(self.main_and_reference_items[idx_])
                    self.chip_ids.append(self.main_and_reference_items[idx_].id)
                if self.visualize:
                    logger.info("WE HAVE MAIN, BEFORE AND EARLIER for this crop")
                    for ax_idx in range(idx, len(self.main_and_reference_items) - 1):
                        if is_ax_empty(ax[0, ax_idx]):  # type: ignore
                            ax[0, ax_idx].set_visible(False)  # type: ignore
                            ax[1, ax_idx].set_visible(False)  # type: ignore
                break
        if self.visualize:
            plt.tight_layout()
            plt.show()

        if len(self.reference_indices) < 2:  # noqa
            succeeded = False
            if self.visualize:
                logger.info("WE HAVE NOT FOUND ENOUGH CLEAR REFERENCE IMAGES for this crop")
        return succeeded

    def _visualize_reference_chips(
        self,
        ax: npt.NDArray,
        idx: int,
        title_string: str,
        crop_cloud: npt.NDArray,
        crop_cloud_shadow: npt.NDArray,
        cloud_ratio: float,
        cloud_shadow_ratio: float,
        nodata_perc: float,
    ) -> None:
        rgb_image = self.crop_bands[idx][[self.bands.index(b) for b in ["red", "green", "blue"]]]
        rgb_image = np.transpose(rgb_image, (1, 2, 0)) / 10000  # type: ignore
        norm_rgb = 0.4
        rgb_image = (255 * (rgb_image / norm_rgb)).clip(0, 255).astype(np.uint8)

        # Plot RGB image
        ax[0, idx - 1].imshow(rgb_image, interpolation="nearest")
        ax[0, idx - 1].set_title(title_string, fontsize=13)

        # Plot RGB with clouds and cloud shadows overlayed only on the valid main px
        ax[1, idx - 1].imshow(rgb_image, interpolation="nearest")
        ax[1, idx - 1].imshow(
            np.ma.masked_where(
                ~LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], self.main_qa),
                np.ones_like(self.main_qa),
            ),
            cmap="Wistia",
            interpolation="none",
            alpha=0.35,
        )
        ax[1, idx - 1].imshow(
            np.ma.masked_where(crop_cloud != 1, crop_cloud), cmap="spring", interpolation="none", alpha=0.4
        )
        ax[1, idx - 1].imshow(
            np.ma.masked_where(crop_cloud_shadow != 1, crop_cloud_shadow),
            cmap="autumn",
            interpolation="none",
            alpha=0.4,
        )
        ax[1, idx - 1].set_title(
            f"Clouds: {cloud_ratio:.1f}%\n Shadows: {cloud_shadow_ratio:.1f}%\nNodata: {nodata_perc:.1f}%", fontsize=13
        )


class LandsatTile(BaseModel):
    """Helper for tile operations."""

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    img: npt.NDArray[np.int16]
    qa: npt.NDArray[np.int16]  # Landsat QA_PIXEL band
    id: str
    solar_zenith_angle: float
    mtl_data: LandsatImageMetadataFile
    bands: list[str]

    def convert_band_dn_values(self) -> None:
        """
        Convert band DN values to reflectance/brightness values.

        - All bands except LWIR11, LWIR12, QA_PIXEL to reflectance values
        - LWIR11, LWIR12 to brightness values
        - QA_PIXEL remains unchanged
        """
        # Get indices for different band types
        lwir11_idx = self.bands.index("lwir11") if "lwir11" in self.bands else None
        lwir12_idx = self.bands.index("lwir12") if "lwir12" in self.bands else None
        qa_pixel_idx = self.bands.index("qa_pixel") if "qa_pixel" in self.bands else None

        rescaling = self.mtl_data.LANDSAT_METADATA_FILE.LEVEL1_RADIOMETRIC_RESCALING

        # Convert all bands except LWIR and QA to reflectance
        for i, band in enumerate(self.bands):
            if i in [lwir11_idx, lwir12_idx, qa_pixel_idx]:
                continue

            # Get band number from STAC asset mapping
            band_num = LANDSAT_STAC_ASSET_MAP[band].replace("B", "")
            mult = getattr(rescaling, f"REFLECTANCE_MULT_BAND_{band_num}")
            add = getattr(rescaling, f"REFLECTANCE_ADD_BAND_{band_num}")

            self.img[i] = LandsatGranuleAccess.convert_band_dn_to_reflectance_values(
                self.img[i], self.solar_zenith_angle, mult, add
            )

        thermal_constants = self.mtl_data.LANDSAT_METADATA_FILE.LEVEL1_THERMAL_CONSTANTS

        # Convert LWIR bands to brightness
        self.img[lwir11_idx] = LandsatGranuleAccess.convert_thermal_band_dn_to_brightness_values(
            self.img[lwir11_idx],
            rescaling.RADIANCE_MULT_BAND_10,
            rescaling.RADIANCE_ADD_BAND_10,
            thermal_constants.K1_CONSTANT_BAND_10,
            thermal_constants.K2_CONSTANT_BAND_10,
        )

        self.img[lwir12_idx] = LandsatGranuleAccess.convert_thermal_band_dn_to_brightness_values(
            self.img[lwir12_idx],
            rescaling.RADIANCE_MULT_BAND_11,
            rescaling.RADIANCE_ADD_BAND_11,
            thermal_constants.K1_CONSTANT_BAND_11,
            thermal_constants.K2_CONSTANT_BAND_11,
        )

    def hide_nodata_px(self) -> None:
        """
        Set pixels to nodata (zero) if they meet nodata conditions.

        Nodata px conditions:
        1. Marked as FILL in the QA_PIXEL band
        2. Have 0 values in any reflectance/brightness band (from reflectance/brightness conversion)
        Updates QA band to mark these as FILL.
        """
        qa_nodata_mask = LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], self.qa)
        reflectance_nodata_mask = (self.img[:-1] == 0).sum(axis=0) > 0
        nodata_mask = qa_nodata_mask | reflectance_nodata_mask

        # Set reflectance/brightness bands to 0
        self.img[:-1, nodata_mask] = 0

        # Set QA band to FILL value for all nodata pixels
        self.img[-1, nodata_mask] = LandsatQAValues.FILL.value  # modify the img QA band so we don't recompute nodata
        self.qa[nodata_mask] = LandsatQAValues.FILL.value

    def hide_main_nodata_px_on_reference_img(self, main_qa: npt.NDArray) -> None:
        """Hide the main nodata px for reference images."""
        # Nodata mask using main_qa FILL value (considers both original nodata px and reflectance values = 0)
        nodata_mask_on_main = LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], main_qa)

        # Apply mask to cropped regions of the arrays
        self.shadows_combined[nodata_mask_on_main] = 0
        self.clouds_combined[nodata_mask_on_main] = 0

        # Calculate overlap percentage using QA band
        valid_pixels_main = ~LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], main_qa)
        valid_pixels_ref = ~LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], self.qa)

        # Compute overlapping valid pixels
        overlap_pixels = (valid_pixels_main & valid_pixels_ref).sum()

        self.overlap_with_main = 100 * (overlap_pixels / valid_pixels_main.sum()) if valid_pixels_main.sum() > 0 else 0

    def prepare_clouds(self) -> None:
        """Extract clouds and cloud shadows from QA_PIXEL band."""
        cloud_labels = [
            LandsatQAValues.CLOUD,
            LandsatQAValues.DILATED_CLOUD,
            LandsatQAValues.CIRRUS,
            LandsatQAValues.CLOUD_CONFIDENCE_MEDIUM,
            LandsatQAValues.CLOUD_CONFIDENCE_HIGH,
            LandsatQAValues.CIRRUS_CONFIDENCE_MEDIUM,
            LandsatQAValues.CIRRUS_CONFIDENCE_HIGH,
        ]
        cloud_mask = LandsatGranuleAccess.get_mask_from_qa_pixel(cloud_labels, self.qa).astype(np.uint8)

        shadow_labels = [
            LandsatQAValues.CLOUD_SHADOW,
            LandsatQAValues.CLOUD_SHADOW_CONFIDENCE_MEDIUM,
            LandsatQAValues.CLOUD_SHADOW_CONFIDENCE_HIGH,
        ]
        shadow_mask = LandsatGranuleAccess.get_mask_from_qa_pixel(shadow_labels, self.qa).astype(np.uint8)
        shadow_mask[cloud_mask == 1] = 0  # deactivate cloud flags, avoid double counting clouds as shadows

        nodata_mask = LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], self.qa)
        cloud_mask[nodata_mask] = 0  # deactivate nodata flags
        shadow_mask[nodata_mask] = 0  # deactivate nodata flags

        # Store the masks
        self.clouds_combined = cloud_mask
        self.shadows_combined = shadow_mask

    def calculate_metadata(self) -> None:
        """Calculate nodata, cloud, cloud shadow percentages for current tile."""
        # Calculate valid pixels (not FILL)
        valid_pixels = (~LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], self.qa)).astype(np.uint8)
        self.valid_px = float(valid_pixels.sum()) + 0.001  # Avoid division by zero

        # Calculate cloud and shadow percentages
        cloud_combined_sum = float(self.clouds_combined.sum())
        shadows_combined_sum = float(self.shadows_combined.sum())
        self.cloud_combined_perc = 100 * cloud_combined_sum / self.valid_px
        self.cloud_shadow_combined_perc = 100 * shadows_combined_sum / self.valid_px

        # Calculate nodata percentage
        self.no_data_perc = (
            100 * LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], self.qa).sum() / self.qa.size
        )

    def visualize(self, bands: list[str], main_qa: npt.NDArray | None = None) -> None:
        """Visualize tile as RGB with clouds and cloud shadows overlayed."""
        start = time.time()
        overlap_with_main_string = "" if main_qa is None else f"{self.overlap_with_main:.1f}% overlap with Main"

        # NOTE: calculate_metadata() and prepare_clouds() should be called before visualize()
        logger.info(f"{self.id}")
        logger.info("#" * 100)
        logger.info(f"Clouds: {self.cloud_combined_perc:.1f}%")
        logger.info(f"Cloud Shadows: {self.cloud_shadow_combined_perc:.1f}%")
        logger.info("#" * 100)
        logger.info(f"No Data: {self.no_data_perc:.1f}%")
        logger.info("#" * 100)

        # Create RGB image
        rgb_image = (
            np.transpose(
                self.img[[bands.index("red"), bands.index("green"), bands.index("blue")], :],
                (1, 2, 0),
            )
            / 10000
        )
        norm_rgb = 0.4
        rgb_image = (255 * (rgb_image / norm_rgb)).clip(0, 255).astype(np.uint8)

        # Create visualization plots
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax = cast(np.ndarray, ax)  # for mypy

        # Plot RGB image
        ax[0].imshow(rgb_image, interpolation="nearest")
        ax[0].set_title(f"RGB {overlap_with_main_string}", fontsize=25)

        # Plot RGB with clouds and shadows
        ax[1].imshow(rgb_image, interpolation="nearest")
        ax[1].imshow(
            np.ma.masked_where(self.clouds_combined != 1, self.clouds_combined),
            cmap="spring",
            interpolation="none",
            alpha=0.5,
        )
        ax[1].imshow(
            np.ma.masked_where(self.shadows_combined != 1, self.shadows_combined),
            cmap="autumn",
            interpolation="none",
            alpha=0.5,
        )
        ax[1].set_title(
            f"Clouds: {self.cloud_combined_perc:.1f}%\nCloud shadows: {self.cloud_shadow_combined_perc:.1f}%",
            fontsize=25,
        )

        plot_qa_band(self.qa, ax[2], add_legend=False, title="QA_PIXEL", title_fontsize=20)

        plt.tight_layout()
        plt.show()

        logger.info(f"Visualizing took {time.time() - start:.1f}s")


class LandsatDataGeneration(BaseDataGeneration):
    """Data generation pipeline for Landsat data."""

    def __init__(
        self,
        landsat_tile_id: str,
        bands: list[str],
        time_delta_days: int,
        nb_reference_ids: int,
        **kwargs: Any,
    ) -> None:
        self.landsat_tile_id = landsat_tile_id
        self.tile_components = LandsatGranuleAccess.parse_landsat_tile_id(self.landsat_tile_id)
        self.date = datetime.datetime.strptime(self.tile_components["acquisition_date"], "%Y%m%d")
        self.bands = bands
        self.time_delta_days = time_delta_days
        self.nb_reference_ids = nb_reference_ids

        # Crop counts
        self.non_overlapping_count = 0
        self.overlapping_count = 0
        self.too_much_main_nodata_count = 0
        self.succeed_5perc_count = 0
        self.failed_5perc_count = 0
        self.reference_indices_all: list[list[int]] = []

        # By default no visualizations, useful for debugging
        self.visualize_tiles = False
        self.visualize_crops = False
        self.visualize_crops_show_frac = 0.04  # 0.04 = Show random 4% of chips
        self.visualize_insertion = False

        # Create config object for base class
        config = DataGenerationConfig(**kwargs)
        super().__init__(config=config)

        # Initialize producing area/val regions polygons to check which chips overlap what
        self.hassi_poly = (
            gpd.read_file(f"{PROJECT_ROOT}/src/data/ancillary/hassi.geojson").geometry.iloc[0].buffer(1.5).simplify(0.1)
        )
        self.permian_poly = (
            gpd.read_file(f"{PROJECT_ROOT}/src/data/ancillary/permian.geojson").geometry.iloc[0].simplify(0.01)
        )
        self.marcellus_poly = gpd.read_file(f"{PROJECT_ROOT}/src/data/ancillary/marcellus.geojson").geometry.iloc[0]

        # self.producing_union = Union of all producing areas, TODO: Insert as we can't share the one we were using

    @property
    def hash_id(self) -> str:
        """Get unique scene identifier for hash computation."""
        return self.landsat_tile_id

    @property
    def scene_id(self) -> str:
        """Get unique scene identifier for file naming."""
        return self.landsat_tile_id

    @timer(phase="download_data", accumulator=MEASUREMENTS, verbose=True, logger=logger)
    def download_data(self) -> dict:  # noqa: PLR0912, PLR0915
        """Download Landsat data."""
        start_time = self.date - datetime.timedelta(days=self.time_delta_days)
        end_time = self.date + datetime.timedelta(days=1)  # to include self.date

        try_count = 0
        while True:
            if try_count == 5:  # noqa
                break
            try:
                # TODO: Remove WRS path/row constraint to get all available data for crops
                all_potential_items = filter_landsat_granules_by_quality(
                    self.tile_components["wrs_path"],
                    self.tile_components["wrs_row"],
                    start_time,
                    end_time,
                    self.quality_thresholds["main_tile_cloud"],
                    self.quality_thresholds["main_tile_nodata"][1],
                    self.quality_thresholds["reference_tile_cloud_shadow"][1],
                    abs_client=self.abs_client,
                    producing_union=self.producing_union,
                    s3_client=self.s3_client,
                )
                break
            except Exception as err:
                logger.error("filter_landsat_granules_by_quality failed, try again")
                logger.error(traceback.print_exception(None, err, err.__traceback__))
                try_count += 1
        # We need a minimum of 3 (one main tile, two reference tiles)
        assert len(all_potential_items) >= 3, f"not enough items found for tile {self.landsat_tile_id}"  # noqa

        cloud_shadow_reference_perc_max = 100 * self.quality_thresholds["reference_tile_cloud_shadow"][1]
        id_to_metadata: dict[str, dict] = {}
        reference_items: list[LandsatGranuleAccess] = []
        main_reference_bands: list[npt.NDArray] = []
        main_reference_clouds: list[npt.NDArray] = []
        main_reference_cloud_shadows: list[npt.NDArray] = []
        break_loop = False
        found_main_id = False
        lt_2perc_reference_img_nb = 0
        main_meta: dict = {}  # Initialise as empty instead of None so mypy is happy
        main_qa: npt.NDArray | None = None

        # Remove duplicate LS IDs by using the most recent processing dates
        filtered_item_ids = select_best_landsat_items_by_quality([item.id for item in all_potential_items])

        # Map back to the original objects
        all_potential_items = [item for item in all_potential_items if item.id in filtered_item_ids]

        for item in all_potential_items:
            start = time.time()

            try:
                # Now prefetch all bands since we know we want this item
                item.prefetch_l1(self.s3_client, self.abs_client)
            except Exception as err:
                logger.info(f"prefetch_l1() failed, skipping ({item.id})")
                logger.info(traceback.print_exception(None, err, err.__traceback__))
                continue
            logger.info(f"Transfering data from S3 to ABS took {time.time() - start:.1f}s")

            start = time.time()
            try:
                img = item.get_bands(self.bands, abs_client=self.abs_client)
                # Get metadata for grid alignment
                band_meta = item.get_raster_meta(self.bands[0], abs_client=self.abs_client)
            except Exception as err:
                logger.info("get_bands() failed, skipping this ID")
                logger.info(traceback.print_exception(None, err, err.__traceback__))
                continue
            logger.info(f"Loading {len(self.bands)} bands from ABS took {time.time() - start:.1f}s")

            if found_main_id:
                # Verify CRS matches before attempting grid alignment
                assert band_meta["crs"] == main_meta["crs"], f"CRS mismatch: {band_meta['crs']} != {main_meta['crs']}"

                # Only align if transforms or shapes differ
                if (
                    band_meta["transform"] != main_meta["transform"]
                    or band_meta["height"] != main_meta["height"]
                    or band_meta["width"] != main_meta["width"]
                ):
                    logger.info(f"Aligning {item.id} to main image grid")
                    img = align_raster_to_grid(img, band_meta, main_meta)
                    logger.info("Alignment complete")

            # Read in MTL.json to get conversion scaling factors
            mtl_data = item.get_mtl_data(abs_client=self.abs_client)

            # All relevant bands and qa masks loaded
            # --> set nodata px, prepare clouds, tile metadata and optionally visualize
            tile = LandsatTile(
                img=img,
                qa=img[-1].copy(),  # FIXME: [-1] seems error prone for identifying qa/SCL band
                id=item.id,
                solar_zenith_angle=item.solar_angle,
                mtl_data=mtl_data,
                bands=self.bands,
            )
            # Convert bands to reflectance/brightness values before any other operations
            tile.convert_band_dn_values()

            # Now, compute metrics/masks using reflectance values
            tile.hide_nodata_px()
            tile.prepare_clouds()

            if found_main_id:  # for reference images, we only care about the overlap with main
                # NOTE: We pass the main_tile's MODIFIED QA band here, so it includes FILL values for the original
                # nodata px AND reflectance values = 0 (which are also nodata). It is modified because it has been
                # processed by hide_nodata_px()
                assert main_qa is not None, "main_qa should be set when found_main_id is True"
                tile.hide_main_nodata_px_on_reference_img(main_qa)

            tile.calculate_metadata()
            if self.visualize_tiles:
                tile.visualize(self.bands, main_qa if found_main_id else None)

            id_to_metadata[item.id] = {}
            for k, v in tile.model_dump(
                exclude=[  # type: ignore
                    "clouds",
                    "clouds_combined",
                    "id",
                    "img",
                    "mtl_data",
                    "qa",
                    "shadows",
                    "shadows_combined",
                    "solar_zenith_angle",
                    "valid_px",
                ]
            ).items():
                id_to_metadata[item.id][k] = v

            if found_main_id:
                # Dont use reference images with > e.g. 80% clouds + cloud shadows
                if tile.cloud_combined_perc + tile.cloud_shadow_combined_perc < cloud_shadow_reference_perc_max:
                    reference_items.append(item)
                    logger.info(f"USE {item.id} AS REFERENCE IMAGE #{len(reference_items)}")
                    if len(reference_items) == self.nb_reference_ids:
                        logger.info(
                            f"We have {self.nb_reference_ids} reference images and will stop searching for more"
                        )
                        break_loop = True

                    # If we have 3 reference images with
                    # a) >99% overlap with the main tiles
                    # b) <2% clouds/cloud shadows
                    # we can create chips completely with them and can stop looking for more reference imgs here
                    if tile.overlap_with_main > 99 and tile.cloud_combined_perc + tile.cloud_shadow_combined_perc < 2.0:  # noqa
                        lt_2perc_reference_img_nb += 1
                    if lt_2perc_reference_img_nb == EARLY_STOP_CLEAR_REFERENCE_IMGS:
                        logger.info(
                            "We have 3 reference images with < 2% cloud+shadows, we don't need more reference images"
                        )
                        break_loop = True
                else:
                    logger.info(
                        f"Too much clouds/cloud shadows ({tile.cloud_combined_perc:.1f}% / "
                        f"{tile.cloud_shadow_combined_perc:.1f}%) --> DONT USE {item.id} AS A REFERENCE IMAGE"
                    )
                    continue
            else:
                logger.info(f"USE {item.id} AS THE MAIN IMAGE")
                main_item = item
                main_meta = band_meta
                found_main_id = True
                main_qa = tile.qa
            main_reference_bands.append(tile.img)
            main_reference_clouds.append(tile.clouds_combined)
            main_reference_cloud_shadows.append(tile.shadows_combined)
            if break_loop:
                break

        ### Summary of main and reference IDs found
        self.main_and_reference_items = [main_item, *reference_items]
        self.main_and_reference_ids = [k.id for k in self.main_and_reference_items]
        self.dropped_items = [item for item in all_potential_items if item.id not in self.main_and_reference_ids]

        logger.info("\n\n")
        for item in all_potential_items:
            if item.id == main_item.id:
                name = "MAIN     "
            elif item.id in self.main_and_reference_ids[1:]:
                name = "REFERENCE"
            elif item.id in id_to_metadata:
                name = "NOT USED "
            else:  # not considered
                continue
            logger.info(
                f"{name} {item.id} with {id_to_metadata[item.id]['no_data_perc']:5.1f}% nodata, "
                f"{id_to_metadata[item.id]['cloud_combined_perc']:.1f}% Clouds, "
                f"{id_to_metadata[item.id]['cloud_shadow_combined_perc']:.1f}% Cloud Shadows"
            )

        self.id_to_metadata = id_to_metadata
        return {
            "main_reference_bands": main_reference_bands,
            "main_reference_clouds": main_reference_clouds,
            "main_reference_cloud_shadows": main_reference_cloud_shadows,
        }

    def generate_crops(self, data: dict) -> Iterator[dict[str, Any]]:
        """Generate crops from Landsat data."""
        main_reference_bands = data["main_reference_bands"]
        main_reference_clouds = data["main_reference_clouds"]
        main_reference_cloud_shadows = data["main_reference_cloud_shadows"]

        # Define crop grid
        _, height, width = main_reference_bands[0].shape
        n_crops_x = math.ceil(width / self.crop_size)
        n_crops_y = math.ceil(height / self.crop_size)

        for row, col in list(itertools.product(range(n_crops_y), range(n_crops_x))):
            crop_x = col * self.crop_size
            crop_y = row * self.crop_size

            # Crop Landsat tile to crop_size x crop_size
            # Pad with 0s (nodata) to ensure the crop is divisible by crop_size
            crops_bands = [
                pad_crop_to_size(
                    whole_tile[:, crop_y : crop_y + self.crop_size, crop_x : crop_x + self.crop_size], self.crop_size
                )
                for whole_tile in main_reference_bands
            ]
            crops_clouds = [
                pad_crop_to_size(
                    whole_tile[crop_y : crop_y + self.crop_size, crop_x : crop_x + self.crop_size], self.crop_size
                )
                for whole_tile in main_reference_clouds
            ]
            crops_cloud_shadows = [
                pad_crop_to_size(
                    whole_tile[crop_y : crop_y + self.crop_size, crop_x : crop_x + self.crop_size], self.crop_size
                )
                for whole_tile in main_reference_cloud_shadows
            ]

            yield {
                "crops_bands": crops_bands,
                "crops_clouds": crops_clouds,
                "crops_cloud_shadows": crops_cloud_shadows,
                "crop_x": crop_x,
                "crop_y": crop_y,
            }

    def generate_synthetic_data_items(
        self, plume_files: npt.NDArray, crops: Iterator[dict[str, Any]], position_by_source: bool
    ) -> Iterator[MultiTemporalPlumesDataItem]:
        """Generate synthetic data items with plumes."""
        self.main_item = self.main_and_reference_items[0]

        # helpers for checking overlap with producing areas
        transform = self.main_item.get_raster_meta("nir08", abs_client=self.abs_client)["transform"]
        transformer = Transformer.from_crs(self.main_item.crs, "EPSG:4326", always_xy=True)
        crs_4326 = CRS.from_epsg(4326)
        # for dilating the exclusion mask for inserting plumes
        dilation_square = square(2)

        for crop_data in crops:
            crop_x = crop_data["crop_x"]
            crop_y = crop_data["crop_y"]

            overlap, region_overlap = self.overlaps_with_producing_area(
                crop_x, crop_y, transform, transformer, crs_4326
            )
            if not overlap:
                continue

            if self.visualize_crops and np.random.random() > self.visualize_crops_show_frac:
                continue

            self.overlapping_count += 1

            chips = LandsatChipSelection(
                crop_bands=crop_data["crops_bands"],
                crop_clouds=crop_data["crops_clouds"],
                crop_cloud_shadows=crop_data["crops_cloud_shadows"],
                main_and_reference_items=self.main_and_reference_items,
                bands=self.bands,
                crop_x=crop_x,
                crop_y=crop_y,
                main_id=self.main_item.id,
                dilation_square=dilation_square,
                visualize=self.visualize_crops,
            )

            skip = chips.prepare_main_chip()
            if skip:  # too much nodata in main crop
                self.too_much_main_nodata_count += 1
                if self.visualize_crops:
                    logger.info("Too much nodata")
                continue

            # Find clear view reference chips
            succeeded = chips.find_reference_chips(self.quality_thresholds["reference_crop_cloud_shadow"][1])

            if succeeded:
                self.succeed_5perc_count += 1
                self.reference_indices_all.append(chips.reference_indices)
                # Select plumes for this chip
                chosen_plume_files, plume_arrays, plume_emissions = self._select_plumes(
                    plume_files, self.visualize_crops
                )

                # Create data item from chips
                data_item = MultiTemporalPlumesDataItem.get_data_items_from_crops(
                    main_and_reference_items=chips.chip_items,
                    bands=self.bands,
                    hapi_data_path=self.hapi_data_path,
                    crops=chips.chip_bands,
                    crops_clouds=chips.chip_clouds,
                    crops_cloud_shadows=chips.chip_cloud_shadows,
                    crop_x=crop_x,
                    crop_y=crop_y,
                    crop_size=self.crop_size,
                    plume_arrays=plume_arrays,
                    plume_files=chosen_plume_files.tolist(),
                    plume_emissions=plume_emissions,
                    exclusion_mask_plumes=chips.exclusion_mask_plumes,
                    rng=self.rng,
                    transformation_params=self.transformation_params,
                    satellite_id=SatelliteID.LANDSAT,
                    swir16_band_name="swir16",
                    swir22_band_name="swir22",
                    rgb_image_main=chips.rgb_image_main,
                    visualize=self.visualize_insertion,
                    tile_id_to_metadata=self.id_to_metadata,
                    region_overlap=region_overlap,
                    position_by_source=position_by_source,
                )
                yield data_item
            else:
                self.failed_5perc_count += 1

    def overlaps_with_producing_area(
        self, crop_x: int, crop_y: int, transform: Affine, transformer: Transformer, crs_4326: CRS
    ) -> tuple[bool, str]:
        """Return True if the chip overlaps the producing areas."""
        region_overlap = "Other"
        # Calculate bounds in native CRS
        x_min, y_min = rasterio.transform.xy(transform, crop_y + self.crop_size, crop_x, offset="center")  # bottom-left
        x_max, y_max = rasterio.transform.xy(transform, crop_y, crop_x + self.crop_size, offset="center")  # top-right

        # Create a polygon in native tile CRS
        polygon_crop = Polygon(
            [
                (x_min, y_min),  # bottom-left
                (x_max, y_min),  # bottom-right
                (x_max, y_max),  # top-right
                (x_min, y_max),  # top-left
                (x_min, y_min),  # close the loop
            ]
        )

        # Transform the polygon to EPSG:4326
        polygon_crop_4326 = Polygon([transformer.transform(x, y) for x, y in polygon_crop.exterior.coords])

        intersection_with_producing = self.producing_union.intersection(polygon_crop_4326)
        if intersection_with_producing.area == 0:
            self.non_overlapping_count += 1
            if self.visualize_crops and np.random.random() < 0.05:  # Only show this log in 5% to not spam #noqa
                logger.info("The crop has 0km² overlapping area with producing areas and is skipped")
            return False, region_overlap

        # Get overlap area with producing areas in real km²
        transformer_aea = Transformer.from_crs(
            crs_4326,
            CRS(
                proj="aea",
                lat_1=intersection_with_producing.bounds[1],
                lat_2=intersection_with_producing.bounds[3],
            ),
            always_xy=True,
        )

        geom_aea = ops.transform(transformer_aea.transform, intersection_with_producing)
        area_km2 = geom_aea.area / 1e6

        # a full 128x128 px crop with 20m resolution has 6.5536km². Let's discard crops that have less than 10% overlap
        if area_km2 < self.crop_size * self.crop_size * (self.target_spatial_resolution / 1000) ** 2 * 0.1:
            self.non_overlapping_count += 1
            if self.visualize_crops:
                logger.info("The crop has less than 10% overlapping area with producing areas and is skipped")
            return False, region_overlap

        if self.hassi_poly.intersection(polygon_crop_4326).area > 0:
            region_overlap = "Hassi"
        elif self.permian_poly.intersection(polygon_crop_4326).area > 0:
            region_overlap = "Permian"
        elif self.marcellus_poly.intersection(polygon_crop_4326).area > 0:
            region_overlap = "Marcellus"
        return True, region_overlap

    # NOTE: we have this method to conveniently get a generator of data items, which is used in other scripts
    def generate_data_items(
        self,
        plume_files: npt.NDArray,
        position_by_source: bool,
    ) -> Iterator[MultiTemporalPlumesDataItem]:
        """Generate data items from Sentinel-2 data."""
        # 1. Download data
        data = self.download_data()

        # 2. Generate crops
        crops = self.generate_crops(data)

        # 3. Generate synthetic data items
        return self.generate_synthetic_data_items(
            plume_files=plume_files,
            crops=crops,
            position_by_source=position_by_source,
        )

    def report(self) -> None:
        """
        After chipping is complete, report on how many chips
        1) succeeded end-to-end
        2) did not overlap producing areas
        3) did have too many nodata main px
        4) did not find enough clear view reference chips
        Also, report on which indices from all reference images we used on average.
        """  # noqa
        nb_chips_total = self.overlapping_count + self.non_overlapping_count + 0.0001
        self.end_to_end_chip_success_perc = 100 * self.succeed_5perc_count / nb_chips_total
        logger.info(
            f"{self.succeed_5perc_count:4.0f}/{nb_chips_total:.0f} = {self.end_to_end_chip_success_perc:6.1f}%"
            f" did succeed end-to-end"
        )
        logger.info("#" * 100)

        self.non_overlapping_perc = 100 * self.non_overlapping_count / nb_chips_total
        logger.info(
            f"{self.non_overlapping_count:4}/{nb_chips_total:.0f} = {self.non_overlapping_perc:6.1f}%"
            f" did not overlap producing areas"
        )
        self.too_much_main_nodata_perc = (
            100 * self.too_much_main_nodata_count / (nb_chips_total - self.non_overlapping_count)
        )
        logger.info(
            f"{self.too_much_main_nodata_count:4}/{nb_chips_total - self.non_overlapping_count:4.0f} = "
            f"{self.too_much_main_nodata_perc:6.1f}% did not have enough valid main px"
        )
        logger.info("#" * 100)

        self.reference_success_perc = (
            100 * self.succeed_5perc_count / (self.succeed_5perc_count + self.failed_5perc_count + 0.0001)
        )
        logger.info(
            f"Of the remaining {nb_chips_total - self.non_overlapping_count - self.too_much_main_nodata_count:.0f}: "
            f"{self.reference_success_perc:.1f}% = {self.succeed_5perc_count:4.0f} "
            f"succeeded to find good (< 5% bad px) reference chips ( ==> {self.failed_5perc_count:4.0f} failed)"
        )
        logger.info("#" * 100)
        logger.info("#" * 100)
        if len(self.reference_indices_all) > 0:
            logger.info(
                f"We have {len(self.main_and_reference_ids) - 1} reference tiles where 1 = the closest in time to "
                f"the main ID and {len(self.main_and_reference_ids) - 1} = the furthest in time."
            )
            logger.info(
                "The oldest reference tile we used for any chip was at position "
                f"{max([max(k) for k in self.reference_indices_all if len(k) > 0])}"
            )
            logger.info(
                f"On average, the indices we have chosen reference tiles from are on position "
                f"{np.mean([np.mean(k) for k in self.reference_indices_all if len(k) > 0]):.1f} +/- "
                f"{np.std([np.mean(k) for k in self.reference_indices_all if len(k) > 0]):.1f}"
            )

    def save_json_summary(self, data_df: pd.DataFrame, save_cloud: bool = True, save_local: bool = False) -> None:
        """Save a .json with a report summary to quickly know what happened."""
        report: dict[str, int | float | list] = {}
        # add metadata of chip exclusion/reference chip selection
        report["end_to_end_chip_success_perc"] = round(self.end_to_end_chip_success_perc, 2)
        report["non_overlap_prod_area_perc"] = round(self.non_overlapping_perc, 2)
        report["too_much_main_nodata_perc"] = round(self.too_much_main_nodata_perc, 2)
        report["reference_success_perc"] = round(self.reference_success_perc, 2)

        report["crop_size"] = self.crop_size
        report["nb_reference_ids"] = self.nb_reference_ids
        report["main_and_reference_ids"] = self.main_and_reference_ids
        report["nb_chips_overlap_prod_area"] = self.overlapping_count
        report["nb_chips_non_overlap_prod_area"] = self.non_overlapping_count
        report["nb_chips_too_much_main_nodata"] = self.too_much_main_nodata_count
        report["nb_chips_success"] = self.succeed_5perc_count
        report["nb_chips_reference_failed"] = self.failed_5perc_count

        # main ID metadata
        if not data_df.empty:
            for col in [
                "exclusion_perc",
                "how_many_plumes_we_wanted",
                "how_many_plumes_we_inserted",
                "tile_cloud_combined_perc_main",
                "tile_cloud_shadow_combined_perc_main",
                "tile_no_data_perc_main",
            ]:
                if "tile" in col:
                    report[col] = round(data_df[col].iloc[0], 3)
                else:
                    report[col] = round(data_df[col].mean(), 3)

        report_data = json.dumps(report, indent=4)
        if save_cloud and self.storage_options is not None:
            cloud_parquet_path = str(self._get_parquet_path())
            logger.info(f"{cloud_parquet_path=}")
            # skip json for test for now
            if "data" in cloud_parquet_path:
                blob_name = "data" + cloud_parquet_path.split("data")[1]
                blob_name = blob_name.replace(".parquet", ".json")
                blob_client = self.abs_client.get_blob_client(container=self.container_name, blob=blob_name)

                # Upload JSON data
                blob_client.upload_blob(report_data, overwrite=True)

        if save_local:
            with open("test.json", "wb") as f:
                f.write(report_data.encode("utf-8"))

    def save_parquet(
        self,
        data_items: Iterator[BasePlumesDataItem],
        save_cloud: bool = True,
        save_local: bool = False,
    ) -> None:
        """Save data items to parquet file."""
        parquet_path = self._get_parquet_path()

        # Convert data items to DataFrame
        data_df = pd.DataFrame([item.to_dict() for item in data_items])
        self.report()
        self.save_json_summary(data_df, save_cloud, save_local)

        if data_df.empty:
            logger.info("No valid data items to write. Skipping parquet creation.")
            return

        # to avoid ArrowInvalid: Could not convert ... with type ListConfig: ...
        data_df["bands"] = data_df["bands"].apply(list)

        # add metadata of chip exclusion/reference chip selection
        data_df["nb_chips_overlap_prod_area"] = self.overlapping_count
        data_df["nb_chips_non_overlap_prod_area"] = self.non_overlapping_count
        data_df["non_overlap_prod_area_perc"] = self.non_overlapping_perc

        data_df["nb_chips_too_much_main_nodata"] = self.too_much_main_nodata_count
        data_df["too_much_main_nodata_perc"] = self.too_much_main_nodata_perc

        data_df["nb_chips_success"] = self.succeed_5perc_count
        data_df["nb_chips_reference_failed"] = self.failed_5perc_count
        data_df["reference_success_perc"] = self.reference_success_perc
        data_df["end_to_end_chip_success_perc"] = self.end_to_end_chip_success_perc

        data_df["reference_indices_chosen"] = self.reference_indices_all
        data_df["main_and_reference_dates"] = data_df["main_and_reference_ids"].apply(
            lambda x: [
                datetime.datetime.strptime(
                    LandsatGranuleAccess.parse_landsat_tile_id(k)["acquisition_date"], "%Y%m%d"
                ).strftime("%Y-%m-%d")
                for k in x
            ]
        )

        if self.plume_type == PlumeType.CARBONMAPPER:
            # Merge emission rates for AVIRIS plumes
            aviris_plumes = pd.concat(
                (
                    pd.read_csv(f"{PROJECT_ROOT}/src/data/ancillary/aviris_plumes_training.csv"),
                    pd.read_csv(f"{PROJECT_ROOT}/src/data/ancillary/aviris_plumes_validation.csv"),
                ),
                ignore_index=True,
            )
            aviris_plumes["con_tif"] = aviris_plumes["con_tif"].apply(lambda x: f"azureml://{x}")
            # 100 plumes do not have an emission rate
            aviris_plumes["emission_auto"] = aviris_plumes["emission_auto"].fillna(
                aviris_plumes["emission_auto"].median()
            )
            aviris_plumes["emission_auto"] = aviris_plumes["emission_auto"].round(1)
            plume_to_emission = (
                aviris_plumes[["con_tif", "emission_auto"]].set_index("con_tif")["emission_auto"].to_dict()
            )
            data_df["plume_emissions"] = data_df["plume_files"].apply(
                lambda x: [plume_to_emission[k] for k in x] if len(x) > 0 else np.nan
            )

        if "validation_" in str(self.out_dir):
            # Set all chips to one of Hassi, Marcellus or Permian as we want to validate only in these groups
            val_region = [k for k in data_df["region_overlap"].unique().tolist() if k != "Other"]
            if len(val_region) > 0:
                data_df["region_overlap"] = val_region[0]

        if save_cloud:
            data_df.to_parquet(
                str(parquet_path),
                compression="zstd",
                compression_level=9,
                row_group_size=1,
                write_statistics=False,  # row-group statistics -> meaningless when row groups are 1 row.
                write_page_index=False,  # groups the page statistics into one place for more efficient IO
                store_schema=False,  # False - do not write Arrow schema to file.  Will effect recreation of some types.
                use_dictionary=False,  # don't use dictionary encoding.  Doesn't reduce our file size, uses more memory.
                storage_options=self.storage_options,
            )
            logger.info(f"Parquet file with {len(data_df)} rows written to {parquet_path}")

        if save_local:
            data_df.to_parquet(
                "test.parquet",
                compression="zstd",
                compression_level=9,
                row_group_size=1,
                write_statistics=False,  # row-group statistics -> meaningless when row groups are 1 row.
                write_page_index=False,  # groups the page statistics into one place for more efficient IO
                store_schema=False,  # False - do not write Arrow schema to file.  Will effect recreation of some types.
                use_dictionary=False,  # don't use dictionary encoding.  Doesn't reduce our file size, uses more memory.
            )


####################################################
################ UTILITY FUNCTIONS #################
####################################################


# TODO: remove WRS path/row constraint to use all available data
def filter_landsat_granules_by_quality(  # noqa: PLR0913 (too-many-arguments)
    wrs_path: str,
    wrs_row: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    cloud_coverage_threshold_range: tuple[float, float],
    nodata_limit: float,
    cloud_shadow_limit: float,
    producing_union: Polygon | MultiPolygon | None,
    s3_client: S3Client,
    abs_client: BlobServiceClient,
) -> list[LandsatGranuleAccess]:
    """
    Query and filter Landsat granules based on quality metrics (nodata, clouds, shadows).

    This function performs two steps:
    1. Queries the STAC catalog for Landsat granules matching the spatial and temporal criteria
    2. Filters the granules based on quality metrics by analyzing their QA bands

    Arguments:
    - wrs_path: WRS path of tile; like "111"
    - wrs_row: WRS row of tile; like "222"
    - start_time: Start datetime to search from
    - end_time: End datetime to search until
    - cloud_coverage_threshold_range: tuple (lower limit, upper limit) of cloud coverage as a decimal
    - nodata_limit: upper limit of nodata fraction as a decimal
    - cloud_shadow_limit: upper limit of cloud + cloud shadow fraction as a decimal
    - producing_union: Producing area to check the amount of overlap with Landsat tiles
    - abs_client: Azure Blob Storage client
    """
    # Get initial list of items from STAC catalog
    pystac_items = query_landsat_catalog_for_tile(
        wrs_path,
        wrs_row,
        start_time,
        end_time,
        cloud_cover_range=cloud_coverage_threshold_range,
    )
    main_and_reference_items = [LandsatGranuleAccess(item) for item in pystac_items]

    mask_kwargs = {"abs_client": abs_client}
    printed_overlap = False
    filtered_items = []
    for item in main_and_reference_items:
        try:
            # Prefetch just the QA band to get access to nodata/clouds/cloud shadow
            item.prefetch_l1(s3_client, abs_client, bands_to_transfer=["qa_pixel"])

            # Calculate nodata
            mask_nodata = item.get_mask([LandsatQAValues.FILL], **mask_kwargs)
            mean_mask_nodata = np.mean(mask_nodata)

            # Calculate valid px mask
            valid_pixels = ~mask_nodata
            valid_pixel_count: int = np.sum(valid_pixels)

            # Calculate cloud and shadow masks
            # NOTE: This filtering is different from how we do it with S2. In S2 we compute cloud shadow/cloud for the
            # ref tiles only where the main tile has valid px. Here we compute it for all valid px in the ref tiles.

            cloud_labels = [
                LandsatQAValues.CLOUD,
                LandsatQAValues.DILATED_CLOUD,
                LandsatQAValues.CIRRUS,
                LandsatQAValues.CLOUD_CONFIDENCE_MEDIUM,
                LandsatQAValues.CLOUD_CONFIDENCE_HIGH,
                LandsatQAValues.CIRRUS_CONFIDENCE_MEDIUM,
                LandsatQAValues.CIRRUS_CONFIDENCE_HIGH,
            ]
            mask_cloud = item.get_mask(cloud_labels, **mask_kwargs)
            mean_mask_cloud = np.sum(mask_cloud & valid_pixels) / valid_pixel_count if valid_pixel_count > 0 else 1.0

            shadow_labels = [
                LandsatQAValues.CLOUD_SHADOW,
                LandsatQAValues.CLOUD_SHADOW_CONFIDENCE_MEDIUM,
                LandsatQAValues.CLOUD_SHADOW_CONFIDENCE_HIGH,
            ]
            mask_shadow = item.get_mask(shadow_labels, **mask_kwargs)
            mask_shadow[mask_cloud] = 0  # deactivate cloud flags
            mean_mask_shadow = np.sum(mask_shadow & valid_pixels) / valid_pixel_count if valid_pixel_count > 0 else 1.0

        except Exception as err:
            logger.error(f"item.get_mask() failed, skip this ID {item.id}")
            logger.error(traceback.print_exception(None, err, err.__traceback__))
            continue

        if mean_mask_nodata >= nodata_limit:
            logger.info(f"{item.id} with {100 * mean_mask_nodata:6.1f}% nodata --> DONT USE")
            continue

        if mean_mask_cloud + mean_mask_shadow >= cloud_shadow_limit:
            logger.info(
                f"{item.id} with {100 * mean_mask_cloud:6.1f}% / {100 * mean_mask_shadow:.1f}% clouds/cloud shadows "
                "--> DONT USE"
            )
            continue

        # If we get here, both quality checks passed
        logger.info(
            f"{item.id} with {100 * mean_mask_nodata:6.1f}% nodata, "
            f"{100 * mean_mask_cloud:6.1f}% clouds, {100 * mean_mask_shadow:6.1f}% shadows --> USE"
        )
        filtered_items.append(item)

        if producing_union is not None and not printed_overlap:
            # Prefetch the NIR band. NOTE: this can be any band.
            item.prefetch_l1(s3_client, abs_client, bands_to_transfer=["nir08"])
            raster_meta = item.get_raster_meta("nir08", abs_client)
            transform = raster_meta["transform"]
            height = raster_meta["height"]
            width = raster_meta["width"]
            transformer = Transformer.from_crs(item.crs, "EPSG:4326", always_xy=True)

            # Calculate bounds in LS CRS
            x_min, y_min = rasterio.transform.xy(transform, height, 0, offset="center")  # bottom-left
            x_max, y_max = rasterio.transform.xy(transform, 0, width, offset="center")  # top-right
            # Create a polygon in LS CRS
            polygon_tile = box(x_min, y_min, x_max, y_max)

            # Transform the polygon to EPSG:4326
            polygon_tile_4326 = Polygon([transformer.transform(x, y) for x, y in polygon_tile.exterior.coords])

            intersection_with_producing = producing_union.intersection(polygon_tile_4326)
            logger.info(f"Intersection with producing: {intersection_with_producing.area:.2f}")
            printed_overlap = True

    return filtered_items


def select_best_landsat_items_by_quality(items: list[str], verbose: bool = False) -> list[str]:  # noqa PLR0912
    """For each sensor and date, select the best quality and most recent processed Landsat items.

    This function assumes there can be multiple WRS path/rows.

    Example 1:
    LC09_L1TP_027037_20250418_20250418_02_T1
    LC09_L1TP_027038_20250418_20250418_02_T1
    We want to use only one of them as they are both from LS9 and the same date. Processing date is the same,
    so which one does not matter.

    Example 2:
    LC09_L1GT_027038_20250128_20250128_02_T2
    LC09_L1TP_027037_20250128_20250128_02_T1
    We want to use the T1 item as it is better quality. The T2 item is not used.

    T1 collection items are preferred over T2.
    If there are multiple T1/T2 items, we take the one with the most recent processing date.
    """
    grouped_items = defaultdict(list)

    # Group by all keys EXCEPT processing_date
    for item_id in items:
        parsed = LandsatGranuleAccess.parse_landsat_tile_id(item_id)
        key = (
            parsed["satellite"],
            parsed["sensor"],
            parsed["acquisition_date"],
            parsed["collection_number"],
        )
        grouped_items[key].append(item_id)

    selected_items = []
    for key, scene_items in grouped_items.items():
        if verbose:
            print("#" * 100)
            print(key, scene_items)
        # Case 1: No duplicates
        if len(scene_items) == 1:
            selected_items.append(scene_items[0])
            if verbose:
                logger.info(f"Use single ID {scene_items[0]}")
            continue

        # Case 2: Only one T1 collection item --> Take it
        t1_items = [
            item_
            for item_ in scene_items
            if LandsatGranuleAccess.parse_landsat_tile_id(item_)["collection_category"] == "T1"
        ]
        t1_sum = len(t1_items)
        if t1_sum == 1:
            for item_ in scene_items:
                parsed = LandsatGranuleAccess.parse_landsat_tile_id(item_)
                if parsed["collection_category"] == "T1":
                    best_item = item_
                    if verbose:
                        logger.info(f"Use single T1: {best_item}")
                    break
        # Case 3: Multiple T1 collection items --> Pick the T1 with the most recent processing date
        elif t1_sum > 1:
            best_item = max(
                t1_items,
                key=lambda item_: int(LandsatGranuleAccess.parse_landsat_tile_id(item_)["processing_date"]),
            )
            if verbose:
                logger.info(f"Using most recent T1 ID {best_item}")
        # Case 4: Multiple T2 collection items --> Pick the T2 with the most recent processing date
        else:
            best_item = max(
                scene_items,
                key=lambda item_: int(LandsatGranuleAccess.parse_landsat_tile_id(item_)["processing_date"]),
            )
            if verbose:
                logger.info(f"Using most recent T2 ID {best_item}")
        selected_items.append(best_item)

        # Log skipped duplicates
        for item_ in scene_items:
            if item_ != best_item:
                logger.info(f"Skipping duplicate ID {item_}, keep {best_item}")
        if verbose:
            print("#" * 100)
    return selected_items


def align_raster_to_grid(source_array: npt.NDArray, source_meta: dict, target_meta: dict) -> npt.NDArray:
    """
    Aligns a source raster to match the grid and spatial extent of a target raster.

    This ensures that both rasters have the same transform, CRS, and extent so they
    can be indexed equivalently. Pixel resolution remains unchanged.

    This function assumes the input `source_array` has a band dimension,
    even for single-band rasters (i.e., shape should be (bands, height, width)).

    Args:
        source_array (np.ndarray): Source raster array (single-band or multi-band).
        source_meta (dict): Metadata of the source raster.
        target_meta (dict): Metadata of the target raster.

    Return:
    -------
        np.ndarray: Aligned raster array that matches the target raster grid.
    """
    # A multiband image tensor should have three dimensions: (bands, rows, cols)
    N_DIMENSIONS_MULTIBAND_IMAGE = 3
    assert len(source_array.shape) == N_DIMENSIONS_MULTIBAND_IMAGE, "Band dimension needs to be included"
    num_bands, _, _ = source_array.shape

    aligned_array = np.zeros((num_bands, target_meta["height"], target_meta["width"]), dtype=source_array.dtype)

    # Align each band separately
    for i in range(num_bands):
        reproject(
            source=source_array[i],
            destination=aligned_array[i],
            src_transform=source_meta["transform"],
            src_crs=source_meta["crs"],
            dst_transform=target_meta["transform"],
            dst_crs=target_meta["crs"],
            resampling=Resampling.nearest,
        )

    return aligned_array


def pad_crop_to_size(crop: np.ndarray, crop_size: int) -> np.ndarray:
    """Pad the crop to crop_size if it's smaller (edge case)."""
    h, w = crop.shape[-2:]
    pad_h = crop_size - h
    pad_w = crop_size - w
    if pad_h <= 0 and pad_w <= 0:
        return crop.copy()  # No padding needed

    NUM_DIMS_3D = 3
    NUM_DIMS_2D = 2

    if crop.ndim == NUM_DIMS_3D:
        return np.pad(crop, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
    elif crop.ndim == NUM_DIMS_2D:
        return np.pad(crop, ((0, pad_h), (0, pad_w)), mode="constant")
    else:
        raise ValueError("Crop must be 2D or 3D")


# Thank you Cursor
def plot_qa_band(
    qa_band: npt.NDArray,
    ax: plt.Axes | None = None,
    add_legend: bool = True,
    figsize: tuple[int, int] = (10, 10),
    title: str = "QA_PIXEL",
    title_fontsize: int = 20,
) -> plt.Axes:
    """
    Plot QA band categories using a priority-based approach to handle overlapping flags.

    Since Landsat QA bands use bitwise flags, a single pixel can have multiple flags active
    simultaneously (e.g., both cloud and cloud shadow). To avoid double-counting and ensure
    clear visualization, flags are processed in priority order, where higher priority flags
    (e.g., No Data) deactivate lower priority flags for the same pixels.

    Note:
        - Value 0 in the output image means: no flag matched at all.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Define QA flags in order of priority (highest first)
    priority_flags = [
        (flag, LANDSAT_QA_COLORS_DICT[flag], label)
        for flag, label in [
            (LandsatQAValues.FILL, "No Data"),
            (LandsatQAValues.CLOUD, "Cloud"),
            (LandsatQAValues.DILATED_CLOUD, "Dilated Cloud"),
            (LandsatQAValues.CIRRUS, "Cirrus"),
            (LandsatQAValues.CLOUD_SHADOW, "Cloud Shadow"),
            (LandsatQAValues.WATER, "Water"),
            (LandsatQAValues.SNOW, "Snow"),
            (LandsatQAValues.CLEAR, "Clear"),
        ]
    ]

    # Assign each flag a unique value starting from 1
    # Value 0 in qa_viz will represent "no flag matched"
    flag_to_value = {flag: i + 1 for i, (flag, _, _) in enumerate(priority_flags)}

    # Initialize output image with 0s (background / unmatched)
    qa_viz = np.zeros_like(qa_band)

    # Create masks for each flag
    masks = {flag: LandsatGranuleAccess.get_mask_from_qa_pixel([flag], qa_band) for flag, _, _ in priority_flags}

    # Apply masks sequentially, deactivating lower priority flags
    for i, (current_flag, _, _) in enumerate(priority_flags):
        current_mask = masks[current_flag].copy()

        # Deactivate this flag's pixels in all lower priority masks
        for lower_flag, _, _ in priority_flags[i + 1 :]:
            masks[lower_flag][current_mask] = 0

        # Set the visualization value for this flag
        qa_viz[current_mask] = flag_to_value[current_flag]

    # First color is for background (value 0 = unmatched pixels)
    cmap_colors = ["lightgray"] + [color for _, color, _ in priority_flags]
    custom_cmap = plt.cm.colors.ListedColormap(cmap_colors)

    # Plot
    ax.imshow(
        qa_viz,
        cmap=custom_cmap,
        interpolation="nearest",
        vmin=0,  # vmin=0 to include unmatched px
        vmax=len(priority_flags),
    )
    ax.set_title(title, fontsize=title_fontsize)
    ax.axis("off")

    if add_legend:
        # Create legend elements
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color, label=label) for _, color, label in priority_flags
        ]

        ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5), title="QA Flags")

    return ax
