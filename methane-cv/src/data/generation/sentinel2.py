"""Sentinel-2 data generation pipeline."""

import datetime
import itertools
import json
import os
import time
import traceback
from collections.abc import Iterator
from typing import Any, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import torch
from azure.storage.blob import BlobServiceClient
from matplotlib.axes import Axes
from pydantic import BaseModel
from pyproj import CRS, Transformer
from rasterio.transform import Affine
from shapely import ops
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import transform, unary_union
from skimage.morphology import dilation, square

from src.data.common.data_item import BasePlumesDataItem, MultiTemporalPlumesDataItem
from src.data.common.sim_plumes import PlumeType
from src.data.generation.base import BaseDataGeneration, DataGenerationConfig
from src.data.sentinel2 import (
    BAND_RESOLUTIONS,
    query_sentinel2_catalog_for_tile,
)
from src.data.sentinel2 import SceneClassificationLabel as SCLabel
from src.data.sentinel2_l1c import Sentinel2L1CItem
from src.plotting.plotting_functions import (
    CMAP,
    S2_LAND_COVER_CLASSIFICATIONS,
)
from src.utils import PROJECT_ROOT
from src.utils.parameters import SatelliteID
from src.utils.profiling import MEASUREMENTS, timer
from src.utils.utils import setup_logging

MAX_MAIN_NODATA_PERCENTAGE = 80
OMNICLOUD_NODATA = 255
EARLY_STOP_CLEAR_REFERENCE_IMGS = 3

logger = setup_logging()


class S2ChipSelection(BaseModel):
    """Helper class for data generation."""

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    crop_bands: list[npt.NDArray[np.int16]]
    crop_clouds: list[npt.NDArray[np.bool_]]
    crop_cloud_shadows: list[npt.NDArray[np.bool_]]
    main_and_reference_items: list[Sentinel2L1CItem]
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
        # Use all crop_main's independent on clouds/cloud shadows
        h, w = self.crop_clouds[0].shape[:2]
        self.main_crop_band = self.crop_bands[0].copy()
        self.main_scl = self.main_crop_band[-1, :]
        main_nodata_perc = 100 * (self.main_scl == 0).sum() / (h * w)
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

        # Create the mask where we will not insert plumes
        # clouds, cloud shadows + SCL: nodata=0, saturated or defective (=1), dark area pixels (=2),
        # cloud shadows=3, water=6, unclassified=7
        self.main_clouds = self.crop_clouds[0].copy()
        self.main_cloud_shadows = self.crop_cloud_shadows[0].copy()
        self.exclusion_mask_plumes = (
            (self.main_clouds == 1)
            | (self.main_cloud_shadows == 1)
            | (self.main_scl == SCLabel.NO_DATA.value)
            | (self.main_scl == SCLabel.SATURATED_OR_DEFECTIVE.value)
            | (self.main_scl == SCLabel.CAST_SHADOWS.value)
            | (self.main_scl == SCLabel.CLOUD_SHADOWS.value)
            | (self.main_scl == SCLabel.WATER.value)
            | (self.main_scl == SCLabel.UNCLASSIFIED.value)
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
        s2_date = self.main_id.split("_")[2]
        s2_date = f"{s2_date[:4]}-{s2_date[4:6]}-{s2_date[6:8]}"
        self.rgb_image_main = (
            np.transpose(
                self.main_crop_band[
                    [
                        self.bands.index("B04"),
                        self.bands.index("B03"),
                        self.bands.index("B02"),
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
        ax[0].imshow(self.rgb_image_main, interpolation="nearest")
        ax[0].set_title(f"Main RGB, {main_nodata_perc:.1f}% nodata\n{s2_date}", fontsize=25)

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

        ax[2].imshow(self.rgb_image_main, interpolation="nearest")
        ax[2].imshow(
            np.ma.masked_where(self.exclusion_mask_plumes != 1, self.exclusion_mask_plumes),
            cmap="spring",
            interpolation="none",
            alpha=0.5,
        )
        exclusion_perc = 100 * self.exclusion_mask_plumes.sum() / (h * w)
        ax[2].set_title(f"Exclusion for plumes: {exclusion_perc:.1f}%", fontsize=25)

        im = ax[3].imshow(self.main_scl, cmap=CMAP, interpolation="nearest")
        ax[3].set_title("SCL", fontsize=20)
        im.set_clim(-0.5, 11.5)  # Set limits to match label edges
        plt.tight_layout()
        plt.show()

        logger.info(
            f"MAIN      {self.crop_x=}, {self.crop_y=} with {main_cloud_ratio:5.1f}% clouds, "
            f"{main_cloud_shadow_ratio:5.1f}% cloud shadows, {main_nodata_perc:.1f}% nodata "
            f"--> USE      ({self.main_id[11:-16]})"
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
            s2_date = id_.split("_")[2]
            s2_date = f"{s2_date[:4]}-{s2_date[4:6]}-{s2_date[6:8]}"

            crop_cloud = self.crop_clouds[idx].copy()
            crop_cloud_shadow = self.crop_cloud_shadows[idx].copy()
            scl_ = self.crop_bands[idx][-1]

            # number of px where the main crop and the reference crop have data
            valid_and_overlapping_px = float(((scl_ != 0) & (self.main_scl != 0)).sum())
            # number of px where the main crop has data
            main_valid_px = float((self.main_scl != 0).sum())
            # Of the px where both main and reference crop have data, how much % is cloudy?
            cloud_ratio = 100 * float(crop_cloud.sum()) / (valid_and_overlapping_px + 0.01)
            # Of the px where both main and reference crop have data, how much % is cloud shadows?
            cloud_shadow_ratio = 100 * float(crop_cloud_shadow.sum()) / (valid_and_overlapping_px + 0.01)

            # Of the px where the main crop has data, how much nodata % has the reference crop?
            nodata_perc = 100 * ((scl_ == 0) & (self.main_scl != 0)).sum() / main_valid_px

            if cloud_ratio + cloud_shadow_ratio + nodata_perc < 100 * reference_crop_cloud_shadow_max:
                if self.visualize:
                    logger.info(
                        f"REFERENCE {s2_date}, {self.crop_x=}, {self.crop_y=} with {cloud_ratio:5.1f}% clouds, "
                        f"{cloud_shadow_ratio:5.1f}% cloud shadows, {nodata_perc:6.1f}% nodata"
                        f" --> USE      ({id_[11:-16]})"
                    )
                self.reference_indices.append(idx)
                title_string = f"USE\n{s2_date}"
            else:
                if self.visualize:
                    logger.info(
                        f"REFERENCE {s2_date}, {self.crop_x=}, {self.crop_y=} with {cloud_ratio:5.1f}% clouds, "
                        f"{cloud_shadow_ratio:5.1f}% cloud shadows, {nodata_perc:6.1f}% nodata"
                        f" --> DONT USE ({id_[11:-16]})"
                    )
                title_string = f"DONT USE\n{s2_date}"

            if self.visualize:
                ax = cast(np.ndarray, ax)
                self._visualize_reference_chips(
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

    def _visualize_reference_chips(  # type: ignore
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
        rgb_image = self.crop_bands[idx][[self.bands.index(b) for b in ["B04", "B03", "B02"]]]
        rgb_image = np.transpose(rgb_image, (1, 2, 0)) / 10000  # type: ignore
        norm_rgb = 0.4
        rgb_image = (255 * (rgb_image / norm_rgb)).clip(0, 255).astype(np.uint8)

        ax[0, idx - 1].imshow(rgb_image, interpolation="nearest")
        ax[0, idx - 1].set_title(title_string, fontsize=13)
        ax[1, idx - 1].imshow(rgb_image, interpolation="nearest")
        ax[1, idx - 1].imshow(
            np.ma.masked_where(self.main_scl != 0, self.main_scl), cmap="Wistia", interpolation="none", alpha=0.35
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


class S2Tile(BaseModel):
    """Helper for tile operations."""

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    img: npt.NDArray[np.int16]
    scl: npt.NDArray[np.int16]
    probs: npt.NDArray[np.int16]
    id: str

    def hide_nodata_px(self) -> None:
        """Set all pixels to nan (0/255) if a single band is 0 until we understand this better."""
        nodata_mask = (self.img == 0).sum(axis=0) > 0
        self.scl[nodata_mask] = 0
        self.probs[:, nodata_mask] = OMNICLOUD_NODATA  # for omnicloud, we use 255 as a nodata value
        self.img[:, nodata_mask] = 0

    def hide_main_nodata_px_on_reference_img(self, main_scl: npt.NDArray) -> None:
        """Hide the main nodata px for reference images."""
        self.clouds_omni[main_scl == 0] = 0
        self.shadows_omni[main_scl == 0] = 0
        self.scl[main_scl == 0] = 0
        self.omnicloud_argmax[main_scl == 0] = 0
        self.shadows_combined[main_scl == 0] = 0
        self.clouds_combined[main_scl == 0] = 0

        self.overlap_with_main = 100 * ((self.scl != 0) & (main_scl != 0)).sum() / (main_scl != 0).sum()

    def prepare_clouds(self, omnicloud_cloud_t: int = 35, omnicloud_shadow_t: int = 30) -> None:
        """Sum up thin/thick omnicloud predictions, threshold and combine with SCL clouds."""
        self.probs = self.probs.astype(np.float32)
        self.probs[self.probs == OMNICLOUD_NODATA] = np.nan
        self.omnicloud_argmax = np.argmax(self.probs, axis=0)

        # Sum thin and thick cloud probabilities together to threshold the sum, instead of the separate classes
        probs_3cls = self.probs[:3, :, :].copy()
        probs_3cls[1, :, :] = probs_3cls[1, :, :] + probs_3cls[2, :, :]  # Add thin + thick probabilities
        probs_3cls[2, :, :] = self.probs[3, :, :].copy()

        # threshold the sum of thin+thick cloud probabilities
        self.clouds_omni = (probs_3cls[1, :, :] > omnicloud_cloud_t).astype(np.uint8)
        self.shadows_omni = ((probs_3cls[2, :, :] > omnicloud_shadow_t) & (self.clouds_omni != 1)).astype(np.uint8)
        # Set clouds as Omniclouds OR Thick/Thin SCL Clouds to catch some Omnicloud FNs
        self.clouds_scl_thick_thin = (
            (self.scl == SCLabel.CLOUD_MEDIUM_PROBABILITY.value) | (self.scl == SCLabel.CLOUD_HIGH_PROBABILITY.value)
        ).astype(np.uint8)
        self.clouds_combined = ((self.clouds_omni == 1) | (self.clouds_scl_thick_thin == 1)).astype(np.uint8)
        # If the cloud shadow threshold is reached and it's not a cloud, classify as cloud shadow
        self.shadows_combined = ((probs_3cls[2, :, :] > omnicloud_shadow_t) & (self.clouds_combined != 1)).astype(
            np.uint8
        )

    def calculate_metadata(self) -> None:
        """Calculate nodata, cloud, cloud shadow percentages for current tile."""
        self.valid_px = float((self.scl != 0).sum()) + 0.001

        shadows_combined_sum = float(self.shadows_combined.sum())
        shadows_omni_sum = float(self.shadows_omni.sum())

        cloud_omni_sum = float(self.clouds_omni.sum())
        cloud_combined_sum = float(self.clouds_combined.sum())

        self.cloud_omni_perc = 100 * cloud_omni_sum / self.valid_px
        self.cloud_combined_perc = 100 * cloud_combined_sum / self.valid_px
        self.cloud_shadow_omni_perc = 100 * shadows_omni_sum / self.valid_px
        self.cloud_shadow_combined_perc = 100 * shadows_combined_sum / self.valid_px
        self.clouds_scl = (
            (self.scl == SCLabel.CLOUD_MEDIUM_PROBABILITY.value)
            | (self.scl == SCLabel.CLOUD_HIGH_PROBABILITY.value)
            | (self.scl == SCLabel.THIN_CIRRUS.value)
        ).astype(np.uint8)  # cirrus added
        cloud_scl_sum = float(self.clouds_scl.sum())
        self.SCL_clouds_cirrus_thin_thick_perc = 100 * cloud_scl_sum / self.valid_px
        self.no_data_perc = 100 * (self.scl == 0).sum() / (self.scl.shape[0] * self.scl.shape[1])
        for cls_idx, class_name in enumerate(S2_LAND_COVER_CLASSIFICATIONS):
            setattr(
                self,
                f"SCL_{class_name}_perc",
                100 * (self.scl == cls_idx).sum() / self.valid_px,
            )
        self.SCL_NO_DATA_perc = 100 * (self.scl == 0).sum() / (self.scl.shape[0] * self.scl.shape[1])

    def visualize(self, bands: list[str], main_scl: npt.NDArray | None = None) -> None:  # noqa PLR0915
        """Visualize tile as RGB with clouds and cloud shadows overlayed."""
        start = time.time()
        overlap_with_main_string = "" if main_scl is None else f"{self.overlap_with_main:.1f}% overlap with Main"
        shadows_scl = (self.scl == SCLabel.CLOUD_SHADOWS.value).astype(np.uint8)
        shadows_scl_sum = float(shadows_scl.sum())
        cloud_shadow_scl_perc = 100 * shadows_scl_sum / self.valid_px

        logger.info(f"{self.id}")
        logger.info("#" * 100)
        logger.info(
            f"Clouds   : {self.SCL_clouds_cirrus_thin_thick_perc:.1f}% (SCL)   vs {self.cloud_omni_perc:.1f}% (Omni)"
            f" vs {self.cloud_combined_perc:.1f}% (Combined)"
        )
        logger.info(
            f"Cloud Shadows : {cloud_shadow_scl_perc:.1f}% (SCL)   vs.   {self.cloud_shadow_omni_perc:.1f}% (Omni)"
            f"vs. {self.cloud_shadow_combined_perc:.1f}% (Combined)"
        )
        logger.info("#" * 100)
        logger.info(f"SCL No Data       : {self.no_data_perc:.1f}%")
        logger.info(
            f"  SCL Cloud Medium: {100 * (self.scl == SCLabel.CLOUD_MEDIUM_PROBABILITY.value).sum() / self.valid_px:.1f}%"  # noqa
        )
        logger.info(
            f"  SCL Cloud High  : {100 * (self.scl == SCLabel.CLOUD_HIGH_PROBABILITY.value).sum() / self.valid_px:.1f}%"
        )
        logger.info(f"  SCL Cirrus      : {100 * (self.scl == SCLabel.THIN_CIRRUS.value).sum() / self.valid_px:.1f}%")
        logger.info(f"SCL Unclassified  : {100 * (self.scl == SCLabel.UNCLASSIFIED.value).sum() / self.valid_px:.1f}%")
        logger.info("#" * 100)
        logger.info(f"  Omni Cloud Thick : {100 * (self.omnicloud_argmax == 1).sum() / self.valid_px:.1f}%")
        logger.info(f"  Omni Cloud Thin  : {100 * (self.omnicloud_argmax == 2).sum() / self.valid_px:.1f}%")  # noqa
        logger.info("#" * 100)
        logger.info("#" * 100)

        rgb_image = (
            np.transpose(
                self.img[[bands.index("B04"), bands.index("B03"), bands.index("B02")], :],
                (1, 2, 0),
            )
            / 10000
        )
        # increase max viz value if the image is really bright outside nodata values
        norm_rgb = 0.4
        rgb_image = (255 * (rgb_image / norm_rgb)).clip(0, 255).astype(np.uint8)
        # COMPARING OMNICLOUD VS SCL
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax = cast(np.ndarray, ax)  # for mypy
        ax[0].imshow(rgb_image, interpolation="nearest")
        ax[0].set_title(f"RGB {overlap_with_main_string}", fontsize=25)

        ax[1].imshow(rgb_image, interpolation="nearest")
        ax[1].imshow(
            np.ma.masked_where(self.clouds_omni != 1, self.clouds_omni),
            cmap="spring",
            interpolation="none",
            alpha=0.5,
        )
        ax[1].imshow(
            np.ma.masked_where(self.shadows_omni != 1, self.shadows_omni),
            cmap="autumn",
            interpolation="none",
            alpha=0.5,
        )
        ax[1].imshow(
            np.ma.masked_where(self.scl != 0, self.scl),
            cmap="Wistia",
            interpolation="none",
            alpha=0.35,
        )
        ax[1].set_title(
            f"OmniCloudMask: {self.cloud_omni_perc:.1f}%\n Cloud shadows: {self.cloud_shadow_omni_perc:.1f}%",
            fontsize=25,
        )

        ax[2].imshow(rgb_image, interpolation="nearest")
        ax[2].imshow(
            np.ma.masked_where(self.clouds_combined != 1, self.clouds_combined),
            cmap="spring",
            interpolation="none",
            alpha=0.5,
        )
        ax[2].imshow(
            np.ma.masked_where(self.shadows_combined != 1, self.shadows_combined),
            cmap="autumn",
            interpolation="none",
            alpha=0.5,
        )
        ax[2].imshow(
            np.ma.masked_where(self.scl != 0, self.scl),
            cmap="Wistia",
            interpolation="none",
            alpha=0.35,
        )
        ax[2].set_title(
            f"SCL + Omni combined: {self.cloud_combined_perc:.1f}%\nCloud shadows: "
            f"{self.cloud_shadow_combined_perc:.1f}%",
            fontsize=25,
        )
        plt.tight_layout()
        plt.show()

        f, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax = cast(np.ndarray, ax)  # for mypy
        ax[0].imshow(rgb_image, interpolation="nearest")
        ax[0].imshow(
            np.ma.masked_where(self.scl != 0, self.scl),
            cmap="Wistia",
            interpolation="none",
            alpha=0.35,
        )
        ax[0].imshow(
            np.ma.masked_where(self.clouds_scl != 1, self.clouds_scl),
            cmap="spring",
            interpolation="none",
            alpha=0.5,
        )
        ax[0].imshow(
            np.ma.masked_where(shadows_scl != 1, shadows_scl),
            cmap="autumn",
            interpolation="none",
            alpha=0.5,
        )
        ax[0].set_title(
            f"SCL CloudMask: {self.SCL_clouds_cirrus_thin_thick_perc:.1f}%\n"
            f"Cloud shadows: {cloud_shadow_scl_perc:.1f}%",
            fontsize=25,
        )
        im = ax[1].imshow(self.scl, cmap=CMAP, interpolation="nearest")
        ax[1].set_title("SCL", fontsize=20)
        im.set_clim(-0.5, 11.5)  # Set limits to match label edges
        plt.tight_layout()
        plt.show()
        logger.info(f"Visualizing took {time.time() - start:.1f}s")


class S2DataGeneration(BaseDataGeneration):
    """Data generation pipeline for Sentinel-2 data."""

    WHOLE_SIZE = BAND_RESOLUTIONS["B11"]

    def __init__(
        self,
        sentinel_MGRS: str,
        sentinel_date: datetime.datetime,
        bands: list[str],
        time_delta_days: int,
        nb_reference_ids: int,
        omnicloud_cloud_t: int,
        omnicloud_shadow_t: int,
        **kwargs: Any,
    ) -> None:
        self.sentinel_MGRS = sentinel_MGRS
        self.sentinel_date = sentinel_date
        self.bands = bands
        self.time_delta_days = time_delta_days
        self.nb_reference_ids = nb_reference_ids
        self.omnicloud_cloud_t = omnicloud_cloud_t
        self.omnicloud_shadow_t = omnicloud_shadow_t

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

        self.norm_rgb = 0.4
        # Create config object for base class
        config = DataGenerationConfig(**kwargs)
        super().__init__(config=config)

        self.preload_models: list[Any] = []  # omnicloud models come in here when we need them

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
        return f"{self.sentinel_MGRS}_{self.sentinel_date.isoformat()}"

    @property
    def scene_id(self) -> str:
        """Get unique scene identifier combining MGRS tile and date."""
        date_without_hours = self.sentinel_date.isoformat()
        if "T" in date_without_hours:
            date_without_hours = date_without_hours.split("T")[0]
        return f"{self.sentinel_MGRS}_{date_without_hours}"

    def get_producing_union_polygon_transformed(self, chip_crs: CRS) -> Polygon:
        """Get the producing union polygon transformed to the chip CRS."""
        transformer = Transformer.from_crs("EPSG:4326", chip_crs, always_xy=True)
        return transform(transformer.transform, self.producing_union)

    def get_target_producing_regions_transformed(self, chip_crs: CRS) -> dict[str, Polygon]:
        """Get the target producing regions."""
        transformer = Transformer.from_crs("EPSG:4326", chip_crs, always_xy=True)
        return {
            "Hassi": transform(transformer.transform, self.hassi_poly),
            "Permian": transform(transformer.transform, self.permian_poly),
            "Marcellus": transform(transformer.transform, self.marcellus_poly),
        }

    @timer(phase="download_data", accumulator=MEASUREMENTS, verbose=True, logger=logger)
    def download_data(self) -> dict:  # noqa: PLR0912, PLR0915
        """Download Sentinel-2 data."""
        start_time = self.sentinel_date - datetime.timedelta(days=self.time_delta_days)
        end_time = self.sentinel_date + datetime.timedelta(days=1)  # to include self.sentinel_date

        try_count = 0
        while True:
            if try_count == 5:  # noqa
                break
            try:
                all_potential_items = get_sentinel2_ids_in_timerange(
                    self.sentinel_MGRS,
                    start_time,
                    end_time,
                    self.quality_thresholds["main_tile_cloud"],
                    self.quality_thresholds["main_tile_nodata"][1],
                    abs_client=self.abs_client,
                    whole_size=self.WHOLE_SIZE,
                    producing_union=self.producing_union,
                )
                break
            except Exception as err:
                logger.error("get_sentinel2_ids_in_timerange failed, try again")
                logger.error(traceback.print_exception(None, err, err.__traceback__))
                try_count += 1
        # We need a minimum of 3 (one main tile, two reference tiles)
        assert len(all_potential_items) >= 3, f"not enough items found for tile {self.sentinel_MGRS}"  # noqa

        cloud_shadow_reference_perc_max = 100 * self.quality_thresholds["reference_tile_cloud_shadow"][1]
        s2_id_to_metadata: dict[str, dict] = {}
        reference_items: list[Sentinel2L1CItem] = []
        main_reference_bands: list[npt.NDArray] = []
        main_reference_clouds: list[npt.NDArray] = []
        main_reference_cloud_shadows: list[npt.NDArray] = []
        break_loop = False
        found_main_id = False
        lt_2perc_reference_img_nb = 0
        for item in all_potential_items:
            if found_main_id:  # check for duplicate IDs with different processing times
                found_duplicate = False
                for item_ in [main_item, *reference_items]:  # type: ignore # noqa
                    id_wo_processing_time = "_".join(item.id.split("_")[:-1])
                    if id_wo_processing_time == "_".join(item_.id.split("_")[:-1]):
                        logger.info(f"Reference ID {item.id} is a duplicate with {item_.id}, skipping.")
                        found_duplicate = True
                        break
                if found_duplicate:
                    continue

            start = time.time()
            try:
                item.observation_angle  # noqa
            except Exception as err:
                logger.info(f"item.observation_angle failed, skipping ({item.id})")
                logger.info(traceback.print_exception(None, err, err.__traceback__))
                continue

            try:
                item.prefetch_l1c(self.s3_client, self.abs_client)
            except Exception as err:
                logger.info(f"prefetch_l1c() failed, skipping ({item.id})")
                logger.info(traceback.print_exception(None, err, err.__traceback__))
                continue
            logger.info(f"Transfering data from S3 to ABS took {time.time() - start:.1f}s")

            start = time.time()
            try:
                img = item.get_bands(
                    self.bands,
                    out_height=self.WHOLE_SIZE,
                    out_width=self.WHOLE_SIZE,
                    abs_client=self.abs_client,
                )
            except Exception as err:
                logger.info("get_bands() failed, skipping this ID")
                logger.info(traceback.print_exception(None, err, err.__traceback__))
                continue
            logger.info(f"Loading {len(self.bands)} bands from ABS took {time.time() - start:.1f}s")

            omnicloud_exists = item.check_omnicloud_on_abs(self.abs_client)
            if omnicloud_exists:
                logger.info("OmniCloud exists on ABS --> Download")
                probs = item.get_omnicloud(
                    out_height=self.WHOLE_SIZE,
                    out_width=self.WHOLE_SIZE,
                    abs_client=self.abs_client,
                )
            else:
                logger.info("OmniCloud does NOT exist on ABS --> Predict and Upload")
                probs = self.predict_and_save_omnicloud(img, item)

            # Tile Bands and Omnicloud loaded
            # --> set nodata px, prepare clouds, tile metadata and optionally visualize
            tile = S2Tile(img=img, scl=img[-1].copy(), probs=probs, id=item.id)
            tile.hide_nodata_px()
            tile.prepare_clouds(self.omnicloud_cloud_t, self.omnicloud_shadow_t)
            if found_main_id:  # for reference images, we only care about the overlap with main
                main_scl = main_reference_bands[0][-1]
                tile.hide_main_nodata_px_on_reference_img(main_scl)
            tile.calculate_metadata()
            if self.visualize_tiles:
                tile.visualize(self.bands, main_scl if found_main_id else None)

            s2_id_to_metadata[item.id] = {}
            for k, v in tile.model_dump(
                exclude=[  # type: ignore
                    "img",
                    "scl",
                    "probs",
                    "id",
                    "omnicloud_argmax",
                    "clouds_omni",
                    "shadows_omni",
                    "clouds_scl_thick_thin",
                    "clouds_combined",
                    "shadows_combined",
                    "clouds_scl",
                ]
            ).items():
                s2_id_to_metadata[item.id][k] = v

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
                found_main_id = True
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
            elif item.id in s2_id_to_metadata:
                name = "NOT USED "
            else:  # not considered
                continue
            logger.info(
                f"{name} {item.id} with {s2_id_to_metadata[item.id]['SCL_NO_DATA_perc']:5.1f}% nodata, "
                f"{s2_id_to_metadata[item.id]['cloud_combined_perc']:5.1f}% Clouds, "
                f"{s2_id_to_metadata[item.id]['cloud_shadow_combined_perc']:5.1f}% Cloud Shadows"
            )

        self.s2_id_to_metadata = s2_id_to_metadata
        return {
            "main_reference_bands": main_reference_bands,
            "main_reference_clouds": main_reference_clouds,
            "main_reference_cloud_shadows": main_reference_cloud_shadows,
        }

    def predict_and_save_omnicloud(self, img: npt.NDArray, item: Sentinel2L1CItem) -> npt.NDArray:
        """Predict and save Omnicloud as uint8 scaled to 0-100 with nodata as 255."""
        from omnicloudmask import predict_from_array
        from omnicloudmask.model_utils import load_model_from_weights

        if len(self.preload_models) == 0:
            # Download the model weights from Azure instead of from the official API which crashes easily
            weights_paths = [
                "PM_model_2.2.10_RG_NIR_509_regnety_004.pycls_in1k_PT_state.pth",
                "PM_model_2.2.10_RG_NIR_509_convnextv2_nano.fcmae_ft_in1k_PT_state.pth",
            ]
            for timm_model_name, weights_path in zip(["regnety_004", "convnextv2_nano"], weights_paths, strict=False):
                if not os.path.exists(weights_path):
                    self.fs.download(f"data/omnicloud_model_weight/{weights_path}", ".")
                    logger.info(f"Downloaded {weights_path}")
                self.preload_models.append(
                    load_model_from_weights(
                        model_name=timm_model_name,
                        weights_path=weights_path,
                        device=torch.device("cuda"),
                    )
                )

        start = time.time()
        probs = predict_from_array(
            input_array=img[
                [
                    self.bands.index("B04"),
                    self.bands.index("B03"),
                    self.bands.index("B8A"),
                ]
            ],  # Red=B04, Green=B03 and NIR=B8A
            patch_size=1700,
            patch_overlap=300,
            batch_size=1,
            inference_device=torch.device("cuda"),
            export_confidence=True,
            softmax_output=True,
            custom_models=self.preload_models,
        )
        logger.info(f"Predicting with OmniCloud took {time.time() - start:.1f}s")

        # Save 4 classes probabilities scaled to 0-100 and as uint8 to save space
        # set nan values to 255, outside of normal 0-100 values
        probs[np.isnan(probs)] = OMNICLOUD_NODATA
        # Scale all probabilities to 0-100 --> round them
        probs[probs != OMNICLOUD_NODATA] = np.round(probs[probs != OMNICLOUD_NODATA] * 100, 0)
        # Use the much smaller dtype uint8
        probs = probs.astype(np.uint8)

        with rasterio.open(item.item.assets["B8A"].href) as ds:
            profile = ds.profile
        profile["nodata"] = OMNICLOUD_NODATA
        profile["dtype"] = "uint8"
        # 4 classes = clear view/thick/thin clouds/cloud shadows
        profile["count"] = 4

        omni_local_path = "OmniCloud.tif"
        with rasterio.open(omni_local_path, "w", **profile) as dst:
            dst.write(probs)

        item.transfer_omnicloud_to_abs(omni_local_path, self.abs_client)
        return probs

    def generate_crops(self, data: dict) -> Iterator[dict[str, Any]]:
        """Generate crops from Sentinel-2 data."""
        main_reference_bands = data["main_reference_bands"]
        main_reference_clouds = data["main_reference_clouds"]
        main_reference_cloud_shadows = data["main_reference_cloud_shadows"]

        for row, col in list(itertools.product(range(self.WHOLE_SIZE // self.crop_size), repeat=2)):
            crop_x = col * self.crop_size
            crop_y = row * self.crop_size

            # Crop sentinel-2 tile to crop_size x crop_size
            crops_bands = [
                whole_tile[
                    :,
                    crop_x : crop_x + self.crop_size,
                    crop_y : crop_y + self.crop_size,
                ].copy()
                for whole_tile in main_reference_bands
            ]
            crops_clouds = [
                whole_tile[
                    crop_x : crop_x + self.crop_size,
                    crop_y : crop_y + self.crop_size,
                ].copy()
                for whole_tile in main_reference_clouds
            ]
            crops_cloud_shadows = [
                whole_tile[
                    crop_x : crop_x + self.crop_size,
                    crop_y : crop_y + self.crop_size,
                ].copy()
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
        self, plume_files: npt.NDArray, crops: Iterator[dict[str, Any]], position_by_source: bool = False
    ) -> Iterator[MultiTemporalPlumesDataItem]:
        """Generate synthetic data items with plumes."""
        self.main_item = self.main_and_reference_items[0]

        # helpers for checking overlap with producing areas
        transform = self.main_item.get_raster_meta("B8A")["transform"]
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
                self.non_overlapping_count += 1
                continue

            if self.visualize_crops and np.random.random() > self.visualize_crops_show_frac:
                continue

            self.overlapping_count += 1

            chips = S2ChipSelection(
                crop_bands=crop_data["crops_bands"],
                crop_clouds=crop_data["crops_clouds"],
                crop_cloud_shadows=crop_data["crops_cloud_shadows"],
                main_and_reference_items=self.main_and_reference_items,
                bands=self.bands,
                hapi_data_path=self.hapi_data_path,
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
                    satellite_id=SatelliteID.S2,
                    swir16_band_name="B11",
                    swir22_band_name="B12",
                    rgb_image_main=chips.rgb_image_main,
                    visualize=self.visualize_insertion,
                    tile_id_to_metadata=self.s2_id_to_metadata,
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
        # Calculate bounds in S2 CRS
        x_min, y_min = rasterio.transform.xy(transform, crop_y + self.crop_size, crop_x, offset="center")  # bottom-left
        x_max, y_max = rasterio.transform.xy(transform, crop_y, crop_x + self.crop_size, offset="center")  # top-right

        # Create a polygon in S2 CRS
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
        # print(f"Overlap = {area_km2:.2f} km²")

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
        position_by_source: bool = False,
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
        data_df["num_plumes"] = data_df["plume_files"].apply(len)

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
                datetime.datetime.strptime(k.split("_")[2].split("T")[0], "%Y%m%d").strftime("%Y-%m-%d") for k in x
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
        elif self.plume_type == PlumeType.GAUSSIAN:
            # For Gaussian plumes, verify that each row has the correct number of emission rates
            # and that they are all valid floats
            assert (
                data_df["plume_emissions"].apply(len) == data_df["num_plumes"]
            ).all(), "Each row should have emission rates matching its number of plumes"
            assert (
                data_df["plume_emissions"]
                .apply(lambda x: all(isinstance(e, float) and not np.isnan(e) for e in x))
                .all()
            ), "All emission rates should be valid floats"

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


def get_sentinel2_ids_in_timerange(
    mgrs: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    cloud_coverage_threshold_range: tuple[float, float],
    nodata_limit: float,
    whole_size: int,
    producing_union: Polygon | MultiPolygon | None,
    abs_client: BlobServiceClient,
) -> list[Sentinel2L1CItem]:
    """
    Return all IDs of a given MGRS in a date_query with less than `nodata_limit` nodata.

    Arguments:
    - mgrs: mgrs of tile; like "10TDT"
    - start_time: Start datetime to search from
    - end_time: End datetime to search until
    - cloud_coverage_threshold_range: tuple (lower limit, upper limit) of cloud coverage as a decimal
    - nodata_limit: upper limit of nodata fraction as a decimal
    - whole_size: Width and height size of a S2 tile on 20m resolution
    - producing_union: Producing area to check the amount of overlap with S2 tiles
    """
    pystac_items = query_sentinel2_catalog_for_tile(
        mgrs,
        start_time,
        end_time,
        cloud_cover_range=cloud_coverage_threshold_range,
    )
    main_and_reference_items = [Sentinel2L1CItem(item) for item in pystac_items]

    mask_size = 366  # to approximate the whole tile mask for nodata, we use a smaller size
    mask_kwargs = {"abs_client": abs_client}
    printed_overlap = False
    filtered_items = []
    for item in main_and_reference_items:
        try:
            mask_nodata = item.get_mask([SCLabel.NO_DATA], mask_size, mask_size, **mask_kwargs)
            mean_mask_nodata = np.mean(mask_nodata)
        except Exception as err:
            logger.error(f"item.get_mask() failed, skip this ID {item.id}")
            logger.error(traceback.print_exception(None, err, err.__traceback__))
            continue

        if mean_mask_nodata < nodata_limit:
            logger.info(f"{item.id} with {100 * mean_mask_nodata:6.1f}% nodata --> USE")
            filtered_items.append(item)

            if producing_union is not None and not printed_overlap:
                transform = item.get_raster_meta("B8A")["transform"]
                transformer = Transformer.from_crs(item.crs, "EPSG:4326", always_xy=True)

                # Calculate bounds in S2 CRS
                x_min, y_min = rasterio.transform.xy(transform, whole_size, 0, offset="center")  # bottom-left
                x_max, y_max = rasterio.transform.xy(transform, 0, whole_size, offset="center")  # top-right
                # Create a polygon in S2 CRS
                polygon_tile = box(x_min, y_min, x_max, y_max)

                # Transform the polygon to EPSG:4326
                polygon_tile_4326 = Polygon([transformer.transform(x, y) for x, y in polygon_tile.exterior.coords])
                intersection_with_producing = producing_union.intersection(polygon_tile_4326)
                logger.info(f"Intersection with producing: {intersection_with_producing.area:.2f}")
                printed_overlap = True
        else:
            logger.info(f"{item.id} with {100 * mean_mask_nodata:6.1f}% nodata --> DONT USE")

    return filtered_items
