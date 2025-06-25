"""Data structures and functions to create synthetic Sentinel 2 satellite imagery with simulated methane plumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr
from pydantic import AnyUrl

from src.data.common.sim_plumes import create_simulated_bands
from src.data.landsat_data import LandsatGranuleAccess, LandsatQAValues
from src.data.sentinel2 import SceneClassificationLabel
from src.data.sentinel2_l1c import Sentinel2L1CItem
from src.utils.parameters import (
    PLUME_STRENGTH_MEDIUM_MIN_FRAC_CUTOFF,
    PLUME_STRENGTH_STRONG_MIN_FRAC_CUTOFF,
    PLUME_STRENGTH_WEAK_MIN_FRAC_CUTOFF,
    CropSnapshots,
    SatelliteID,
)


def get_frac(swir16: npt.NDArray, swir16_o: npt.NDArray, swir22: npt.NDArray, swir22_o: npt.NDArray) -> npt.NDArray:
    """
    Compute the fractional reduction in the band ratio.

    Arguments
    ---------
    swir16: band 11 reflectance
    swir16_o: methane-free (aka background) band 11 reflectance
    swir22: band 12 reflectance
    swir22_o: methane-free (aka background) band 12 reflectance

    The band ratio is swir22/swir16 and frac is defined as the fractional change, so
    frac = ((swir22/swir16) - (swir22_o/swir16_o)) / (swir22_o/swir16_o)
         = (swir22/swir22_o) / (swir16/swir16_o) - 1
    """
    return (swir22 / swir22_o) / (swir16 / swir16_o) - 1


@dataclass
class BasePlumesDataItem:
    """Base class for satellite data items with methane plumes."""

    crop_main: npt.NDArray
    # And lets make the target gamma*enhancement so the computer vision doesn't need to understand any physics
    target: npt.NDArray
    plume_files: list[str]
    plume_emissions: list[float]
    bands: list[str] | list[int]
    satellite_id: SatelliteID
    size: int
    crop_x: int
    crop_y: int
    # TODO: would it be better to store original bands like this?
    # orig_bands: (
    #     npt.NDArray
    # )  # Original bands before plume addition. Assumption here is bands are ordered in ascending order (B1, B2, ...)

    def to_dict(self) -> dict:
        """Convert the dataclass instance to a dictionary."""
        result = {}
        # Using vars() to get all attributes including dynamically added ones
        for name, val in vars(self).items():
            if isinstance(val, np.ndarray):
                result[name] = val.tobytes()
            else:
                result[name] = val
        return result


@dataclass
class MultiTemporalPlumesDataItem(BasePlumesDataItem):
    """
    Sentinel-2 specific implementation with multiple temporal snapshots.

    A dataclass for an item in a synthetic dataset, described here: https://git.orbio.earth/orbio/pipeline-v2/-/milestones/13#tab-issues.

    It contains 3 snaphots of an area (a crop from Sentinel 2), where (band 11 and 12 in) the crop_main is modified by
    patching a plume on it, using some methods from the sim_creation stage in https://git.orbio.earth/orbio/pipeline-v2.

    Fields
    ------
    crop_main: numpy array of main snapshot, of shape (b, size, size)
    crop_before: numpy array of "before" snapshot (t - 1), of shape (b, size, size)
    crop_earlier: numpy array of "earlier" snapshot (t - 2), of shape (b, size, size),
        where b is the number of bands saved.
    orig_swir16: original band 11 of the main snapshot
    orig_swir22: original band 12 of the main snapshot
    target: The target values in arbitrary units.
        In the case of FRAC this is computed as (sim_band_ratio - band_ratio) / band_ratio,
        where band_ratio = orig_swir22 / orig_swir16 and sim_band_ratio is
        the same ratio with the simulated bands
    cloud_main: mask of combined clouds of the main snapshot (Combined = Omni cloud OR SCL Thin/Thick clouds)
    cloud_before: mask of combined clouds of the "before" snapshot
    cloud_earlier: mask of combined clouds of the "earlier" snapshot
    cloudshadow_main: mask of omnicloud cloud shadows of the main snapshot
    cloudshadow_before: mask of omnicloud cloud shadows of the "before" snapshot
    cloudshadow_earlier: mask of omnicloud cloud shadows of the "earlier" snapshot
    exclusion_mask_plumes: mask where no plumes were inserted
    main_and_reference_ids: Main ID at idx=0, then before ID at idx=1, then earlier ID at idx=2
    plume_files: list of file paths of the plumes (can be length 0)
    bands: list of saved band names, used as indices in the crop arrays
    size: Integer spatial size of the square crop in pixels (e.g. 128 for 128x128 crops)
    crop_x: X-coordinate (column) of top-left corner of this crop in the original full scene
    crop_y: Y-coordinate (row) of top-left corner of this crop in the original full scene

    how_many_plumes_we_wanted: Number of plumes we wanted to insert in this chip
    how_many_plumes_we_inserted: Number of plumes we actually inserted
    plumes_inserted_idxs: Indices of the plumes we inserted, match with 'plume_files'
    plume_sizes: List of number of methane px inserted for each plume
    frac_abs_sum: Absolute sum of all frac values of target
    min_frac: Minimum frac of target
    plume_category: Based on minimum frac, was this a "weak", "medium" or "strong" plume?

    exclusion_perc: e.g. 33.8% of px were excluded from inserting methane into
    region_overlap: One of ["Hassi", "Marcellus", "Permian", "Other"]
    """

    crop_before: npt.NDArray[np.int16]
    crop_earlier: npt.NDArray[np.int16]
    orig_swir16: npt.NDArray[np.int16]
    orig_swir22: npt.NDArray[np.int16]

    cloud_main: npt.NDArray[np.bool_]
    cloud_before: npt.NDArray[np.bool_]
    cloud_earlier: npt.NDArray[np.bool_]
    cloudshadow_main: npt.NDArray[np.bool_]
    cloudshadow_before: npt.NDArray[np.bool_]
    cloudshadow_earlier: npt.NDArray[np.bool_]
    exclusion_mask_plumes: npt.NDArray[np.bool_]
    main_and_reference_ids: list[str]

    how_many_plumes_we_wanted: int
    how_many_plumes_we_inserted: int
    plumes_inserted_idxs: list[int]
    plume_sizes: list[int]
    frac_abs_sum: float
    min_frac: float
    plume_category: str
    plume_emissions: list[float]

    exclusion_perc: float
    region_overlap: str

    @classmethod
    def get_data_items_from_crops(  # noqa: PLR0913, PLR0912
        cls,
        main_and_reference_items: list[Sentinel2L1CItem] | list[LandsatGranuleAccess],
        bands: list[str],
        hapi_data_path: AnyUrl,
        crops: list[npt.NDArray],
        crops_clouds: list[npt.NDArray],
        crops_cloud_shadows: list[npt.NDArray],
        crop_x: int,
        crop_y: int,
        crop_size: int,
        plume_arrays: list[npt.NDArray],
        plume_files: list[str],
        plume_emissions: list[float],
        exclusion_mask_plumes: npt.NDArray,
        rng: np.random.Generator,
        transformation_params: dict[str, float],
        satellite_id: SatelliteID,
        swir16_band_name: str,
        swir22_band_name: str,
        tile_id_to_metadata: dict[str, dict],
        band_frac_offset: float = 10.0,
        rgb_image_main: npt.NDArray | None = None,
        visualize: bool = False,
        region_overlap: str = "Other",
        position_by_source: bool = False,
    ) -> MultiTemporalPlumesDataItem:
        """
        Create a MultiTemporalPlumesDataItem from a list of nearby Sentinel 2 crops.

        Add simulated plumes to the target crop.
        """
        MAIN_IDX = CropSnapshots.CROP_MAIN
        BEFORE_IDX = CropSnapshots.CROP_BEFORE
        EARLIER_IDX = CropSnapshots.CROP_EARLIER
        index_swir16 = bands.index(swir16_band_name)
        index_swir22 = bands.index(swir22_band_name)

        orig_swir16 = np.array(crops[MAIN_IDX][index_swir16])
        orig_swir22 = np.array(crops[MAIN_IDX][index_swir22])
        offset_swir16 = orig_swir16 + band_frac_offset
        offset_swir22 = orig_swir22 + band_frac_offset

        sim_swir16, sim_swir22, plumes_inserted_idxs = create_simulated_bands(
            main_and_reference_items[MAIN_IDX],
            cropped_swir16=offset_swir16,
            cropped_swir22=offset_swir22,
            hapi_data_path=hapi_data_path,
            plume_arrs=plume_arrays,
            exclusion_mask_plumes=exclusion_mask_plumes,
            rng=rng,
            transformation_params=transformation_params,
            position_by_source=position_by_source,
        )

        # compute FRAC
        reference_frac = get_frac(
            swir16=sim_swir16,
            swir16_o=offset_swir16,
            swir22=sim_swir22,
            swir22_o=offset_swir22,
        )

        # now remove the offset from the simulated bands and clip to zero (otherwise we're giving
        # the neural network information that wouldn't actually be present in a real image)
        np.clip(sim_swir16 - band_frac_offset, a_min=0.0, a_max=None, out=sim_swir16)
        np.clip(sim_swir22 - band_frac_offset, a_min=0.0, a_max=None, out=sim_swir22)
        # and also round to the nearest integer, again so the neural network can't use
        # non-integer values as a way to detect methane
        np.round(sim_swir16, decimals=0, out=sim_swir16)
        np.round(sim_swir22, decimals=0, out=sim_swir22)
        # Modify the main snapshot methane bands
        if visualize:
            ratio_before_insertion = (
                crops[MAIN_IDX][bands.index(swir22_band_name)] / crops[MAIN_IDX][bands.index(swir16_band_name)]
            ).copy()
        crops[MAIN_IDX][index_swir16] = sim_swir16
        crops[MAIN_IDX][index_swir22] = sim_swir22

        if visualize:
            cls.visualize_plumes_insertion(
                ratio_before_insertion,
                crops,
                bands,
                MAIN_IDX,
                rgb_image_main,
                exclusion_mask_plumes,
                reference_frac,
                swir16_band_name,
                swir22_band_name,
            )

        # Set Plume category based on min frac, used especially for validation
        ## With multiple plumes inserted (training), this is inflated down, so only useful
        ## if only one plume was inserted (validation)
        plume_category = "no_plume"
        min_frac = reference_frac.min()
        if min_frac < PLUME_STRENGTH_STRONG_MIN_FRAC_CUTOFF:
            plume_category = "strong"
        elif min_frac < PLUME_STRENGTH_MEDIUM_MIN_FRAC_CUTOFF:
            plume_category = "medium"
        elif min_frac < PLUME_STRENGTH_WEAK_MIN_FRAC_CUTOFF:
            plume_category = "weak"

        exclusion_perc = (
            100 * exclusion_mask_plumes.sum() / (exclusion_mask_plumes.shape[0] * exclusion_mask_plumes.shape[1])
        )

        data_item = cls(
            crop_main=crops[MAIN_IDX].astype(np.int16),
            crop_before=crops[BEFORE_IDX].astype(np.int16),
            crop_earlier=crops[EARLIER_IDX].astype(np.int16),
            orig_swir16=orig_swir16.astype(np.int16),
            orig_swir22=orig_swir22.astype(np.int16),
            target=reference_frac.astype(np.float32),
            cloud_main=crops_clouds[MAIN_IDX].astype(bool),
            cloud_before=crops_clouds[BEFORE_IDX].astype(bool),
            cloud_earlier=crops_clouds[EARLIER_IDX].astype(bool),
            cloudshadow_main=crops_cloud_shadows[MAIN_IDX].astype(bool),
            cloudshadow_before=crops_cloud_shadows[BEFORE_IDX].astype(bool),
            cloudshadow_earlier=crops_cloud_shadows[EARLIER_IDX].astype(bool),
            exclusion_mask_plumes=exclusion_mask_plumes,
            main_and_reference_ids=[
                main_and_reference_items[MAIN_IDX].id,
                main_and_reference_items[BEFORE_IDX].id,
                main_and_reference_items[EARLIER_IDX].id,
            ],
            bands=bands,
            size=crop_size,
            crop_x=crop_x,
            crop_y=crop_y,
            satellite_id=satellite_id,
            # plume metadata
            plume_files=[plume_files[i] for i in plumes_inserted_idxs],
            plume_emissions=[plume_emissions[i] for i in plumes_inserted_idxs],
            how_many_plumes_we_wanted=len(plume_arrays),
            how_many_plumes_we_inserted=len(plumes_inserted_idxs),
            plumes_inserted_idxs=plumes_inserted_idxs,
            plume_sizes=[
                (k > PLUME_STRENGTH_WEAK_MIN_FRAC_CUTOFF).sum()
                for i, k in enumerate(plume_arrays)
                if i in plumes_inserted_idxs
            ],
            frac_abs_sum=np.sum(np.abs(reference_frac)),
            plume_category=plume_category,
            min_frac=min_frac,
            exclusion_perc=exclusion_perc,
            region_overlap=region_overlap,  # Overlap with Hassi/Marcellus/Permian
        )

        # Dynamically add transformation parameters as attributes
        for key, value in transformation_params.items():
            setattr(data_item, key, value)

        # Add chip metadata
        for name, idx in zip(
            ["main", "before", "earlier"],
            [MAIN_IDX, BEFORE_IDX, EARLIER_IDX],
            strict=False,
        ):
            if satellite_id == SatelliteID.S2:
                valid_px = (crops[MAIN_IDX][-1] != 0).sum()

                # For each SCL value, calculate percentage only on valid pixels
                for cls_idx, class_name in enumerate(SceneClassificationLabel):
                    setattr(
                        data_item,
                        f"chip_SCL_{class_name.name}_perc_{name}",
                        100 * (crops[idx][-1] == cls_idx).sum() / valid_px,
                    )
                setattr(
                    data_item,
                    f"chip_SCL_NO_DATA_perc_{name}",
                    100 * (crops[idx][-1] == 0).sum() / (crops[idx][-1].shape[0] * crops[idx][-1].shape[1]),
                )
            elif satellite_id == SatelliteID.LANDSAT:
                # Calculate nodata px for current temporal snapshot
                # Again, at this point FILL in QA represents both the original FILL values and the pixels where
                # any reflectance/brightness band was 0 (set in LandsatTile.hide_nodata_px)
                nodata_mask = LandsatGranuleAccess.get_mask_from_qa_pixel([LandsatQAValues.FILL], crops[idx][-1])
                total_px = float(crops[idx][-1].size)

                # Set nodata percentage for current temporal snapshot
                setattr(
                    data_item,
                    f"chip_QA_NO_DATA_perc_{name}",
                    100 * nodata_mask.sum() / total_px,
                )

                # Calculate QA percentages for current temporal snapshot relative to main's valid pixels
                nodata_mask_main = LandsatGranuleAccess.get_mask_from_qa_pixel(
                    [LandsatQAValues.FILL], crops[MAIN_IDX][-1]
                )
                valid_px = float((~nodata_mask_main).sum())

                metadata_to_write = [
                    LandsatQAValues.SNOW,
                    LandsatQAValues.WATER,
                ]
                for qa_value in metadata_to_write:
                    qa_mask = LandsatGranuleAccess.get_mask_from_qa_pixel([qa_value], crops[idx][-1])
                    # Only count QA values on main's valid pixels
                    valid_qa_mask = qa_mask & (~nodata_mask_main)
                    setattr(
                        data_item,
                        f"chip_QA_{qa_value.name}_perc_{name}",
                        100 * valid_qa_mask.sum() / valid_px,
                    )

            # Cloud and shadow percentages (these masks have already been processed to account for nodata)
            # Same calculation for both S2 and Landsat
            setattr(
                data_item,
                f"chip_cloud_combined_perc_{name}",
                100 * (crops_clouds[idx] == 1).sum() / valid_px,
            )
            setattr(
                data_item,
                f"chip_cloud_shadow_combined_perc_{name}",
                100 * (crops_cloud_shadows[idx] == 1).sum() / valid_px,
            )

        # Add tile metadata
        for name, id_ in zip(
            ["main", "before", "earlier"],
            [
                main_and_reference_items[MAIN_IDX].id,
                main_and_reference_items[BEFORE_IDX].id,
                main_and_reference_items[EARLIER_IDX].id,
            ],
            strict=False,
        ):
            for key, value in tile_id_to_metadata[id_].items():
                setattr(data_item, f"tile_{key}_{name}", value)

        return data_item

    @staticmethod
    def visualize_plumes_insertion(  # (too-many-arguments)
        ratio_before_insertion: npt.NDArray,
        crops: list[npt.NDArray],
        bands: list[str],
        MAIN_IDX: int,
        rgb_image_main: npt.NDArray | None,
        exclusion_mask_plumes: npt.NDArray,
        reference_frac: npt.NDArray,
        swir16_band_name: str,
        swir22_band_name: str,
    ) -> None:
        """Visualizes the ratio before and after plume insertion."""
        index_swir16 = bands.index(swir16_band_name)
        index_swir22 = bands.index(swir22_band_name)
        ratio_vmin, ratio_vmax = 0.4, 0.7
        ratio_after_insertion = crops[MAIN_IDX][index_swir22] / crops[MAIN_IDX][index_swir16]

        f, ax = plt.subplots(1, 4, figsize=(30, 7))
        ax = cast(np.ndarray, ax)  # for mypy

        ax[0].imshow(rgb_image_main, interpolation="nearest")
        ax[0].imshow(
            np.ma.masked_where(exclusion_mask_plumes != 1, exclusion_mask_plumes),
            cmap="spring",
            interpolation="none",
            alpha=0.2,
        )
        exclusion_perc = (
            100 * exclusion_mask_plumes.sum() / (exclusion_mask_plumes.shape[0] * exclusion_mask_plumes.shape[1])
        )
        ax[0].set_title(f"Exclusion mask: {exclusion_perc:.1f}%", fontsize=25)

        ax[1].imshow(rgb_image_main, interpolation="nearest")
        ax[1].imshow(
            ratio_before_insertion,
            vmin=ratio_vmin,
            vmax=ratio_vmax,
            interpolation="nearest",
        )
        ax[1].set_title(
            f"Before Insertion\nMin {np.nanmin(ratio_before_insertion):.3f}, Max "
            f"{np.nanmax(ratio_before_insertion):.3f}, Mean {np.nanmean(ratio_before_insertion):.3f}",
            fontsize=12,
        )
        ax[2].imshow(
            ratio_after_insertion,
            vmin=ratio_vmin,
            vmax=ratio_vmax,
            interpolation="nearest",
        )
        ax[2].set_title(
            f"After Insertion\nMin {np.nanmin(ratio_after_insertion):.3f}, Max "
            f"{np.nanmax(ratio_after_insertion):.3f}, Mean {np.nanmean(ratio_after_insertion):.3f}",
            fontsize=12,
        )

        ax[3].imshow(
            reference_frac.astype(np.float32),
            vmin=-0.04,
            vmax=0.04,
            cmap="RdBu",
            interpolation="nearest",
        )
        plt.tight_layout()
        plt.show()


@dataclass
class MonoTemporalPlumesDataItem(BasePlumesDataItem):
    """A data item containing synthetic plume data for a single time point.

    For EMIT data, the target variable represents the effective methane column density,
    calculated as the product of:
    - The plume concentration (enhancement) in mol/m²
    - The path length factor (gamma), computed from viewing geometry as:
        1/cos(θ_sensor) + 1/cos(θ_sun)
      where θ_sensor is the sensor zenith angle and θ_sun is the solar zenith angle

    This product accounts for the actual path length that light travels through the
    plume based on the viewing geometry. For example, if the sensor or sun are at
    oblique angles, the light path through the plume is longer, resulting in
    stronger absorption.

    Fields
    ------
    modified_crop: Radiance data with synthetic plumes applied, representing the modified spectral radiance after
                  simulating methane absorption. Shape is (bands, height, width).
    target: The effective methane column density in mol/m², calculated as enhancement * gamma. The enhancement
           represents the plume concentration, while gamma accounts for the viewing geometry path length effects.
    mask: Boolean mask indicating invalid/masked pixels (e.g. clouds, shadows, bad data). True indicates masked pixels
          where plumes will not be placed.
    emit_id: Identifier string for the EMIT scene, following format: EMIT_L1B_RAD_001_YYYYMMDDTHHMMSS_ORBITID_SCENEID
    plume_files: List of file paths to the plume templates used to generate the synthetic methane plumes in this crop
    bands: List of strings identifying the spectral bands included in the radiance data.
    size: Integer spatial size of the square crop in pixels (e.g. 128 for 128x128 crops)
    crop_x: X-coordinate (column) of top-left corner of this crop in the original full EMIT scene
    crop_y: Y-coordinate (row) of top-left corner of this crop in the original full EMIT scene
    main_cloud_ratio: Float between 0-1 indicating the fraction of pixels covered by clouds in this crop
    region_overlap: One of ["Hassi", "Colorado", "Permian", "Other"]
    """

    mask_main: npt.NDArray[np.bool_]
    granule_item_id: str
    main_cloud_ratio: float
    region_overlap: str
    plume_emissions: list[float]

    @classmethod
    def create_data_item(  # noqa: PLR0913 (too-many-arguments)
        cls,
        modified_crop: xr.DataArray,
        target: npt.NDArray,
        mask: xr.DataArray,
        granule_id: str,
        plume_files: list[str],
        plume_emissions: list[float],
        bands: list[str],
        size: int,
        crop_x: int,
        crop_y: int,
        main_cloud_ratio: float,
        transformation_params: dict[str, float],
        region_overlap: str = "Other",
    ) -> MonoTemporalPlumesDataItem:
        """Create a MonoTemporalPlumesDataItem from a crop with simulated plumes."""
        data_item = cls(
            crop_main=modified_crop.transpose("bands", "downtrack", "crosstrack").values,
            target=target,
            mask_main=mask.values,  # type: ignore
            plume_files=plume_files,
            plume_emissions=plume_emissions,
            bands=bands,
            size=size,
            crop_x=crop_x,
            crop_y=crop_y,
            main_cloud_ratio=main_cloud_ratio,
            granule_item_id=granule_id,
            satellite_id=SatelliteID.EMIT,
            region_overlap=region_overlap,
        )

        # Add transformation parameters as attributes
        for key, value in transformation_params.items():
            setattr(data_item, key, value)

        return data_item
