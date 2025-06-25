"""Add simulated plumes over Satelliteimages."""

from __future__ import annotations

import math
import tempfile
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import numpy.typing as npt
import rasterio
import torch
from affine import Affine
from azure.storage.blob import BlobServiceClient
from pydantic import AnyUrl, BaseModel, Field, TypeAdapter
from rasterio.warp import Resampling, reproject
from scipy.ndimage import gaussian_filter, rotate
from torchvision.transforms.v2 import InterpolationMode, Resize

from src.azure_wrap.ml_client_utils import (
    download_blob_directly,
)
from src.data.landsat_data import LandsatGranuleAccess
from src.data.sentinel2 import Sentinel2Item
from src.data.sentinel2_l1c import Sentinel2L1CItem
from src.utils.radtran_utils import RadTranLookupTable
from src.utils.utils import setup_logging

logger = setup_logging()

####################################################
################ COMMON FUNCTIONS ##################
####################################################


class PlumeType(str, Enum):
    """Types of plume datasets available."""

    RECYCLED = "recycled"
    CARBONMAPPER = "carbonmapper"
    NO_METHANE = "no_methane"
    GAUSSIAN = "gaussian"

    def __str__(self) -> str:
        """Get string representation."""
        return self.value

    @classmethod
    def list(cls) -> list[str]:
        """Return a list of the valid values for PlumeType."""
        return list(map(lambda c: c.value, cls))  # type: ignore[attr-defined]


def build_plume_transform_params(
    plume_type: str | PlumeType,
    psf_sigma: float,
    target_spatial_resolution: int,
    **kwargs: Any,
) -> PlumeTransformParams:
    """Build the correct plume transform params model for the plume type."""
    # Let's just let pydantic work its magic rather than using if/else or match here
    adapter = TypeAdapter(AbstractPlumeTransformParamsModel)  # type: ignore
    return adapter.validate_python(
        dict(
            plume_type=plume_type,
            psf_sigma=psf_sigma,
            target_spatial_resolution=target_spatial_resolution,
            **kwargs,
        )
    )


class PlumeTransformParams(ABC):
    """Abstract class for a PlumeDataset.

    Parameters
    ----------
    target_spatial_resolution: Target pixel resolution of the sensor used for data generation in meters
        (example 20 = sentinel-2).
    plumes_in_ppm_m: if plumes are in ppm_m we need to convert to mol/m².
    psf_sigma: best fit gaussian sigma for the target sensor point spread function.
    upscale: whether to upscale the plume
    transform: whether to apply transformations (rotation) to the plume
    """

    target_spatial_resolution: int
    plumes_in_ppm_m: bool
    psf_sigma: float
    upscale: bool
    transform: bool
    plume_type: PlumeType


# TODO: ask why we need separate RecycledPlumeTransformParams & AVIRISPlumeTransformParams
# if they don't hold the actual params
class RecycledPlumeTransformParams(PlumeTransformParams, BaseModel):
    """The transform parameters for recycled S2 plumes."""

    psf_sigma: float
    target_spatial_resolution: int
    plumes_in_ppm_m: bool = False
    upscale: bool = False
    transform: bool = False  # will not rotate plumes
    plume_type: Literal[PlumeType.RECYCLED] = PlumeType.RECYCLED


class AVIRISPlumeTransformParams(PlumeTransformParams, BaseModel):
    """The transform parameters for AVIRIS S2 plumes."""

    psf_sigma: float
    target_spatial_resolution: int
    plumes_in_ppm_m: bool = True
    upscale: bool = True
    transform: bool = True
    plume_type: Literal[PlumeType.CARBONMAPPER] = PlumeType.CARBONMAPPER


class GaussianPlumeTransformParams(PlumeTransformParams, BaseModel):
    """The transform parameters for Gaussian plumes."""

    psf_sigma: float
    target_spatial_resolution: int
    plumes_in_ppm_m: bool = False
    upscale: bool = True
    transform: bool = True
    plume_type: Literal[PlumeType.GAUSSIAN] = PlumeType.GAUSSIAN


# Set up this abstract class so pydantic can parse to the correct plume transform params class.
AbstractPlumeTransformParamsModel = Annotated[
    RecycledPlumeTransformParams | AVIRISPlumeTransformParams | GaussianPlumeTransformParams,
    Field(discriminator="plume_type"),
]


###############################################################
# NOTE: Convert sim methods to assume mol/m² for consistency!!!
###############################################################

DENSITY_AIR_GROUND_LEVEL = 1.2250  # kg/m³
M_AIR = 0.0289644  # kg/mol
UNIT_MULTIPLIER = 1e-6  # 1/ppm


def convert_ppmm_to_mol_m2(retrieval_ppm_m: np.ndarray) -> np.ndarray:
    """Convert raster from parts per million meter (ppmm) to mols per meter^2 (mol_m2).

    Calculations explained in this issue: https://git.orbio.earth/orbio/orbio/-/issues/812#note_68616
    """
    ppm_m_to_mol_factor = UNIT_MULTIPLIER * DENSITY_AIR_GROUND_LEVEL / M_AIR  # mol/(m³ * ppm)

    retrieval_mol_m2 = retrieval_ppm_m * ppm_m_to_mol_factor

    return retrieval_mol_m2


def load_and_transform_plume_arr(
    plume_tiff_path: str | Path,
    blob_service_client: BlobServiceClient,
    container_name: str,
    plume_transform_params: PlumeTransformParams,
    rotation_degrees: int,
    concentration_rescale_value: float,
) -> npt.NDArray:
    """Load a plume raster, randomly rotate the plume and resample the plume to the resolution of the target satellite.

    Notes
    -----
    1) Sim plumes are either stored in ppm/m or in mol/m2, for our sim calculations, we want the values in mol/m2
        so there is an additional transformation if the plumes are in ppm/m.

    2) Satellites use gaussian point spread values to assign pixel values, this code emulates this logic by using
        a gaussian filter with sigma = best fit to the satellite point spread function (s2 = 10.8). We then divide the
        `psf_sigma` by the spatial resolution of the plume to resample the plume values to the target sensor resolution
        as accuratly as possible.

    Parameters
    ----------
    plume_tiff_path: Path to the input raster file.
    blob_service_client: Client to access the Azure Blob Storage
    container_name: name of the Blob Storage container
    plume_transform_params: transformation parameters for the plume (e.g. recycled or aviris)
    rotation_degrees: rotation angle in degrees.  Low-resolution plumes, should be rotated by multiples of 90 degrees.
    concentration_rescale_value: Rescale methane concentration values by this factor (this is used to
        match the range of plume emission rates to a target range)

    Return
    ------
        Transformed plume array in mol/m2.

    """
    local_tiff_path, temp_dir = get_local_path_for_tiff(
        str(plume_tiff_path),
        blob_service_client,
        container_name,
        plume_transform_params.plume_type,
    )

    with rasterio.open(local_tiff_path) as src:
        # Read the input band and replace NaN values with 0

        # the retrievals are stored in mol/m² in the database,
        # according to https://git.orbio.earth/orbio/wiki/-/wikis/Products/Pipeline/Pipeline-Docs/Pipeline-Stages-Overview
        plume_retrieval = src.read(1)

    # Our plumes must not contain NaNs, and cannot realistically contain negative values.
    # Negative values on the edge of plumes were occuring for some AVIRIS plumes as an
    # artificat of one of Carbon Mapper's retrieval algorithms.
    plume_retrieval = np.nan_to_num(plume_retrieval, nan=0).clip(min=0)

    if plume_transform_params.plumes_in_ppm_m:
        plume_retrieval = convert_ppmm_to_mol_m2(plume_retrieval)

    plume_retrieval *= concentration_rescale_value

    if plume_transform_params.upscale:
        plume_retrieval = upscale_rotate_plume(
            plume_retrieval,
            src.crs,
            src.transform,
            plume_transform_params,
            rotation_degrees,
        )

    assert not np.isnan(plume_retrieval).any()

    return plume_retrieval


def upscale_rotate_plume(
    band: np.ndarray,
    crs: rasterio.crs.CRS,
    transform: Affine,
    plume_transform_params: PlumeTransformParams,
    rotation_degrees: int,
) -> np.ndarray:
    """Upscale and randomly rotate the plume to the target sensor resolution."""
    # rescale methane concentration values by target value
    plume_resolution = transform.a

    # Pad the plume with 1 target sensor pixel on each side to avoid information loss
    expand_factor = math.ceil(plume_transform_params.target_spatial_resolution / plume_resolution)

    band_padded = np.pad(
        band,
        ((expand_factor, expand_factor), (expand_factor, expand_factor)),
        mode="constant",
        constant_values=0.0,
    )

    # Calculate scale factor based on target resolution
    scale_factor = plume_transform_params.target_spatial_resolution / plume_resolution  # Assuming square pixels

    # Calculate the max dim so we can create a square transformation
    # bounding box so we can rotate the plume as we like
    height, width = band_padded.shape
    max_dim = max(height, width)

    # pad the new bounding box by rescale * 2 (this makes sure we keep the rotated plume in bounds)
    scaled_dim = math.ceil(max_dim / scale_factor) * 2  # Adjust dimensions

    # Define affine transformations
    scaling_transform = Affine.scale(scale_factor)
    rotation_transform = Affine.rotation(rotation_degrees) if plume_transform_params.transform else Affine.rotation(0.0)

    # TODO: @zani, if we scale first then it's being translated still by the nonscaled version?
    translation_transform = Affine.translation(-max_dim, -max_dim)

    # Combine right to left transforms: scaling -> translation -> rotation
    dst_transform = transform * rotation_transform * translation_transform * scaling_transform

    # Apply Gaussian smoothing before resampling to emulate the point spread function
    sigma = plume_transform_params.psf_sigma / plume_resolution
    band_smoothed = gaussian_filter(band_padded, sigma=sigma)

    # Create an empty array for the output
    transformed_band = np.zeros((scaled_dim, scaled_dim), dtype=band_smoothed.dtype)

    # Our EMIT plumes do not have a geographic CRS as they are available in EPSG:4326 and we don't want
    # to introduce any distortions by reporjecting them, so we simply drop the CRS information and set
    # the Affine transformation to indicate the pixels are 60m. A null CRS will cause an error in the below
    # project function, but as we are not changing the CRS we simply pass in a fake one, here web psuedo mercator.
    crs = crs or "EPSG:3857"

    # Reproject the raster band to the new transformation
    reproject(
        source=band_smoothed,
        destination=transformed_band,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=crs,
        resampling=Resampling.bilinear,
    )

    if np.all(transformed_band == 0):
        logger.warning("After reprojecting, the plume has become all 0s. We will filter this plume out")
        return transformed_band

    plume = trim_zero_padding_from_array(transformed_band)

    assert not np.isnan(plume).any()

    return plume


def trim_zero_padding_from_array(array: np.ndarray) -> np.ndarray:
    """Trim zero padding around a simulated plume array."""
    # Find indices of non-zero elements along each axis
    non_zero_rows = np.where(np.any(array, axis=1))[0]
    non_zero_cols = np.where(np.any(array, axis=0))[0]

    # Slice the array using the found indices to trim zero padding
    trimmed_plume = array[
        non_zero_rows.min() : non_zero_rows.max() + 1,
        non_zero_cols.min() : non_zero_cols.max() + 1,
    ]

    return trimmed_plume


def check_for_invalid_plume_placement(
    mask: np.ndarray,
    pos_x: int,
    pos_y: int,
    flipped_plume_x: int,
    flipped_plume_y: int,
    max_px: int,
) -> bool:
    """Check for valid plume placement."""
    return (
        np.sum(
            mask[
                pos_x : pos_x + flipped_plume_x,
                pos_y : pos_y + flipped_plume_y,
            ]
        )
        > max_px
    )


def check_plume_too_large_for_crop(tile_x: int, tile_y: int, plume_x: int, plume_y: int) -> bool:
    """Check if the plume fits in the crop.

    Returns
    -------
        True if the plume is too large for the crop, False otherwise.
    """
    return plume_x > tile_x or plume_y > tile_y


def randomly_position_sim_plume_by_source(  # noqa: PLR0915
    sim_plumes: list[tuple[np.ndarray, np.ndarray]],
    tile_band: np.ndarray,
    exclusion_mask_plumes: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Flip the simulated plume array and mask randomly and inserts the plume in a random position in the tile grid.

    For simplicity, the "source" of the plume is considered as the pixel with the highest concentration.

    The source is randomly positioned, and then the plume is allowed to extend beyond the image bounds,
    in which case the plume is trimmed.

    We allow the plume to overlap with the exclusion areas, as long as the source (brightest pixel)
    is not in an excluded area.


    Parameters
    ----------
    sim_plumes: A list of tuples of 1 or more plumes where the first tuple item is the plume retrieval and the
        second is the plume mask
    tile_band: the b12 band of the scene the simulated methane will be inserted into
    exclusion_mask_plumes: binary mask indicating cloudy pixels (1 = cloudy, 0 = clear)
    rng: The numpy random number generator to use

    Returns
    -------
        Two-dimensional array, the same size as the target tile, containing the simulated methane plume
    """
    assert tile_band.shape == exclusion_mask_plumes.shape, "Tile band and exclusion mask must have the same shape"
    tile_x, tile_y = tile_band.shape
    sim_tile_grid = np.zeros((tile_x, tile_y), dtype=np.float32)
    mask_tile_grid = np.zeros((tile_x, tile_y), dtype=np.bool_)

    valid_positions = np.where((exclusion_mask_plumes == 0) & (tile_band >= 0))

    if len(valid_positions[0]) == 0:
        logger.warning("No valid positions found for plume source placement.")
        return sim_tile_grid, mask_tile_grid, []

    plumes_inserted_idxs = []
    for plume_idx, plume in enumerate(sim_plumes):
        for i in range(1000):
            sim_retrieval = trim_zero_padding_from_array(plume[0])
            sim_mask = trim_zero_padding_from_array(plume[1])
            assert sim_retrieval.shape == sim_mask.shape

            mirror_flip_direction = rng.integers(low=0, high=2)

            # Apply a mirror flip, 0 = no flip, 1 = horizontal flip
            if mirror_flip_direction == 0:
                sim_retrieval = np.fliplr(sim_retrieval)
                sim_mask = np.fliplr(sim_mask)

            plume_x, plume_y = sim_retrieval.shape

            # Sample from indices within buffer of the image boundaries that are not masked

            # Randomly select one of the valid positions
            idx = rng.integers(0, len(valid_positions[0]))
            pos_source_x = valid_positions[0][idx]
            pos_source_y = valid_positions[1][idx]

            plume_source_x, plume_source_y = np.unravel_index(np.argmax(sim_retrieval), sim_retrieval.shape)
            pos_x = pos_source_x - plume_source_x
            pos_y = pos_source_y - plume_source_y

            # Clip plume to fit within tile boundaries
            if pos_x < 0:
                sim_retrieval = sim_retrieval[-pos_x:, :]
                sim_mask = sim_mask[-pos_x:, :]
                plume_x = sim_retrieval.shape[0]
                pos_x = 0
            if pos_y < 0:
                sim_retrieval = sim_retrieval[:, -pos_y:]
                sim_mask = sim_mask[:, -pos_y:]
                plume_y = sim_retrieval.shape[1]
                pos_y = 0
            if pos_x + plume_x > tile_x:
                sim_retrieval = sim_retrieval[: tile_x - pos_x, :]
                sim_mask = sim_mask[: tile_x - pos_x, :]
                plume_x = sim_retrieval.shape[0]
            if pos_y + plume_y > tile_y:
                sim_retrieval = sim_retrieval[:, : tile_y - pos_y]
                sim_mask = sim_mask[:, : tile_y - pos_y]
                plume_y = sim_retrieval.shape[1]
            exclusion_mask_plumes_crop = exclusion_mask_plumes[pos_x : pos_x + plume_x, pos_y : pos_y + plume_y]
            methane_proportion_visible = sim_retrieval[exclusion_mask_plumes_crop == 0].sum() / plume[0].sum()
            minimum_proportion = 0.8
            if methane_proportion_visible < minimum_proportion:
                logger.warning(
                    f"{i} - Proportion of visible methane is too low "
                    f"({methane_proportion_visible} < {minimum_proportion}): retrying"
                )
                continue
            else:
                # we've found a valid plume placement, break out of the inner loop
                break
        else:
            logger.warning("Plume placement failed after 1000 attempts")
            continue  # continue outer loop - do not attempt to insert a plume

        sim_tile_grid[pos_x : pos_x + plume_x, pos_y : pos_y + plume_y] += sim_retrieval
        mask_tile_grid[pos_x : pos_x + plume_x, pos_y : pos_y + plume_y] |= sim_mask
        plumes_inserted_idxs.append(plume_idx)

    logger.info(f"Inserted {len(plumes_inserted_idxs)}/{len(sim_plumes)} plumes")
    return sim_tile_grid, mask_tile_grid, plumes_inserted_idxs


def randomly_position_sim_plume(  # noqa
    sim_plumes: list[tuple[np.ndarray, np.ndarray]],
    tile_band: np.ndarray,
    exclusion_mask_plumes: np.ndarray,
    rng: np.random.Generator,
    random_rotate: bool = False,
    allow_overlap: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Flip the simulated plume array and mask randomly and inserts the plume in a random position in the tile grid.

    The code uses the cloud mask to ensure the placement of the plume is valid.

    Parameters
    ----------
    sim_plumes: A list of tuples of 1 or more plumes where the first tuple item is the plume retrieval and the
        second is the plume mask
    tile_band: the b12 band of the scene the simulated methane will be inserted into
    exclusion_mask_plumes: binary mask indicating cloudy pixels
    rng: The numpy random number generator to use
    random_rotate: If True, randomly rotate
    allow_overlap: If True, allow for 5% plume size overlap with exclusion areas, increasing to 20% if we cant place it

    Returns
    -------
        Two-dimensional array, the same size as the target tile, containing the simulated methane plume
    """
    tile_x, tile_y = tile_band.shape
    sim_tile_grid = np.zeros((tile_x, tile_y), dtype=np.float32)
    mask_tile_grid = np.zeros((tile_x, tile_y), dtype=np.float32)

    plumes_inserted_idxs = []
    for plume_idx, plume in enumerate(sim_plumes):
        sim_retrieval = trim_zero_padding_from_array(plume[0])
        sim_mask = trim_zero_padding_from_array(plume[1])

        px_for_placement = (exclusion_mask_plumes == 0).sum()
        px_sim_plume = sim_mask.shape[0] * sim_mask.shape[1]
        # Allow optionally for 7.5% plume size overlap with
        px_plume = sim_mask.sum()
        acceptable_overlap_px_count = px_plume * 0.075 if allow_overlap else 0

        if px_for_placement <= px_sim_plume:
            continue

        assert sim_retrieval.shape == sim_mask.shape

        # Try to place plume
        for k in range(15000):
            if k == 2999 and allow_overlap:  # noqa
                acceptable_overlap_px_count = px_plume * 0.125  # for hard to place plumes, allow 12.5% overlap
            if k == 5999 and allow_overlap:  # noqa
                acceptable_overlap_px_count = px_plume * 0.185
            if k == 8999 and allow_overlap:  # noqa
                acceptable_overlap_px_count = px_plume * 0.255
            if k == 11999 and allow_overlap:  # noqa
                acceptable_overlap_px_count = px_plume * 0.335

            mirror_flip_direction = rng.integers(low=0, high=2)

            # Apply a mirror flip, 0 = no flip, 1 = horizontal flip
            if mirror_flip_direction == 0:
                sim_retrieval = np.fliplr(sim_retrieval)
                sim_mask = np.fliplr(sim_mask)

            if random_rotate:
                angle = rng.uniform(0, 360)
                # reshape=False keeps the original dimensions
                sim_retrieval = rotate(sim_retrieval, angle, reshape=False, mode="constant")
                sim_mask = rotate(sim_mask, angle, reshape=False, mode="constant")

            plume_x, plume_y = sim_retrieval.shape

            if check_plume_too_large_for_crop(tile_x, tile_y, plume_x, plume_y):
                # TODO: how can we track this in MLFlow so it's not lost in the logs?
                logger.error(f"Plume ({plume_x}, {plume_y}) is too large for tile ({tile_x}, {tile_y})")
                break  # will not fit no matter how many times we try

            # Calculate random position so that the rotated plume fits in the tile
            pos_x = rng.integers(low=0, high=tile_x - plume_x + 1)
            pos_y = rng.integers(low=0, high=tile_y - plume_y + 1)

            # Check if plume placement overlaps existing plume
            if check_for_invalid_plume_placement(mask_tile_grid, pos_x, pos_y, plume_x, plume_y, 0):
                continue

            # Check if plume placement overlaps clouds
            if check_for_invalid_plume_placement(
                exclusion_mask_plumes, pos_x, pos_y, plume_x, plume_y, acceptable_overlap_px_count
            ):
                continue

            # Check if plume placement overlaps an area with missing swath
            if check_for_invalid_plume_placement(
                tile_band == 0, pos_x, pos_y, plume_x, plume_y, acceptable_overlap_px_count
            ):
                continue

            # Plume placement is valid, insert plume and break loop
            sim_tile_grid[pos_x : pos_x + plume_x, pos_y : pos_y + plume_y] = sim_retrieval
            mask_tile_grid[pos_x : pos_x + plume_x, pos_y : pos_y + plume_y] = sim_mask
            plumes_inserted_idxs.append(plume_idx)
            break
    return sim_tile_grid, mask_tile_grid, plumes_inserted_idxs


def transform_and_position_plumes(
    plume_arrs: list[np.ndarray],
    tile_band: npt.NDArray,
    exclusion_mask_plumes: npt.NDArray,
    rng: np.random.Generator,
    transformation_params: dict[str, float] | None = None,
    position_by_source: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Generate plume enhancement for a single crop.

    Parameters
    ----------
    plume_arrs : list[np.ndarray]
        List of plume arrays to transform and position
    tile_band : npt.NDArray
        The target band to insert plumes into
    exclusion_mask_plumes : npt.NDArray
        Binary mask indicating areas where plumes cannot be placed (e.g. clouds)
    rng : np.random.Generator
        Random number generator for reproducible results
    transformation_params : dict[str, float] | None, optional
        Parameters to transform the plumes, including:
        - modulate: Factor to multiply plume values by (default: 1.0)
        - resize: Factor to resize plume by (default: 1.0)
    position_by_source : bool, default=False
        If True, positions plumes by their source point (highest concentration pixel).
        This allows plumes to extend beyond image bounds and be trimmed, creating more
        realistic plume placements where the source is visible but the plume may be
        partially outside the image. If False, positions the entire plume within bounds.
        Note: position_by_source allows an inserted plume overlap with exclusion areas
        and with other plumes.
    """
    if transformation_params is None:
        transformation_params = {}

    # Get transformation parameters, default to 1.0 (no change in array) if parameters are not specified
    modulate_factor = transformation_params.get("modulate", 1.0)
    resize_factor = transformation_params.get("resize", 1.0)

    # Apply modulation and resizing to the plume arrays
    modulated_plumes = [arr * modulate_factor for arr in plume_arrs]
    clipped_and_masked_plumes = [(arr.clip(min=0.0), (arr > 0)) for arr in modulated_plumes]
    sim_plumes = [resize(plume, resize_factor) for plume in clipped_and_masked_plumes]

    # Generate enhancement
    if position_by_source:
        methane_enhancement_molperm2, methane_enhancement_mask, plumes_inserted_idxs = (
            randomly_position_sim_plume_by_source(
                sim_plumes=sim_plumes,
                tile_band=tile_band,
                exclusion_mask_plumes=exclusion_mask_plumes,
                rng=rng,
            )
        )
    else:
        methane_enhancement_molperm2, methane_enhancement_mask, plumes_inserted_idxs = randomly_position_sim_plume(
            sim_plumes=sim_plumes,
            tile_band=tile_band,
            exclusion_mask_plumes=exclusion_mask_plumes,
            rng=rng,
        )

    # TODO: what happens if the plumes aren't able to be placed in the tile? Currently we return all 0s. For the S2
    # pipeline this did not result in any errors — i.e we essentially return the original bands as the sim bands and in
    # the parquet file even though we specify plume files for the crop it could be the plume was not successfully placed
    # in the crop. For EMIT however, we will get an error if the methane enhancement is all 0s (see add_plumes method).
    # TODO: log that plume couldn't be inserted and continue to next plume.  Make sure to not count plume as inserted.
    return methane_enhancement_molperm2, methane_enhancement_mask, plumes_inserted_idxs


def get_local_path_for_tiff(
    tiff_path: str,
    blob_service_client: BlobServiceClient,
    container_name: str,
    plume_type: PlumeType,
) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Return local filepath to plume tiff, if needed download the tiff from Azure first."""
    if tiff_path.startswith("azureml://"):
        tiff_name = tiff_path.split("/")[-1]

        match plume_type:
            # TODO: prefix of blob name is hard coded
            # FIXME: regenerate the recycled plumes catalogue with the whole blob storage path
            # then we can find the relative path to the tiff in the blob store and remove the hardcoded prefixes
            case PlumeType.RECYCLED:
                tiff_blob_name = f"orbio-data-exports-dev/unzipped_complete2/usa_v2.2.0/{tiff_name}"
            case PlumeType.CARBONMAPPER:
                tiff_blob_name = f"carbonmapper_plumes/{tiff_name}"
            case PlumeType.GAUSSIAN:
                tiff_blob_name = f"gaussian_plumes/plumes/{tiff_name}"
            case _:
                raise ValueError(f"Invalid plume type: {plume_type}")

        # Create a temporary directory for this specific TIFF file
        temp_dir = tempfile.TemporaryDirectory()
        local_dir = Path(temp_dir.name)
        local_download_filepath = local_dir / tiff_name

        # TODO: should we use download_from_blob() instead? and remove download_blob_directly()?
        download_blob_directly(tiff_blob_name, local_download_filepath, blob_service_client, container_name)
        return (local_download_filepath, temp_dir)
    else:
        return (Path(tiff_path), None)


def resize(arr: tuple[npt.NDArray, npt.NDArray], zoom: float) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Slightly convoluted method to scale down a numpy array, using torch transforms.

    We use average interpolation (bilinear) to avoid scaling the plume to all 0.
    """
    plume, _ = arr
    orig_h, orig_w = plume.shape
    new_h, new_w = orig_h * zoom, orig_w * zoom
    new_h = max(round(new_h), 1)
    new_w = max(round(new_w), 1)

    interp = InterpolationMode.BILINEAR
    resizer = Resize((new_h, new_w), interpolation=interp)

    tensor = torch.Tensor(plume[np.newaxis, :, :])
    resized_tensor = resizer(tensor)
    resized_plume = resized_tensor.numpy()[0, ...]

    mask = resized_plume != 0.0
    return resized_plume, mask


####################################################
############### SENTINEL2 FUNCTIONS ################
####################################################

# NOTE: _radtran_memo is a global variable and used to cache the radtran lookup tables. If removed as a global variable,
# create_simulated_bands will take forever to run
_radtran_memo: dict = {}


def create_simulated_bands(
    granule_item: Sentinel2Item | Sentinel2L1CItem | LandsatGranuleAccess,
    cropped_swir16: npt.NDArray,
    cropped_swir22: npt.NDArray,
    hapi_data_path: AnyUrl,
    plume_arrs: list[npt.NDArray],
    exclusion_mask_plumes: npt.NDArray,
    rng: np.random.Generator,
    transformation_params: dict[str, float] | None = None,
    position_by_source: bool = False,
) -> tuple[npt.NDArray, npt.NDArray, list[int]]:
    """
    Create simulated band 11 and 12 by randomly placing the plumes on a (crop of a) Sentinel 2 image.

    The core logic here is computing the modified image (i.e. reflectances) as if there would have been that amount of
    methane emission there.

    Arguments:
    - granule_item: Sentinel2L1CItem | LandsatGranuleAccess | pystac Item class, to get metadata
    - cropped_band11: npt.NDArray, a crop of band 11
    - cropped_band12: npt.NDArray, same crop of band 12
    - plume_arrs: List[npt.NDArray], the plumes as numpy arrays (giving a concentration of methane in the atmosphere
        column, measured in mol/m²)
    - exclusion_mask_plumes: npt.NDArray, in order to find places with no cloud
    - rng: numpy.random.Generator, the random number generator to use
    """
    methane_enhancement_molperm2, methane_enhancement_mask, plumes_inserted_idxs = transform_and_position_plumes(
        plume_arrs=plume_arrs,
        tile_band=cropped_swir22.squeeze(),
        exclusion_mask_plumes=exclusion_mask_plumes,
        rng=rng,
        transformation_params=transformation_params,
        position_by_source=position_by_source,
    )

    # Check if lookup table exists for this granule
    if granule_item.id in _radtran_memo:
        lookup_table = _radtran_memo[granule_item.id]
    else:
        logger.info(f"...creating radtran lookup table for {granule_item.id}")

        lookup_table = RadTranLookupTable.from_params(
            instrument=granule_item.instrument,
            solar_angle=granule_item.solar_angle,
            observation_angle=granule_item.observation_angle,
            hapi_data_path=hapi_data_path,
            min_ch4=0.0,
            max_ch4=200.0,  # this value was selected based on the common value ranges of the sim plume datasets
            spacing_resolution=40000,
            ref_band=granule_item.swir16_band_name,
            band=granule_item.swir22_band_name,
            full_sensor_name=granule_item.sensor_name,
        )

        _radtran_memo[granule_item.id] = lookup_table  # Store in cache

    try:
        nB_band, nB_ref_band = lookup_table.lookup(methane_enhancement_molperm2)
        sim_swir16 = nB_ref_band * cropped_swir16.squeeze()
        sim_swir22 = nB_band * cropped_swir22.squeeze()

    except IndexError as e:
        print(f"Exception occurred: {e}. Maybe try increasing max_ch4")

    return sim_swir16, sim_swir22, plumes_inserted_idxs
