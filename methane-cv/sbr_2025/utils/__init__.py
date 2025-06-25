"""Utilities and such."""

from datetime import datetime
from typing import Any

import numpy as np
from affine import Affine

# For converting mol/m2 to ppb
MOLES_AIR_PER_M2 = 356_624  # mol/m²
MOL_M2_TO_PPB_FACTOR = MOLES_AIR_PER_M2 / 1e9  # mol/m2/ppb


def select_reference_tiles_from_dates_str(
    reference_data: list[dict[str, Any]], before_date: str, earlier_date: str
) -> list[dict]:
    """Select two tiles for before and earlier to use as reference for predictions."""
    date_index_map = {x["tile_item"].time.isoformat(): i for i, x in enumerate(reference_data)}

    before_index = date_index_map[before_date]
    earlier_index = date_index_map[earlier_date]
    return [reference_data[before_index], reference_data[earlier_index]]


def select_reference_tiles_from_datetimes(
    reference_data: list[dict[str, Any]], before_date: datetime, earlier_date: datetime
) -> list[dict]:
    """Select two tiles for before and earlier to use as reference for predictions."""
    date_index_map = {x["tile_item"].time: i for i, x in enumerate(reference_data)}

    before_index = date_index_map[before_date]
    earlier_index = date_index_map[earlier_date]
    return [reference_data[before_index], reference_data[earlier_index]]


def intersects_center(
    min_row: int, min_col: int, max_row: int, max_col: int, img_size: int = 128, buffer: int = 3
) -> bool:
    """Check intersection of bbox with buffered center of an image."""
    # Calculate center of the image
    center_row = img_size / 2
    center_col = img_size / 2

    # Expand the check to include a buffer zone around the center
    return (
        min_row <= center_row + buffer
        and max_row >= center_row - buffer
        and min_col <= center_col + buffer
        and max_col >= center_col - buffer
    )


def convert_mol_m2_to_ppb(retrieval_mol_m2: np.ndarray) -> np.ndarray:
    """Convert from mol/m2 to ppb (https://git.orbio.earth/orbio/orbio/-/issues/796#note_69475)."""
    retrieval_ppb = retrieval_mol_m2 / MOL_M2_TO_PPB_FACTOR  # units: (mol/m2) / (mol/m2/ppb) = ppb
    return retrieval_ppb


def update_transform_for_crop(original_transform: Affine, crop_start_x: int, crop_start_y: int) -> Affine:
    """Update the transform for a crop."""
    return original_transform * Affine.translation(crop_start_x, crop_start_y)
