"""Plotting related functions."""

import copy
import math
from datetime import datetime
from enum import Enum
from itertools import combinations_with_replacement
from typing import cast

import numpy as np
import pyproj
import rasterio
import rasterio.plot
import torch
from azure.storage.blob import BlobServiceClient
from lib.models.schemas import WatershedParameters
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from rasterio.windows import from_bounds
from skimage.feature import peak_local_max
from torch import nn

from sbr_2025 import BANDS, BANDS_LANDSAT
from sbr_2025.utils import select_reference_tiles_from_datetimes
from sbr_2025.utils.prediction import Prediction, get_center_buffer, predict, predict_retrieval
from src.data.landsat_data import LandsatGranuleAccess
from src.data.sentinel2_l1c import Sentinel2L1CItem
from src.plotting.plotting_functions import grid16
from src.training.loss_functions import TwoPartLoss
from src.training.transformations import BaseBandExtractor
from src.utils.parameters import SatelliteID
from src.utils.radtran_utils import RadTranLookupTable


def get_rgb_bands(x_tensor: np.ndarray, bands: list[str]) -> np.ndarray:
    """Extract and stack RGB bands from input tensor for Sentinel-2.

    Args:
        x_tensor: Input tensor containing band data with shape (bands, height, width).
        bands: List of band names of x_tensor.

    Returns
    -------
        RGB image array with shape (height, width, 3).
    """
    rgb_indices = [bands.index(band) for band in ("B04", "B03", "B02")]
    rgb_image = np.transpose(x_tensor[rgb_indices], (1, 2, 0)) / 10000  # Normalize
    return rgb_image


def get_band_ratio(x_tensor: np.ndarray, bands: list[str]) -> np.ndarray:
    """Calculate B12/B11 band ratio for Sentinel-2.

    Args:
        x_tensor: Input tensor containing band data with shape (bands, height, width).
        bands: List of band names  of x_tensor.

    Returns
    -------
        Band ratio array with shape (height, width).
    """
    b11 = x_tensor[bands.index("B11")]
    b12 = x_tensor[bands.index("B12")]
    return b12 / b11


def get_rgb_bands_landsat(x_tensor: np.ndarray, bands: list[str]) -> np.ndarray:
    """Extract and stack RGB bands from input tensor for Landsat.

    Args:
        x_tensor: Input tensor containing band data with shape (bands, height, width).
        bands: List of band names of x_tensor.

    Returns
    -------
        RGB image array with shape (height, width, 3).
    """
    rgb_indices = [bands.index(band) for band in ("red", "green", "blue")]
    rgb_image = np.transpose(x_tensor[rgb_indices], (1, 2, 0)) / 10000  # Normalize
    return rgb_image


def get_band_ratio_landsat(x_tensor: np.ndarray, bands: list[str]) -> np.ndarray:
    """Calculate swir22/swir16 band ratio for Landsat.

    Args:
        x_tensor: Input tensor containing band data with shape (bands, height, width).
        bands: List of band names of x_tensor.

    Returns
    -------
        Band ratio array with shape (height, width).
    """
    b11 = x_tensor[bands.index("swir16")]
    b12 = x_tensor[bands.index("swir22")]
    return b12 / b11


class Colorbar(str, Enum):
    """How the colorbar should be set."""

    SHARE = "share"
    INDIVIDUAL = "individual"
    EXTENT = "extent"


def plot_rgb_ratio(
    data_item: dict, ratio_colorbar: Colorbar | tuple[float, float], idx: int, satellite_id: SatelliteID
) -> Figure:
    """Plot the RGB and B12/B11 Ratios for the selected chips.

    Args:
        data_items: a list of data items, where main is the first item and the rest are reference items in temporal order
        ratio_colorbar: how to set the color bar for the ratio plots
            FIXME: this isn't yet implemented
            - Colorbar.SHARE - all ratio plots share the same min and max values, which are determined from the min and max values for all ratios being plotted
            - Colorbar.INDIVIDUAL - ratio plot colorbars use the min and max value of each ratio plot
            - tuple - the min and max values for colorbar (min, max)
        idx: the number of timeseries the tile is from main tile
        satellite_id: Satellite type (S2 or LANDSAT).

    Returns
    -------
        Matplotlib figure with RGB and ratio plots.

    Raises
    ------
        ValueError: If colorbar is EXTENT or satellite_id is invalid.
    """  # noqa: E501
    if ratio_colorbar == Colorbar.EXTENT:
        raise ValueError("To set an extent for the colorbar pass in a tuple of (min, max) values to use.")

    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(10, 5))

    item = data_item["tile_item"]
    if satellite_id == SatelliteID.S2:
        rgb = get_rgb_bands(data_item["crop_arrays"], BANDS)
        ratio = get_band_ratio(data_item["crop_arrays"], BANDS)
    elif satellite_id == SatelliteID.LANDSAT:
        rgb = get_rgb_bands_landsat(data_item["crop_arrays"], BANDS_LANDSAT)
        ratio = get_band_ratio_landsat(data_item["crop_arrays"], BANDS_LANDSAT)
    else:
        raise ValueError(f"Satellite type {satellite_id.value} not handled.")

    # RGB Plot
    plt.subplot(1, 2, 1)
    plt.title(
        f"""RGB (t-{idx} | {item.time.date().isoformat()})
            Min {rgb.min():.3f}, Max {rgb.max():.3f}, Mean {rgb.mean():.3f}""",
        fontsize=10,
    )
    plt.imshow(
        (rgb / 0.35 * 255).clip(0, 255).astype(np.uint8),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    plt.colorbar()

    # Ratio Plot
    plt.subplot(1, 2, 2)
    plt.title(
        f"""B12/B11 Ratio (t-{idx} | {item.time.date().isoformat()})
                Min {ratio.min():.3f}, Max {ratio.max():.3f}, Mean {ratio.mean():.3f}""",
        fontsize=10,
    )
    plt.imshow(ratio, interpolation="nearest", vmin=ratio.min(), vmax=ratio.max())
    plt.colorbar()

    return fig


def is_ax_empty(ax: Axes) -> bool:
    """Check if a matplotlib axis is empty."""
    return not (ax.lines or ax.collections or ax.patches or ax.images)


def plot_all_rgb(data_items: list[dict], satellite_id: SatelliteID) -> None:
    """Plot RGB images for all data items in a grid.

    Args:
        data_items: List of dictionaries containing 'crop_arrays' and 'tile_item'.
        satellite_id: Satellite type (S2 or LANDSAT).

    Raises
    ------
        ValueError: If satellite_id is invalid.
    """
    fig, ax = plt.subplots(5, 9, figsize=(35, 20))
    ax = cast(np.ndarray, ax)  # for mypy
    ax = ax.flatten()  # type: ignore[union-attr]
    for i, data in enumerate(data_items[: len(ax)]):
        item = data["tile_item"]
        if satellite_id == SatelliteID.S2:
            rgb = get_rgb_bands(data["crop_arrays"], BANDS)
        elif satellite_id == SatelliteID.LANDSAT:
            rgb = get_rgb_bands_landsat(data["crop_arrays"], BANDS_LANDSAT)
        else:
            raise ValueError(f"Satellite type {satellite_id.value} not handled.")

        ax[i].set_title(f"t-{i} | {item.time.date().isoformat()}", fontsize=18)
        ax[i].imshow(
            (rgb / 0.35 * 255).clip(0, 255).astype(np.uint8),
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
    # Hide empty axes
    for ax_idx in range(len(ax)):
        if is_ax_empty(ax[ax_idx]):  # type: ignore
            ax[ax_idx].set_visible(False)  # type: ignore
    plt.tight_layout()
    plt.show()


def plot_all_ratio(data_items: list[dict], satellite_id: SatelliteID) -> None:
    """Plot B12/B11 ratios for all data items in a grid.

    Args:
        data_items: List of dictionaries containing 'crop_arrays' and 'tile_item'.
        satellite_id: Satellite type (S2 or LANDSAT).

    Raises
    ------
        ValueError: If satellite_id is invalid.
    """
    fig, ax = plt.subplots(5, 9, figsize=(35, 20))
    ax = cast(np.ndarray, ax)  # for mypy
    ax = ax.flatten()  # type: ignore[union-attr]
    for i, data in enumerate(data_items[: len(ax)]):
        item = data["tile_item"]
        if satellite_id == SatelliteID.S2:
            ratio = get_band_ratio(data["crop_arrays"], BANDS)
        elif satellite_id == SatelliteID.LANDSAT:
            ratio = get_band_ratio_landsat(data["crop_arrays"], BANDS_LANDSAT)
        else:
            raise ValueError(f"Satellite type {satellite_id.value} not handled.")

        if i == 0:
            vmin = np.percentile(ratio, 0.5)
            vmax = np.percentile(ratio, 99.5)
            print(f"Using vmin={vmin:.3f} and vmax={vmax:.3f} for all ratios")

        ax[i].set_title(f"t-{i} | {item.time.date().isoformat()}, M {ratio.mean():.3f}", fontsize=18)
        ax[i].imshow(ratio, interpolation="nearest", vmin=vmin, vmax=vmax)

    # Hide empty axes
    for ax_idx in range(len(ax)):
        if is_ax_empty(ax[ax_idx]):  # type: ignore
            ax[ax_idx].set_visible(False)  # type: ignore
    plt.tight_layout()
    plt.show()


def get_all_pairs(items: range) -> list[tuple[int, int]]:
    """Generate all unique pairs from a range of indices."""
    return list(combinations_with_replacement(items, 2))  # type: ignore


def plot_ratio_diffs(
    data_items: list[dict],
    ratio_colorbar: Colorbar | tuple[float, float],
    satellite_id: SatelliteID,
    mean_adjust: bool = True,
) -> Figure:
    """Plot the B12/B11 Ratio differences for all the pairs of the selected chips.

    The older chip is always subtracted from the more recent chip.

    Args:
        data_items: a list of data items, where main is the first item and the rest are reference items in temporal order
        ratio_colorbar: how to set the color bar for the ratio plots
            FIXME: this isn't implemented.
            - Colorbar.SHARE - all ratio plots share the same min and max values, which are determined from the min and max values for all ratios being plotted
            - Colorbar.INDIVIDUAL - ratio plot colorbars use the min and max value of each ratio plot
            - tuple - the min and max values for colorbar (min, max)
        mean_adjust: whether to adjust for the difference in means.  This can make the differences more apparent if one chips is overall darker / lighter than the other
        satellite_id: Satellite type (S2 or LANDSAT).
    """  # noqa: E501
    # match ratio_diff_colorbar:
    #     case Colorbar.SHARE:
    #         # Use the min / max of all main and reference tiles so colors are consistent across plots
    #         vmin = min(ratio_earlier.min(), ratio_before.min(), ratio_main.min())
    #         vmax = max(ratio_earlier.max(), ratio_before.max(), ratio_main.max())

    #         vmin_main_earlier = vmin
    #         vmax_main_earlier = vmax
    #         vmin_main_before = vmin
    #         vmax_main_before = vmax
    #         vmin_before_earlier = vmin
    #         vmax_before_earlier = vmax
    #     case Colorbar.INDIVIDUAL:
    #         vmin_main_earlier = ratio_earlier.min()
    #         vmax_main_earlier = ratio_earlier.max()
    #         vmin_main_before = ratio_before.min()
    #         vmax_main_before = ratio_before.max()
    #         vmin_before_earlier = ratio_earlier.min()
    #         vmax_before_earlier = ratio_earlier.max()
    #     case (float(), float()):
    #         assert ratio_diff_colorbar[0] < ratio_diff_colorbar[1]
    #         vmin_main_earlier = ratio_diff_colorbar[0]
    #         vmax_main_earlier = ratio_diff_colorbar[1]
    #         vmin_main_before = ratio_diff_colorbar[0]
    #         vmax_main_before = ratio_diff_colorbar[1]
    #         vmin_before_earlier = ratio_diff_colorbar[0]
    #         vmax_before_earlier = ratio_diff_colorbar[1]
    #     case _:
    #         typing.assert_never(ratio_diff_colorbar)

    # Plot ratio diff
    num_items = len(data_items)
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(5 * num_items, 5 * num_items))

    for i, (a, b) in enumerate(get_all_pairs(range(num_items))):
        data_a = data_items[a]
        data_item_a = data_a["tile_item"]
        data_b = data_items[b]
        data_item_b = data_b["tile_item"]
        print(f"Pair {i}: {data_item_a.time.date().isoformat()}, {data_item_b.time.date().isoformat()}")

        if satellite_id == SatelliteID.S2:
            ratio_a = get_band_ratio(data_a["crop_arrays"], BANDS)
            ratio_b = get_band_ratio(data_b["crop_arrays"], BANDS)
        elif satellite_id == SatelliteID.LANDSAT:
            ratio_a = get_band_ratio_landsat(data_a["crop_arrays"], BANDS_LANDSAT)
            ratio_b = get_band_ratio_landsat(data_b["crop_arrays"], BANDS_LANDSAT)
        else:
            raise ValueError(f"Satellite type {satellite_id.value} not handled.")

        a_b_ratio_diff = ratio_a - ratio_b

        vmin = ratio_diff_min if (ratio_diff_min := a_b_ratio_diff.min()) < 0.0 else -1e-10
        vmax = ratio_diff_max if (ratio_diff_max := a_b_ratio_diff.max()) > 0.0 else 1e-10

        # adjust for difference in means
        if mean_adjust:
            if (mean_diff := a_b_ratio_diff.mean()) < 0:
                a_b_ratio_diff = a_b_ratio_diff + np.abs(mean_diff)
            else:
                a_b_ratio_diff = a_b_ratio_diff - np.abs(mean_diff)

        rows, cols = len(data_items), len(data_items)
        plot_number = a * num_items + b + 1
        plt.subplot(rows, cols, plot_number)
        plt.title(
            f"""Ratio Diff (t-{a} - t-{b})
            ({data_item_a.time.date().isoformat()} | {data_item_b.time.date().isoformat()})
            Min {a_b_ratio_diff.min():.3f}, Max {a_b_ratio_diff.max():.3f}, Mean {a_b_ratio_diff.mean():.3f}""",
            fontsize=10,
        )
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        plt.imshow(a_b_ratio_diff, interpolation="nearest", norm=norm, cmap="seismic")
        plt.colorbar()
        grid16()
    return fig


def retrieval_plots(
    rgb_main: np.ndarray,
    ratio_main: np.ndarray,
    binary_probability: np.ndarray,
    rescaled_retrieval: np.ndarray,
    marginal_retrieval: np.ndarray | None,
    center_plume_mask: np.ndarray | None,
    watershed_segmentation_params: WatershedParameters,
    date_main: str,
    show_rgb_ratio: bool = True,
) -> None:
    """Plot RGB/Ratio (optionally) and binary, marginal and mask retrievals."""
    center_y = binary_probability.shape[0] // 2
    center_x = binary_probability.shape[1] // 2

    if show_rgb_ratio:
        f, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax = cast(np.ndarray, ax)  # for mypy
        ax[0].set_title(f"RGB (t=main {date_main})", fontsize=18)
        ax[0].imshow(
            (rgb_main / 0.35 * 255).clip(0, 255).astype(np.uint8),
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        grid16(ax[0])
        ax[0].scatter(center_x, center_y, color="green", marker="x")

        ax[1].set_title(f"B12/B11 Ratio (t=main)\nMean {ratio_main.mean():.3f}", fontsize=18)
        vmin_main, vmax_main = ratio_main.min(), ratio_main.max()
        img = ax[1].imshow(ratio_main, interpolation="nearest", vmin=vmin_main, vmax=vmax_main)
        grid16()
        ax[1].scatter(center_x, center_y, color="green", marker="x")

        plt.tight_layout()
        plt.show()

    f, ax = plt.subplots(1, 4, figsize=(30, 8))
    ax = cast(np.ndarray, ax)  # for mypy
    fontsize = 20

    ax[2].set_title(f"marginal_retrieval: {marginal_retrieval.sum():.1f}", fontsize=fontsize)  # type:ignore
    ax[2].scatter(center_x, center_y, color="green", marker="x")
    img = ax[2].imshow(marginal_retrieval, vmin=-0.2, vmax=0.2, cmap="RdBu_r", interpolation="nearest")  # type: ignore
    plt.colorbar(img, ax=ax[2])
    grid16(ax[2])

    ax[0].set_title(f"rescaled_retrieval: {rescaled_retrieval.sum():.1f}", fontsize=fontsize)
    ax[0].scatter(center_x, center_y, color="green", marker="x")
    img = ax[0].imshow(rescaled_retrieval, vmin=-0.2, vmax=0.2, cmap="RdBu_r", interpolation="nearest")
    grid16(ax[0])
    plt.colorbar(img, ax=ax[0])

    center_buffer = 10
    ax[1].set_title(
        f"Binary Probability\n Center Max: {get_center_buffer(binary_probability, center_buffer).max():.3f})",
        fontsize=fontsize,
    )
    ax[1].scatter(center_x, center_y, color="green", marker="x")
    img = ax[1].imshow(
        binary_probability,
        vmin=0.0,
        vmax=1.0,
        cmap="hot_r",
        interpolation="nearest",
    )
    plt.colorbar(img, ax=ax[1])
    grid16(ax[1])

    ax[3].imshow(
        (rgb_main / 0.35 * 255).clip(0, 255).astype(np.uint8),
        vmin=0.0,
        vmax=1.0,
        interpolation="none",
    )
    ax[3].scatter(center_x, center_y, color="green", marker="x")
    masked_marg_retrieval = np.ma.masked_where(center_plume_mask == False, marginal_retrieval)  # noqa

    marker_thresh = watershed_segmentation_params.marker_threshold
    floor_thresh = watershed_segmentation_params.watershed_floor_threshold
    ax[3].set_title(f"Marg. Retrieval\nmarker_t={marker_thresh}, floor_t={floor_thresh})", fontsize=fontsize)
    img = ax[3].imshow(
        masked_marg_retrieval,  # type: ignore
        vmin=0.0,
        vmax=0.2,
        cmap="hot_r",
        interpolation="none",
    )
    plt.colorbar(img, ax=ax[3])
    grid16(ax[3])

    plt.tight_layout()
    plt.show()


def all_error_analysis_plots(  # noqa: PLR0913, PLR0915
    rgb_main: np.ndarray,
    rgb_before: np.ndarray,
    rgb_earlier: np.ndarray,
    ratio_main: np.ndarray,
    ratio_before: np.ndarray,
    ratio_earlier: np.ndarray,
    predicted_frac: np.ndarray,
    predicted_mask: np.ndarray,
    conditional_pred: np.ndarray,
    binary_probability: np.ndarray,
    conditional_retrieval: np.ndarray,
    masked_conditional_retrieval: np.ndarray,
    rescaled_retrieval: np.ndarray | None,
    marginal_retrieval: np.ndarray | None,
    watershed_segmentation_params: WatershedParameters,
    dates: tuple[str, str, str],
    ratio_colorbar: Colorbar | tuple[float, float],
    ratio_diff_colorbar: Colorbar | tuple[float, float],
) -> None:
    """Plot RGB, frac, ratio, retrievals, watershed segmentation, and binary probability."""
    date_main, date_before, date_earlier = dates

    center_y = binary_probability.shape[0] // 2
    center_x = binary_probability.shape[1] // 2

    alpha = 0.5
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(32, 15))
    fig.suptitle(
        f"""date: {date_main} """,
        fontsize=24,
        y=0.97,
        x=0.5,
    )

    #########################
    # Column 1: Plot RGB
    #########################
    plt.subplot(3, 6, 1)
    plt.title(
        f"""RGB (t=earlier {date_earlier})
            Min {rgb_earlier.min():.3f}, Max {rgb_earlier.max():.3f}, Mean {rgb_earlier.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(
        (rgb_earlier / 0.35 * 255).clip(0, 255).astype(np.uint8),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    grid16()

    plt.subplot(3, 6, 7)
    plt.title(
        f"""RGB (t=before {date_before})
            Min {rgb_before.min():.3f}, Max {rgb_before.max():.3f}, Mean {rgb_before.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(
        (rgb_before / 0.35 * 255).clip(0, 255).astype(np.uint8),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    grid16()

    plt.subplot(3, 6, 13)
    plt.title(
        f"""RGB (t=main {date_main})
            Min {rgb_main.min():.3f}, Max {rgb_main.max():.3f}, Mean {rgb_main.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(
        (rgb_main / 0.35 * 255).clip(0, 255).astype(np.uint8),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    grid16()

    #########################
    # Column 2: Plot Ratios
    #########################
    match ratio_colorbar:
        case Colorbar.SHARE:
            # Use the min / max of all main and reference tiles so colors are consistent across plots
            vmin = min(ratio_earlier.min(), ratio_before.min(), ratio_main.min())
            vmax = max(ratio_earlier.max(), ratio_before.max(), ratio_main.max())

            vmin_earlier = vmin
            vmax_earlier = vmax
            vmin_before = vmin
            vmax_before = vmax
            vmin_main = vmin
            vmax_main = vmax
        case Colorbar.INDIVIDUAL:
            vmin_earlier = ratio_earlier.min()
            vmax_earlier = ratio_earlier.max()
            vmin_before = ratio_before.min()
            vmax_before = ratio_before.max()
            vmin_main = ratio_main.min()
            vmax_main = ratio_main.max()
        case (float(), float()):
            assert isinstance(ratio_diff_colorbar[0], float)
            assert isinstance(ratio_diff_colorbar[1], float)
            assert ratio_diff_colorbar[0] < ratio_diff_colorbar[1]
            vmin_earlier = ratio_colorbar[0]
            vmax_earlier = ratio_colorbar[1]
            vmin_before = ratio_colorbar[0]
            vmax_before = ratio_colorbar[1]
            vmin_main = ratio_colorbar[0]
            vmax_main = ratio_colorbar[1]
        case _:
            raise ValueError("ratio_colorbar not a valid type.")

    # Plot ratio earlier
    plt.subplot(3, 6, 2)
    plt.title(
        f"""B12/B11 Ratio (t-2=earlier)
              Min {ratio_earlier.min():.3f}, Max {ratio_earlier.max():.3f}, Mean {ratio_earlier.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(ratio_earlier, interpolation="nearest", vmin=vmin_earlier, vmax=vmax_earlier)
    grid16()
    plt.colorbar()

    # Plot ratio before
    plt.subplot(3, 6, 8)
    plt.title(
        f"""B12/B11 Ratio (t-1=before)
              Min {ratio_before.min():.3f}, Max {ratio_before.max():.3f}, Mean {ratio_before.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(ratio_before, interpolation="nearest", vmin=vmin_before, vmax=vmax_before)
    grid16()
    plt.colorbar()

    # Plot ratio main
    plt.subplot(3, 6, 14)
    plt.title(
        f"""B12/B11 Ratio (t=main)
              Min {ratio_main.min():.3f}, Max {ratio_main.max():.3f}, Mean {ratio_main.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(ratio_main, interpolation="nearest", vmin=vmin_main, vmax=vmax_main)
    grid16()
    plt.colorbar()

    #########################
    # Column 3: Plot Ratio Diffs Variable
    #########################
    main_earlier_ratio_diff = ratio_main - ratio_earlier
    main_before_ratio_diff = ratio_main - ratio_before
    before_earlier_ratio_diff = ratio_before - ratio_earlier

    match ratio_diff_colorbar:
        case Colorbar.SHARE:
            # Use the min / max of all main and reference tiles so colors are consistent across plots
            vmin = min(main_earlier_ratio_diff.min(), main_before_ratio_diff.min(), before_earlier_ratio_diff.min())
            vmax = max(main_earlier_ratio_diff.max(), main_before_ratio_diff.max(), before_earlier_ratio_diff.max())

            vmin_main_earlier = vmin
            vmax_main_earlier = vmax
            vmin_main_before = vmin
            vmax_main_before = vmax
            vmin_before_earlier = vmin
            vmax_before_earlier = vmax
        case Colorbar.INDIVIDUAL:
            vmin_main_earlier = ratio_earlier.min()
            vmax_main_earlier = ratio_earlier.max()
            vmin_main_before = ratio_before.min()
            vmax_main_before = ratio_before.max()
            vmin_before_earlier = ratio_earlier.min()
            vmax_before_earlier = ratio_earlier.max()
        case (float(), float()):
            assert isinstance(ratio_diff_colorbar[0], float)
            assert isinstance(ratio_diff_colorbar[1], float)
            assert ratio_diff_colorbar[0] < ratio_diff_colorbar[1]
            vmin_main_earlier = ratio_diff_colorbar[0]
            vmax_main_earlier = ratio_diff_colorbar[1]
            vmin_main_before = ratio_diff_colorbar[0]
            vmax_main_before = ratio_diff_colorbar[1]
            vmin_before_earlier = ratio_diff_colorbar[0]
            vmax_before_earlier = ratio_diff_colorbar[1]
        case _:
            raise ValueError("ratio_colorbar not a valid type.")

    # Plot ratio diff
    plt.subplot(3, 6, 3)
    plt.title(
        f"""Ratio Diff (main - earlier)
        Min {main_earlier_ratio_diff.min():.3f}, Max {main_earlier_ratio_diff.max():.3f}, Mean {main_earlier_ratio_diff.mean():.3f}""",  # noqa: E501
        fontsize=15,
    )
    norm = TwoSlopeNorm(vmin=vmin_main_earlier, vcenter=0, vmax=vmax_main_earlier)
    plt.imshow(main_earlier_ratio_diff, interpolation="nearest", norm=norm, cmap="seismic")
    plt.colorbar()
    grid16()

    # Plot ratio ratio diff with prediction mask
    plt.subplot(3, 6, 9)
    plt.title(
        f"""Ratio Diff (main - before)
        Min {main_before_ratio_diff.min():.3f}, Max {main_before_ratio_diff.max():.3f}, Mean {main_before_ratio_diff.mean():.3f}""",  # noqa: E501
        fontsize=15,
    )
    norm = TwoSlopeNorm(vmin=vmin_main_before, vcenter=0, vmax=vmax_main_before)
    plt.imshow(main_before_ratio_diff, interpolation="nearest", norm=norm, cmap="seismic")
    plt.colorbar()
    grid16()

    # Blank Plot (put masked retrieval here?)
    plt.subplot(3, 6, 15)
    plt.title(
        f"""Ratio Diff (before - earlier)
        Min {before_earlier_ratio_diff.min():.3f}, Max {before_earlier_ratio_diff.max():.3f}, Mean {before_earlier_ratio_diff.mean():.3f}""",  # noqa: E501
        fontsize=15,
    )
    norm = TwoSlopeNorm(vmin=vmin_before_earlier, vcenter=0, vmax=vmax_before_earlier)
    plt.imshow(before_earlier_ratio_diff, interpolation="nearest", norm=norm, cmap="seismic")
    plt.colorbar()
    grid16()

    #########################
    # Column 4: Plot Ratio Diffs
    #########################
    # Plot ratio diff
    ratio_diff = ratio_main - (ratio_before + ratio_earlier) / 2
    vmin = ratio_diff_min if (ratio_diff_min := ratio_diff.min()) < 0 else -1e-10
    vmax = ratio_diff_max if (ratio_diff_max := ratio_diff.max()) > 0 else 1e-10

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.subplot(3, 6, 4)
    plt.title(
        f"""Ratio Diff (main - avg(reference))
        Min {ratio_diff.min():.3f}, Max {ratio_diff.max():.3f}, Mean {ratio_diff.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(
        ratio_diff,
        norm=norm,
        interpolation="nearest",
        cmap="seismic",
    )
    plt.colorbar()
    grid16()

    # Plot ratio ratio diff with prediction mask
    plt.subplot(3, 6, 10)
    plt.title("Ratio Diff (main - avg(reference))\n(prediction mask)", fontsize=15)
    plt.scatter(center_x, center_y, color="green", marker="x")
    plt.imshow(
        ratio_diff,
        norm=norm,
        interpolation="nearest",
        cmap="seismic",
    )
    plt.colorbar()
    plt.imshow(np.ma.masked_where(predicted_mask == 0, predicted_mask), cmap="binary_r", alpha=alpha)
    grid16()

    # Blank Plot (put masked retrieval here?)
    plt.subplot(3, 6, 16)
    plt.title("""Masked Conditional Retrieval""", fontsize=15)
    plt.imshow(
        (rgb_main / 0.35 * 255).clip(0, 255).astype(np.uint8),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    plt.scatter(center_x, center_y, color="green", marker="x")
    masked_conditional_retrieval = np.ma.masked_where(predicted_mask == 0, conditional_retrieval)

    plt.imshow(
        masked_conditional_retrieval,  # type: ignore
        vmin=0.0,
        vmax=0.5,
        cmap="hot_r",
        interpolation="nearest",
    )
    plt.colorbar()
    grid16()

    #########################
    # Column 5: Plot Masks
    #########################
    # Plot watershed markers
    plt.subplot(3, 6, 5)
    distance = watershed_segmentation_params.marker_distance
    marker_thresh = watershed_segmentation_params.marker_threshold
    floor_thresh = watershed_segmentation_params.watershed_floor_threshold
    plt.title(
        f"Watershed Markers\n(dist={distance}, marker_t={marker_thresh}, floor_t={floor_thresh})",
        fontsize=15,
    )
    plt.scatter(center_x, center_y, color="green", marker="x")

    marker_coords = peak_local_max(
        binary_probability,
        min_distance=watershed_segmentation_params.marker_distance,
        threshold_abs=watershed_segmentation_params.marker_threshold,
    )

    marker_map = np.zeros_like(binary_probability)
    marker_map[tuple(marker_coords.T)] = 1
    plt.imshow(marker_map, vmin=0, vmax=1, interpolation="none")
    grid16()
    plt.colorbar()

    plt.subplot(3, 6, 11)
    plt.title("rescaled_retrieval", fontsize=15)
    plt.scatter(center_x, center_y, color="green", marker="x")
    if rescaled_retrieval is not None:
        plt.imshow(rescaled_retrieval, vmin=-0.2, vmax=0.2, cmap="RdBu_r", interpolation="nearest")
    else:
        plt.text(
            0.5,
            0.5,
            "No rescaled retrieval data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
    grid16()
    plt.colorbar()

    plt.subplot(3, 6, 17)
    plt.title("marginal_retrieval", fontsize=15)
    plt.scatter(center_x, center_y, color="green", marker="x")
    if marginal_retrieval is not None:
        plt.imshow(marginal_retrieval, vmin=-0.2, vmax=0.2, cmap="RdBu_r", interpolation="nearest")  # type: ignore
    else:
        plt.text(
            0.5,
            0.5,
            "No marginal retrieval data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
    plt.colorbar()
    grid16()

    #########################
    # Column 6: Plot Retrievals
    #########################
    # Plot binary probability
    plt.subplot(3, 6, 6)
    center_buffer = 10
    plt.title(
        f"Binary Probability\n Center Max: {get_center_buffer(binary_probability, center_buffer).max():.3f})",
        fontsize=15,
    )
    plt.scatter(center_x, center_y, color="green", marker="x")
    plt.imshow(
        binary_probability,
        vmin=0.0,
        vmax=1.0,
        cmap="hot_r",
        interpolation="nearest",
    )
    grid16()
    plt.colorbar()

    # Plot Conditional Pred
    plt.subplot(3, 6, 12)
    plt.title("Conditional Prediction", fontsize=15)
    plt.scatter(center_x, center_y, color="green", marker="x")
    plt.imshow(conditional_pred, vmin=-0.025, vmax=0.025, cmap="RdBu_r", interpolation="nearest")
    grid16()
    plt.colorbar()

    # Plot Conditional Retrieval
    plt.subplot(3, 6, 18)
    plt.title("Conditional Retrieval", fontsize=15)
    plt.scatter(center_x, center_y, color="green", marker="x")
    plt.imshow(
        conditional_retrieval,  # type: ignore
        vmin=-0.2,
        vmax=0.2,
        cmap="RdBu_r",
        interpolation="nearest",
    )
    grid16()
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_max_proba_center_buffer_heatmap(  # noqa PLR0915
    predictions: dict[tuple[datetime, datetime], Prediction],
    dates: list[datetime],
    center_buffer: int,
    title: str,
    satellite_id: SatelliteID,
    show_topk: int = 5,
    topk_based_on: str = "max",  # "max" or "sum" or "marg_sum"
    main_date: str = "",
    before_date: datetime | None = None,
    earlier_date: datetime | None = None,
    plot_topk: bool = True,
    plot_max_grid: bool = True,
    plot_binary_grid: bool = False,
    dates_to_exclude: list[str] = [],  # noqa
    extent: tuple[int, int, int, int] = (0, 128, 0, 128),
) -> dict:
    """Create a 2x2 matrix heatmap from a given 2x2 matrix of reference date pairs of probabilities.

    The max probability of the center buffer is used.

    Args:
        predictions: a dictionary of reference dates as tuple keys and containing Predictions values
        dates: reference dates matching the 2x2 arr
        center_buffer: Size of center buffer for probability calculation.
        title: Title suffix for the plot.
        show_topk: Number of top predictions to show from all reference img combinations
        main_date: Date of the main image.
        before_date: Date of the before image.
        earlier_date: Date of the earlier image.
        plot_binary_grid: Whether to plot the binary grid.
    """
    output_preds = {}
    num_items = len(dates)
    arr = np.ones((num_items, num_items)) * -1
    arr_probas = np.ones((num_items, num_items)) * -1
    arr_marg = np.ones((num_items, num_items)) * -1
    for (date_before, date_earlier), prediction in predictions.items():
        center = get_center_buffer(prediction.binary_probability, center_buffer)
        max_proba = 100 * center.max()
        arr_probas[dates.index(date_before), dates.index(date_earlier)] = (
            100 * center.sum() / (center.shape[0] * center.shape[1])
        )
        arr[dates.index(date_before), dates.index(date_earlier)] = max_proba
        center_marg = get_center_buffer(prediction.marginal, center_buffer)
        arr_marg[dates.index(date_before), dates.index(date_earlier)] = np.abs(center_marg).sum() * 100

    xmin, xmax, ymin, ymax = extent
    center_y = prediction.binary_probability[xmin:xmax, ymin:ymax].shape[0] // 2
    center_x = prediction.binary_probability[xmin:xmax, ymin:ymax].shape[1] // 2

    if before_date is not None and earlier_date is not None:
        # Plot normal before/earlier prediction
        normal_preds = predictions[(before_date, earlier_date)]
        center = get_center_buffer(normal_preds.binary_probability, center_buffer)
        output_preds["normal"] = {
            "date_before": before_date,
            "date_earlier": earlier_date,
            "max_prob": 100 * center.max(),
            "sum_prob": 100 * center.sum() / (center.shape[0] * center.shape[1]),
        }
        center_marg = get_center_buffer(normal_preds.marginal, center_buffer)

        if plot_topk:
            if satellite_id == SatelliteID.S2:
                ratio_main = get_band_ratio(normal_preds.x_dict["crop_main"][0], BANDS)[xmin:xmax, ymin:ymax]  # type:ignore
                ratio_before = get_band_ratio(normal_preds.x_dict["crop_before"][0], BANDS)[xmin:xmax, ymin:ymax]  # type:ignore
                ratio_earlier = get_band_ratio(normal_preds.x_dict["crop_earlier"][0], BANDS)[xmin:xmax, ymin:ymax]  # type:ignore
            elif satellite_id == SatelliteID.LANDSAT:
                ratio_main = get_band_ratio_landsat(
                    normal_preds.x_dict["crop_main"][0], BANDS_LANDSAT
                )[  # type:ignore
                    xmin:xmax, ymin:ymax
                ]  # type:ignore
                ratio_before = get_band_ratio_landsat(
                    normal_preds.x_dict["crop_before"][0], BANDS_LANDSAT
                )[  # type:ignore
                    xmin:xmax, ymin:ymax
                ]  # type:ignore
                ratio_earlier = get_band_ratio_landsat(
                    normal_preds.x_dict["crop_earlier"][0], BANDS_LANDSAT
                )[  # type:ignore
                    xmin:xmax, ymin:ymax
                ]  # type:ignore
            vmin = np.percentile(ratio_main, 0.5)
            vmax = np.percentile(ratio_main, 99.5)

            f, ax = plt.subplots(1, 4, figsize=(20, 5))
            ax = cast(np.ndarray, ax)  # for mypy
            earlier_date_title = earlier_date.date().isoformat()
            before_date_title = before_date.date().isoformat()

            ax[0].imshow(ratio_earlier, vmin=vmin, vmax=vmax)
            ax[0].set_title(
                f"Earlier Ratio {earlier_date_title}\nMin {ratio_earlier.min():.2f} Max {ratio_earlier.max():.2f} "
                f"Mean {ratio_earlier.mean():.3f}",
                fontsize=15,
            )
            ax[1].imshow(ratio_before, vmin=vmin, vmax=vmax)
            ax[1].set_title(
                f"Before Ratio {before_date_title}\nMin {ratio_before.min():.2f} Max {ratio_before.max():.2f} "
                f"Mean {ratio_before.mean():.3f}",
                fontsize=15,
            )
            ax[2].imshow(ratio_main, vmin=vmin, vmax=vmax)
            ax[2].set_title(
                f"Main Ratio {main_date}\nMin {ratio_main.min():.2f} Max {ratio_main.max():.2f} "
                f"Mean {ratio_main.mean():.3f}",
                fontsize=15,
            )
            ax[3].imshow(normal_preds.binary_probability[xmin:xmax, ymin:ymax], vmin=0.0, vmax=1.0, cmap="hot_r")

            ax[3].set_title(
                f"NORMAL Center\nsum(Prob): {100 * center.sum() / (center.shape[0] * center.shape[1]):.1f}%, "
                f"Max: {100 * center.max():.0f}%, Marg: {np.abs(center_marg).sum() * 100:.1f}",
                fontsize=16,
            )
            ax[0].scatter(center_x, center_y, color="green", marker="x")
            ax[1].scatter(center_x, center_y, color="green", marker="x")
            ax[2].scatter(center_x, center_y, color="green", marker="x")
            ax[3].scatter(center_x, center_y, color="green", marker="x")

            grid16(ax[0])
            grid16(ax[1])
            grid16(ax[2])
            grid16(ax[3])
        print(
            f"NORMAL  Center sum(Prob): {100 * center.sum() / (center.shape[0] * center.shape[1]):4.1f}%, "
            f"Max: {100 * center.max():3.0f}%, Marg: {np.abs(center_marg).sum() * 100:5.1f}"
            f" ({before_date}, {earlier_date})"
        )

    print("=" * 50)
    # Print top k predictions summaries from all reference img combinations
    if topk_based_on == "sum":
        flat_indices = np.argsort(arr_probas.ravel())[-show_topk:][::-1]
    elif topk_based_on == "marg_sum":
        flat_indices = np.argsort(arr_marg.ravel())[-show_topk:][::-1]
    else:
        flat_indices = np.argsort(arr.ravel())[-show_topk:][::-1]
    row_cols = np.unravel_index(flat_indices, arr_probas.shape)
    for top_index in range(show_topk):
        row_index = row_cols[0][top_index]  # type:ignore
        col_index = row_cols[1][top_index]  # type:ignore

        if dates[row_index] in dates_to_exclude or dates[col_index] in dates_to_exclude:
            print(f"Top-{top_index + 1:2}  --> Skipping as it uses an excluded reference image")
            continue
        max_ = arr[row_index, col_index]
        print(
            f"Top-{top_index + 1:2}  Center sum(Prob): {arr_probas[row_index, col_index]:4.1f}%, "
            f"Max: {max_:3.0f}%, Marg: {arr_marg[row_index, col_index]:5.1f} ({dates[row_index]}, {dates[col_index]})"
        )
        output_preds["top_" + str(top_index + 1)] = {
            "date_before": dates[row_index],
            "date_earlier": dates[col_index],
            "max_prob": max_,
            "sum_prob": arr_probas[row_index, col_index],
        }

    if before_date is not None and earlier_date is not None and plot_topk:
        plt.tight_layout()
        plt.show()

    if plot_topk:
        # Plot top k predictions from all reference img combinations
        for top_index in range(show_topk):
            row_index = row_cols[0][top_index]  # type:ignore
            col_index = row_cols[1][top_index]  # type:ignore
            max_ = arr[row_index, col_index]
            if dates[row_index] in dates_to_exclude or dates[col_index] in dates_to_exclude:
                continue

            data_ = predictions[(dates[row_index], dates[col_index])].x_dict
            if satellite_id == SatelliteID.S2:
                ratio_main = get_band_ratio(data_["crop_main"][0], BANDS)[xmin:xmax, ymin:ymax]  # type:ignore
                ratio_before = get_band_ratio(data_["crop_before"][0], BANDS)[xmin:xmax, ymin:ymax]  # type:ignore
                ratio_earlier = get_band_ratio(data_["crop_earlier"][0], BANDS)[xmin:xmax, ymin:ymax]  # type:ignore
            elif satellite_id == SatelliteID.LANDSAT:
                ratio_main = get_band_ratio_landsat(data_["crop_main"][0], BANDS_LANDSAT)[xmin:xmax, ymin:ymax]  # type:ignore
                ratio_before = get_band_ratio_landsat(data_["crop_before"][0], BANDS_LANDSAT)[xmin:xmax, ymin:ymax]  # type:ignore
                ratio_earlier = get_band_ratio_landsat(data_["crop_earlier"][0], BANDS_LANDSAT)[xmin:xmax, ymin:ymax]  # type:ignore

            vmin = np.percentile(ratio_main, 0.5)
            vmax = np.percentile(ratio_main, 99.5)

            f, ax = plt.subplots(1, 4, figsize=(20, 5))
            ax = cast(np.ndarray, ax)  # for mypy
            ax[0].imshow(ratio_earlier, vmin=vmin, vmax=vmax)
            ax[0].set_title(
                f"Earlier Ratio {dates[col_index]}\nMin {ratio_earlier.min():.2f} Max {ratio_earlier.max():.2f} "
                f"Mean {ratio_earlier.mean():.3f}",
                fontsize=15,
            )
            ax[1].imshow(ratio_before, vmin=vmin, vmax=vmax)
            ax[1].set_title(
                f"Before Ratio {dates[row_index]}\nMin {ratio_before.min():.2f} Max {ratio_before.max():.2f} "
                f"Mean {ratio_before.mean():.3f}",
                fontsize=15,
            )
            ax[2].imshow(ratio_main, vmin=vmin, vmax=vmax)
            ax[2].set_title(
                f"Main Ratio {main_date}\nMin {ratio_main.min():.2f} Max {ratio_main.max():.2f} "
                f"Mean {ratio_main.mean():.3f}",
                fontsize=15,
            )
            ax[3].imshow(
                predictions[(dates[row_index], dates[col_index])].binary_probability[xmin:xmax, ymin:ymax],
                vmin=0.0,
                vmax=1.0,
                cmap="hot_r",
            )
            ax[3].set_title(
                f"Top-{top_index + 1} Center\nsum(Prob): {arr_probas[row_index, col_index]:.1f}%, Max: {max_:.0f}%, "
                f"Marg: {arr_marg[row_index, col_index]:5.1f} ",
                fontsize=15,
            )
            ax[0].scatter(center_x, center_y, color="green", marker="x")
            ax[1].scatter(center_x, center_y, color="green", marker="x")
            ax[2].scatter(center_x, center_y, color="green", marker="x")
            ax[3].scatter(center_x, center_y, color="green", marker="x")

            grid16(ax[0])
            grid16(ax[1])
            grid16(ax[2])
            grid16(ax[3])

            plt.tight_layout()
            plt.show()

    # rotate 90˚ clockwise so that labels can be ascending bottom->top and left->right
    if topk_based_on == "sum":
        arr_to_use = np.rot90(arr_probas, k=3)
    elif topk_based_on == "marg_sum":
        arr_to_use = np.rot90(arr_marg, k=3)
    else:
        arr_to_use = np.rot90(arr, k=3)

    if plot_max_grid:
        size = num_items * 2
        fig, ax = plt.subplots(figsize=(size, size))
        if topk_based_on == "sum":
            vmax_arr = 20.0
        elif topk_based_on == "marg_sum":
            vmax_arr = 30.0
        else:
            vmax_arr = 100.0
        img = ax.imshow(arr_to_use, cmap="hot_r", vmin=0.0, vmax=vmax_arr)
        cbar = fig.colorbar(img, ax=ax)
        cbar.ax.tick_params(labelsize=size)
        if topk_based_on == "sum":
            cbar_label = "Mean Binary Prob"
        elif topk_based_on == "marg_sum":
            cbar_label = "Marg Sum"
        else:
            cbar_label = "Max Probability"
        cbar.set_label(cbar_label, fontsize=size)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(
            range(len(dates)),
            labels=[d.date().isoformat() for d in dates[::-1]],
            fontsize=size,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_yticks(range(len(dates)), labels=[d.date().isoformat() for d in dates], fontsize=size)

        # Loop over data dimensions and create text annotations.
        for i in range(len(dates)):
            for j in range(len(dates)):
                ax.text(
                    j,
                    i,
                    str(int(np.round(arr_to_use[i, j], decimals=0))),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=size,
                )

        ax.set_title(
            f"""{topk_based_on} Binary Probability for Reference Chip Pairs
            (within {center_buffer} pixel buffer from center) - {title}""",
            fontsize=size,
        )
        ax.set_xlabel("Before Date", fontsize=size)
        ax.set_ylabel("Earlier Date", fontsize=size)
        fig.tight_layout()
        plt.show()

    if plot_binary_grid:
        # Plot binary probabilities
        center_y = prediction.binary_probability.shape[0] // 2
        center_x = prediction.binary_probability.shape[1] // 2
        f, ax_arr = plt.subplots(arr.shape[0], arr.shape[1], figsize=(size, size))
        ax_arr = cast(np.ndarray, ax_arr)

        for (date_before, date_earlier), prediction in predictions.items():
            row_index = dates.index(date_earlier)
            col_index = num_items - 1 - dates.index(date_before)
            ax_arr[row_index, col_index].imshow(prediction.binary_probability, vmin=0.0, vmax=1.0, cmap="hot_r")

            center = get_center_buffer(prediction.binary_probability, center_buffer)
            max_proba = center.max()
            sum_proba = 100 * center.sum() / (center.shape[0] * center.shape[1])
            ax_arr[row_index, col_index].set_title(
                f"B={date_before}\nE={date_earlier}\nMax:{max_proba:.2f}, {sum_proba:.0f}%", fontsize=11
            )
            # Turns off x ticks
            ax_arr[row_index, col_index].set_xticks([])
            ax_arr[row_index, col_index].set_yticks([])
            ax_arr[row_index, col_index].scatter(center_x, center_y, color="green", marker="x")
        plt.tight_layout(pad=0.1)
        plt.show()
    return output_preds


def plot_normal_and_avg_strategy(  # noqa: PLR0913, PLR0915
    stac_items: list,
    main_data_all: dict,
    reference_data_all: dict,
    phase0_release_dates: list[str],
    models: list,
    model_ids: list[str],
    band_concatenators: list[BaseBandExtractor],
    device: str,
    lossFn: TwoPartLoss,
    watershed_params: WatershedParameters,
    center_buffer: int,
    use_mean: bool = True,
) -> tuple[dict[datetime, dict], dict[datetime, dict], dict[datetime, int]]:
    """Compare normal and average reference strategies for methane detection.

    Args:
        stac_items: List of STAC items of main IDs
        main_data_all: Dictionary of main chip data by date.
        reference_data_all: Dictionary of reference chip data by date.
        phase0_release_dates: Dates to exclude from analysis.
        models: List of prediction models.
        model_ids: List of model identifiers.
        band_concatenators: List of band concatenators for preprocessing.
        device: Device for model computation (e.g., 'cpu', 'cuda').
        lossFn: Loss function for prediction.
        watershed_params: Parameters for watershed segmentation.
        center_buffer: Size of center buffer for probability calculation.

    Returns
    -------
        Tuple of dictionaries containing max probabilities, sums, and reference counts.
    """
    maxs: dict[datetime, dict] = {}
    sums: dict[datetime, dict] = {}
    avg_ref_counts: dict[datetime, int] = {}
    for item in stac_items:
        target_date = item.datetime.date().isoformat()
        target_date = datetime.strptime(target_date, "%Y-%m-%d")
        maxs[target_date] = {}
        sums[target_date] = {}
        avg_ref_counts[target_date] = 0

        main_data = main_data_all[target_date]
        main_item = main_data["tile_item"]
        reference_data = reference_data_all[target_date]

        print("\n---: Summary")
        print(f"Main tile for {target_date}: USE {main_item.id}")
        avg_ref_count = 0
        for reference in reference_data:
            reference_date = reference["tile_item"].time.date().isoformat()
            if reference_date in phase0_release_dates:
                print(f"         Dont use {reference['tile_item'].id} (SBR Phase 0 release)")
            else:
                avg_ref_count += 1
                if avg_ref_count == 1:
                    print(f"Before : Use      {reference['tile_item'].id}")
                    before_ref = reference
                elif avg_ref_count == 2:  # noqa type:ignore
                    print(f"Earlier: Use      {reference['tile_item'].id}")
                    earlier_ref = reference
                elif avg_ref_count <= 5:  # noqa
                    print(f"Average: Use      {reference['tile_item'].id} for last 10 and 5 images version")
                elif avg_ref_count <= 10:  # noqa
                    print(f"Average: Use      {reference['tile_item'].id} for last 10 images version")

        # Data preparation for predicting with average last 10 reference images
        data_avg = copy.copy(reference_data[0])
        valid_refs = [
            ref for ref in reference_data if ref["tile_item"].time.date().isoformat() not in phase0_release_dates
        ]
        avg_ref_count = min(len(valid_refs), 10)  # Cap at 10
        avg_ref_counts[target_date] = avg_ref_count
        print(f"==> Using {avg_ref_count} for the average reference image")
        if use_mean:
            data_avg["crop_arrays"] = np.mean(
                [
                    ref["crop_arrays"]
                    for ref in valid_refs[:10]  # Limit to first 10 valid references
                ],
                axis=0,
            )
            # Also prepare data for last 5 reference images
            data_avg_last5 = copy.copy(reference_data[0])
            data_avg_last5["crop_arrays"] = np.mean(
                [
                    ref["crop_arrays"]
                    for ref in valid_refs[:5]  # Limit to first 5 valid references
                ],
                axis=0,
            )
        else:
            data_avg["crop_arrays"] = np.median(
                [
                    ref["crop_arrays"]
                    for ref in valid_refs[:10]  # Limit to first 10 valid references
                ],
                axis=0,
            )
            # Also prepare data for last 5 reference images
            data_avg_last5 = copy.copy(reference_data[0])
            data_avg_last5["crop_arrays"] = np.median(
                [
                    ref["crop_arrays"]
                    for ref in valid_refs[:5]  # Limit to first 5 valid references
                ],
                axis=0,
            )

        center_y = data_avg["crop_arrays"].shape[1] // 2
        center_x = data_avg["crop_arrays"].shape[2] // 2
        f, ax = plt.subplots(3, len(model_ids), figsize=(30, 20))
        ax = cast(np.ndarray, ax)

        for model_idx, model_id in enumerate(model_ids):
            # Normal strategy = Predict with last two reference images
            preds_normal = predict(
                main_data,
                [before_ref, earlier_ref],
                watershed_params,
                models[model_idx],
                device,
                band_concatenators[model_idx],
                lossFn,
            )
            center = get_center_buffer(preds_normal.binary_probability, center_buffer)
            max_proba = center.max()
            sum_proba = center.sum()
            maxs[target_date][model_id] = {"normal_refs": max_proba}
            sums[target_date][model_id] = {"normal_refs": sum_proba}
            ax[0, model_idx].imshow(preds_normal.binary_probability, vmin=0.0, vmax=1.0, cmap="hot_r")
            date = target_date.isoformat().split("T")[0]
            ax[0, model_idx].set_title(
                f"{date} - Model {model_id} Normal\nCenter 21x21 Sum: {sum_proba:6.1f}, Max {max_proba:5.3f}",
                fontsize=24,
            )
            ax[0, model_idx].scatter(center_x, center_y, color="green", marker="x")
            grid16(ax[0, model_idx])

            # Predict with average of last 10 reference image
            preds_avg = predict(
                main_data,
                [data_avg, data_avg],
                watershed_params,
                models[model_idx],
                device,
                band_concatenators[model_idx],
                lossFn,
            )
            center = get_center_buffer(preds_avg.binary_probability, center_buffer)
            max_proba = center.max()
            sum_proba = center.sum()
            maxs[target_date][model_id]["avg_refs"] = max_proba
            sums[target_date][model_id]["avg_refs"] = sum_proba
            ax[1, model_idx].imshow(preds_avg.binary_probability, vmin=0.0, vmax=1.0, cmap="hot_r")
            date = target_date.isoformat().split("T")[0]
            ax[1, model_idx].set_title(f"Avg(10 Refs) Sum: {sum_proba:.1f}, Max {max_proba:.3f}", fontsize=24)
            ax[1, model_idx].scatter(center_x, center_y, color="green", marker="x")
            grid16(ax[1, model_idx])

            # Predict with average of last 5 reference image
            preds_avg = predict(
                main_data,
                [data_avg_last5, data_avg_last5],
                watershed_params,
                models[model_idx],
                device,
                band_concatenators[model_idx],
                lossFn,
            )
            center = get_center_buffer(preds_avg.binary_probability, center_buffer)
            max_proba = center.max()
            sum_proba = center.sum()
            maxs[target_date][model_id]["avg_last5_refs"] = max_proba
            sums[target_date][model_id]["avg_last5_refs"] = sum_proba
            ax[2, model_idx].imshow(preds_avg.binary_probability, vmin=0.0, vmax=1.0, cmap="hot_r")
            date = target_date.isoformat().split("T")[0]
            ax[2, model_idx].set_title(f"Avg(5 Refs) Sum: {sum_proba:.1f}, Max {max_proba:.3f}", fontsize=24)
            ax[2, model_idx].scatter(center_x, center_y, color="green", marker="x")
            grid16(ax[2, model_idx])
        plt.tight_layout()
        plt.show()
    return maxs, sums, avg_ref_counts


def plot_normal_and_avg_strategy_summary(
    sums: dict[str, dict],
    avg_ref_counts: dict[str, int],
    model_ids: list[str],
    ylim: tuple[int, int] = (-2, 60),
    buffer_width: int = 21,
    use_mean: bool = True,
    release_dates: list[str] = [],  # noqa
) -> None:
    """Plot summary of normal (before/earlier) and average reference methane detection.

    Args:
        sums: Dictionary of centered sum probabilities by date and model.
        avg_ref_counts: Dictionary of reference counts by date.
        model_ids: List of model identifiers.
        ylim: Y-axis limits for plots in %.
        buffer_width: Width of center buffer for probability calculation.
    """
    method = "avg" if use_mean else "median"
    center_nb_px = buffer_width**2
    times: dict[str, list] = {id_: [] for id_ in model_ids}
    detections: dict[str, list] = {id_: [] for id_ in model_ids}
    detections_10avg: dict[str, list] = {id_: [] for id_ in model_ids}
    detections_5avg: dict[str, list] = {id_: [] for id_ in model_ids}
    avg_ref_counts_ = []
    for date, date_data in sums.items():
        avg_ref_counts_.append(avg_ref_counts[date])
        for model_id in model_ids:
            times[model_id].append(date)
            detections[model_id].append(100 * date_data[model_id]["normal_refs"] / center_nb_px)
            detections_10avg[model_id].append(100 * date_data[model_id]["avg_refs"] / center_nb_px)
            detections_5avg[model_id].append(100 * date_data[model_id]["avg_last5_refs"] / center_nb_px)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    stop_idx = None  # -35 #None #-35
    for detections_, ref_label in zip(
        [detections, detections_10avg, detections_5avg], ["normal", f"{method} 10", f"{method} 5"], strict=False
    ):
        fig, ax = plt.subplots(figsize=(12, 4))
        for model_idx, model_id in enumerate(model_ids):
            ax.plot(
                times[model_id][:stop_idx],
                detections_[model_id][:stop_idx],
                color=colors[model_idx],
                linewidth=1.5,
                marker="o",
                ls="--",
                label=f"Model {model_id} using {ref_label} reference",
            )
        ax.plot(
            times[model_id][:stop_idx],
            np.mean([detections_[model_id] for model_id in model_ids], axis=0)[:stop_idx],
            color="#9467bd",
            linewidth=3,
            marker="o",
            label=f"{method.capitalize()} of all models",
        )
        for date in release_dates:
            ax.axvline(x=datetime.strptime(date, "%Y-%m-%d"), color="lime", linestyle="-", linewidth=1.5)  # type:ignore

        ax.set_ylabel("% Methane in center")
        ax.set_xticks(times[model_id][:stop_idx])  # Set xticks to exactly match the times values
        ax.set_xticklabels(
            [k.isoformat().split("T")[0] for k in times[model_id][:stop_idx]], ha="right"
        )  # Align labels to the right
        ax.tick_params(axis="x", rotation=45)
        plt.title(f"Methane Center {buffer_width}x{buffer_width} Binary Prob % Sum")
        ax.grid(True)
        ax.set_ylim(ylim)
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        times[model_id][:stop_idx],
        avg_ref_counts_[:stop_idx],
        color="#8c564b",
        linewidth=2,
        marker="o",
        ls="--",
    )
    ax.set_ylabel(f"#Reference images used in {method}")
    ax.set_xticks(times[model_id][:stop_idx])
    ax.tick_params(axis="x", rotation=45)
    plt.title(f"How many images are used in the {method} 10 reference")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_wind(  # noqa: PLR0913
    lon: float,
    lat: float,
    main_item: LandsatGranuleAccess | Sentinel2L1CItem,
    abs_client: BlobServiceClient,
    wind_speed_mps: float,
    wind_direction_deg: float,
    satellite_id: SatelliteID,
    arrow_scale_factor: int = 500,
    percentile_lower: float = 0.5,
    percentile_upper: float = 99.5,
) -> None:
    """
    Plot wind direction and speed as an arrow on a raster image.

    Args:
        lon: Longitude of the arrow origin
        lat: Latitude of the arrow origin
        main_item: Main item containing the raster data
        abs_client: Client for accessing the raster data
        wind_speed_mps: Wind speed in meters per second
        wind_direction_deg: Wind direction (degrees clockwise from North, 0=N, 90=E, 180=S, 270=W)
        arrow_scale_factor = 500 # Adjust this to make the arrow visually appropriately sized (in CRS units)
        percentile_lower: Lower percentile for clipping raster visualization values
        percentile_upper: Upper percentile for clipping raster visualization values
    """
    # The core idea is to:
    # Determine the direction of True North at the arrow's location within your specific projected CRS.
    # Calculate the angle of this North vector relative to the positive x-axis of your projected CRS.
    # Adjust the wind angle (given relative to North) by this North angle offset to get the final angle
    # relative to the positive x-axis of your projected CRS.
    # Calculate the arrow components (dx, dy) using this final angle and the wind speed (scaled for visualization).

    # --- 1. Configuration ---
    if satellite_id == SatelliteID.S2:
        main_item.get_raster_as_tmp("B12", abs_client=abs_client)
        raster_path = "tmp.jp2"  # Path to your raster file
    elif satellite_id == SatelliteID.LANDSAT:
        main_item.get_raster_as_tmp("blue", abs_client=abs_client)
        raster_path = "tmp.tif"  # Path to your raster file
    raster_crs = rasterio.open(raster_path).crs

    # Visualization parameters
    arrow_scale_factor = 500  # Adjust this to make the arrow visually appropriately sized (in CRS units)
    # (e.g., if CRS unit is meters, this arrow represents speed visually over 5000m length)

    # --- 2. Determine North Orientation in Raster CRS ---
    # Define Geographic CRS (WGS84)
    crs_geographic = pyproj.CRS("EPSG:4326")

    # Create transformer from Geographic (Lat/Lon) to Raster CRS
    transformer_to_raster = pyproj.Transformer.from_crs(crs_geographic, raster_crs, always_xy=True)

    # # Transform the arrow origin from Raster CRS to Geographic
    arrow_origin_x, arrow_origin_y = transformer_to_raster.transform(lon, lat)

    # Define a point slightly North in Geographic coordinates
    delta_lat = 0.001  # Small latitude offset (in degrees)
    lat_north = lat + delta_lat
    lon_north = lon

    # Transform this slightly North point back to the Raster CRS
    x_north, y_north = transformer_to_raster.transform(lon_north, lat_north)

    # Calculate the vector representing North in the Raster CRS
    north_vector_x = x_north - arrow_origin_x
    north_vector_y = y_north - arrow_origin_y

    # Calculate the angle of the North vector relative to the positive X-axis of the Raster CRS
    # This angle tells us how 'map North' is rotated relative to the grid's horizontal axis
    # Using atan2 ensures the angle is in the correct quadrant (-pi to pi)
    north_angle_map_rad = math.atan2(north_vector_y, north_vector_x)

    # --- 3. Calculate Arrow Components (dx, dy) in Raster CRS ---

    # Convert wind direction (clockwise from North) to radians
    wind_direction_rad = math.radians(wind_direction_deg)

    # Calculate the final angle of the wind arrow relative to the map's positive X-axis
    # Start with the angle of North on the map, then rotate clockwise by the wind direction
    # Remember: standard math angles are counter-clockwise from +X axis.
    # Angle of North on map = north_angle_map_rad
    # Wind angle relative to North (clockwise) = wind_direction_rad
    # Final angle relative to map's +X axis = north_angle_map_rad - wind_direction_rad
    final_arrow_angle_rad = north_angle_map_rad - wind_direction_rad

    # Calculate arrow components (dx, dy) in the units of the raster CRS
    # Scale the length by wind speed and the visualization scale factor
    arrow_length = wind_speed_mps * arrow_scale_factor  # Visual length in CRS units
    arrow_dx = arrow_length * math.cos(final_arrow_angle_rad)
    arrow_dy = arrow_length * math.sin(final_arrow_angle_rad)

    # --- 4. Plot the Raster and the Arrow ---

    # Define the zoomed-in area (in meters, assuming raster CRS units are meters)
    zoom_size_meters = 2000  # Half-width of the square region (e.g., 2000m gives a 4km x 4km area)

    # Calculate the bounding box for the zoomed-in area
    zoom_left = arrow_origin_x - zoom_size_meters
    zoom_right = arrow_origin_x + zoom_size_meters
    zoom_bottom = arrow_origin_y - zoom_size_meters
    zoom_top = arrow_origin_y + zoom_size_meters
    zoom_extent = [zoom_left, zoom_right, zoom_bottom, zoom_top]

    # Open the raster again to crop to the zoomed-in area
    with rasterio.open(raster_path) as src:
        # Define a window to read only the zoomed-in portion
        window = from_bounds(zoom_left, zoom_bottom, zoom_right, zoom_top, src.transform)
        raster_data = src.read(1, window=window)  # Read the cropped data
        window_transform = src.window_transform(window)  # Get the transform for the cropped window

        # Verify that the window contains data
        if raster_data.size == 0:
            raise ValueError("No raster data found in the specified zoom window. Try increasing zoom_size_meters.")

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # Calculate percentile-based value range to clip extreme values (e.g., clouds)
    vmin = np.percentile(raster_data[~np.isnan(raster_data)], percentile_lower)
    vmax = np.percentile(raster_data[~np.isnan(raster_data)], percentile_upper)

    # Display the raster
    rasterio.plot.show(
        raster_data,
        ax=ax,
        transform=window_transform,
        cmap="viridis",
        extent=zoom_extent,
        vmin=vmin,  # Set minimum value for colormap
        vmax=vmax,  # Set maximum value for colormap
    )

    # Add the wind arrow
    ax.arrow(
        arrow_origin_x,  # x start
        arrow_origin_y,  # y start
        arrow_dx,  # change in x
        arrow_dy,  # change in y
        head_width=arrow_length * 0.2,  # Adjust head width based on arrow length
        head_length=arrow_length * 0.4,  # Adjust head length based on arrow length
        fc="red",  # Fill color
        ec="red",  # Edge color
        length_includes_head=True,  # Make length include the head
    )

    # Optional: Add a North arrow for reference (pointing straight up on the plot means nothing here)
    # Optional: Add text label for wind speed/direction
    ax.text(
        arrow_origin_x * 0.9999 + arrow_dx * 0.0,
        arrow_origin_y * 0.9999 + arrow_dy * 0.0,
        f"{wind_speed_mps:.2f} m/s @ {wind_direction_deg:.3f}°N",
        color="red",
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5", alpha=0.6),
    )
    # Set aspect ratio based on data coordinates (important for projected CRS)
    ax.set_aspect("equal", adjustable="box")

    # Set labels and title
    ax.set_xlabel(f"Easting ({raster_crs.linear_units})")
    ax.set_ylabel(f"Northing ({raster_crs.linear_units})")
    ax.set_title(f"Zoomed-In Raster with Wind Arrow (CRS: {raster_crs.to_epsg()})")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.show()


def validate_pred_retrievals(  # noqa: PLR0913, PLR0915
    combinations_to_visualize: list[dict],
    main_data: dict,
    reference_data: list[dict],
    model_ids: list[str],
    watershed_params: WatershedParameters,
    models: list[nn.Module],
    device: torch.device,
    band_concatenators: list[BaseBandExtractor],
    lossFn: TwoPartLoss,
    lookup_table: RadTranLookupTable,
    wind_speed: float,
    satellite_id: SatelliteID,
    max_distance_pixels: int = 10,
    pixel_width: int = 30,
    show_plots: bool = False,
    simple_plots: bool = False,
    extent: tuple[int, int, int, int] = (0, 128, 0, 128),
    skip_plot_if_no_central_plume: bool = True,
) -> None:
    """Validate and visualize retrieval emission rates."""
    xmin, xmax, ymin, ymax = extent

    emission_rates, Ls, IMEs = [], [], []
    rgb_ratio_shown = False
    for emission_ensemble_selection in combinations_to_visualize:
        model_id = emission_ensemble_selection["model_id"]
        model_idx = model_ids.index(model_id)

        before_date = emission_ensemble_selection["date_before"]
        earlier_date = emission_ensemble_selection["date_earlier"]
        reference_chips = select_reference_tiles_from_datetimes(
            reference_data, before_date=before_date, earlier_date=earlier_date
        )
        before_data = reference_chips[0]
        earlier_data = reference_chips[1]

        L, IME, emission_rate, preds, rescaled_retrieval, center_plume_mask = predict_retrieval(
            main_data,
            reference_chips,
            watershed_params,
            models[model_idx],
            device,
            band_concatenators[model_idx],
            lossFn,
            lookup_table,
            max_distance_pixels,
            pixel_width,
            wind_speed,
        )
        if emission_rate == -1:
            print(
                f"Model {model_id:3} for B={before_date} and E={earlier_date}: No central plume found "
                f"within {max_distance_pixels} pixels"
            )
            if skip_plot_if_no_central_plume:
                continue

        emission_rates.append(emission_rate)
        Ls.append(L)
        IMEs.append(IME)
        print(
            f"Model {model_id:3} for B={before_date} and E={earlier_date}: L={L:6.1f}, IME={IME:7.1f}, "
            f"Emission Rate: {emission_rate:4.0f}"
        )

        if show_plots:
            if satellite_id == SatelliteID.S2:
                rgb_main = get_rgb_bands(main_data["crop_arrays"], BANDS)
                ratio_main = get_band_ratio(main_data["crop_arrays"], BANDS)
            elif satellite_id == SatelliteID.LANDSAT:
                rgb_main = get_rgb_bands_landsat(main_data["crop_arrays"], BANDS_LANDSAT)
                ratio_main = get_band_ratio_landsat(main_data["crop_arrays"], BANDS_LANDSAT)
            else:
                raise ValueError(f"Unsupported satellite ID: {satellite_id}")

            date_main = main_data["tile_item"].time.date().isoformat()

            if simple_plots:
                retrieval_plots(
                    rgb_main=rgb_main[xmin:xmax, ymin:ymax],
                    ratio_main=ratio_main[xmin:xmax, ymin:ymax],
                    binary_probability=preds.binary_probability[xmin:xmax, ymin:ymax],
                    rescaled_retrieval=rescaled_retrieval[xmin:xmax, ymin:ymax],
                    marginal_retrieval=preds.marginal_retrieval[xmin:xmax, ymin:ymax],  # type:ignore
                    center_plume_mask=center_plume_mask[xmin:xmax, ymin:ymax],
                    watershed_segmentation_params=watershed_params,
                    date_main=date_main,
                    show_rgb_ratio=not rgb_ratio_shown,
                )
                rgb_ratio_shown = True
            else:
                if satellite_id == SatelliteID.S2:
                    rgb_before = get_rgb_bands(before_data["crop_arrays"], BANDS)
                    rgb_earlier = get_rgb_bands(earlier_data["crop_arrays"], BANDS)
                    ratio_before = get_band_ratio(before_data["crop_arrays"], BANDS)
                    ratio_earlier = get_band_ratio(earlier_data["crop_arrays"], BANDS)
                elif satellite_id == SatelliteID.LANDSAT:
                    rgb_before = get_rgb_bands_landsat(before_data["crop_arrays"], BANDS_LANDSAT)
                    rgb_earlier = get_rgb_bands_landsat(earlier_data["crop_arrays"], BANDS_LANDSAT)
                    ratio_before = get_band_ratio_landsat(before_data["crop_arrays"], BANDS_LANDSAT)
                    ratio_earlier = get_band_ratio_landsat(earlier_data["crop_arrays"], BANDS_LANDSAT)

                date_before = before_data["tile_item"].time.date().isoformat()
                date_earlier = earlier_data["tile_item"].time.date().isoformat()

            all_error_analysis_plots(
                rgb_main=rgb_main[xmin:xmax, ymin:ymax],
                rgb_before=rgb_before[xmin:xmax, ymin:ymax],
                rgb_earlier=rgb_earlier[xmin:xmax, ymin:ymax],
                ratio_main=ratio_main[xmin:xmax, ymin:ymax],
                ratio_before=ratio_before[xmin:xmax, ymin:ymax],
                ratio_earlier=ratio_earlier[xmin:xmax, ymin:ymax],
                predicted_frac=preds.marginal[xmin:xmax, ymin:ymax],
                predicted_mask=preds.mask[xmin:xmax, ymin:ymax],
                conditional_pred=preds.conditional[xmin:xmax, ymin:ymax],
                binary_probability=preds.binary_probability[xmin:xmax, ymin:ymax],
                watershed_segmentation_params=watershed_params,
                conditional_retrieval=preds.conditional_retrieval[xmin:xmax, ymin:ymax],  # type:ignore
                masked_conditional_retrieval=preds.masked_conditional_retrieval[xmin:xmax, ymin:ymax],  # type:ignore
                dates=(date_main, date_before, date_earlier),
                ratio_colorbar=Colorbar.SHARE,
                ratio_diff_colorbar=Colorbar.SHARE,
                rescaled_retrieval=rescaled_retrieval[xmin:xmax, ymin:ymax],
                marginal_retrieval=preds.marginal_retrieval[xmin:xmax, ymin:ymax],  # type:ignore
            )
    print("#" * 100)
    print("Summary")
    print("#" * 100)
    print(f"L              : {np.mean(Ls):7.1f} +/- {np.std(Ls):6.1f}")
    print(f"IME            : {np.mean(IMEs):7.1f} +/- {np.std(IMEs):6.1f}")
    print(f"Q=Emission_rate: {np.mean(emission_rates):7.1f} +/- {np.std(emission_rates):6.1f}")
    print("#" * 100)
    print("#" * 100)
