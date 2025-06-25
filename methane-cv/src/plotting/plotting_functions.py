import matplotlib
import numpy as np
import torch
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator

from src.data.landsat_data import LandsatQAValues
from src.models.varon_models import get_orbio_frac_from_varon
from src.utils.parameters import SatelliteID

# Classification table colors
COLORS = [
    "#000000",  # NO_DATA
    "#FF0000",  # SATURATED_OR_DEFECTIVE
    "#804000",  # DARK_AREA_PIXELS
    "#808080",  # CLOUD_SHADOWS
    "#00FF00",  # VEGETATION
    "#FFFF00",  # NOT_VEGETATED
    "#0000FF",  # WATER
    "#9E2283",  # UNCLASSIFIED
    "#C0C0C0",  # CLOUD_MEDIUM_PROBABILITY
    "#FFFFFF",  # CLOUD_HIGH_PROBABILITY
    "#00FFFF",  # THIN_CIRRUS
    "#FF00FF",  # SNOW
]

# Class names
S2_LAND_COVER_CLASSIFICATIONS = [
    "NO_DATA",
    "SATURATED_OR_DEFECTIVE",
    "DARK_AREA_PIXELS",
    "CLOUD_SHADOWS",
    "VEGETATION",
    "NOT_VEGETATED",
    "WATER",
    "UNCLASSIFIED",
    "CLOUD_MEDIUM_PROBABILITY",
    "CLOUD_HIGH_PROBABILITY",
    "THIN_CIRRUS",
    "SNOW",
]

# Generate a colormap
CMAP = ListedColormap(COLORS)

LANDSAT_QA_COLORS_DICT = {
    LandsatQAValues.FILL: "#000000",  # no data, black
    LandsatQAValues.DILATED_CLOUD: "#E6E6E6",  # light gray
    LandsatQAValues.CIRRUS: "#E6FFFF",  # very light cyan
    LandsatQAValues.CLOUD: "#FFFFFF",  # pure white
    LandsatQAValues.CLOUD_SHADOW: "#999999",  # medium gray
    LandsatQAValues.SNOW: "#E6F3FF",  # very light blue
    LandsatQAValues.CLEAR: "#FFFF00",  # yellow
    LandsatQAValues.WATER: "#000099",  # deep blue
    LandsatQAValues.CLOUD_CONFIDENCE_LOW: "#F2F2F2",  # slightly gray white
    LandsatQAValues.CLOUD_CONFIDENCE_MEDIUM: "#F8F8F8",  # nearly white
    LandsatQAValues.CLOUD_CONFIDENCE_HIGH: "#FFFFFF",  # pure white
    LandsatQAValues.CLOUD_SHADOW_CONFIDENCE_LOW: "#B3B3B3",  # light gray
    LandsatQAValues.CLOUD_SHADOW_CONFIDENCE_MEDIUM: "#808080",  # medium gray
    LandsatQAValues.CLOUD_SHADOW_CONFIDENCE_HIGH: "#4D4D4D",  # dark gray
    LandsatQAValues.SNOW_ICE_CONFIDENCE_LOW: "#E6F3FF",  # very light blue
    LandsatQAValues.SNOW_ICE_CONFIDENCE_MEDIUM: "#B3D9FF",  # light blue
    LandsatQAValues.SNOW_ICE_CONFIDENCE_HIGH: "#66B2FF",  # stronger blue
    LandsatQAValues.CIRRUS_CONFIDENCE_LOW: "#F2FFFF",  # very light cyan
    LandsatQAValues.CIRRUS_CONFIDENCE_MEDIUM: "#CCF2F2",  # light cyan
    LandsatQAValues.CIRRUS_CONFIDENCE_HIGH: "#99E6E6",  # stronger cyan
}
LANDSAT_QA_CMAP = ListedColormap(list(LANDSAT_QA_COLORS_DICT.values()))

# the presumed band mapping of the above spectra, in case wavelengths are not available
EMIT_COLOR_BAND_RANGES = {"red": slice(33, 49), "green": slice(17, 24), "blue": slice(10, 14)}
S2_COLOR_BANDS = {"red": "B04", "green": "B03", "blue": "B02"}
LANDSAT_COLOR_BANDS = {"red": "red", "green": "green", "blue": "blue"}

# EMIT band ratio ranges identified by using the below figure for adjacent SWIR ranges
# without and with methane absorption, respectively, and universally without water
# vapour absorption. These were then refined by looking at the RADTRAN absorption estimates
# in the EMIT-radtran.ipynb notebook. These are *not* the same SWIR ranges as Sentinel-2 uses.
# https://www.nature.com/articles/s41598-023-44918-6/figures/2
# EMIT bands 217-231 and 244-261 correspond to wavelengths 2000-2100nm and 2200-2400nm.
EMIT_SWIR_RATIO_RANGES = {"swir1": slice(217, 231), "swir2": slice(244, 261)}
S2_SWIR_RATIO_BANDS = {"swir1": "B11", "swir2": "B12"}
LANDSAT_SWIR_RATIO_BANDS = {"swir1": "swir16", "swir2": "swir22"}

# A multiband image tensor should have three dimensions: (bands, rows, cols)
# We use this to assert we have not been passed an invalid input
# e.g. an image without bands (2-dimensional), or a series of images such as a batch (4-dimensional)
N_DIMENSIONS_MULTIBAND_IMAGE = 3


def get_rgb_from_xarray(x: xr.DataArray, satellite_id: SatelliteID) -> xr.DataArray:
    """Get the RGB representation of a satellite image."""
    assert (
        len(x.dims) == N_DIMENSIONS_MULTIBAND_IMAGE
    ), f"Expected 3-dimensional tensor (bands, rows, cols) but received a {len(x.shape)}-dimensional one."

    if satellite_id == SatelliteID.S2:
        rgb_image = x.sel(bands=list(S2_COLOR_BANDS.values())) / 10_000
    elif satellite_id == SatelliteID.LANDSAT:
        rgb_image = x.sel(bands=list(LANDSAT_COLOR_BANDS.values())) / 10_000
    elif satellite_id == SatelliteID.EMIT:
        color_channels = []
        for band_name, band_range in EMIT_COLOR_BAND_RANGES.items():
            color_spectra = x.sel(bands=band_range)
            if color_spectra.size == 0:
                raise KeyError(f"No spectral overlap with band range {band_range}.")
            color_channel = color_spectra.mean(dim="bands").expand_dims(dim={"bands": [band_name]})
            color_channels.append(color_channel)
        rgb_image = xr.concat(color_channels, dim="bands")
    else:
        raise ValueError(f"Unhandled satellite id {satellite_id}.")

    return rgb_image.transpose("y", "x", "bands")


def get_swir_ratio_from_xarray(x: xr.DataArray, satellite_id: SatelliteID) -> xr.DataArray:
    """Get the ratio between a portion of the SWIR spectrum that absorbs methane and one that does not."""
    assert (
        len(x.dims) == N_DIMENSIONS_MULTIBAND_IMAGE
    ), f"Expected 3-dimensional tensor (bands, rows, cols) but received a {len(x.shape)}-dimensional one."

    if satellite_id == SatelliteID.S2:
        swir1 = x.sel(bands=S2_SWIR_RATIO_BANDS["swir1"])
        swir2 = x.sel(bands=S2_SWIR_RATIO_BANDS["swir2"])
    elif satellite_id == SatelliteID.LANDSAT:
        swir1 = x.sel(bands=LANDSAT_SWIR_RATIO_BANDS["swir1"])
        swir2 = x.sel(bands=LANDSAT_SWIR_RATIO_BANDS["swir2"])
    elif satellite_id == SatelliteID.EMIT:
        # select bands corresponding to our SWIR spectral ranges so we can check if they are empty before
        # taking the mean (otherwise xarray will expand the empty array to a full one of nans when we
        # take the mean)
        swir_spectra = {
            k: x.sel(bands=spectra) for k, spectra in EMIT_SWIR_RATIO_RANGES.items() if x.sel(bands=spectra).size > 0
        }
        swir1 = swir_spectra["swir1"].mean(dim="bands")
        swir2 = swir_spectra["swir2"].mean(dim="bands")
    else:
        raise ValueError(f"Unhandled satellite id {satellite_id}.")

    return swir2 / swir1


def get_rgb_from_tensor(x_tensor: torch.Tensor, bands: list[str], batch_idx: int) -> torch.Tensor:
    """
    Get the RGB for a given tile from batch idex batch_idx.

    x_tensor has shape (batch_num, bands, height, width)
    """
    rgb_bands = [bands.index("B04"), bands.index("B03"), bands.index("B02")]
    rgb_image = x_tensor[batch_idx, rgb_bands, ...].moveaxis(0, -1) / 10000
    return rgb_image


def get_band_ratio_from_tensor(x_tensor: torch.Tensor, bands: list[str], batch_idx: int):
    """
    Get the B12/B11 band ratio for a given tile from batch index batch_idx.

    x_tensor has shape (batch_num, bands, height, width)
    """
    b11_image = x_tensor[batch_idx, bands.index("B11"), ...]
    b12_image = x_tensor[batch_idx, bands.index("B12"), ...]

    ratio = b12_image / b11_image

    return ratio


def get_item_from_list(data: list, batch_idx: int) -> torch.Tensor:
    """Get data item from list."""
    return data[batch_idx]


def calculate_mbmp_frac(
    b11_t: torch.Tensor, b12_t: torch.Tensor, b11_r: torch.Tensor, b12_r: torch.Tensor
) -> torch.Tensor:
    """
    Get an approximation of the multi-band multi-pass (MBMP) frac from Varon 2021.
    The Varon-inspired frac is defined as
        $frac_mbmp = \frac{c_t * B12_t - B11_t}{B11_t} - \frac{c_r * B12_r - B11_r}{B11_r}$
    where t (r) denotes the target (reference) date. The scaling factors c_r and c_t in Varon 2021
    are defined via a least squares fit of B12 against B11. For computational reasons, we instead
    use
        c_{t/r} = np.nanmedian(B11_{t/r} / B12_{t/r}).
    """
    b12_t_safe = np.where(b12_t == 0, np.nan, b12_t)
    b12_r_safe = np.where(b12_r == 0, np.nan, b12_r)
    b11_t_safe = np.where(b11_t == 0, np.nan, b11_t)
    b11_r_safe = np.where(b11_t == 0, np.nan, b11_r)

    c_t = np.nanmedian(b11_t / b12_t_safe)
    c_r = np.nanmedian(b11_r / b12_r_safe)

    mbmp = c_t * b12_t / b11_t_safe - c_r * b12_r / b11_r_safe

    return mbmp


def plot_mbmp_frac_from_tensor(
    x_target_tensor: torch.Tensor,
    x_reference_tensor: torch.Tensor,
    bands: list[str],
    batch_idx: int,
    vmax_pred: float,
    swir16_band_name: str,
    swir22_band_name: str,
) -> matplotlib.image.AxesImage:
    """
    Plot an approximation of the MBMP frac from Varon 2021 for a given tile from batch index batch_idx.

    Frac here is defined as
        frac_mbmp = c_target * B12_target / B11_target - c_ref B12_ref / B11_ref
    with
        c_target = nanmedian(B11_target / B12_target), c_ref = nanmedian(B11_ref / B12_ref

    x_target_tensor and x_reference_tensor have shape (batch_num, bands, height, width)
    """
    swir16_t = x_target_tensor[batch_idx, bands.index(swir16_band_name), ...]
    swir22_t = x_target_tensor[batch_idx, bands.index(swir22_band_name), ...]

    swir16_r = x_reference_tensor[batch_idx, bands.index(swir16_band_name), ...]
    swir22_r = x_reference_tensor[batch_idx, bands.index(swir22_band_name), ...]

    mbmp = calculate_mbmp_frac(swir16_t, swir22_t, swir16_r, swir22_r)
    mbmp_orbio = get_orbio_frac_from_varon(mbmp)

    out = plt.imshow(mbmp_orbio, cmap="RdBu", vmin=-vmax_pred, vmax=vmax_pred)
    return out


def plot_frac(
    frac: torch.Tensor | np.ndarray, colorbar_swapped: bool = False, **kwargs
) -> tuple[matplotlib.image.AxesImage, float]:
    """
    Plot the frac predicted by a NN model
    """
    try:
        # if it's a tensor, convert it to a numpy array
        frac = frac.cpu()
        frac = frac.numpy()
    except:
        pass
    if "vmax" in kwargs:
        vmax = kwargs["vmax"]
        kwargs["vmin"] = -vmax
    else:
        vmax = max(float(np.abs(frac).max()), 0.01)
        kwargs["vmax"] = vmax
        kwargs["vmin"] = -vmax

    if colorbar_swapped:
        out = plt.imshow(frac, cmap="RdBu_r", interpolation="nearest", **kwargs)
    else:
        out = plt.imshow(frac, cmap="RdBu", interpolation="nearest", **kwargs)

    return out, vmax


def grid16(ax: matplotlib.axes.Axes | None = None) -> None:
    """Apply a grid with 16x16 blocks to the current active plot."""
    if ax is None:
        ax = plt.gca()

    ax.xaxis.set_major_locator(MultipleLocator(32))
    ax.xaxis.set_minor_locator(MultipleLocator(16))

    ax.yaxis.set_major_locator(MultipleLocator(32))
    ax.yaxis.set_minor_locator(MultipleLocator(16))

    ax.grid(True, which="both")
