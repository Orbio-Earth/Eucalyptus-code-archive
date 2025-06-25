"""Shared utility functions for data generation across satellites."""

from pathlib import Path

import numpy as np
import torch
import xarray as xr

from src.azure_wrap.blob_storage_sdk_v2 import download_from_blob


def download_plume_catalog(plume_catalog: str) -> Path:
    """Download or locate the plume_catalog."""
    if plume_catalog.startswith("azureml://"):
        local_dir = Path.home() / "localfiles" / "data" / "plumes"
        local_dir = local_dir.expanduser()
        download_from_blob(plume_catalog, local_dir, recursive=False)
        plume_catalog_filename = Path(plume_catalog).name
        local_plume_catalog_path = local_dir / plume_catalog_filename

    else:
        local_plume_catalog_path = Path(plume_catalog).expanduser()

    if not local_plume_catalog_path.exists():
        raise ValueError(f"Plume catalog not found in: {local_plume_catalog_path}")

    return local_plume_catalog_path


def tensor_to_dataarray(tensor: torch.Tensor, bands: list[str] | list[int]) -> xr.DataArray:
    """Convert a tensor to an xarray DataArray.

    Args:
        tensor: The tensor to convert. Must be 3-dimensional: a single image with shape (bands, y, x),
            or 4-dimensional: a batch of images with shape (images, bands, y, x).
        bands: A list of bands available in the image(s).

    Returns: The tensor converted to an xarray DataArray.
    """
    three_dimensional = 3
    four_dimensional = 4

    if tensor.is_cuda:
        tensor = tensor.cpu()

    if len(tensor.shape) == three_dimensional:
        # assume single image with shape (bands, y, x)
        dims = ["bands", "y", "x"]
        coords = {
            "bands": bands,
            "y": np.arange(tensor.shape[1]),
            "x": np.arange(tensor.shape[2]),
        }
    elif len(tensor.shape) == four_dimensional:
        # assume batch of images with shape (images, bands, y, x)
        dims = ["images", "bands", "y", "x"]
        coords = {
            "images": np.arange(tensor.shape[0]),
            "bands": bands,
            "y": np.arange(tensor.shape[2]),
            "x": np.arange(tensor.shape[3]),
        }
    else:
        raise ValueError(
            "Expected 3- or 4-dimensional tensor (single image or batch of images, respectively)."
            f" Received a {len(tensor.shape)}-dimensional tensor."
        )

    return xr.DataArray(tensor.numpy(), dims=dims, coords=coords)
