"""Tests for src.data.dataset.py."""

import logging
from collections.abc import Callable

import dask.dataframe as dd
import pyarrow as pa
import torch
from azureml.fsspec import AzureMachineLearningFileSystem
from torch.utils.data import DataLoader

from src.data.dataset import ArrowDataset, DaskDataFrameDataset, collate_rowgroups
from src.training.transformations import ConcatenateSnapshots
from src.utils.parameters import (
    SATELLITE_COLUMN_CONFIGS,
    TARGET_COLUMN,
    SatelliteID,
)

logger = logging.getLogger(__name__)


# FIXME: Hack for now to get tests passing, should refactor these tests to be satellite agnostic
# Identity transform for EMIT data, ConcatenateSnapshots for others
def get_transform_function(sat_key: SatelliteID, concatenator_config: dict) -> Callable:
    """Get the appropriate transform function for each satellite type."""

    def transform(x: tuple[dict[str, torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform function for different satellite data."""
        inputs, target = x
        if sat_key == SatelliteID.EMIT:
            return inputs["crop_main"], target

        if sat_key not in concatenator_config:
            raise ValueError(f"Unknown satellite type: {sat_key}")

        return ConcatenateSnapshots(**concatenator_config[sat_key])(x)

    return transform


def test_local_dask_dataframe_dataset(
    local_file: str, sat_key: SatelliteID, concatenator_config: dict[SatelliteID, dict]
) -> None:
    """Tests that parquet file is correctly transfomed into a DaskDataFrameDataset and loaded in batches."""
    dask_dataframe = dd.read_parquet(local_file, dtype_backend="pyarrow", split_row_groups=1, filesystem=None)

    columns_config = SATELLITE_COLUMN_CONFIGS[sat_key]
    ds = DaskDataFrameDataset(
        dask_dataframe,
        columns_config,
        target_column=TARGET_COLUMN,
        transform=get_transform_function(sat_key, concatenator_config),
    )

    dataloader = DataLoader(ds, batch_size=2, collate_fn=collate_rowgroups, shuffle=True)
    inputs_batch, labels_batch = next(iter(dataloader))

    expected_length = 4
    assert len(inputs_batch.shape) == expected_length
    assert len(labels_batch.shape) == expected_length


def test_amlfs_dask_dataframe_dataset(
    remote_files: dict[SatelliteID, str],
    sat_key: SatelliteID,
    fs: AzureMachineLearningFileSystem,
    concatenator_config: dict[SatelliteID, dict],
) -> None:
    """Tests that parquet file on AMLFS is correctly transfomed into a DaskDataFrameDataset and loaded in batches."""
    remote_file = remote_files[sat_key]
    dask_dataframe = dd.read_parquet(remote_file, dtype_backend="pyarrow", split_row_groups=1, filesystem=fs)

    columns_config = SATELLITE_COLUMN_CONFIGS[sat_key]
    ds = DaskDataFrameDataset(
        dask_dataframe,
        columns_config,
        target_column=TARGET_COLUMN,
        transform=get_transform_function(sat_key, concatenator_config),
    )

    dataloader = DataLoader(ds, batch_size=2, collate_fn=collate_rowgroups, shuffle=True)
    inputs_batch, labels_batch = next(iter(dataloader))

    expected_length = 4
    assert len(inputs_batch.shape) == expected_length
    assert len(labels_batch.shape) == expected_length


def test_local_arrow_dataset(
    local_file: str, sat_key: SatelliteID, concatenator_config: dict[SatelliteID, dict]
) -> None:
    """Tests that parquet file is correctly transfomed into a ArrowDataset and loaded in batches."""
    parquet_dataset = pa.dataset.dataset(
        local_file,
        format="parquet",
        exclude_invalid_files=True,
    )

    columns_config = SATELLITE_COLUMN_CONFIGS[sat_key]
    ds = ArrowDataset(
        parquet_dataset,
        columns_config,
        target_column=TARGET_COLUMN,
        transform=get_transform_function(sat_key, concatenator_config),
    )

    dataloader = DataLoader(ds, batch_size=2, collate_fn=collate_rowgroups, shuffle=True)
    inputs_batch, labels_batch = next(iter(dataloader))

    expected_length = 4

    assert len(inputs_batch.shape) == expected_length
    assert len(labels_batch.shape) == expected_length


def test_amlfs_arrow_dataset(
    remote_files: dict[SatelliteID, str],
    sat_key: SatelliteID,
    fs: AzureMachineLearningFileSystem,
    concatenator_config: dict[SatelliteID, dict],
) -> None:
    """Tests that parquet file on AMLFS is correctly transfomed into an ArrowDataset and loaded in batches."""
    remote_file = remote_files[sat_key]
    parquet_dataset = pa.dataset.dataset(
        remote_file,
        format="parquet",
        exclude_invalid_files=True,
        filesystem=fs,
    )

    columns_config = SATELLITE_COLUMN_CONFIGS[sat_key]
    ds = ArrowDataset(
        parquet_dataset,
        columns_config,
        target_column=TARGET_COLUMN,
        transform=get_transform_function(sat_key, concatenator_config),
        filesystem=fs,
    )

    dataloader = DataLoader(ds, batch_size=2, collate_fn=collate_rowgroups, shuffle=True)
    inputs_batch, labels_batch = next(iter(dataloader))

    expected_length = 4
    assert len(inputs_batch.shape) == expected_length
    assert len(labels_batch.shape) == expected_length
