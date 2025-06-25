"""Custom pytorch Datasets."""

import functools
import json
import logging
import os
import time
import warnings
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from typing import Any

import dask.dataframe as dd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from azureml.fsspec import AzureMachineLearningFileSystem
from torch.utils.data import Dataset, Subset

from src.training.transformations import ConcatenateSnapshots

logger = logging.getLogger(__name__)


def collate_rowgroups(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process the batch to return from the DataLoader.

    The inputs and targets are unpacked into separate lists and then concatenated into torch tensors.

    Return
    ------
        (labels, indices)
    """
    # Unpack the batch into separate lists for inputs and targets
    X_list, Y_list = zip(*batch, strict=False)
    return torch.cat(X_list), torch.cat(Y_list)


@dataclass
class Partition:
    """Store information about the partition to be able to retrieve it."""

    file: str
    row_group: int


@functools.lru_cache(maxsize=5000)
def create_parquet_file(file: str, filesystem: AzureMachineLearningFileSystem | None) -> pq.ParquetFile:
    """Cache the creation of the same pq.ParquetFile for each partition read.

    Creating a new instance of pq.ParquetFile adds 50% overhead, but it's not pickleable
    so we can't store the files already transformed into ParquetFiles as a class attribute.

    This needs to be a function and not a class method because it could lead to a memory leak
    - https://docs.astral.sh/ruff/rules/cached-instance-method/
    """
    return pq.ParquetFile(file, filesystem=filesystem)


class ArrowDataset(Dataset):
    """Implement a PyTorch dataset using arrow to access rowgroups of .parquet files.

    Arguments:
    ---------
    dataset:
        an Arrow dataset, for example obtained from `pyarrow.dataset.dataset()
    columns_config:
        dictionary containing the name, shape, and dtype of the arrays.
    target_column:
        name of target column
    transform:
        function (or callable) transforming the inputs and/or target
        This could include things like log-transforms, normalization, choice of bands, choice of time steps, etc.
        The input to this function is the output of the `parse_inputs` method and is a
        dictionary of type Dict[str, torch.Tensor].
    parquet_to_rowgroups_mapping_path:
        path to a json file containing the mapping of row groups to files.
        If this file exists, it will be used to create the partition mapping much faster.
        Otherwise, the mapping will be created from the parquet files one by one.
    """

    def __init__(
        self,
        dataset: ds.FileSystemDataset,
        columns_config: dict[str, dict[str, Any]],
        target_column: str,
        transform: ConcatenateSnapshots | Callable,
        filesystem: AzureMachineLearningFileSystem | None = None,
        parquet_to_rowgroups_mapping_path: str = "parquet_rowgroups_data_aviris_X.json",
    ):
        """Initialize an ArrowDataset."""
        super().__init__()
        self.dataset = dataset
        self.target_col = target_column
        columns_config = copy(columns_config)

        # Extract target config
        target_config = columns_config.pop(target_column)
        self.target_shape = target_config["shape"]
        self.target_dtype = target_config["dtype"]

        column_names = self.dataset.schema.names
        if target_column not in column_names:
            if "target_frac" in column_names:
                # This is a legacy file, switch target column to the old name.
                self.target_col = target_column = "target_frac"
            else:
                raise KeyError(
                    f"Column {target_column} is not present in this parquet file. "
                    f"Available columns: {column_names}. "
                    "To use a legacy column name, modify TARGET_COLUMN in parameters.py."
                )

        self.orig_swir16_band_name = "orig_swir16"
        self.orig_swir22_band_name = "orig_swir22"
        if "orig_swir16" in columns_config:
            # This is a legacy file, switch swir16/swir22 columns to the old names.
            if "orig_swir16" not in column_names:
                columns_config["orig_band_11"] = columns_config["orig_swir16"]
                columns_config.pop("orig_swir16")
                self.orig_swir16_band_name = "orig_band_11"
            if "orig_swir22" not in column_names:
                columns_config["orig_band_12"] = columns_config["orig_swir22"]
                columns_config.pop("orig_swir22")
                self.orig_swir22_band_name = "orig_band_12"

        # Extract input configs
        self.input_columns = []
        self.input_shapes = []
        self.input_dtypes = []

        for col, config in columns_config.items():
            self.input_columns.append(col)
            self.input_shapes.append(config["shape"])
            self.input_dtypes.append(config["dtype"])
            if col not in column_names:
                raise KeyError(f"Column {col} is not present in this parquet file. Available columns: {column_names}. ")
        self.transform = transform
        self.filesystem = filesystem

        self.partitions: list[Partition] = self.create_partition_mapping(
            self.dataset, parquet_to_rowgroups_mapping_path
        )

    def __repr__(self) -> str:
        """Str representation."""
        return f"ArrowDataset({self.dataset.files})"

    def __len__(self) -> int:
        """Return the number of row_groups in the dataset.  This should correspond to RecordBatches."""
        return len(self.partitions)

    def create_partition_mapping(
        self, dataset: ds.Dataset, parquet_to_rowgroups_mapping_path: str = "parquet_rowgroups_data_aviris_X.json"
    ) -> list[Partition]:
        """Create a mapping of partition number to the file and row_group."""
        partitions = []
        if os.path.exists(parquet_to_rowgroups_mapping_path):
            # If the mapping exists, this is much faster
            with open(parquet_to_rowgroups_mapping_path) as json_file:
                parquet_rowgroups = json.load(json_file)
            for file in dataset.files:
                num_row_groups = parquet_rowgroups[file.split("/")[-1]]
                for row_group in range(num_row_groups):
                    partitions.append(Partition(file, row_group))
        else:
            start = time.time()
            for file in dataset.files:
                num_row_groups = pq.ParquetFile(file, filesystem=self.filesystem).num_row_groups
                for row_group in range(num_row_groups):
                    partitions.append(Partition(file, row_group))
            logger.info(f"Setting up the partitions manually took {time.time() - start:.1f}s")
        return partitions

    def get_partition(self, idx: int, columns: list[str] | None = None) -> pa.Table:
        """Return a RecordBatch for the associated row group.

        Partitions are mappings to files & row groups within files.

        During training, we want to randomly sample examples for each batch.  For efficiency (because random access ]
        is really not efficient), we instead sample row_groups, with 10 rows per row group.  Dask handily handles these
        groupings by considering each row_group a partition (unless the dataset gets repartitioned differently).  Arrow,
        does a similar thing, by considering each row_group a RecordBatch by default. Where things break down with an
        Arrow dataset is that it doesn't store the information about each row_group nor expose an API to retrieve
        specific row groups (it does for rows though).  The pq.ParquetFile does provide an API for
        accessing specific row_groups.  So we simpoly create our own mapping with a list[Partition], where Partition
        stores the information necessary to retrive a specific row_group.  Then, we can simply randomly sample the list.
        """
        partition = self.partitions[idx]

        parquet_file = create_parquet_file(partition.file, self.filesystem)
        return parquet_file.read_row_group(partition.row_group, columns)

    def read_bytes_into_array(self, input_array: bytes, dtype: torch.dtype, shape: tuple) -> torch.Tensor:
        """Interpret numpy bytes and convert to torch tensor.

        Bytes are saved using numpy's .tobytes() so we read them back into a numpy array
        before converting to a torch tensor.
        """
        with warnings.catch_warnings():  # (ignore warning about non-writable buffer)
            warnings.simplefilter("ignore", category=UserWarning)
            tensor = torch.frombuffer(input_array, dtype=dtype).reshape(shape)

        assert tensor.shape == shape
        return tensor

    def parse_row_group_bytes(self, batch: pa.Table, dtype: torch.dtype, column: str, shape: tuple) -> list:
        """Parse the bytes in each row in the column into a tensor.

        Iterate over the rows.
        """
        column_array = batch.column(column)

        arrays = []
        for row in column_array:
            arrays.append(self.read_bytes_into_array(row.as_py(), dtype=dtype, shape=shape))

        return arrays

    def parse_target(self, df: pa.Table) -> torch.Tensor:
        """Parse the bytes for the target."""
        return torch.stack(self.parse_row_group_bytes(df, self.target_dtype, self.target_col, self.target_shape))

    def parse_inputs(self, df: pa.Table) -> dict[str, torch.Tensor]:
        """Parse the bytes for each input column into a torch tensor.

        Iterate over the columns.

        Return
        ------
        A mapping of band name to the data.
        """
        inputs = {}
        for col, shape, dtype in zip(self.input_columns, self.input_shapes, self.input_dtypes, strict=True):
            inputs[col] = torch.stack(self.parse_row_group_bytes(df, dtype, col, shape))
        return inputs

    def get_untransformed_data(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Parse the raw bytes into torch tensors for the inputs and the target."""
        # Use PyArrow to retrieve the data at the given index
        sample = self.get_partition(idx, [*list(self.input_columns), self.target_col])

        # The inputs and target are stored as byte strings,
        # so we need to parse them into pytorch tensors of the right shape
        target = self.parse_target(sample)
        inputs = self.parse_inputs(sample)

        return inputs, target

    def get_metadata(self, idx: int, columns: list[str]) -> dict:
        """Retrieve data from given columns as a dictionary.

        Will be of the form {"column": [<data>]}
        """
        partition = self.partitions[idx]
        table = self.get_partition(idx, columns)

        metadata = table.to_pydict()
        num_rows = len(table)
        metadata.update(
            {
                "file": [partition.file] * num_rows,
                "row_group": [partition.row_group] * num_rows,
                "dataset_idx": [idx] * num_rows,
            }
        )
        return metadata

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve the transformed input and target."""
        untransformed_input, untransformed_target = self.get_untransformed_data(idx)
        return self.transform((untransformed_input, untransformed_target))


class DaskDataFrameDataset(Dataset):
    """Implements the pytorch Dataset class for our recycled data.

    See the pytorch documentation [here](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
    for background on Dataset and Dataloader classes.

    The implementation uses a map-style dataset instead of an iterable dataset.
    See [the pytorch documentation](https://pytorch.org/docs/stable/data.html#dataset-types)
    for a bit of background on this distinction.
    The map-style dataset will allow us to shuffle the samples,
    so we do not always see the same tiles in the same order in every epoch.
    This is quite important as we are carving a large number of samples out of each Sentinel 2 MGRS scene,
    so there isn't very much diversity in contiguous samples.
    If we don't shuffle, the neural network will converge before being exposed to scenes with a different topography,
    with the same tiles being effectively given more weight in every epoch.

    It's important to understand that each index in the map-style dataset actually returns a row **group**
    from the original parquet files.
    This is for efficiency, as reading just one row at a time would be wasteful.
    These contiguous rows therefore form a single "sample".
    I think this is fairly standard, especially for parquet files
    to take advantage of the structure of the parquet format.
    See the [ParquetDataFrameLoader](https://pytorch.org/data/main/generated/torchdata.datapipes.iter.ParquetDataFrameLoader.html#parquetdataframeloader)
    type in `torchdata` for an example of this,
    and have a look at the [source here](https://github.com/pytorch/data/blob/d727f63289ae18ec3989c3bca4cb7e890506cce2/torchdata/datapipes/iter/util/dataframemaker.py#L95).
    """

    def __init__(
        self,
        dask_dataframe: dd.DataFrame,
        columns_config: dict[str, dict[str, Any]],
        target_column: str,
        transform: ConcatenateSnapshots | Callable,
    ):
        """Initialize a DaskDataFrameDataset.

        Arguments:
        ---------
        dask_dataframe: dask.dataframe.DataFrame
            a dask dataframe object, for example obtained from `dask.dataframe.read_parquet()
        columns_config:
            dictionary containing the name, shape, and dtype of the arrays.
        target_column:
            name of target column
        transform:
            function (or callable) transforming the predictors and/or target
            This could include things like log-transforms, normalization, choice of bands,
            choice of time steps, etc.
            The input to this function is the output of the `parse_inputs` method,
            and is a dictionary of type Dict[str, torch.Tensor].
            For Sentinel-2 data, a sensible normalization choice is to divide by 10,000.
        """
        super().__init__()
        self.dask_dataframe = dask_dataframe
        self.target_col = target_column
        columns_config = copy(columns_config)

        # Extract target config
        target_config = columns_config.pop(target_column)
        self.target_shape = target_config["shape"]
        self.target_dtype = target_config["dtype"]

        # Extract input configs
        self.input_columns = []
        self.input_shapes = []
        self.input_dtypes = []
        for col, config in columns_config.items():
            self.input_columns.append(col)
            self.input_shapes.append(config["shape"])
            self.input_dtypes.append(config["dtype"])
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of partitions in the dask dataframe.

        Note: the number of partitions is not the number of rows.  To calculate the number of rows would require
        computing the whole dataframe, which we generally would like to avoid since these are large.
        """
        return self.dask_dataframe.npartitions

    def read_bytes_into_array(self, input_array: bytes, dtype: torch.dtype, shape: tuple) -> torch.Tensor:
        """Interpret numpy bytes and convert to torch tensor.

        Bytes are saved using numpy's .tobytes() so we read them back into a numpy array
        before converting to a torch tensor.
        """
        with warnings.catch_warnings():  # (ignore warning about non-writable buffer)
            warnings.simplefilter("ignore", category=UserWarning)
            tensor = torch.frombuffer(input_array, dtype=dtype).reshape(shape)

        assert tensor.shape == shape
        return tensor

    def parse_target(self, df: dd.DataFrame) -> torch.Tensor:
        """Parse the bytes for the target."""
        target = df[self.target_col]

        return torch.stack(
            [self.read_bytes_into_array(b, dtype=self.target_dtype, shape=self.target_shape) for b in target]
        )

    def parse_inputs(self, df: dd.DataFrame) -> dict[str, torch.Tensor]:
        """Parse the bytes for each band in the columns.

        Return
        ------
        A mapping of band name to the data.
        """
        inputs_df = df[list(self.input_columns)]
        inputs = {}
        for col, shape, dtype in zip(self.input_columns, self.input_shapes, self.input_dtypes, strict=True):
            # the raw bytes in each cell in the column are fed into the function `read_bytes_into_array` with
            # arguments for the cell's dtype and shape.  we then stack the resulting arrays into a tensor
            inputs[col] = torch.stack(
                inputs_df[col].apply(self.read_bytes_into_array, dtype=dtype, shape=shape).tolist()
            )
        return inputs

    def get_untransformed_data(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Parse the raw bytes into torch tensors for the inputs and the target."""
        # Use Dask DataFrame to retrieve the data at the given index
        sample = self.dask_dataframe.get_partition(idx)[[*list(self.input_columns), self.target_col]].compute()

        # The inputs and target are stored as byte strings, so we need to parse them
        # into pytorch tensors of the right shape
        inputs = self.parse_inputs(sample)
        target = self.parse_target(sample)

        return inputs, target

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve the transformed input and target."""
        untransformed_input, untransformed_target = self.get_untransformed_data(idx)
        return self.transform((untransformed_input, untransformed_target))


def collate_rowgroups_diagnostics(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor | dict]:
    """Process the batch to return from the DataLoader.

    The inputs and targets are unpacked into separate lists and then concatenated into torch tensors.

    Return
    ------
        {
            "subset_idx": torch.Tensor,
            "partition": torch.Tensor,
            "untransformed_input": dict[str, torch.Tensor],
            "untransformed_target": torch.Tensor,
            "X": torch.Tensor,
            "y": torch.Tensor
        }
    """
    crop_earlier: list[torch.Tensor] = []
    crop_main: list[torch.Tensor] = []
    crop_before: list[torch.Tensor] = []
    orig_swir16: list[torch.Tensor] = []
    orig_swir22: list[torch.Tensor] = []
    for b in batch:
        input = b["untransformed_input"]
        crop_earlier.extend(input["crop_earlier"])  # type: ignore
        crop_main.extend(input["crop_main"])  # type: ignore
        crop_before.extend(input["crop_before"])  # type: ignore
        orig_swir16.extend(input["orig_swir16"])  # type: ignore
        orig_swir22.extend(input["orig_swir22"])  # type: ignore

    untransformed_input = {
        "crop_earlier": torch.cat(crop_earlier),
        "crop_main": torch.cat(crop_main),
        "crop_before": torch.cat(crop_before),
        "orig_swir16": torch.cat(orig_swir16),
        "orig_swir22": torch.cat(orig_swir22),
    }

    return {
        "subset_idx": torch.tensor([x["subset_idx"] for x in batch]),
        "partition": torch.tensor([x["partition"] for x in batch]),
        "untransformed_input": untransformed_input,
        "untransformed_target": torch.cat([x["untransformed_target"] for x in batch]),
        "X": torch.cat([x["X"] for x in batch]),
        "y": torch.cat([x["y"] for x in batch]),
    }


class DiagnosticsDataset(Dataset):
    """Wrapper class for a Subset dataset that returns the original index alongside untransformed and transformed Xy."""

    def __init__(self, subset: Subset):
        self.subset = subset
        self.wrapped = subset.dataset

    def __getitem__(self, subset_idx: int) -> dict[str, Any]:
        """Get the untransformed and tranformed data at the given subset index."""
        # Translate subset index to the original dataset index
        partition = self.subset.indices[subset_idx]
        untransformed_input, untransformed_target = self.wrapped.get_untransformed_data(partition)  # type: ignore
        X, y = self.wrapped.transform((untransformed_input, untransformed_target))  # type: ignore
        return {
            "subset_idx": subset_idx,
            "partition": partition,
            "untransformed_input": untransformed_input,
            "untransformed_target": untransformed_target,
            "X": X,
            "y": y,
        }

    def __len__(self) -> int:
        """Return the number of samples in the subset."""
        return self.subset.__len__()


class ValidationDataset(ArrowDataset):
    """Extend ArrowDataset to also return "region_overlap", "emissions"."""

    def __init__(
        self,
        dataset: ds.FileSystemDataset,
        columns_config: dict[str, dict[str, Any]],
        target_column: str,
        transform: ConcatenateSnapshots | Callable,
        filesystem: AzureMachineLearningFileSystem | None = None,
        parquet_to_rowgroups_mapping_path: str = "parquet_rowgroups_data_aviris_X.json",
    ):
        """Initialize an ValidationDataset."""
        super().__init__(
            dataset, columns_config, target_column, transform, filesystem, parquet_to_rowgroups_mapping_path
        )

    def parse_scalar(self, df: pa.Table, column: str) -> list:
        """Parse the bytes for the target."""
        column_array = df.column(column)
        out = []
        for row in column_array:
            out.append(row.as_py())
        return out

    def get_untransformed_data(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, list, list]:  # type:ignore
        """Parse the raw bytes into torch tensors for the inputs and the target."""
        # Use PyArrow to retrieve the data at the given index
        columns = [*self.input_columns, self.target_col, *["plume_emissions", "region_overlap"]]

        sample = self.get_partition(idx, columns)

        # The inputs and target are stored as byte strings, need to parse them into pytorch tensors of the right shape
        target = self.parse_target(sample)
        inputs = self.parse_inputs(sample)
        region_overlaps = self.parse_scalar(sample, "region_overlap")
        emissions = self.parse_scalar(sample, "plume_emissions")

        return inputs, target, region_overlaps, emissions

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], list, list]:  # type:ignore
        """Retrieve the transformed input and target."""
        untransformed_input, untransformed_target, region_overlaps, emissions = self.get_untransformed_data(idx)
        return self.transform((untransformed_input, untransformed_target)), region_overlaps, emissions


def collate_rowgroups_val(
    batch: list[tuple],
) -> dict[str, torch.Tensor | list]:
    """Process the batch to return from the DataLoader."""
    X_list: list[torch.Tensor] = []
    y_list: list[torch.Tensor] = []
    flattened_emissions_list: list[float] = []
    region_overlaps_list: list[str] = []
    for Xy, region, batch_emissions in batch:
        X, y = Xy
        X_list.append(X)  # type: ignore
        y_list.append(y)  # type: ignore
        region_overlaps_list.extend(region)  # type: ignore
        for chip_emissions in batch_emissions:
            if chip_emissions is None or len(chip_emissions) == 0:
                # For AVIRIS plumes, the chip_emissions are just None,
                # whereas for Gaussian plumes, it's an empty list. Sorry.
                # In both cases, we set the emissions for the chip to 0.0.
                flattened_emissions_list.append(0.0)  # type: ignore
            else:
                # The expectation here is that there is only one emission rate per chip.
                assert len(chip_emissions) == 1, "Expected only one emission rate per chip"
                flattened_emissions_list.extend(chip_emissions)  # type: ignore
    return {
        "X": torch.cat(X_list),
        "y": torch.cat(y_list),
        "region_overlap": region_overlaps_list,
        "emission": flattened_emissions_list,
    }
