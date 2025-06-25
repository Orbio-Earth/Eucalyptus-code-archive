"""Benchmarks for data loading.

Run only benchmarks:
```bash
pytest . --benchmark-only
```

Run only partition (row_group) benchmarks:
```bash
pytest . -m partition
```

Run only DataLoader benchmarks:
```bash
pytest . -m dataloader
```
"""

import random

import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from azureml.fsspec import AzureMachineLearningFileSystem
from pytest_benchmark.fixture import BenchmarkFixture
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import ArrowDataset, DaskDataFrameDataset, collate_rowgroups
from src.tests import MAIN_S2_BANDS, SNAPSHOTS, TEMPORAL_S2_BANDS
from src.training.transformations import ConcatenateSnapshots
from src.utils.parameters import S2_BANDS, SATELLITE_COLUMN_CONFIGS, TARGET_COLUMN

TOTAL_ROW_GROUPS = 100


def read_partitions_arrow(parquet_file: pq.ParquetFile, partitions: list[int]) -> None:
    """Read the partitions into memory."""
    for part in partitions:
        _ = parquet_file.read_row_group(part)


def read_partition_dask(ddf: dd.DataFrame, partitions: list[int]) -> None:
    """Read the partitions into memory."""
    for part in partitions:
        ddf.get_partition(part).compute()


def read_row_groups(dataset: Dataset, partitions: list[int]) -> None:
    """Read a random set of row_groups into memory."""
    for part in partitions:
        _ = dataset[part]


def read_batch(dataloader: DataLoader) -> None:
    """Read the batch from DataLoader into memory."""
    _, _ = next(iter(dataloader))


########################################
# Partition / Row Group
########################################
@pytest.mark.benchmark(group="partition", disable_gc=True)
@pytest.mark.partition
@pytest.mark.parametrize("num_partitions", [1, 16, 32])
def test_arrow_local(benchmark: BenchmarkFixture, local_s2_file: str, num_partitions: int) -> None:
    """Benchmark reading a single row_group using PyArrow on file stored locally."""
    parquet_file = pq.ParquetFile(local_s2_file, filesystem=None)
    partitions = [random.randint(0, TOTAL_ROW_GROUPS) for _ in range(0, num_partitions)]

    benchmark(read_partitions_arrow, parquet_file, partitions)


@pytest.mark.benchmark(group="partition", disable_gc=True)
@pytest.mark.partition
@pytest.mark.parametrize("num_partitions", [1, 16, 32])
def test_arrow_amlfs(
    benchmark: BenchmarkFixture, fs: AzureMachineLearningFileSystem, remote_s2_file: str, num_partitions: int
) -> None:
    """Benchmark reading a single row_group using PyArrow on file stored on AML filesystem."""
    parquet_file = pq.ParquetFile(remote_s2_file, filesystem=fs)

    partitions = [random.randint(0, TOTAL_ROW_GROUPS) for _ in range(0, num_partitions)]

    benchmark(read_partitions_arrow, parquet_file, partitions)


@pytest.mark.benchmark(group="partition", disable_gc=True)
@pytest.mark.partition
@pytest.mark.parametrize("num_partitions", [1, 16, 32])
def test_dask_local(benchmark: BenchmarkFixture, local_s2_file: str, num_partitions: list[int]) -> None:
    """Benchmark reading a single row_group using Dask on file stored locally."""
    ddf = dd.read_parquet(local_s2_file, dtype_backend="pyarrow", split_row_groups=1, filesystem=None)

    partitions = [random.randint(0, TOTAL_ROW_GROUPS) for _ in range(0, num_partitions)]

    benchmark(read_partition_dask, ddf, partitions)


@pytest.mark.benchmark(group="partition", disable_gc=True)
@pytest.mark.partition
@pytest.mark.parametrize("num_partitions", [1, 16, 32])
def test_dask_amlfs(
    benchmark: BenchmarkFixture, fs: AzureMachineLearningFileSystem, remote_s2_file: str, num_partitions: list[int]
) -> None:
    """Benchmark reading a single row_group using Dask on file stored on AML filesystem."""
    ddf = dd.read_parquet(remote_s2_file, dtype_backend="pyarrow", split_row_groups=1, filesystem=fs)

    partitions = [random.randint(0, TOTAL_ROW_GROUPS) for _ in range(0, num_partitions)]

    benchmark(read_partition_dask, ddf, partitions)


########################################
# Loading batches with Dataset
########################################
@pytest.mark.benchmark(
    group="dataset",
    disable_gc=True,
    min_rounds=2,
)
@pytest.mark.dataloader
@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_arrow_dataset_local(benchmark: BenchmarkFixture, local_s2_file: str, batch_size: int) -> None:
    """Benchmark ArrowDataset on file stored locally."""
    parquet_dataset = pa.dataset.dataset(
        local_s2_file,
        format="parquet",
        exclude_invalid_files=True,
        filesystem=None,
    )

    band_concatenator = ConcatenateSnapshots(
        snapshots=SNAPSHOTS,
        all_available_bands=S2_BANDS,
        temporal_bands=TEMPORAL_S2_BANDS,
        main_bands=MAIN_S2_BANDS,
        scaling_factor=1 / 10_000,
    )
    s2_columns_config = SATELLITE_COLUMN_CONFIGS["s2"]
    dataset = ArrowDataset(
        parquet_dataset,
        s2_columns_config,
        target_column=TARGET_COLUMN,
        transform=band_concatenator,
        filesystem=None,
    )

    partitions = [random.randint(0, TOTAL_ROW_GROUPS) for _ in range(0, batch_size)]
    benchmark(read_row_groups, dataset, partitions)


@pytest.mark.benchmark(
    group="dataset",
    disable_gc=True,
    min_rounds=2,
)
@pytest.mark.dataloader
@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_arrow_dataset_amlfs(
    benchmark: BenchmarkFixture, fs: AzureMachineLearningFileSystem, remote_s2_file: str, batch_size: int
) -> None:
    """Benchmark ArrowDataset on file stored on AML filesystem."""
    parquet_dataset = pa.dataset.dataset(
        remote_s2_file,
        format="parquet",
        exclude_invalid_files=True,
        filesystem=fs,
    )

    band_concatenator = ConcatenateSnapshots(
        snapshots=SNAPSHOTS,
        all_available_bands=S2_BANDS,
        temporal_bands=TEMPORAL_S2_BANDS,
        main_bands=MAIN_S2_BANDS,
        scaling_factor=1 / 10_000,
    )

    s2_columns_config = SATELLITE_COLUMN_CONFIGS["s2"]

    dataset = ArrowDataset(
        parquet_dataset,
        s2_columns_config,
        target_column=TARGET_COLUMN,
        transform=band_concatenator,
        filesystem=fs,
    )

    partitions = [random.randint(0, TOTAL_ROW_GROUPS) for _ in range(0, batch_size)]
    benchmark(read_row_groups, dataset, partitions)


@pytest.mark.benchmark(
    group="dataset",
    disable_gc=True,
    min_rounds=2,
)
@pytest.mark.dataloader
@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_dask_dataset_local(benchmark: BenchmarkFixture, local_s2_file: str, batch_size: int) -> None:
    """Benchmark DaskDataFrameDataset on file stored locally."""
    ddf = dd.read_parquet(local_s2_file, dtype_backend="pyarrow", split_row_groups=1, filesystem=None)

    band_concatenator = ConcatenateSnapshots(
        snapshots=SNAPSHOTS,
        all_available_bands=S2_BANDS,
        temporal_bands=TEMPORAL_S2_BANDS,
        main_bands=MAIN_S2_BANDS,
        scaling_factor=1 / 10_000,
    )

    s2_columns_config = SATELLITE_COLUMN_CONFIGS["s2"]

    dataset = DaskDataFrameDataset(
        ddf,
        s2_columns_config,
        target_column=TARGET_COLUMN,
        transform=band_concatenator,
    )

    partitions = [random.randint(0, TOTAL_ROW_GROUPS) for _ in range(0, batch_size)]
    benchmark(read_row_groups, dataset, partitions)


@pytest.mark.benchmark(group="dataset", min_rounds=2, disable_gc=True)
@pytest.mark.dataloader
@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_dask_dataset_amlfs(
    benchmark: BenchmarkFixture, fs: AzureMachineLearningFileSystem, remote_s2_file: str, batch_size: int
) -> None:
    """Benchmark DaskDataFrameDataset on file stored on AML filesystem."""
    ddf = dd.read_parquet(remote_s2_file, dtype_backend="pyarrow", split_row_groups=1, filesystem=fs)

    band_concatenator = ConcatenateSnapshots(
        snapshots=SNAPSHOTS,
        all_available_bands=S2_BANDS,
        temporal_bands=TEMPORAL_S2_BANDS,
        main_bands=MAIN_S2_BANDS,
        scaling_factor=1 / 10_000,
    )

    s2_columns_config = SATELLITE_COLUMN_CONFIGS["s2"]

    dataset = DaskDataFrameDataset(
        ddf,
        s2_columns_config,
        target_column=TARGET_COLUMN,
        transform=band_concatenator,
    )

    partitions = [random.randint(0, TOTAL_ROW_GROUPS) for _ in range(0, batch_size)]
    benchmark(read_row_groups, dataset, partitions)


########################################
# Loading batches with DataLoader
# this isn't totally representative of real training as here we're benchmarking single batches
# and turn off prefetching and multiple workers so we can benchmark individually.
# real training uses multiple workers and prefetches ahead so to benchmark we would need to
# run through a large sample or the whole training data.
########################################
@pytest.mark.benchmark(
    group="dataloader",
    disable_gc=True,
    min_rounds=2,
)
@pytest.mark.dataloader
@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_arrow_dataloader_local(benchmark: BenchmarkFixture, local_s2_file: str, batch_size: int) -> None:
    """Benchmark ArrowDataset on file stored locally."""
    parquet_dataset = pa.dataset.dataset(
        local_s2_file,
        format="parquet",
        exclude_invalid_files=True,
        filesystem=None,
    )

    band_concatenator = ConcatenateSnapshots(
        snapshots=SNAPSHOTS,
        all_available_bands=S2_BANDS,
        temporal_bands=TEMPORAL_S2_BANDS,
        main_bands=MAIN_S2_BANDS,
        scaling_factor=1 / 10_000,
    )

    s2_columns_config = SATELLITE_COLUMN_CONFIGS["s2"]

    dataset = ArrowDataset(
        parquet_dataset,
        s2_columns_config,
        target_column=TARGET_COLUMN,
        transform=band_concatenator,
        filesystem=None,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_rowgroups,
        shuffle=True,
        num_workers=1,
        persistent_workers=False,
        prefetch_factor=1,
        multiprocessing_context="forkserver",
    )

    benchmark(read_batch, dataloader)


@pytest.mark.benchmark(
    group="dataloader",
    disable_gc=True,
    min_rounds=2,
)
@pytest.mark.dataloader
@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_arrow_dataloader_amlfs(
    benchmark: BenchmarkFixture, fs: AzureMachineLearningFileSystem, remote_s2_file: str, batch_size: int
) -> None:
    """Benchmark ArrowDataset on file stored on AML filesystem."""
    parquet_dataset = pa.dataset.dataset(
        remote_s2_file,
        format="parquet",
        exclude_invalid_files=True,
        filesystem=fs,
    )

    band_concatenator = ConcatenateSnapshots(
        snapshots=SNAPSHOTS,
        all_available_bands=S2_BANDS,
        temporal_bands=TEMPORAL_S2_BANDS,
        main_bands=MAIN_S2_BANDS,
        scaling_factor=1 / 10_000,
    )

    s2_columns_config = SATELLITE_COLUMN_CONFIGS["s2"]

    dataset = ArrowDataset(
        parquet_dataset,
        s2_columns_config,
        target_column=TARGET_COLUMN,
        transform=band_concatenator,
        filesystem=fs,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_rowgroups,
        shuffle=True,
        num_workers=1,
        persistent_workers=False,
        prefetch_factor=1,
        multiprocessing_context="forkserver",
    )

    benchmark(read_batch, dataloader)


@pytest.mark.benchmark(
    group="dataloader",
    disable_gc=True,
    min_rounds=2,
)
@pytest.mark.dataloader
@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_dask_dataloader_local(benchmark: BenchmarkFixture, local_s2_file: str, batch_size: int) -> None:
    """Benchmark DaskDataFrameDataset on file stored locally."""
    ddf = dd.read_parquet(local_s2_file, dtype_backend="pyarrow", split_row_groups=1, filesystem=None)

    band_concatenator = ConcatenateSnapshots(
        snapshots=SNAPSHOTS,
        all_available_bands=S2_BANDS,
        temporal_bands=TEMPORAL_S2_BANDS,
        main_bands=MAIN_S2_BANDS,
        scaling_factor=1 / 10_000,
    )

    s2_columns_config = SATELLITE_COLUMN_CONFIGS["s2"]

    dataset = DaskDataFrameDataset(
        ddf,
        s2_columns_config,
        target_column=TARGET_COLUMN,
        transform=band_concatenator,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_rowgroups,
        shuffle=True,
        num_workers=1,
        persistent_workers=False,
        prefetch_factor=1,
        multiprocessing_context="forkserver",
    )

    benchmark(read_batch, dataloader)


@pytest.mark.benchmark(group="dataloader", min_rounds=2, disable_gc=True)
@pytest.mark.dataloader
@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_dask_dataloader_amlfs(
    benchmark: BenchmarkFixture, fs: AzureMachineLearningFileSystem, remote_s2_file: str, batch_size: int
) -> None:
    """Benchmark DaskDataFrameDataset on file stored on AML filesystem."""
    ddf = dd.read_parquet(remote_s2_file, dtype_backend="pyarrow", split_row_groups=1, filesystem=fs)

    band_concatenator = ConcatenateSnapshots(
        snapshots=SNAPSHOTS,
        all_available_bands=S2_BANDS,
        temporal_bands=TEMPORAL_S2_BANDS,
        main_bands=MAIN_S2_BANDS,
        scaling_factor=1 / 10_000,
    )

    s2_columns_config = SATELLITE_COLUMN_CONFIGS["s2"]

    dataset = DaskDataFrameDataset(
        ddf,
        s2_columns_config,
        target_column=TARGET_COLUMN,
        transform=band_concatenator,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_rowgroups,
        shuffle=True,
        num_workers=1,
        persistent_workers=False,
        prefetch_factor=1,
        multiprocessing_context="forkserver",
    )

    benchmark(read_batch, dataloader)
