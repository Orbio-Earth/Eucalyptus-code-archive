# ruff: noqa: I001
"""Fixtures for testing module."""

import hashlib
import logging
import pickle
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import xarray as xr

from src.azure_wrap.azure_path import AzureBlobPath
from src.azure_wrap.blob_storage_sdk_v2 import DATASTORE_URI, download_from_blob
from src.azure_wrap.ml_client_utils import (
    get_abfs_output_directory,
    get_default_blob_storage,
    get_storage_options,
    initialize_blob_service_client,
    initialize_ml_client,
)
from src.data.common.sim_plumes import PlumeType
from src.data.emit_data import EmitGranuleAccess
from src.data.generate import SATELLITE_CLASSES
from src.tests import MAIN_LANDSAT_BANDS, MAIN_S2_BANDS, SNAPSHOTS, TEMPORAL_LANDSAT_BANDS, TEMPORAL_S2_BANDS
from src.tests.generate_test_data import generate_test_data_for_satellite
from src.utils.git_utils import GIT_REPO_ROOT
from src.utils.parameters import (
    LANDSAT_BANDS,
    S2_BANDS,
    SatelliteID,
)
from src.utils.utils import initialize_s3_client
from collections.abc import Generator

# Note: these imports are at the end, so warnings are suppressed in src/__init__.py first
import mlflow  # isort: skip
from azure.ai.ml import MLClient  # isort: skip
from azure.storage.blob import BlobServiceClient  # isort: skip
from azureml.fsspec import AzureMachineLearningFileSystem  # isort: skip
from azure.core.exceptions import HttpResponseError  # isort: skip


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s (%(filename)s:%(lineno)d)",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def local_out_dir() -> Path:
    """Local directory where generated test data is stored during the test session."""
    out_dir = GIT_REPO_ROOT / "test" / "out" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture(scope="module")
def local_in_dir(plume_type: PlumeType) -> Path:
    """Local directory where input data necessary for tests is stored.

    This data is tracked in git for debugging purposes.
    # TODO: Don't think test data needs to be committed to the repo, all the test data either
    # get regenerated during testing or downloaded from Azure
    # we could use the built-in pytest temp dir
    """
    match plume_type:
        case PlumeType.RECYCLED:
            local_in_dir = GIT_REPO_ROOT / "methane-cv" / "test" / "in" / "data" / "recycled_plumes"
        case PlumeType.CARBONMAPPER:
            local_in_dir = GIT_REPO_ROOT / "methane-cv" / "test" / "in" / "data" / "aviris_plumes"

    return local_in_dir


@pytest.fixture(scope="session")
def ml_client() -> MLClient:
    """Initialize and return an MLClient instance."""
    try:
        ml_client = initialize_ml_client()
        # the next line is needed to trigger the HttpResponseError
        # if the ml_client is not authenticated properly
        initialize_s3_client(ml_client)
    except HttpResponseError:
        ml_client = initialize_ml_client(force_msi=True)
    return ml_client


@pytest.fixture(scope="module")
def abs_client(ml_client: MLClient) -> BlobServiceClient:
    """Initialize and return an Azure Blob Storage client."""
    return initialize_blob_service_client(ml_client)


@pytest.fixture(scope="session")
def blob_container(ml_client: MLClient) -> str:
    """Retrieve the default blob storage container name from the MLClient."""
    def_blob_storage = get_default_blob_storage(ml_client)
    return def_blob_storage.container_name


@pytest.fixture(scope="session")
def storage_options(ml_client: MLClient) -> dict:
    """Retrieve and return storage options for the Azure Blob Storage."""
    return get_storage_options(ml_client)


@pytest.fixture(scope="session")
def azure_data_dir(ml_client: MLClient) -> AzureBlobPath:
    """Return the Azure Blob File System (ABFS) directory path for the test directory."""
    return get_abfs_output_directory(ml_client, Path("test/dummy_directory"))


@pytest.fixture(name="fs", scope="module")
def filesystem() -> AzureMachineLearningFileSystem:
    """AML file system handle."""
    return AzureMachineLearningFileSystem(DATASTORE_URI)


@pytest.fixture(scope="module")
def remote_files() -> dict[SatelliteID, str]:
    """
    Map of satellite types to their respective files stored on AML file store.

    FIXME: this is error prone— what if the files are deleted?
    FIXME: this is error prone— we need to update the remote files if there is a change in parquet structure
    """
    return {
        SatelliteID.S2: "data/aviris/S2/training_test_1157/10SEJ_2017-10-12.parquet",
        SatelliteID.EMIT: "data/test_refactor/emit/test_gorrono/EMIT_L1B_RAD_001_20240127T195840_2402713_006.parquet",
        SatelliteID.LANDSAT: "data/carbonmapper/LANDSAT/training_1157_2025_03_20_sanity_check2/LC09_L1TP_128037_20220209_20230428_02_T1.parquet",
    }


@pytest.fixture(scope="module")
def local_file(
    remote_files: dict[SatelliteID, str],
    sat_key: SatelliteID,
    fs: AzureMachineLearningFileSystem,
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    """Will download the test file into a temporary directory and return the path.

    We use the built-in `tmp_path_factory` fixture so we can download the datafile to a temp directory
    and not need to clean it up after tests.
    - https://docs.pytest.org/en/stable/how-to/tmp_path.html#the-tmp-path-factory-fixture
    """
    remote_file = remote_files[sat_key]
    table = pq.read_table(remote_file, filesystem=fs)

    local_file = Path(tmp_path_factory.mktemp("data") / "local_file.parquet")

    pq.write_table(table, local_file, row_group_size=10, compression="zstd", compression_level=9)

    assert local_file.exists()

    return str(local_file)


# TODO: created these fixtures so we run the benchmarks only for s2 files. There is probably a more elegant solution
# instead of creating more fixtures.
@pytest.fixture(scope="module")
def remote_s2_file() -> str:
    """
    Map of satellite types to their respective files stored on AML file store.

    FIXME: this is error prone— what if the files are deleted?
    """
    return "data/recycled_plumes/training_L1C_MGRS_near_OG_selection_2000-zstd_compression_9/modulate_1.0_resize_1.0/cloud_bucket_30/recycled_plumes_10TDS_2023-09-02.parquet"


@pytest.fixture(scope="module")
def local_s2_file(
    remote_s2_file: str,
    fs: AzureMachineLearningFileSystem,
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    """Will download the test file into a temporary directory and return the path.

    We use the built-in `tmp_path_factory` fixture so we can download the datafile to a temp directory
    and not need to clean it up after tests.
    - https://docs.pytest.org/en/stable/how-to/tmp_path.html#the-tmp-path-factory-fixture
    """
    table = pq.read_table(remote_s2_file, filesystem=fs)

    local_file = Path(tmp_path_factory.mktemp("data") / "local_s2_file.parquet")

    pq.write_table(table, local_file, row_group_size=10, compression="zstd", compression_level=9)

    assert local_file.exists()

    return str(local_file)


#########################
### UTILITY FUNCTIONS ###
#########################


def delete_blob_data(container_name: str, directory_name: str, abs_client: BlobServiceClient) -> None:
    """Delete all data in the specified directory within the Azure blob container."""
    # Get a client to interact with the specified container
    container_client = abs_client.get_container_client(container_name)

    # List all blobs in the directory
    blob_list = container_client.list_blobs(name_starts_with=directory_name)
    for blob in blob_list:
        print(f"Deleting blob: {blob.name}")
        blob_client = container_client.get_blob_client(blob.name)
        blob_client.delete_blob()

    print("Deletion complete.")


@pytest.fixture(scope="module", name="local_data_dir")
def local_out_dir_generate(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Local directory to store the generated test parquet files."""
    local_dir = Path(tmp_path_factory.mktemp("recycled_plumes") / "test_generate")
    local_dir.mkdir()
    assert local_dir.exists()
    return local_dir


@pytest.fixture(scope="session", params=SATELLITE_CLASSES.keys())
def sat_key(request: pytest.FixtureRequest) -> str:
    """Fixture that provides satellite keys."""
    return request.param


@pytest.fixture(scope="module", params=[PlumeType("recycled")])  # , PlumeType("aviris")
def plume_type(request: pytest.FixtureRequest) -> PlumeType:
    """Fixture that provides satellite keys."""
    return request.param


@pytest.fixture(scope="module")
def local_sat_dir(sat_key: SatelliteID, local_data_dir: Path) -> Path:
    """Fixture that provides satellite-specific local output directories."""
    local_sat_dir = local_data_dir / str(sat_key)
    local_sat_dir.mkdir(parents=True, exist_ok=True)
    assert local_sat_dir.exists()
    return local_sat_dir


@pytest.fixture(scope="module")
def local_test_data(local_sat_dir: Path, local_in_dir: Path, sat_key: SatelliteID, plume_type: PlumeType) -> None:
    """Refresh the local output directory with fresh test parquet data for a controlled testing environment."""
    if sat_key == SatelliteID.EMIT and plume_type == PlumeType.CARBONMAPPER:
        pytest.skip("Unimplemented configuration")

    # Allow mlflow to log to the same parameter multiple times during testing
    with mlflow.start_run():  # noqa: SIM117
        with mlflow.start_run(nested=True):
            generate_test_data_for_satellite(
                local_sat_dir, local_in_dir, sat_key=sat_key, plume_type=plume_type, storage_options={}
            )


# TODO: use temp dir
@pytest.fixture(scope="module")
def azure_in_dir(local_in_dir: Path, plume_type: PlumeType) -> Path:
    """Local directory to store the unzipped tiff files from the test Azure Blob Storage."""
    local_dir = local_in_dir / "azure" / "plumes"
    local_dir.mkdir(exist_ok=True)

    # TODO: the recycled and aviris plumes are stored differently
    # - copy a subset of aviris plumes to the 'test/aviris' prefix
    # - move the recycled test plumes to prefix 'test/recycled' prefix
    # - update recycled and aviris plume json files to store the same file patterns
    #   - recycled only stores a shortened (and invalid) uri
    #   - aviris stores the full uri
    match plume_type:
        case PlumeType.RECYCLED:
            # FIXME: This file was deleted and does not exist anymore
            download_from_blob("test/unzipped_data/plumes/catalog_condensed.json", local_dir, recursive=False)
        case PlumeType.CARBONMAPPER:
            download_from_blob("carbonmapper_plumes/_emit_plume_uris_validation.json", local_dir, recursive=False)
    return local_dir.parent


@pytest.fixture(scope="module")
def azure_sat_dir(sat_key: SatelliteID) -> Path:
    """Fixture that provides satellite-specific Azure output directories."""
    return Path("test/dummy_directory/test_generate") / str(sat_key)


@pytest.fixture(scope="module")
def azure_test_data(
    azure_sat_dir: AzureBlobPath,
    azure_in_dir: Path,
    sat_key: str,
    plume_type: PlumeType,
    abs_client: BlobServiceClient,
    blob_container: str,
) -> Generator[None, None, None]:
    """Refresh the test ABS output directory with test parquet data for a controlled testing environment."""
    if sat_key == SatelliteID.EMIT and plume_type == PlumeType.CARBONMAPPER:
        pytest.skip("Unimplemented configuration")

    # Allow mlflow to log to the same parameter multiple times during testing
    with mlflow.start_run():  # noqa: SIM117
        with mlflow.start_run(nested=True):
            generate_test_data_for_satellite(
                azure_sat_dir, azure_in_dir, sat_key=sat_key, plume_type=plume_type, storage_options=None
            )

    # Tests
    yield

    # Cleanup
    delete_blob_data(blob_container, str(azure_sat_dir), abs_client)


@pytest.fixture
def concatenator_config() -> dict[SatelliteID, dict]:
    """Fixture providing ConcatenateSnapshots configurations for different satellites."""
    return {
        SatelliteID.S2: {
            "snapshots": SNAPSHOTS,
            "all_available_bands": S2_BANDS,
            "temporal_bands": TEMPORAL_S2_BANDS,
            "main_bands": MAIN_S2_BANDS,
            "scaling_factor": 1 / 10_000,
        },
        SatelliteID.LANDSAT: {
            "snapshots": SNAPSHOTS,
            "all_available_bands": LANDSAT_BANDS,
            "temporal_bands": TEMPORAL_LANDSAT_BANDS,
            "main_bands": MAIN_LANDSAT_BANDS,
            "scaling_factor": 1 / 10_000,
        },
    }


# =====================
# EarthAccess Fixtures
# =====================


@pytest.fixture(scope="session", autouse=True)
def _memoize_emit_dataset() -> None:
    """Monkey-patch EmitGranuleAccess._get_dataset with disk caching for testing.

    This fixture modifies the EmitGranuleAccess._get_dataset method to cache its results
    to disk during testing. The cache persists between test sessions and is stored in
    .cache/emit_datasets in the project root.
    """
    # Create a persistent cache directory in the project root
    cache_dir = GIT_REPO_ROOT / ".cache" / "emit_datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    original_get_dataset = EmitGranuleAccess._get_dataset

    def disk_cached_get_dataset(self: EmitGranuleAccess, product_type: str, group: str | None = None) -> xr.Dataset:
        # Create a unique cache key based on the input parameters
        cache_key = f"{self.id}_{product_type}_{group}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = cache_dir / f"emit_cache_{cache_hash}.pkl"

        # Try to load from cache first
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    logger.info(f"Loading cached EMIT dataset from {cache_file}")
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                # If cache is corrupted, regenerate it
                logger.warning(f"Corrupted cache file found at {cache_file}, regenerating...")
                cache_file.unlink(missing_ok=True)

        # If not in cache, get the dataset and cache it
        logger.info(f"Downloading EMIT dataset for {product_type} (group={group})")
        dataset = original_get_dataset(self, product_type, group)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(dataset, f)
            logger.info(f"Cached EMIT dataset to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache dataset: {e}")

        return dataset

    # Apply the monkey-patch
    EmitGranuleAccess._get_dataset = disk_cached_get_dataset


@pytest.fixture(scope="session", autouse=True)
def _earthaccess_auth(ml_client: MLClient) -> None:
    """Authenticate with earthaccess once per test session."""
    from src.utils.utils import earthaccess_login

    earthaccess_login(ml_client)
