"""Tests for src.training.tranformations.py."""

import logging
from typing import Any

import dask.dataframe as dd
import pytest
import torch
from _pytest.python_api import RaisesContext
from torch.utils.data import DataLoader

from src.tests import MAIN_LANDSAT_BANDS, MAIN_S2_BANDS, SNAPSHOTS, TEMPORAL_LANDSAT_BANDS, TEMPORAL_S2_BANDS
from src.training.training_script import data_preparation
from src.training.transformations import (
    ConcatenateSnapshots,
    CustomHorizontalFlip,
    MethaneModulator,
    MonotemporalBandExtractor,
    Rotate90,
)
from src.utils.parameters import EMIT_BANDS, LANDSAT_BANDS, S2_BANDS, SatelliteID

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("azure")
logger.setLevel(logging.ERROR)

#######################
### SETUP FUNCTIONS ###
#######################

TEST_FILE_PATH = "test/dummy_directory/test_generate/recycled_plumes_test.parquet"


# NOTE: test_generate.py needs to run before this test can be run
@pytest.fixture(scope="session")
def parquet_glob_uri() -> dict[SatelliteID, str]:
    """
    URI of the test file.

    FIXME: these are the same files as in remote_files
    """
    return {
        SatelliteID.S2: "azureml://subscriptions/6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab/resourcegroups/orbio-ml-rg/workspaces/orbio-ml-ml-workspace/datastores/workspaceblobstore/paths/data/aviris/S2/training_test_1157/10SEJ_2017-10-12.parquet",
        SatelliteID.EMIT: "azureml://subscriptions/6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab/resourcegroups/orbio-ml-rg/workspaces/orbio-ml-ml-workspace/datastores/workspaceblobstore/paths/data/test_refactor/emit/test_gorrono/EMIT_L1B_RAD_001_20240127T195840_2402713_006.parquet",
        SatelliteID.LANDSAT: "azureml://subscriptions/6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab/resourcegroups/orbio-ml-rg/workspaces/orbio-ml-ml-workspace/datastores/workspaceblobstore/paths/data/carbonmapper/LANDSAT/training_1157_2025_03_20_sanity_check2/LC09_L1TP_128037_20220209_20230428_02_T1.parquet",
    }


@pytest.fixture(scope="session")
def parquet_file(parquet_glob_uri: str, sat_key: SatelliteID, tmp_path_factory: pytest.TempPathFactory) -> str:
    """Will download the test file into a temporary directory and return the path.

    Pyarrow is able to read from Azure Blob Storage directly, but it's a bit more involved.
    - https://arrow.apache.org/docs/python/parquet.html#reading-a-parquet-file-from-azure-blob-storage
    """
    file_name = str(tmp_path_factory.mktemp("data") / TEST_FILE_PATH)

    df = dd.read_parquet(parquet_glob_uri[sat_key], dtype_backend="pyarrow", split_row_groups=1)
    df.to_parquet(
        file_name,
        compression="zstd",
        compression_level=9,
        row_group_size=10,
        storage_options={},
    )

    return file_name


@pytest.fixture(scope="session")
def satellite_transformation_configs() -> dict[SatelliteID, dict[str, Any]]:
    return {
        SatelliteID.S2: {
            "band_extractor": ConcatenateSnapshots(
                snapshots=SNAPSHOTS,
                all_available_bands=S2_BANDS,
                temporal_bands=TEMPORAL_S2_BANDS,
                main_bands=MAIN_S2_BANDS,
                scaling_factor=1 / 10_000,
            ),
            "swir16_band_name": "B11",
            "swir22_band_name": "B12",
            "all_available_bands": S2_BANDS,
        },
        SatelliteID.EMIT: {
            "band_extractor": MonotemporalBandExtractor(
                band_indices=EMIT_BANDS,
                scaling_factor=0.1,
            ),
        },
        SatelliteID.LANDSAT: {
            "band_extractor": ConcatenateSnapshots(
                snapshots=SNAPSHOTS,
                all_available_bands=LANDSAT_BANDS,
                temporal_bands=TEMPORAL_LANDSAT_BANDS,
                main_bands=MAIN_LANDSAT_BANDS,
                scaling_factor=1 / 10_000,
            ),
            "swir16_band_name": "swir16",
            "swir22_band_name": "swir22",
            "all_available_bands": LANDSAT_BANDS,
        },
    }


def create_test_data(
    batch_size: int = 2, image_size: tuple[int, int] = (128, 128), num_bands: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a randomized test dataset with target."""
    X = torch.rand(batch_size, num_bands, *image_size)
    y = torch.rand(batch_size, num_bands, *image_size)
    return X, y


######################
### TEST FUNCTIONS ###
######################


def test_rotate90_consistency() -> None:
    """Tests that the Rotate90 applies the same 90° rotation to both inputs and targets."""
    torch.manual_seed(42)
    rotate = Rotate90()

    X, y = create_test_data()
    rotated_X1, rotated_X2 = rotate((X.clone(), X.clone()))

    assert torch.equal(rotated_X1, rotated_X2)


def test_rotate90_repeatability() -> None:
    """Tests that the Rotate90 transformation applies the same rotation to repeated inputs under a fixed seed."""
    # Initialize the Rotate90 transformation with the seed before each use
    rotate1 = Rotate90()
    rotate2 = Rotate90()

    X, y = create_test_data()

    torch.manual_seed(42)
    rotated_X1, rotated_y1 = rotate1((X.clone(), y.clone()))
    torch.manual_seed(42)
    rotated_X2, rotated_y2 = rotate2((X.clone(), y.clone()))

    # Check for repeatability with the same seed
    assert torch.equal(rotated_X1, rotated_X2), "Rotation not repeatable for X"
    assert torch.equal(rotated_y1, rotated_y2), "Rotation not repeatable for y"


# TODO: remove this test? a different seed can still give same rotation?
# Idea is to replicate logic that each epoch should apply a different transformation
def test_rotate90_variability() -> None:
    """Tests that Rotate90 produces different rotations for the same inputs when initialized with different seeds."""
    rotate1 = Rotate90()
    rotate2 = Rotate90()

    X, y = create_test_data()

    torch.manual_seed(42)
    rotated_X1, rotated_y1 = rotate1((X.clone(), y.clone()))
    torch.manual_seed(43)
    rotated_X2, rotated_y2 = rotate2((X.clone(), y.clone()))

    # Check that different seeds produce different rotations
    assert not torch.equal(rotated_X1, rotated_X2), "Different seeds should produce different rotations for X"
    assert not torch.equal(rotated_y1, rotated_y2), "Different seeds should produce different rotations for y"


def test_custom_horizontal_flip_consistency() -> None:
    """Tests that the CustomHorizontalFlip applies the same flip to both inputs and targets."""
    torch.manual_seed(42)
    flip = CustomHorizontalFlip()

    X, y = create_test_data()
    for _ in range(10):
        flipped_X1, flipped_X2 = flip((X.clone(), X.clone()))
        assert torch.equal(flipped_X1, flipped_X2), "Flip mismatch"


def test_custom_horizontal_flip_repeatability() -> None:
    """Tests that the CustomHorizontalFlip transformation applies the same flip to repeated inputs for a fixed seed."""
    flip = CustomHorizontalFlip()

    X, y = create_test_data()

    torch.manual_seed(42)
    flipped_X1, flipped_y1 = flip((X.clone(), y.clone()))
    torch.manual_seed(42)
    flipped_X2, flipped_y2 = flip((X.clone(), y.clone()))

    # Check for repeatability with the same seed
    assert torch.equal(flipped_X1, flipped_X2), "Flip not repeatable for X"
    assert torch.equal(flipped_y1, flipped_y2), "Flip not repeatable for y"


# TODO: remove this test? a different seed can still give same flip?
# Idea is to replicate logic that each epoch should apply a different transformation
def test_custom_horizontal_flip_variability() -> None:
    """Tests CustomHorizontalFlip produces different flips for the same inputs when initialized with different seeds."""
    flip = CustomHorizontalFlip()

    X, y = create_test_data()

    torch.manual_seed(42)
    flipped_X1, flipped_y1 = flip((X.clone(), y.clone()))
    torch.manual_seed(43)
    flipped_X2, flipped_y2 = flip((X.clone(), y.clone()))

    # Check that different seeds produce different rotations
    assert not torch.equal(flipped_X1, flipped_X2), "Different seeds should produce different flips for X"
    assert not torch.equal(flipped_y1, flipped_y2), "Different seeds should produce different flips for y"


def test_data_preparation_determinism(
    parquet_file: str, sat_key: SatelliteID, satellite_transformation_configs: dict[SatelliteID, dict[str, Any]]
) -> None:
    """
    Ensures that the data preparation process is deterministic with a fixed seed.

    This should lead to identical batches of data being loaded repeatedly.

    NOTE: we only do this test for S2. Is it useful to do it for other satellites?
    The other satellites only differ in their band_extractor.
    """
    # Prepare the dataset with a specific seed
    torch.manual_seed(42)
    band_concatenator = satellite_transformation_configs[sat_key]["band_extractor"]
    datasets, _, _, _ = data_preparation(parquet_file, band_concatenator, satellite_id=sat_key)

    train_dataloader = DataLoader(datasets["train"], batch_size=1, shuffle=False)
    monitoring_dataloader = DataLoader(datasets["train_monitoring"], batch_size=1, shuffle=False)

    # Get initial data
    initial_inputs_train, initial_targets_train = next(iter(train_dataloader))
    initial_inputs_monitoring, initial_targets_monitoring = next(iter(monitoring_dataloader))

    # Reset and repeat
    torch.manual_seed(42)
    datasets_repeat, _, _, _ = data_preparation(parquet_file, band_concatenator, satellite_id=sat_key)
    train_dataloader_repeat = DataLoader(datasets_repeat["train"], batch_size=1, shuffle=False)
    monitoring_dataloader_repeat = DataLoader(datasets_repeat["train_monitoring"], batch_size=1, shuffle=False)
    repeated_inputs_train, repeated_targets_train = next(iter(train_dataloader_repeat))
    repeated_inputs_monitoring, repeated_targets_monitoring = next(iter(monitoring_dataloader_repeat))

    # Check for equality
    assert torch.equal(
        initial_inputs_train, repeated_inputs_train
    ), "Initial and repeated inputs for training set should be identical"
    assert torch.equal(
        initial_targets_train, repeated_targets_train
    ), "Initial and repeated targets for training set should be identical"
    assert torch.equal(
        initial_inputs_monitoring, repeated_inputs_monitoring
    ), "Initial and repeated inputs for training set should be identical"
    assert torch.equal(
        initial_targets_monitoring, repeated_targets_monitoring
    ), "Initial and repeated targets for training set should be identical"


# TODO: Add a test to check if correct transformations are applied to train/monitoring set


@pytest.mark.parametrize(
    "modulate",
    [
        (float("nan")),
        (0.0),
        (0.5),
        (1.0),
    ],
)
def test_methane_modulator_initialisation(
    modulate: float, sat_key: SatelliteID, satellite_transformation_configs: dict[SatelliteID, dict[str, Any]]
) -> None:
    """Test MethaneModulator initialisation."""
    if sat_key == SatelliteID.EMIT:
        pytest.skip("MethaneModulator does not support EMIT")

    all_available_bands = satellite_transformation_configs[sat_key]["all_available_bands"]
    swir16_band_name = satellite_transformation_configs[sat_key]["swir16_band_name"]
    swir22_band_name = satellite_transformation_configs[sat_key]["swir22_band_name"]

    _ = MethaneModulator(
        modulate,
        all_available_bands,
        swir16_band_name,
        swir22_band_name,
    )


@pytest.mark.parametrize(
    "modulate, expectation",
    [
        (float("inf"), pytest.raises(ValueError)),
        (-1.0, pytest.raises(ValueError)),
        (2.0, pytest.raises(ValueError)),
    ],
)
def test_methane_modulator_initialisation_error(
    modulate: float,
    expectation: RaisesContext,
    sat_key: SatelliteID,
    satellite_transformation_configs: dict[SatelliteID, dict[str, Any]],
) -> None:
    """Test MethaneModulator raises error for incorrect initialisation."""
    if sat_key == SatelliteID.EMIT:
        pytest.skip("MethaneModulator does not support EMIT")

    all_available_bands = satellite_transformation_configs[sat_key]["all_available_bands"]
    swir16_band_name = satellite_transformation_configs[sat_key]["swir16_band_name"]
    swir22_band_name = satellite_transformation_configs[sat_key]["swir22_band_name"]

    with expectation:
        _ = MethaneModulator(
            modulate,
            all_available_bands,
            swir16_band_name,
            swir22_band_name,
        )
