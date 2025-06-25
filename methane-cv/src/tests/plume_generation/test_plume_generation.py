"""Tests for generating plumes in generate.py."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.data.generation.plumes.gaussian_plume import GaussianPlume
from src.data.generation.plumes.generate import (
    create_plume_plot,
    generate_plume,
    sample_wind_data,
    wind_direction,
    wind_speed,
    PlumeMetaData,
    FailedPlume,
)


################################################################################
# Tests calculating wind speed and direction
################################################################################
@pytest.mark.parametrize(
    "x, y, expected",
    [
        (np.array([0.0, 3.0, 7.0]), np.array([0.0, 4.0, 24.0]), np.array([0.0, 5.0, 25.0])),
        (np.array([0.0, -3.0, -7.0]), np.array([0.0, -4.0, -24.0]), np.array([0.0, 5.0, 25.0])),
    ],
)
def test_wind_speed(x: np.ndarray, y: np.ndarray, expected: float) -> None:
    """Test wind_speed."""
    assert np.allclose(wind_speed(x, y), expected)


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (np.array([0.0, 1.0, 1.0]), np.array([1.0, 1.0, 0.0]), np.array([0.0, 45.0, 90.0])),
        (
            np.array([1.0, 0.0, -1.0, -1.0, -1.0]),
            np.array([-1.0, -1.0, -1.0, 0.0, 1.0]),
            np.array([135.0, 180.0, -135.0, -90.0, -45.0]),
        ),
    ],
)
def test_wind_direction(x: np.ndarray, y: np.ndarray, expected: float) -> None:
    """Test wind_direction."""
    assert np.allclose(wind_direction(x, y), expected)


################################################################################
# Tests sampling wind data
################################################################################
def test_sample_wind_data(wind_data_filepath: Path) -> None:
    """Test the sample_wind_data function."""
    rng = np.random.default_rng(seed=1)

    duration = 200

    # Sample the selected file to get wind data for the specified duration
    wind_df = sample_wind_data(wind_data_filepath, duration=duration, rng=rng)

    assert isinstance(wind_df, pd.DataFrame)

    # Check that sampled duration is within 5 seconds of requested duration (5 seconds is the typical sampling rate of the files)
    sampled_duration = wind_df.timestamp.max() - wind_df.timestamp.min()
    assert (pd.Timedelta(seconds=duration - 5) <= sampled_duration) or (
        sampled_duration <= pd.Timedelta(seconds=duration + 5)
    )


def test_sample_wind_data_with_gaps(wind_data_filepath_with_gaps: Path) -> None:
    """Test the sample_wind_data function."""
    rng = np.random.default_rng(seed=1)

    duration = 200

    # Sample the selected file to get wind data for the specified duration
    wind_df = sample_wind_data(wind_data_filepath_with_gaps, duration=duration, rng=rng)

    assert isinstance(wind_df, pd.DataFrame)

    # Check that sampled duration is within 5 seconds of requested duration (5 seconds is the typical sampling rate of the files)
    sampled_duration = wind_df.timestamp.max() - wind_df.timestamp.min()
    assert (pd.Timedelta(seconds=duration - 5) <= sampled_duration) or (
        sampled_duration <= pd.Timedelta(seconds=duration + 5)
    )


def test_sample_wind_data_large_duration_raises_error(wind_data_filepath: Path) -> None:
    """Test the sample_wind_data function with a duration larger than the maximum duration."""
    rng = np.random.default_rng(seed=1)

    duration = 300
    max_duration = 100

    with pytest.raises(ValueError):
        _ = sample_wind_data(wind_data_filepath, duration=duration, rng=rng, max_duration=max_duration)


def test_sample_wind_data_large_time_gap_raises_error(wind_data_filepath: Path) -> None:
    """Test the sample_wind_data function with a duration larger than the maximum duration."""
    rng = np.random.default_rng(seed=1)

    duration = 300
    max_time_gap = 1

    with pytest.raises(ValueError):
        _ = sample_wind_data(wind_data_filepath, duration=duration, rng=rng, max_time_gap=max_time_gap)


################################################################################
# Tests for generating plumes
################################################################################
def test_generate_plume(plume: GaussianPlume, wind_data_filepath: Path) -> None:
    """Test the generate method of the GaussianPlume class."""
    wind_files = [wind_data_filepath]

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        plume_metadata = generate_plume(
            plume_id=1,
            plume=plume,
            wind_files=wind_files,
            seed=42,
            output_dir=output_dir,
            save_plots=True,
            filesystem=None,
            upload_plumes=False,
            upload_plots=False,
        )

        if isinstance(plume_metadata, PlumeMetaData):
            assert (output_dir / "gaussian_plume_0000001.tif").is_file()
            assert (output_dir / "gaussian_plume_0000001.png").is_file()
        elif isinstance(plume_metadata, FailedPlume):
            raise Exception(plume_metadata.error_message)


def test_plume_generation_is_reproducible(plume: GaussianPlume, wind_data_filepath: pd.DataFrame) -> None:
    """Test that we can regenerate the same plume from the same seed and plume_id."""
    wind_files = [wind_data_filepath]

    plume_id = 1
    seed = 42
    plume_metadata_1 = generate_plume(
        plume_id=plume_id,
        plume=plume,
        wind_files=wind_files,
        seed=seed,
        output_dir=Path(),
        save_plots=False,
        filesystem=None,
        upload_plumes=False,
        upload_plots=False,
    )

    plume_metadata_2 = generate_plume(
        plume_id=plume_id,
        plume=plume,
        wind_files=wind_files,
        seed=seed,
        output_dir=Path(),
        save_plots=False,
        filesystem=None,
        upload_plumes=False,
        upload_plots=False,
    )

    assert plume_metadata_1 == plume_metadata_2


################################################################################
# Tests for plotting
################################################################################
def test_create_plume_plot(wind_data_filepath: pd.DataFrame) -> None:
    """Test the save_plume_plot function."""
    duration = 300
    rng = np.random.default_rng(seed=1)
    wind_df = sample_wind_data(wind_data_filepath, duration=duration, rng=rng)

    plume = GaussianPlume(
        spatial_resolution=4.0,
        temporal_resolution=0.1,
        duration=duration,
        crop_size=512,
        dispersion_coeff=0.6,
        emission_rate=1000.0,
    )

    concentration, X, Y = plume.generate(wind_df=wind_df, OU_sigma_fluctuations=0.0, OU_correlation_time=1.0)
    plot = create_plume_plot(X, Y, concentration, 512)

    assert isinstance(plot, Figure)
