"""Data generation script for synthetic Gaussian plumes used in training data generation."""

from __future__ import annotations

import argparse
import multiprocessing
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import rasterio
import tqdm
from azureml.fsspec import AzureMachineLearningFileSystem
from matplotlib.figure import Figure
from rasterio.transform import from_origin

from src.azure_wrap.blob_storage_sdk_v2 import DATASTORE_URI, upload_dir
from src.data.generation.plumes.gaussian_plume import GaussianPlume
from src.utils import PROJECT_ROOT
from src.utils.utils import setup_logging

WIND_BLOB = "sbr_wind_data"
# Always upload plumes to this directory.  This matches the directory pattern of the AVIRIS plumes.
# We can delete files on blob store now so we can clean up existing files if we need to regenerate.
# We can also add functionality to expose uploading to a sub blob of "gaussian_plumes", that might
# necessitate changing the data gen code to handle or just updating the paths in the
# training_plumes.json / validation_plumes.json files
GAUSSIAN_PLUME_BLOB = Path("gaussian_plumes")
LOCAL_OUTPUT_DIR = "gaussian_plumes"


@dataclass
class WindMetaData:
    """Class that holds metadata about the sampled wind data."""

    filename: Path
    duration: int
    row: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    avg_wind_speed: float
    avg_wind_direction: float

    @classmethod
    def from_dataframe(cls, filename: Path, wind_df: pd.DataFrame, duration: int) -> WindMetaData:
        """Create a WindMetaData object from a dataframe."""
        return cls(
            filename=filename,
            duration=duration,
            row=wind_df.index.values[0],
            start_time=wind_df.timestamp.min(),
            end_time=wind_df.timestamp.max(),
            avg_wind_speed=wind_speed(wind_df.speed_x, wind_df.speed_y).mean(),
            avg_wind_direction=wind_direction(wind_df.speed_x, wind_df.speed_y).mean(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary but don't include the wind_df."""
        return asdict(self)


@dataclass
class PlumeMetaData:
    """Class that holds metadata about the sampled wind data."""

    plume_id: int
    wind_filename: Path
    plume_filename: str
    seed: int
    plume: GaussianPlume
    wind: WindMetaData

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary but don't include the wind_df."""
        return asdict(self)


@dataclass
class FailedPlume:
    """Class that holds metadata about a plume that failed to generate."""

    plume_id: int
    seed: int
    error: str
    error_message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary but don't include the wind_df."""
        return asdict(self)


def sample_wind_data(
    filename: Path,
    duration: int,
    filesystem: AzureMachineLearningFileSystem | None = None,
    rng: np.random.Generator | None = None,
    max_duration: int = 3600,
    max_time_gap: int = 10,
) -> pd.DataFrame:
    """Sample a time period of wind data from the wind data file.

    The median interval between timestamps is calculated and then we sample the duration interval
    based on the median time difference between samples. This ensures we get exactly the number of
    rows needed to cover the specified duration, avoiding issues with timestamp rounding or
    inconsistent intervals.

    Parameters
    ----------
        filename: Path to the wind data file on Azure Blob Storage
        duration: The duration of the wind data to sample (seconds)
        rng: Optional numpy random number generator to use for sampling
        max_duration: the maximum allowed duration to sample (seconds).
        max_time_gap: the maximum time gap between samples (seconds)

    Returns
    -------
        A pandas DataFrame containing the sampled wind data with a time range matching exactly
        the requested duration
    """
    if duration > max_duration:
        raise ValueError(f"Duration, {duration}, is greater than maximum allowed, {max_duration}")

    df = pd.read_parquet(filename, filesystem=filesystem)
    if df["timestamp"].diff().max() > pd.Timedelta(seconds=max_time_gap):
        # there is a gap in the sampled wind data, so get another random sample
        raise ValueError(
            f"Sampled wind data in {filename} is not continuous. Biggest gap: {df['timestamp'].diff().max()}"
        )

    # Calculate the total duration available in the wind data file in seconds
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()
    wind_data_seconds = (max_time - min_time).total_seconds()

    # if the duration is greater than the total duration of the wind data, sample another file
    if duration > wind_data_seconds:
        raise ValueError(
            f"Duration is greater than the total duration of the wind data. Duration: {duration} seconds, Total duration: {wind_data_seconds} seconds"  # noqa: E501
        )

    start_range = wind_data_seconds - duration  # the range of start times from 0

    # select a random period of time within the time range.
    if rng is None:
        rng = np.random.default_rng()
    random_seconds = rng.random() * start_range
    start_time = min_time + pd.Timedelta(seconds=random_seconds)

    end_time = start_time + pd.Timedelta(seconds=duration)

    # select the subset of the wind data that falls within the specified time range
    return df[df["timestamp"].between(start_time, end_time)]


def create_plume_plot(X: np.ndarray, Y: np.ndarray, concentration: np.ndarray, crop_size: int) -> Figure:
    """Save a plot of the generated plume."""
    lim = crop_size // 2

    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(
        concentration,
        extent=[X.min(), X.max(), Y.min(), Y.max()],  # type: ignore
        vmin=0.0,
        vmax=concentration.max() / 2,
        interpolation="nearest",
        origin="lower",
        cmap="plasma",
        aspect="equal",
    )
    plt.colorbar(label="Concentration (mol/m2)")
    plt.contour(
        X,
        Y,
        concentration,
        colors="white",
        alpha=0.1,
        levels=np.linspace(concentration.min(), concentration.max(), 30),
    )
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("Concentration Field from Gaussian Puffs")
    plt.grid(True, color="white", alpha=0.3)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.tight_layout()

    return fig


def wind_speed(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate wind speed from x and y components."""
    return np.sqrt(np.square(x) + np.square(y))


def wind_direction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate wind direction from x and y components."""
    return np.degrees(np.arctan2(x, y))


def _generate_plume(
    plume_id: int,
    plume: GaussianPlume,
    wind_files: list[Path],
    seed: int,
    output_dir: Path,
    save_plots: bool,
    filesystem: AzureMachineLearningFileSystem | None,
    upload_plumes: bool,
    upload_plots: bool,
) -> PlumeMetaData:
    """Generate a plume using given Gaussian plume initial conditions and sampled wind data."""
    # Use a seed so we can change the generations
    # add the plume_id to the seed for easy replication
    rng = np.random.default_rng(seed + plume_id)

    # Randomly select a file from the available wind files
    selected_file = rng.choice(wind_files, replace=False)  # type: ignore

    # Sample the selected file to get wind data for the specified duration
    wind_df = sample_wind_data(selected_file, duration=plume.duration, rng=rng, filesystem=filesystem)

    wind_data = WindMetaData.from_dataframe(
        filename=selected_file,
        wind_df=wind_df,
        duration=plume.duration,
    )

    concentration, X, Y = plume.generate(wind_df=wind_df)

    ##################################################
    # Save concentration tiff (the generated plume)
    ##################################################
    # prefix leading zeros so files are sorted in the UI
    plume_filename = f"gaussian_plume_{plume_id:07d}.tif"
    plume_filepath = output_dir / plume_filename

    # Since this is a simulated plume, it has no real location,
    # unlike the AVIRIS plumes which came from somewhere.
    # Still, we give it a generic CRS and a position on the lovely and exotic Null Island (0, 0),
    # so the plume is saved as a valid geotiff, to make our lives easier downstream
    # (so real and simulated plumes can be handled the same way).
    west = 0.0 - (plume.crop_size // 2) * plume.spatial_resolution
    north = 0.0 + (plume.crop_size // 2) * plume.spatial_resolution
    transform = from_origin(west, north, plume.spatial_resolution, plume.spatial_resolution)

    with rasterio.open(
        plume_filepath,
        "w",
        driver="GTiff",
        height=concentration.shape[0],
        width=concentration.shape[1],
        count=1,
        dtype=concentration.dtype,
        # EPSG:3857 is being used as a generic stand-in CRS that works globally
        # and has units of meters.
        crs="EPSG:3857",
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(concentration, 1)

    if upload_plumes:
        upload_dir(plume_filepath.as_posix(), (GAUSSIAN_PLUME_BLOB / "plumes").as_posix(), recursive=False)

    if save_plots:
        plot_filepath = plume_filepath.with_suffix(".png").as_posix()
        fig = create_plume_plot(X, Y, concentration, plume.crop_size)
        fig.savefig(plot_filepath, dpi=300)
        plt.close(fig)
        if upload_plots:
            upload_dir(plot_filepath, (GAUSSIAN_PLUME_BLOB / "plots").as_posix(), recursive=False)

    return PlumeMetaData(
        plume_id=plume_id,
        wind_filename=selected_file,
        plume_filename=plume_filename,
        seed=seed,
        plume=plume,
        wind=wind_data,
    )


def generate_plume(
    plume_id: int,
    plume: GaussianPlume,
    wind_files: list[Path],
    seed: int,
    output_dir: Path,
    save_plots: bool,
    filesystem: AzureMachineLearningFileSystem | None,
    upload_plumes: bool = True,
    upload_plots: bool = True,
) -> PlumeMetaData | FailedPlume:
    """Wrap the function for generating a plume in a try/except block to handle errors.

    This is to enable erros to be logged during multiprocessing.  If an error is uncaught during
    multiprocessing, the process may not return and let the processing pool continue.
    """
    try:
        return _generate_plume(
            plume_id=plume_id,
            plume=plume,
            wind_files=wind_files,
            seed=seed,
            output_dir=output_dir,
            save_plots=save_plots,
            filesystem=filesystem,
            upload_plumes=upload_plumes,
            upload_plots=upload_plots,
        )
    except Exception as err:
        # multiprocessing.imap supports exceptions directly, but there was a bug in unpickling
        # one of the Azure Machine Learning SDK's exceptions.  This is a workaround, so we don't
        # pass the exception itself, just its string representation.
        return FailedPlume(plume_id=plume_id, seed=seed, error=type(err).__name__, error_message=f"{err!r}")


def main(  # noqa: PLR0913 (too-many-arguments)
    num_plumes: int,
    spatial_resolution: float,
    temporal_resolution: float,
    crop_size: int,
    duration: int,
    dispersion_coeff: float,
    emission_rate: float,
    OU_sigma_fluctuations: float,
    OU_correlation_time: float,
    seed: int,
    output_dir: Path,
    git_revision_hash: str,
    azure_cluster: bool,
    save_plots: bool = False,
    start_plume_id: int = 0,
) -> None:
    """Generate a synthetic Gaussian plume dataset."""
    # Log parameters with MLflow
    run_params = {
        "OU_correlation_time": OU_correlation_time,
        "OU_sigma_fluctuations": OU_sigma_fluctuations,
        "azure_cluster": azure_cluster,
        "crop_size": crop_size,
        "dispersion_coeff": dispersion_coeff,
        "duration": duration,
        "emission_rate": emission_rate,
        "git_revision_hash": git_revision_hash,
        "num_plumes": num_plumes,
        "output_dir": str(output_dir),
        "save_plots": save_plots,
        "seed": seed,
        "spatial_resolution": spatial_resolution,
        "temporal_resolution": temporal_resolution,
        "wind_blob": WIND_BLOB,
    }

    # Setup MLflow run
    mlflow.log_params(run_params)

    output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fs = AzureMachineLearningFileSystem(DATASTORE_URI)
    wind_files = list(fs.glob(f"{WIND_BLOB}/**/*.parquet"))
    # Log wind file info
    mlflow.log_param("num_wind_files", len(wind_files))

    if not wind_files:
        raise FileNotFoundError(f"No wind data files found in {WIND_BLOB}")

    plume = GaussianPlume(
        spatial_resolution=spatial_resolution,
        temporal_resolution=temporal_resolution,
        duration=duration,
        crop_size=crop_size,
        dispersion_coeff=dispersion_coeff,
        emission_rate=emission_rate,
        OU_sigma_fluctuations=OU_sigma_fluctuations,
        OU_correlation_time=OU_correlation_time,
    )

    # Generate 'num_plumes' using multiprocessing
    plume_metadata: list[PlumeMetaData] = []
    failed_plumes: list[FailedPlume] = []  # holds plume_ids that failed to generate
    worker = partial(
        generate_plume,
        plume=plume,
        wind_files=wind_files,
        seed=seed,
        output_dir=output_dir,
        save_plots=save_plots,
        filesystem=fs,
    )

    # Generate plume IDs from start_plume_id to start_plume_id + num_plumes - 1
    plume_id_range = range(start_plume_id, start_plume_id + num_plumes)

    # 💡 pro tip: use spawn if something in the worker is going to use threading (and maybe IO?).
    # It's slower to initialize (it spawn a whole new python process)
    # but since we're using a pool that's fairly negliable.
    # this script was hanging on the pd.read_parquet(wind_file, filesystem=AzureMachineLearningFileSystem(DATASTORE_URI)
    # using the default spawn context fork.
    with multiprocessing.get_context("spawn").Pool(processes=multiprocessing.cpu_count()) as p:
        for metadata in tqdm.tqdm(p.imap(worker, plume_id_range), total=num_plumes):
            match metadata:
                case PlumeMetaData():
                    plume_metadata.append(metadata)
                case FailedPlume():
                    failed_plumes.append(metadata)
                    logger.error(f"Failed to generate plume {metadata.plume_id} with error: {metadata.error}")
                case _:
                    raise ValueError(f"Unexpected metadata type: {type(metadata)}")

    # Save the plume metadata to a file after flattening the metadata
    filename = output_dir / "_plume_attributes.csv"
    plume_df = pd.json_normalize([x.to_dict() for x in plume_metadata], sep="_")
    plume_df.to_csv(filename, index=False)
    upload_dir(filename.as_posix(), GAUSSIAN_PLUME_BLOB.as_posix(), recursive=False)

    # Save the failed plume metadatas to a file after flattening the metadata
    filename = output_dir / "_failed_plumes.csv"
    plume_df = pd.json_normalize([x.to_dict() for x in failed_plumes], sep="_")
    plume_df.to_csv(filename, index=False)
    upload_dir(filename.as_posix(), GAUSSIAN_PLUME_BLOB.as_posix(), recursive=False)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic Gaussian plumes to be inserted into training data."
    )
    parser.add_argument(
        "--num_plumes",
        type=int,
        required=True,
        help="The spatial size of a pixel in the output raster, e.g. 20.0",
    )
    parser.add_argument(
        "--start_plume_id",
        type=int,
        default=1,
        help="Starting ID for plume generation (default: 1)",
    )
    parser.add_argument(
        "--spatial_resolution",
        type=float,
        required=True,
        help="The spatial size of a pixel in the output raster, e.g. 20.0",
    )
    parser.add_argument(
        "--temporal_resolution",
        type=float,
        required=True,
        help="the size of the time steps.  e.g. 5 seconds",
    )
    parser.add_argument(
        "--duration",
        type=int,
        required=True,
        help="the duration for which to run the simulation for. e.g. 600 seconds",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        required=True,
        help="Size of crops taken from satellite images",
    )
    parser.add_argument(
        "--dispersion_coeff",
        type=float,
        required=True,
        help="coefficient controlling puff dispersion rate",
    )
    parser.add_argument(
        "--emission_rate",
        type=float,
        required=True,
        help="emission rate of the puff (kg/hr)",
    )
    parser.add_argument(
        "--OU_sigma_fluctuations",
        type=float,
        required=True,
        help="Parameter of the Ornstein-Uhlenbeck process (m/s)",
    )
    parser.add_argument(
        "--OU_correlation_time",
        type=float,
        required=True,
        help="Parameter of the Ornstein-Uhlenbeck process (s)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        required=True,
        help="Seed for random number generation",
    )
    parser.add_argument(
        "--out_dir",
        required=False,
        default=LOCAL_OUTPUT_DIR,
        type=Path,
        help=f"Output directory on local machine.  Files are uploaded to '{GAUSSIAN_PLUME_BLOB}' Azure Blob Storage.",
    )
    parser.add_argument(
        "--git_revision_hash",
        type=str,
        required=True,
        help="Git revision hash for tracking code version",
    )
    parser.add_argument(
        "--azure_cluster",
        action="store_true",
        help="Whether running on Azure cluster",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Whether running on Azure cluster",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logger = setup_logging()
    args = parse_args()

    main(
        num_plumes=args.num_plumes,
        spatial_resolution=args.spatial_resolution,
        temporal_resolution=args.temporal_resolution,
        duration=args.duration,
        crop_size=args.crop_size,
        dispersion_coeff=args.dispersion_coeff,
        emission_rate=args.emission_rate,
        OU_sigma_fluctuations=args.OU_sigma_fluctuations,
        OU_correlation_time=args.OU_correlation_time,
        seed=args.random_seed,
        output_dir=args.out_dir,
        git_revision_hash=args.git_revision_hash,
        azure_cluster=args.azure_cluster,
        save_plots=args.save_plots,
        start_plume_id=args.start_plume_id,
    )
