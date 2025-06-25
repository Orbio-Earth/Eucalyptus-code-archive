"""Base pipeline class for data generation."""

import abc
import hashlib
import json
from collections.abc import Iterator
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

import mlflow
import numpy as np
import numpy.typing as npt
import rasterio
from azure.ai.ml import MLClient
from botocore.client import BaseClient
from mypy_boto3_s3 import S3Client
from pydantic import AnyUrl, BaseModel, Field, field_validator
from rasterio.transform import Affine
from scipy.stats import pareto
from shapely.geometry import Polygon

from src.azure_wrap.azure_path import AzureBlobPath
from src.azure_wrap.ml_client_utils import (
    create_ml_client_config,
    get_abfs_output_directory,
    get_azure_ml_file_system,
    get_default_blob_storage,
    get_storage_options,
    initialize_blob_service_client,
    initialize_ml_client,
)
from src.data.common.data_item import BasePlumesDataItem
from src.data.common.sim_plumes import (
    PlumeType,
    build_plume_transform_params,
    load_and_transform_plume_arr,
)
from src.data.common.utils import download_plume_catalog
from src.utils.utils import initialize_s3_client, setup_logging

logger = setup_logging()


class DataGenerationConfig(BaseModel):
    """Configuration for data generation."""

    model_config = {"arbitrary_types_allowed": True}  # allow arbitrary types for MLClient/S3Client

    plume_catalog: str = Field(
        ...,
        description="URI or local path to the plume catalog (typically JSON) containing plume file paths",
    )
    plume_type: PlumeType | None = Field(None, description="Type of plume dataset to use (RECYCLED or AVIRIS)")
    plume_proba_dict: dict[int, float] = Field(..., description="Probabilites for number of inserted plumes")
    out_dir: str = Field(
        ...,
        description="Output directory for the generated data (can be local or Azure)",
    )
    crop_size: int = Field(..., description="Image crop size")
    quality_thresholds: dict[str, tuple[float, float]] = Field(
        ..., description="Quality thresholds (min, max) for different regions"
    )
    random_seed: int = Field(..., description="Random seed for reproducibility")
    transformation_params: dict[str, float] = Field(..., description="Parameters for plume transformation")
    ml_client: MLClient | None = Field(None, description="Azure ML client")
    s3_client: S3Client | BaseClient | None = Field(None, description="S3 client")
    storage_options: dict | None = Field(None, description="Storage options (needed for saving to Azure)")
    azure_cluster: bool = Field(..., description="Are we running in an Azure cluster?")
    git_revision_hash: str = Field(..., description="Git revision hash")
    test: bool = Field(..., description="Whether to run in test mode (only generate a few crops)")
    psf_sigma: float = Field(
        ...,
        description="Sigma of gaussian filter used to rescale AVIRIS plume resolution to that of the target sensor",
    )
    target_spatial_resolution: int = Field(
        ...,
        description=(
            "The spatial resolution of the imagery into which the plumes are being inserted, so that plumes can be"
            " rescaled accordingly."
        ),
    )
    concentration_rescale_value: float | None = Field(
        ...,
        description=(
            "The factor by which to rescale the concentration/enhancement values in the plumes when inserting"
            " into imagery.  Should not be set for Gaussian Plumes (it will be overwritten otherwise)"
        ),
    )
    simulated_emission_rate: float | None = Field(
        default=None,
        description=(
            "The simulated emission rate of the Gaussian plumes. Used to determine the concentration rescale value."
        ),
    )
    min_emission_rate: float = Field(
        ...,
        description=("The minimum emission rate to resample to for the Gaussian plumes."),
    )
    hapi_data_path: AnyUrl = Field(
        ...,
        description="Path to HAPI spectral data",
    )

    @field_validator("quality_thresholds")
    def validate_quality_thresholds(cls, v: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        """Validate quality thresholds."""
        for key, (min_val, max_val) in v.items():
            # Validate min value
            if not 0 <= min_val <= 1:
                raise ValueError(f"Minimum quality threshold for {key} should be between 0 and 1")
            # Validate max value
            if not 0 <= max_val <= 1:
                raise ValueError(f"Maximum quality threshold for {key} should be between 0 and 1")
            # Validate min <= max
            if min_val > max_val:
                raise ValueError(
                    f"Minimum threshold ({min_val}) cannot be greater than maximum threshold ({max_val}) for {key}"
                )
        return v


class BaseDataGeneration(abc.ABC):
    """
    Base class for data generation.

    This class handles infrastructure setup internally to simplify usage.
        You can pass relative Azure paths or URIs as inputs, and the class will:
        1. Initialize all necessary Azure/AWS clients
        2. Download the plume catalog to local disk
        3. Set up the output directory with proper Azure paths
        4. Configure random seed for reproducibility

    Note: specify storage options as None if you want to save to Azure Blob Storage. To save to local disk, you can pass
    an empty dict.
    """

    out_dir: Path | AzureBlobPath

    NON_SERIALIZABLE_ATTRS: ClassVar[set[str]] = {
        "config",
        "ml_client",
        "storage_options",
        "fs",
        "s3_client",
        "abs_client",
        "def_blob_storage",
        "plume_catalog_local_path",
        "abfs_out_dir",
        "container_name",
        "rng",
        "producing_union",
        "hassi_poly",
        "permian_poly",
        "marcellus_poly",
        "plume_files",
        "preload_models",
    }
    NUM_SAMPLES_FOR_TESTING = 3

    def __init__(self, config: DataGenerationConfig) -> None:
        """Initialize data generation pipeline with configuration."""
        self.config = config
        self.crop_size = self.config.crop_size
        self.quality_thresholds = self.config.quality_thresholds
        self.transformation_params = self.config.transformation_params
        self.random_seed = self.config.random_seed
        self.git_revision_hash = self.config.git_revision_hash
        self.test = self.config.test
        self.plume_type = config.plume_type
        self.psf_sigma = config.psf_sigma
        self.target_spatial_resolution = config.target_spatial_resolution
        self.concentration_rescale_value = config.concentration_rescale_value
        self.simulated_emission_rate = config.simulated_emission_rate
        self.min_emission_rate = config.min_emission_rate
        self.plume_proba_dict = config.plume_proba_dict
        self.hapi_data_path = config.hapi_data_path

        # simulated emission rate should be set for Gaussian plumes and not set for other plume types
        # the concentration rescale value should also not be set as it will be calculated and overwritten
        if self.plume_type == PlumeType.GAUSSIAN:
            self.position_by_source = True
            assert (
                self.simulated_emission_rate
            ), f"For Gaussian plumes, simulated_emission_rate must be set - {self.simulated_emission_rate}"
            assert (
                self.concentration_rescale_value is None
            ), f"For Gaussian plumes, concentration_rescale_value must not be set - {self.concentration_rescale_value}"
        else:
            self.position_by_source = False

        # Set up Azure if needed
        if self.config.azure_cluster:
            create_ml_client_config()

        # Initialize clients
        self.ml_client = self.config.ml_client or initialize_ml_client(self.config.azure_cluster)
        self.s3_client = self.config.s3_client or initialize_s3_client(self.ml_client)
        self.storage_options = (
            get_storage_options(self.ml_client) if self.config.storage_options is None else self.config.storage_options
        )
        self.fs = get_azure_ml_file_system(self.ml_client)
        self.abs_client = initialize_blob_service_client(self.ml_client)
        self.def_blob_storage = get_default_blob_storage(self.ml_client)
        self.container_name = self.def_blob_storage.container_name

        # Download plume catalog to disk
        self.plume_catalog = self.config.plume_catalog
        self.plume_catalog_local_path = download_plume_catalog(self.config.plume_catalog)

        # Set up output directory. If storage_options is provided, we save the output locally using
        # the specified out_dir path. Otherwise, we save to Azure Blob Storage.
        if self.config.storage_options is None:
            self.out_dir = get_abfs_output_directory(self.ml_client, Path(self.config.out_dir))
        else:
            self.out_dir = Path(self.config.out_dir)

        self._setup_random_seed()  # ensure reproducibility for the same scene
        self.plume_files = self.load_plumes()

    def __call__(self, log_params: bool = True) -> None:
        """Run the data generation pipeline."""
        # Check if parquet exists first
        parquet_file = self._get_parquet_path()
        if self.fs.exists(parquet_file):
            raise ValueError(f"Parquet file already exists at {parquet_file}!")  # Raise error so job fails

        if log_params:
            self._log_params_to_mlflow()

        data = self.download_data()
        tile_level_data = self.prepare_tile_level_data(data)
        crops = self.generate_crops(data)
        data_items = self.generate_synthetic_data_items(
            self.plume_files, crops, position_by_source=self.position_by_source, **tile_level_data
        )
        self.save_parquet(data_items)

    def prepare_tile_level_data(self, data: Any) -> dict[str, Any]:
        """Prepare data that is shared across all crops from a single satellite tile/scene.

        This method processes data that is common to all crops generated from a single
        satellite scene, avoiding redundant computation during crop generation. For example,
        sensor parameters that are constant across the entire scene can be prepared here.

        Parameters
        ----------
        data : Any
            Raw data downloaded for the satellite scene/tile

        Returns
        -------
        dict[str, Any]
            Dictionary containing tile-level data needed for crop generation
        """
        return {}

    # Public methods
    def load_plumes(self) -> npt.NDArray:
        """Load plume catalog."""
        with open(self.plume_catalog_local_path) as file:
            plume_files = json.load(file)
        plume_files_array = np.array(plume_files)
        return plume_files_array

    # Protected helper methods
    def _setup_random_seed(self) -> None:
        """
        Set random seed based on hash ID.

        The final random seed is deterministically derived from both the initial random seed
        and the hash ID to ensure reproducibility for the same scene.
        """
        hash_input = self.hash_id
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        self.random_seed = (hash_value + self.random_seed) % 2**32
        self.rng = np.random.default_rng(self.random_seed)

    def _select_plumes(
        self, plume_files: npt.NDArray[np.ndarray], verbose: bool = False
    ) -> tuple[npt.NDArray, list[np.ndarray], list[float]]:
        """Randomly select plumes from the catalog based on configured probabilities.

        This method samples plume counts from a probability distribution defined in `plume_proba_dict`,
        then randomly selects that many plumes from the available plume files. For Gaussian plumes,
        it also samples emission rates from a Pareto distribution and rescales concentrations accordingly.

        Parameters
        ----------
        plume_files : npt.NDArray[np.ndarray]
            Array of available plume file paths to sample from
        verbose : bool, default=False
            Whether to print additional debug information

        Returns
        -------
        tuple[npt.NDArray, list[np.ndarray], list[float]]
            A tuple containing:
            - Selected plume file paths as a numpy array
            - List of transformed plume arrays
            - List of plume emission rates (nan for non-Gaussian plumes)
        """
        # Extract keys (values to sample) and probabilities
        values = np.array([int(k) for k in list(self.plume_proba_dict.keys())])
        probabilities = np.array([float(k) for k in list(self.plume_proba_dict.values())])

        # Randomly sample a value based on probabilities
        plume_count = int(self.rng.choice(values, p=probabilities))
        if len(plume_files) == 0 or plume_count == 0:
            chosen_plume_files = np.array([])
            plume_arrays = []
            plume_emissions = []
        else:
            chosen_plume_files = self.rng.choice(plume_files, size=plume_count)

        if self.plume_type is None:
            raise ValueError("Plume type must not be None when loading and transforming plumes.")
        plume_transform_params = build_plume_transform_params(
            self.plume_type,
            self.psf_sigma,
            self.target_spatial_resolution,
        )

        # For Gaussian plumes we need to rescale the concentration values to change the emission rate
        # we sample from a pareto distribution so that we have more smaller plumes (to a minimum emission
        # rate of 100kg/h) and fewer larger plumes (clipping to a maximum emission rate of 100,000kg/h)
        if self.plume_type is PlumeType.GAUSSIAN:
            sampled_emission_rates = pareto(scale=self.min_emission_rate, b=1).rvs(
                size=plume_count, random_state=self.rng
            )
            sampled_emission_rates = np.clip(sampled_emission_rates, self.min_emission_rate, 100_000)
            concentration_rescale_array = sampled_emission_rates / self.simulated_emission_rate
            if verbose:
                logger.info(f"Sampled emission rates: {sampled_emission_rates}")
                logger.info(f"Concentration rescale array: {concentration_rescale_array}")
            plume_emissions = sampled_emission_rates.tolist()  # return value
        else:
            assert self.concentration_rescale_value is not None
            concentration_rescale_array = [self.concentration_rescale_value] * len(chosen_plume_files)
            # for non-Gaussian plumes we don't know the emission rates, so we set them to nan
            plume_emissions = [float("nan")] * len(chosen_plume_files)

        plume_arrays = [
            load_and_transform_plume_arr(
                plume_tiff_path=cpf,
                blob_service_client=self.abs_client,
                container_name=self.container_name,
                plume_transform_params=plume_transform_params,
                rotation_degrees=self.rng.integers(0, 360),
                concentration_rescale_value=rsv,
            )
            for cpf, rsv in zip(chosen_plume_files, concentration_rescale_array, strict=False)
        ]

        # Filter out plumes with all-zero arrays and log which were removed
        filtered_plume_arrays = []
        filtered_plume_files = []
        filtered_emission_rates = []

        for plume_arr, plume_file, emission_rate in zip(plume_arrays, chosen_plume_files, plume_emissions, strict=True):
            if np.all(plume_arr == 0):
                logger.warning(f"Filtered out plume file {plume_file} — array became all zeros after transformation.")
                continue
            filtered_plume_arrays.append(plume_arr)
            filtered_plume_files.append(plume_file)
            filtered_emission_rates.append(emission_rate)

        plume_arrays = filtered_plume_arrays
        chosen_plume_files = np.array(filtered_plume_files)
        plume_emissions = filtered_emission_rates

        if verbose:
            for plume_ in plume_arrays:
                logger.info(
                    f"Plume size in px: {(plume_ > 0.00001).sum()} with sum(mol/cm²) = {(plume_.clip(min=0.0)).sum():.3f}"  # noqa
                )
        return chosen_plume_files, plume_arrays, plume_emissions

    def _overlaps_with_producing_area(
        self,
        crop_x: int,
        crop_y: int,
        chip_transform: Affine,
        producing_area_polygon: Polygon,
        target_producing_regions: dict[str, Polygon],
        check_overlap_ratio: bool = True,
        overlap_ratio_threshold: float = 0.1,
    ) -> tuple[bool, str]:
        """Check if the crop overlaps with target producing regions and return the region label and overlap boolean.

        Parameters
        ----------
        crop_x : int
            The x-coordinate of the crop
        crop_y : int
            The y-coordinate of the crop
        chip_transform : Affine
            The affine transform of the source tile
        producing_area_polygon : Polygon
            The polygon of the oil and gass producing areas
        target_producing_regions : dict[str, Polygon]
            The polygon of the target producing regions (e.g. Marcellus, Permian, Marcellus/Colorado)
        check_overlap_ratio : bool, optional
            Whether to eliminate tiles that excede the overlap ratio threshold, by default True
        overlap_ratio_threshold : float, optional
            The threshold for overlap ratio, by default 0.1

        Returns
        -------
        tuple[bool, str]
            A tuple containing a boolean indicating if the crop overlaps with the target producing regions
            and the region label if it does. The region label is either Hassi, Permian, Marcellus/Colorado, or Other.
        """
        region_overlap = "Other"

        # Calculate chip bounds in the native chip CRS
        x_min, y_min = rasterio.transform.xy(
            chip_transform, crop_y + self.crop_size, crop_x, offset="center"
        )  # bottom-left
        x_max, y_max = rasterio.transform.xy(
            chip_transform, crop_y, crop_x + self.crop_size, offset="center"
        )  # top-right

        # Create a polygon in native chip tile CRS
        polygon_crop = Polygon(
            [
                (x_min, y_min),  # bottom-left
                (x_max, y_min),  # bottom-right
                (x_max, y_max),  # top-right
                (x_min, y_max),  # top-left
                (x_min, y_min),  # close the loop
            ]
        )

        # Check if the chip overlaps with the target producing regions
        intersection_with_producing = producing_area_polygon.intersection(polygon_crop)

        overlap_percentage = intersection_with_producing.area / polygon_crop.area

        if check_overlap_ratio and overlap_percentage < overlap_ratio_threshold:  # between 0 and 1
            logger.info(
                f"The crop has less than {(overlap_ratio_threshold * 100):.0f}% overlapping area with \
                producing areas and is skipped"
            )
            return False, region_overlap

        # Check if the chip overlaps with the target producing regions
        for region, region_polygon in target_producing_regions.items():
            if region_polygon.intersection(polygon_crop).area > 0.0:
                region_overlap = region
                break

        return True, region_overlap

    def _get_parquet_path(self) -> Path | AzureBlobPath:
        """Get the Azure Blob Storage path for the parquet file."""
        return self.out_dir / f"{self.scene_id}.parquet"

    def _get_serializable_params(self) -> dict:
        """Get serializable parameters dictionary."""
        return {
            k: self._convert_to_serializable(v)
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v) and k not in self.NON_SERIALIZABLE_ATTRS
        }

    def _log_params_to_mlflow(self) -> None:
        """Log parameters to MLflow."""
        params = self._get_serializable_params()
        for key, value in params.items():
            max_param_len = 500
            if len(str(value)) <= max_param_len:
                mlflow.log_param(key, str(value))
            else:
                logger.error(f"Logging param {key} will not work. It has length {len(str(value))}")

    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """Convert special types to serializable formats."""
        if isinstance(obj, Path | PurePosixPath):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.name
        # Recursively handle containers
        if isinstance(obj, tuple | list):
            return [BaseDataGeneration._convert_to_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {k: BaseDataGeneration._convert_to_serializable(v) for k, v in obj.items()}
        return obj

    # Abstract methods that must be implemented by subclasses
    @property
    @abc.abstractmethod
    def hash_id(self) -> str:
        """
        Get the unique scene identifier for hash computation.

        This ID ensures deterministic data generation for a specific satellite scene:
        - For Sentinel-2: Combination of MGRS tile and date
        - For EMIT: The EMIT Granule ID (tile identifier)
        """
        pass

    @property
    @abc.abstractmethod
    def scene_id(self) -> str:
        """Get the unique scene identifier for file naming."""
        pass

    @abc.abstractmethod
    def download_data(self) -> Any:
        """Download satellite data."""
        return

    @abc.abstractmethod
    def generate_crops(self, data: Any) -> Iterator[Any]:  # TODO: should this be a generator?
        """Generate crops from satellite data."""
        pass

    @abc.abstractmethod
    def generate_synthetic_data_items(
        self,
        plume_files: npt.NDArray,
        crops: Iterator[dict[str, Any]],
        position_by_source: bool,
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[BasePlumesDataItem]:
        """Generate synthetic data items with plumes."""
        pass

    @abc.abstractmethod
    def save_parquet(self, data_items: Iterator[BasePlumesDataItem]) -> None:
        """Save data items to parquet file."""
        pass
