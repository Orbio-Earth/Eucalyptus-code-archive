"""Dispatch Azure ML jobs to generate the training/validation data for a specified satellite.

See configs/README.md for more details on how to run this script.
"""

import itertools
import json
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from azure.ai.ml import command
from azure.ai.ml.entities import Command, Environment
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.azure_wrap.ml_client_utils import (
    ensure_compute_target_exists,
    get_azureml_uri,
    initialize_ml_client,
    make_acceptable_uri,
)
from src.data.common.sim_plumes import PlumeType
from src.utils import IGNORE_PATTERNS, METHANE_CV_ENV, REPO_ROOT
from src.utils.git_utils import get_git_revision_hash
from src.utils.parameters import SatelliteID


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    """Send Azure ML jobs for dataset generation."""
    print("*********** BASIC JOB CONFIGS ************")
    print(f"Satellite: {config.satellite.name}")
    print(f"Plume type: {config.plumes.name}")
    print(f"Generating set: {config.split.name}")

    plume_type = PlumeType(config.plumes.plume_type)
    satellite = SatelliteID(config.satellite.id)

    # Get transformation combinations
    transformations_grid = OmegaConf.to_container(config.satellite_split.transformations_grid, resolve=True)
    transformations = compute_transformation_combinations(transformations_grid)  # type: ignore[arg-type]
    print(f"Generated {len(transformations)} transformation combinations")

    # Initialize Azure ML
    ml_client = initialize_ml_client()
    ensure_compute_target_exists(ml_client, config.satellite.compute_target)
    custom_env = ml_client.environments.get(name=METHANE_CV_ENV, version=1)

    # Set up plume catalog
    plumes_catalog_uri = get_azureml_uri(ml_client, config.plumes_split.catalog_uri)

    acceptable_uri = make_acceptable_uri(str(plumes_catalog_uri))
    print(f"Plumes catalog URI: {plumes_catalog_uri}")
    # Get output directory suffix
    suffix_prompt = (
        f"Enter unique suffix for output directory after 'data/{plume_type}/{satellite}/{config.split.name}_'. "
        "Use hyphens (-) instead of spaces: "
    )
    suffix = input(suffix_prompt)

    # Set up paths and names
    out_base_dir = f"data/{plume_type.value}/{satellite}/{config.split.name}_{suffix}"
    experiment_name = f"{satellite}-{plume_type.value}-{config.split.name}-{suffix}"

    # Get git revision for tracking
    git_revision_hash = get_git_revision_hash()

    # Get satellite-specific queries grouped by cloud coverage
    queries_by_coverage = get_queries_by_cloud_coverage(
        config.satellite_split.tiles_query_files,
        config.satellite.cloud_coverage_threshold,
        satellite,
        config.split.name,
    )

    print("\n*********** SENDING JOBS ************")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy code to temp directory
        shutil.copytree(
            REPO_ROOT,
            tmpdir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*IGNORE_PATTERNS),
        )

        # Generate all job combinations
        all_jobs = [
            (query, transformation_params, cloud_range)
            for cloud_range, queries in queries_by_coverage.items()
            for query in queries
            for transformation_params in transformations
        ]

        if config.test:
            # For a test run, we only do the first two tiles
            all_jobs = all_jobs[:2]
            print("Running in 'test' mode. Limiting to two tiles and passing --test flag to data generation jobs.")

        # Submit jobs
        for query, transformation, cloud_range in tqdm(all_jobs, desc="Submitting jobs"):
            # Create transformation-cloud_coverage-specific output directory
            out_dir = Path(out_base_dir)

            # Get cloud coverage thresholds dict
            quality_thresholds = get_quality_thresholds(config=config, cloud_range=cloud_range, satellite=satellite)

            # Define and submit job
            job = define_generate_job(
                source_dir=tmpdir,
                custom_env=custom_env,
                compute_target=config.satellite.compute_target,
                satellite=satellite,
                plume_type=plume_type,
                query=query,
                plume_catalog=acceptable_uri,
                out_dir=str(out_dir),
                quality_thresholds=quality_thresholds,
                transformation_params=transformation,
                git_revision_hash=git_revision_hash,
                experiment_name=experiment_name,
                config=config,
                test=config.test,
                psf_sigma=config.satellite.psf_sigma,
                concentration_rescale_value=config.plumes.concentration_rescale_value,
                simulated_emission_rate=config.plumes.simulated_emission_rate,
                min_emission_rate=config.plumes.min_emission_rate,
            )

            run = ml_client.jobs.create_or_update(job)
            print(f"Job name: {run.name}")
            print(f"Studio URL: {run.studio_url}")


def define_generate_job(  # noqa: PLR0913 (too many arguments)
    source_dir: str,
    custom_env: Environment,
    compute_target: str,
    satellite: SatelliteID,
    plume_type: PlumeType,
    query: dict[str, Any],
    plume_catalog: str,
    out_dir: str,
    quality_thresholds: dict,
    transformation_params: dict[str, float],
    git_revision_hash: str,
    experiment_name: str,
    config: DictConfig,
    test: bool,
    psf_sigma: float,
    concentration_rescale_value: float,
    simulated_emission_rate: float | None,
    min_emission_rate: float,
) -> Command:
    """Define an Azure ML command job for generating synthetic methane plume datasets.

    Parameters
    ----------
    source_dir : str
        Local directory containing the source code to be uploaded to Azure ML
    custom_env : Environment
        Azure ML environment configuration for the job
    compute_target : str
        Name of the Azure ML compute target to use
    satellite : SatelliteID
        Identifier for the satellite data source (e.g., EMIT, S2)
    plume_type : PlumeType
        Type of plume to use for data generation (e.g., RECYCLED, AVIRIS)
    query : dict[str, Any]
        Satellite-specific query parameters (e.g., MGRS tile and date for S2, emit_id for EMIT)
    plume_catalog : str
        Azure ML URI to the plume catalog file
    out_dir : str
        Output directory path for generated data
    cloud_coverage_thresholds : dict
        Dictionary of cloud coverage thresholds for different crop types
    transformation_params : dict[str, float]
        Parameters for plume transformations (e.g., scale, rotation)
    git_revision_hash : str
        Git commit hash for experiment tracking
    experiment_name : str
        Name of the Azure ML experiment
    config : DictConfig
        Configuration object containing satellite-specific parameters
    test : bool
        If True, runs in test mode with reduced data
    psf_sigma : float
        Point spread function sigma value for gaussian filter used to rescale plume to target sensor resolution
    concentration_rescale_value : float
        The factor by which to scale plumes values (concentrations/enhancements) before inserting into imagery.
    simulated_emission_rate: float | None
        The simulated emission rate of the plume, if known. If None, the emission rate is not known.
    min_emission_rate: float
        The minimum emission rate of the plume.

    Returns
    -------
    Command
        Configured Azure ML command job ready for submission

    Notes
    -----
    The function handles satellite-specific parameters and configurations, setting up
    the appropriate command line arguments for the data generation script.
    """
    # Common parameters for all satellites
    # TODO: do we need to load the plume-specific transformation_params here and pass into the job?

    inputs = {
        "satellite": str(satellite),
        "plume_type": str(plume_type),
        "plume_catalog": plume_catalog,
        "out_dir": out_dir,
        "crop_size": config.crop_size,
        "quality_thresholds": shlex.quote(json.dumps(quality_thresholds)),
        "transformation_params": shlex.quote(json.dumps(transformation_params)),
        "random_seed": config.random_seed,
        "git_revision_hash": git_revision_hash,
        "plume_proba_dict": shlex.quote(json.dumps(dict(config.satellite_split.plume_proba_dict))),
        "psf_sigma": psf_sigma,
        "concentration_rescale_value": concentration_rescale_value,
        "min_emission_rate": min_emission_rate,
        "simulated_emission_rate": simulated_emission_rate,
        "hapi_data_path": config.satellite.hapi_data_path,
    }

    # Add satellite-specific parameters
    if satellite == SatelliteID.S2:
        inputs.update(
            {
                "sentinel_MGRS": query["mgrs"],
                "sentinel_date": query["date"],
                "s2_bands": config.satellite.bands,
                "time_delta_days": config.satellite.time_delta_days,
                "nb_reference_ids": config.satellite.nb_reference_ids,
                "omnicloud_cloud_t": config.satellite.omnicloud_cloud_t,
                "omnicloud_shadow_t": config.satellite.omnicloud_shadow_t,
            }
        )
    elif satellite == SatelliteID.EMIT:
        inputs.update(
            {
                "emit_id": query["emit_id"],
            }
        )
    elif satellite == SatelliteID.LANDSAT:
        inputs.update(
            {
                "landsat_tile_id": query["ID"],
                "landsat_bands": config.satellite.bands,
                "time_delta_days": config.satellite.time_delta_days,
                "nb_reference_ids": config.satellite.nb_reference_ids,
            }
        )
    else:
        raise ValueError(f"Unhandled satellite {satellite}")

    return command(
        code=source_dir,
        command=(
            "cd methane-cv; "
            "conda run -n methane-cv "
            "pip install --no-deps -r requirements.txt; "
            "conda run -n methane-cv "
            f"python -m src.data.generate {'--test' if test else ''} {build_command_args(inputs)}"
        ),
        environment=custom_env,
        inputs=inputs,
        compute=compute_target,
        experiment_name=f"data-gen-{experiment_name}",
        display_name=None,  # let azure ml generate a display name
    )


def build_command_args(inputs: dict[str, Any]) -> str:
    """Build command line arguments string from inputs dictionary."""
    # For each key, we want:
    #   --key ${{inputs.key}}
    #
    # In an f-string:
    # - {{ becomes {
    # - }} becomes }
    #
    # To produce ${ {inputs.key} } literally, we need to write ${{{{inputs.{key}}}}}
    # which after replacement becomes ${{inputs.key}}.

    args = [f"--{key} ${{{{inputs.{key}}}}}" for key in inputs]
    # Always add azure_cluster flag
    args.append("--azure_cluster")
    return " ".join(args)


def get_queries_by_cloud_coverage(
    tiles_query_files: list[str],
    cloud_coverage_threshold: float,
    satellite: SatelliteID,
    split: str,
) -> dict[tuple[float, float], list[dict]]:
    """
    Get queries grouped by cloud coverage ranges.

    For validation set: Uses predefined cloud coverage buckets from data
    For training set: Uses single threshold from config
    """
    df = pd.concat([pd.read_csv(file) for file in tiles_query_files], ignore_index=True)
    # Get satellite-specific queries
    if satellite == SatelliteID.S2:
        queries = df.drop_duplicates(subset=["mgrs", "date"])[["mgrs", "date"]].to_dict("records")
    elif satellite == SatelliteID.EMIT:
        queries = df.drop_duplicates(subset=["emit_id"])[["emit_id"]].to_dict("records")
    elif satellite == SatelliteID.LANDSAT:
        queries = df.drop_duplicates(subset=["ID"])[["ID"]].to_dict("records")
    else:
        raise ValueError(f"Unhandled satellite {satellite}")

    # # Group by cloud coverage - same logic for all satellites
    if split == "validation" and "cloud_cover_min" in df.columns:
        # Use predefined cloud coverage buckets from validation data
        return {
            (float(str(cloud_bucket[0])), float(str(cloud_bucket[1]))): group.to_dict("records")
            for cloud_bucket, group in df.groupby(["cloud_cover_min", "cloud_cover_max"])
        }
    else:
        # Use single threshold from config for training/revamped validation
        return {(0.0, cloud_coverage_threshold): queries}


def get_quality_thresholds(config: DictConfig, cloud_range: tuple[float, float], satellite: SatelliteID) -> dict:
    """
    cloud_range defines main_crop thresholds (from validation buckets or training threshold).

    Other thresholds are satellite-specific
    """
    min_cloud, max_cloud = cloud_range

    # Common for all satellites in validation/training
    thresholds = {
        "main_crop": (min_cloud, max_cloud),
    }

    # Add satellite-specific thresholds
    if satellite in [SatelliteID.S2, SatelliteID.LANDSAT]:
        thresholds.update(
            {
                "main_tile_cloud": (0.0, config.satellite.main_tile_cloud_max),
                "main_tile_nodata": (0.0, config.satellite.main_tile_nodata_max),
                "reference_tile_cloud_shadow": (
                    0.0,
                    config.satellite.reference_tile_cloud_shadow_max,
                ),
                "reference_crop_cloud_shadow": (
                    0.0,
                    config.satellite.reference_crop_cloud_shadow_max,
                ),
            }
        )

    return thresholds


def compute_transformation_combinations(
    transformations_grid: dict,
) -> list[dict[str, float]]:
    """
    Generate all combinations of transformation parameters from a given dictionary.

    Note: We use the Cartersian product to generate the combinations as it does not matter what order the
    transformations are applied.
    """
    keys = list(transformations_grid.keys())
    values = [transformations_grid[key] for key in keys]

    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


if __name__ == "__main__":
    main()
