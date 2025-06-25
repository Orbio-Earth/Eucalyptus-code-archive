"""Dispatch Azure ML jobs to generate the plume data.

This script submits an Azure ML job to run synthetic Gaussian plume data generation.

Example usage:
```bash
python -m src.data.generation.plumes.azure_run_plume_generation \
    --num_plumes 100 \
    --out_dir my_plume_generation \
    --compute_target cpu-cluster
```
"""

import argparse
import shutil
import tempfile

from azure.ai.ml import command

from src.azure_wrap.ml_client_utils import ensure_compute_target_exists, initialize_ml_client
from src.utils import IGNORE_PATTERNS, METHANE_CV_ENV, REPO_ROOT
from src.utils.git_utils import get_git_revision_hash


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run plume generation on Azure ML")

    # Parameters for the plume generation
    parser.add_argument("--spatial_resolution", type=float, default=4.0, help="Spatial resolution in meters")
    parser.add_argument("--temporal_resolution", type=float, default=0.1, help="Temporal resolution in seconds")
    parser.add_argument("--crop_size", type=int, default=512, help="Size of the crop")
    parser.add_argument("--duration", type=int, default=600, help="Duration of the simulation in seconds")
    parser.add_argument("--dispersion_coeff", type=float, default=0.6, help="Dispersion coefficient")
    parser.add_argument("--emission_rate", type=float, default=1000.0, help="Emission rate")
    parser.add_argument("--num_plumes", type=int, default=10, help="Number of plumes to generate")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--OU_sigma_fluctuations",
        type=float,
        default=0.1,
        help="Parameter of the Ornstein-Uhlenbeck process (m/s)",
    )
    parser.add_argument(
        "--OU_correlation_time", type=float, default=1.0, help="Parameter of the Ornstein-Uhlenbeck process (s)"
    )
    parser.add_argument("--save_plots", action="store_true", help="Whether to save visualization plots of the plumes")
    parser.add_argument("--start_plume_id", type=int, default=1, help="Starting ID for plume generation (default: 1)")

    # Azure ML specific parameters
    parser.add_argument("--out_dir", type=str, required=True, help="Name for the output directory")
    parser.add_argument("--compute_target", type=str, default="cpu-cluster", help="Azure ML compute target")

    return parser.parse_args()


def main() -> None:
    """Dispatch an Azure ML job to generate plume data."""
    args = parse_args()

    # Initialize Azure ML client
    ml_client = initialize_ml_client()
    ensure_compute_target_exists(ml_client, args.compute_target)
    custom_env = ml_client.environments.get(name=METHANE_CV_ENV, version=1)

    # Get git revision hash for tracking
    git_revision_hash = get_git_revision_hash()

    # Build command line arguments - always set all parameters
    command_args = [
        f"--spatial_resolution {args.spatial_resolution}",
        f"--temporal_resolution {args.temporal_resolution}",
        f"--crop_size {args.crop_size}",
        f"--duration {args.duration}",
        f"--dispersion_coeff {args.dispersion_coeff}",
        f"--emission_rate {args.emission_rate}",
        f"--num_plumes {args.num_plumes}",
        f"--random_seed {args.random_seed}",
        f"--OU_sigma_fluctuations {args.OU_sigma_fluctuations}",
        f"--OU_correlation_time {args.OU_correlation_time}",
        f"--git_revision_hash {git_revision_hash}",
        f"--out_dir {args.out_dir}",
        f"--start_plume_id {args.start_plume_id}",
        "--azure_cluster",  # Always set this flag when running on Azure
    ]

    if args.save_plots:
        command_args.append("--save_plots")

    command_args_str = " ".join(command_args)

    # Create temporary directory and copy code
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copytree(
            REPO_ROOT,
            tmpdir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*IGNORE_PATTERNS),
        )

        # Define the command job
        command_job = command(
            code=tmpdir,
            command=(
                "cd methane-cv; "
                "conda run -n methane-cv "
                "pip install --no-deps -r requirements.txt; "
                "conda run -n methane-cv --no-capture-output "
                f"python -m src.data.generation.plumes.generate {command_args_str}"
            ),
            environment=custom_env,
            compute=args.compute_target,
            experiment_name=f"plume-generation-{args.out_dir}",
            display_name=None,  # let Azure ML generate a display name
        )

        # Submit the job
        returned_job = ml_client.jobs.create_or_update(command_job)
        print(f"Job name: {returned_job.name}")
        print(f"Job status URL: {returned_job.studio_url}")


if __name__ == "__main__":
    main()
