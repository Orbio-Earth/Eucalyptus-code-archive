"""Run an inference job for a coordinate and time period on Azure ML.

On the cluster, it will run the `azure_run_inference_target_location.py` script.

Sentinel2 Example use for a Hassi site from 2019-11-10 to 2019-12-10:

python -m src.inference.azure_run_inference_target_location --satellite S2 --lat 31.6585 --lon 5.9053 \
    --start_date 2019-11-10 --end_date 2019-12-10 --model_id 1226 --crop_size 128

    
LANDSAT Example use for a customer site from 2025-04-05 to 2025-04-13 with 256x256 crops:

python -m src.inference.azure_run_inference_target_location --satellite LANDSAT --lat 32.496219 --lon -97.554575 \
    --start_date 2025-04-05 --end_date 2025-04-13 --model_id 37 --crop_size 256
"""

import argparse
import shutil
import tempfile
from datetime import datetime

from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential

from src.utils import IGNORE_PATTERNS, METHANE_CV_ENV, REPO_ROOT
from src.utils.parameters import SatelliteID

ml_client = MLClient.from_config(DefaultAzureCredential())
compute_target = "beefy-cpu-cluster"
ml_client.compute.get(compute_target)

custom_env = ml_client.environments.get(name=METHANE_CV_ENV, version=1)


def parse_datetime(s: str) -> datetime:
    """Parse CLI string formatted datetime into datetime."""
    return datetime.strptime(s, "%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--satellite",
        choices=SatelliteID.list(),
        type=SatelliteID,
        required=True,
        help="Satellite data source to use",
    )
    parser.add_argument("--crop_size", required=False, default=128, type=int)
    parser.add_argument("--lat", required=True, type=float)
    parser.add_argument("--lon", required=True, type=float)
    parser.add_argument(
        "--start_date",
        help="The starting date of the analysis in YYYY-MM-DD format.",
        required=True,
        type=parse_datetime,
    )
    parser.add_argument(
        "--end_date",
        help="The end date of the analysis in YYYY-MM-DD format.",
        required=True,
        type=parse_datetime,
    )
    parser.add_argument(
        "--model_id",
        help="The model version to use, for example '70' Assumes its models:/torchgeo_pwr_unet/70.",
        required=True,
        type=str,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copytree(
            REPO_ROOT,
            tmpdir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*IGNORE_PATTERNS),
        )

        command_job = command(
            code=tmpdir,
            command=(
                "cd methane-cv; "
                "conda run -n methane-cv pip install --no-deps -r requirements.txt;"
                "conda run -n methane-cv --no-capture-output python -m src.inference.inference_target_location "
                "--satellite ${{inputs.satellite}} "
                "--lat ${{inputs.lat}} "
                "--lon ${{inputs.lon}} "
                "--start_date ${{inputs.start_date}} "
                "--end_date ${{inputs.end_date}} "
                "--model_id ${{inputs.model_id}} "
                "--crop_size ${{inputs.crop_size}} "
                "--azure_cluster  "
            ),
            environment=custom_env,
            inputs={
                "satellite": args.satellite.value,
                "lat": args.lat,
                "lon": args.lon,
                "start_date": args.start_date.date().isoformat(),
                "end_date": args.end_date.date().isoformat(),
                "model_id": args.model_id,
                "crop_size": args.crop_size,
            },
            compute=compute_target,
            experiment_name="inference_target_location",
            display_name=input("Target location description: "),
        )

        # submit the command and get a URL for the status of the job
        returned_job = ml_client.jobs.create_or_update(command_job)
        print(f"job name: {returned_job.name}")
        print(f"job status url: {returned_job.studio_url}")
