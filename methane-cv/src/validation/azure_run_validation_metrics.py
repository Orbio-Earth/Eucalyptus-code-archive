"""Run a validation for a trained model on AML.

Command example:

```bash

EMIT:

python -m src.validation.azure_run_validation_metrics --model_name models:/torchgeo_pwr_unet_emit/11
-f src/training/configs/emit.yaml --satellite emit

S2:

python -m src.validation.azure_run_validation_metrics --model_name models:/torchgeo_pwr_unet_s2/1226
-f src/training/configs/s2.yaml --probability_threshold 0.246 --satellite s2
```
"""

import argparse
import shutil
import tempfile

from azure.ai.ml import Input, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from omegaconf import OmegaConf

from src.azure_wrap.ml_client_utils import (
    ensure_compute_target_exists,
    get_azureml_uri,
    get_default_blob_storage,
    initialize_blob_service_client,
    initialize_ml_client,
    make_acceptable_uri,
)
from src.training.run_training_job import create_parquet_rowgroups_mapping
from src.utils import IGNORE_PATTERNS, METHANE_CV_ENV, REPO_ROOT


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        help="The model with version to apply, for example 'models:/torchgeo_pwr_unet/1226'.",
        required=True,
        type=str,
    )
    parser.add_argument("--num_crops", required=False, default=15, type=int, help="Number of crops to create plots for")
    parser.add_argument(
        "--probability_threshold", required=False, default=0.25, type=float, help="probability threshold for masking."
    )
    parser.add_argument(
        "--config",
        "-f",
        required=True,
        help=(
            "Config file to use as base. The configs directory is 'src/training/configs' so"
            " any configs there should have that prefix e.g. 'src/training/configs/s2.yaml'"
        ),
    )
    parser.add_argument(
        "--satellite",
        required=True,
        type=str,
        help="Satellite to use for validation. Options: 's2' or 'emit'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)

    ml_client = initialize_ml_client()

    compute_target = "gpu-cluster-64cores" if args.satellite == "s2" else "gpu-cluster-8cores"
    ensure_compute_target_exists(ml_client, compute_target)

    custom_env = ml_client.environments.get(name=METHANE_CV_ENV, version=1)
    mode = InputOutputModes.RO_MOUNT

    valdata_uri_glob = get_azureml_uri(ml_client, config.validation.data.uri)
    acceptable_valdata_uri = make_acceptable_uri(str(valdata_uri_glob))
    print("URI glob for validation data: ", acceptable_valdata_uri)

    # Construct a mapping between .parquet files and num_row_groups to speed up dataset creation in the job
    abs_client = initialize_blob_service_client(ml_client)
    def_blob_storage = get_default_blob_storage(ml_client)
    container_name = def_blob_storage.container_name
    container_client = abs_client.get_container_client(container_name)

    val_folder = config.validation.data.uri.replace("/", "_")
    val_parquet_rowgroups_path = create_parquet_rowgroups_mapping(False, val_folder, config, container_client)

    validation_dataset_uri = Input(type=AssetTypes.URI_FILE, path=acceptable_valdata_uri, mode=mode)  # type: ignore
    inputs = {
        "model_name": args.model_name,
        "validation_dataset_uri": validation_dataset_uri,
        "num_crops": args.num_crops,
        "probability_threshold": args.probability_threshold,
        "config": args.config,
    }

    if args.satellite == "s2":
        inputs["valdata_rowgroup_path"] = val_parquet_rowgroups_path
        command_str = (
            "export NCCL_DEBUG=WARN; "
            "cd methane-cv; "
            "conda run -n methane-cv "
            "pip install --no-deps -r requirements.txt;"
            "conda run -n methane-cv "
            "--no-capture-output "
            "python -m src.validation.validation_metrics_s2 "
            "--model_name ${{inputs.model_name}} "
            "--validation_dataset_uri ${{inputs.validation_dataset_uri}} "
            "--valdata_rowgroup_path ${{inputs.valdata_rowgroup_path}} "
            "--num_crops ${{inputs.num_crops}} "
            "--probability_threshold ${{inputs.probability_threshold}} "
            "--config ${{inputs.config}} "
        )
    elif args.satellite == "emit":
        # FIXME: Broken, needs update in new world of no transformation validation folders
        command_str = (
            "export NCCL_DEBUG=WARN; "
            "cd methane-cv; "
            "conda run -n methane-cv "
            "pip install --no-deps -r requirements.txt;"
            "conda run -n methane-cv "
            "--no-capture-output "
            "python -m src.validation.validation_metrics_emit "
            "--model_name ${{inputs.model_name}} "
            "--validation_dataset_uri ${{inputs.validation_dataset_uri}} "
            "--num_crops ${{inputs.num_crops}} "
            "--probability_threshold ${{inputs.probability_threshold}} "
            "--config ${{inputs.config}} "
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copytree(
            REPO_ROOT,
            tmpdir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*IGNORE_PATTERNS),
        )

        command_job = command(
            code=tmpdir,
            command=command_str,
            environment=custom_env,
            inputs=inputs,
            compute=compute_target,
            experiment_name="validation",
            display_name=None,  # let Azure ML generate a cute random combination of words
            description=input("Enter description of this experiment: "),
            shm_size=config.compute.shared_memory,
        )

        # submit the command and get a URL for the status of the job
        returned_job = ml_client.jobs.create_or_update(command_job)
        print(f"job name: {returned_job.name}")
        print(f"job status url: {returned_job.studio_url}")
