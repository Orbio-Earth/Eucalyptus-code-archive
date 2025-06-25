"""
Dispatch an Azure ML job on a compute cluster that trains a neural network.

On the cluster, it will run the `training_script.py` script.

```
python -m src.training.run_training_job --help
```
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import tempfile

import omegaconf
import pyarrow.parquet as pq
from azure.ai.ml import Input, command
from azure.ai.ml.constants import InputOutputModes
from azure.storage.blob import ContainerClient
from azureml.fsspec import AzureMachineLearningFileSystem
from omegaconf import OmegaConf
from tqdm import tqdm

from src.azure_wrap.blob_storage_sdk_v2 import DATASTORE_URI
from src.azure_wrap.ml_client_utils import (
    ensure_compute_target_exists,
    get_azureml_uri,
    get_default_blob_storage,
    initialize_blob_service_client,
    initialize_ml_client,
    make_acceptable_uri,
)
from src.utils import IGNORE_PATTERNS, METHANE_CV_ENV, REPO_ROOT


def get_num_row_groups(file: str) -> tuple[str, int | None]:
    """Get the number of row groups in a Parquet file."""
    fs = AzureMachineLearningFileSystem(DATASTORE_URI)
    file_ending = file.split("/")[-1]
    try:
        num_row_groups = pq.ParquetFile(file, filesystem=fs).num_row_groups
        print(f"{file_ending}: {num_row_groups=}")
        return file_ending, num_row_groups
    except Exception as e:
        print(f"Error processing {file_ending}: {e}")
        return file_ending, None  # Handle errors gracefully


def process_files_in_parallel(filenames: list[str]) -> dict[str, int]:
    """Get numrowgroups of Parquet files in parallel using multiprocessing."""
    num_workers = mp.cpu_count()
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(get_num_row_groups, filenames), total=len(filenames)))

    parquet_rowgroups = {file: num_row_groups for file, num_row_groups in results if num_row_groups is not None}
    return parquet_rowgroups


def create_parquet_rowgroups_mapping(
    train: bool,
    folder: str,
    config: omegaconf.dictconfig.DictConfig,
    container_client: ContainerClient,
) -> str:
    """Create parquet ending to number of rowgroup dict e.g. {"test.parquet": 333}."""
    parquet_rowgroups_path = f"parquet_rowgroups_{folder}.json"
    uri = config.train.data.uri if train else config.validation.data.uri

    blob_list = container_client.list_blobs(name_starts_with=uri)
    filenames = []
    for blob in blob_list:
        if blob.name.endswith("parquet"):
            filenames.append(blob.name)

    print(f"Train: {train}. Found {len(filenames)} parquet paths in {uri}")

    create_mapping = True
    if os.path.exists(parquet_rowgroups_path):
        with open(parquet_rowgroups_path) as json_file:
            parquet_rowgroups = json.load(json_file)
        # Saved json has same length as our filenames --> We can use the existing mapping
        if len(parquet_rowgroups) == len(filenames):
            create_mapping = False

    if create_mapping:
        print(f"Train: {train}. Getting rowgroups for {len(filenames)} parquet paths")
        parquet_rowgroups = process_files_in_parallel(filenames)
        with open(parquet_rowgroups_path, "w") as json_file:
            json.dump(parquet_rowgroups, json_file, indent=4)

    return parquet_rowgroups_path


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(prog="Training Job", description="Sets up the training job and compute cluster. ")

    parser.add_argument(
        "--config",
        "-f",
        required=True,
        help=(
            "Config file to use as base. The configs directory is 'src/training/configs' so"
            " any configs there should have that prefix e.g. 'src/training/configs/s2.yaml'"
        ),
    )
    parser.add_argument("--experiment", help="Experiment name. If not set, you will be prompted for a name.")
    parser.add_argument(
        "--pretrained_model_identifier",
        type=str,
        help="MLflow model identifier to continue training from, for example models:/torchgeo_pwr_unet/548",
        default=None,
        required=False,
    )

    return parser.parse_args()


def main() -> None:
    """Set up the training job and send to AML for training."""
    args = parse_args()
    ml_client = initialize_ml_client()
    config = OmegaConf.load(args.config)

    print(f"Using compute cluster: {config.compute.compute_target}")

    mode = InputOutputModes.DOWNLOAD

    ensure_compute_target_exists(ml_client, config.compute.compute_target)

    # the environment must be manually updated after the Docker image is built and deployed in CI
    custom_env = ml_client.environments.get(name=METHANE_CV_ENV, version=1)

    traindata_uri_glob = get_azureml_uri(ml_client, config.train.data.uri)
    acceptable_traindata_uri = make_acceptable_uri(str(traindata_uri_glob))
    print("URI glob for training data: ", acceptable_traindata_uri)

    valdata_uri_glob = get_azureml_uri(ml_client, config.validation.data.uri)
    acceptable_valdata_uri = make_acceptable_uri(str(valdata_uri_glob))
    print("URI glob for validation data: ", acceptable_valdata_uri)

    # Construct a mapping between .parquet files and num_row_groups to speed up dataset creation in the job
    abs_client = initialize_blob_service_client(ml_client)
    def_blob_storage = get_default_blob_storage(ml_client)
    container_name = def_blob_storage.container_name
    container_client = abs_client.get_container_client(container_name)

    if acceptable_traindata_uri.endswith(".parquet"):  # Don't need a mapping if we only use one file
        train_parquet_rowgroups_path = "tmp.json"
    else:
        train_folder = config.train.data.uri.replace("/", "_")
        train_parquet_rowgroups_path = create_parquet_rowgroups_mapping(True, train_folder, config, container_client)

    if acceptable_valdata_uri.endswith(".parquet"):  # Don't need a mapping if we only use one file
        val_parquet_rowgroups_path = "tmp.json"
    else:
        val_folder = config.validation.data.uri.replace("/", "_")
        val_parquet_rowgroups_path = create_parquet_rowgroups_mapping(False, val_folder, config, container_client)

    inputs = {
        "traindata_uri_glob": Input(type=config.train.data.data_type, path=acceptable_traindata_uri, mode=mode),  # type: ignore
        "validation_uri_glob": Input(type=config.validation.data.data_type, path=acceptable_valdata_uri, mode=mode),  # type: ignore
        "traindata_rowgroup_path": train_parquet_rowgroups_path,
        "valdata_rowgroup_path": val_parquet_rowgroups_path,
        "max_train_files": config.train.max_train_files,
        "random_state": config.train.random_state,
        "ground_truth_dataset": config.ground_truth.dataset,
        "lr": config.train.optimizer.lr,
        "beta1": config.train.optimizer.beta1,
        "beta2": config.train.optimizer.beta2,
        "eps": config.train.optimizer.eps,
        "MSE_multiplier": config.train.loss.mse_multiplier,
        "binary_threshold": config.train.loss.binary_threshold,
        "probability_threshold": config.validation.probability_threshold,
        "epochs": config.train.epochs,
        "early_patience": config.validation.early_patience,
        "epochs_warmup": config.train.epochs_warmup,
        "validate_every_x": config.validation.validate_every_x,
        "registered_model_name": config.meta.registered_model_name,
        "num_workers": config.compute.num_workers,
        "min_batch_size": config.train.optimizer.min_batch_size,
        "max_batch_size": config.train.optimizer.max_batch_size,
        "train_shrinkage": config.train.train_shrinkage,
        "validation_shrinkage": config.validation.validation_shrinkage,
        "train_monitoring_ratio": config.train.train_monitoring_ratio,
        "modulation_start": config.train.modulation_start,
        "modulation_end": config.train.modulation_end,
        "pretrained_model_identifier": args.pretrained_model_identifier,
        "satellite_id": config.train.satellite_id,
        "model_type": config.model.model_type,
        "model_encoder": config.model.encoder,
        "bands": config.train.bands,
        "snapshots": config.train.snapshots,
    }
    # Base command string
    command_str = (
        "export NCCL_DEBUG=WARN; "
        "cd methane-cv; "
        "conda run -n methane-cv "
        "pip install --no-deps -r requirements.txt;"
        "conda run -n methane-cv "
        "--no-capture-output "
        "python -m src.training.training_script "
        "--traindata_uri_glob ${{inputs.traindata_uri_glob}} "
        "--validation_uri_glob ${{inputs.validation_uri_glob}} "
        "--traindata_rowgroup_path ${{inputs.traindata_rowgroup_path}} "
        "--valdata_rowgroup_path ${{inputs.valdata_rowgroup_path}} "
        "--max_train_files ${{inputs.max_train_files}} "
        "--ground_truth_dataset ${{inputs.ground_truth_dataset}} "
        "--random_state ${{inputs.random_state}} "
        "--lr ${{inputs.lr}} "
        "--beta1 ${{inputs.beta1}} "
        "--beta2 ${{inputs.beta2}} "
        "--eps ${{inputs.eps}} "
        "--MSE_multiplier ${{inputs.MSE_multiplier}} "
        "--binary_threshold ${{inputs.binary_threshold}} "
        "--probability_threshold ${{inputs.probability_threshold}} "
        "--epochs ${{inputs.epochs}} "
        "--early_patience ${{inputs.early_patience}} "
        "--epochs_warmup ${{inputs.epochs_warmup}} "
        "--validate_every_x ${{inputs.validate_every_x}} "
        "--registered_model_name ${{inputs.registered_model_name}} "
        "--num_workers ${{inputs.num_workers}} "
        "--min_batch_size ${{inputs.min_batch_size}} "
        "--max_batch_size ${{inputs.max_batch_size}} "
        "--azure_cluster  "
        "--train_shrinkage ${{inputs.train_shrinkage}} "
        "--validation_shrinkage ${{inputs.validation_shrinkage}} "
        "--train_monitoring_ratio ${{inputs.train_monitoring_ratio}} "
        "--modulation_start ${{inputs.modulation_start}} "
        "--modulation_end ${{inputs.modulation_end}} "
        "--satellite-id ${{inputs.satellite_id}} "
        "--model ${{inputs.model_type}} "
        "--encoder ${{inputs.model_encoder}} "
        "--bands ${{inputs.bands}} "
        "--snapshots ${{inputs.snapshots}} "
    )

    # Add pretrained model argument if provided
    if args.pretrained_model_identifier:
        command_str += "--pretrained_model_identifier ${{inputs.pretrained_model_identifier}} "
    else:
        inputs.pop("pretrained_model_identifier")

    # We need to provide the worker with the required code, which is in the `src` directory.
    # But we also need to provide the whole PROJECT_ROOT (methane-cv) directory
    # because the Dockerfile this is running in is set to WORKDIR REPO_ROOT
    # We are calling the script with `python -m src.training.training_script`
    # so the `src` directory needs to be there, not just its content.
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
            compute=config.compute.compute_target,
            experiment_name=input("Experiment name: ") if not args.experiment else args.experiment,
            display_name=None,  # let Azure ML generate a cute random combination of words
            description=input("Enter description of this experiment: "),
            environment_variables={
                "DATASET_MOUNT_BLOCK_BASED_CACHE_ENABLED": True,
                "DATASET_MOUNT_MEMORY_CACHE_SIZE": 1024,  # default is 128 MB
            },
            # increase the shared memory size (default is much too low, I think 8 GB)
            shm_size=config.compute.shared_memory,
        )

        # submit the command and get a URL for the status of the job
        returned_job = ml_client.jobs.create_or_update(command_job)
        print(f"job name: {returned_job.name}")
        print(f"job status url: {returned_job.studio_url}")


if __name__ == "__main__":
    main()
