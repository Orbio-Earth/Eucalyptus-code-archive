"""Utility functions such as those for loading the model and setting up distributed processing."""

import logging
import os
from typing import Any

import boto3
import earthaccess
import mlflow
import torch
from azure.ai.ml import MLClient
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from azureml.fsspec import AzureMachineLearningFileSystem
from mypy_boto3_s3 import S3Client
from torch import nn

from src.azure_wrap.ml_client_utils import (
    get_azure_ml_file_system,
    get_storage_options,
    initialize_blob_service_client,
    initialize_ml_client,
)
from src.training.transformations import BaseBandExtractor, ConcatenateSnapshots, MonotemporalBandExtractor
from src.utils.parameters import LANDSAT_BANDS, S2_BANDS, SatelliteID

logger = logging.getLogger(__name__)


####################################################
################## MULTIPROCESSING #################
####################################################


def setup_distributed_processing(rank: int, world_size: int, random_state: int) -> None:
    """
    Initialize distributed processing environment and set seed.

    Args
    -----
        rank (int): Rank of the current process.
        world_size (int): Total number of processes for distributed training.
        random_state (int): Random seed for reproducibility.
    """
    # Initialize distributed processing
    torch.distributed.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",  # CPU only works on gloo backend
        rank=rank,
        world_size=world_size,
    )

    # Only set CUDA device if GPUs are available
    if torch.cuda.is_available():
        # Set the device for this process
        torch.cuda.set_device(rank)

    # Set the seed for reproducibility
    torch.manual_seed(random_state + rank)


def setup_device_and_distributed_model(model: nn.Module, rank: int, world_size: int) -> tuple[nn.Module, torch.device]:
    """
    Set up the model for distributed training on multiple GPUs or CPU.

    Args
    ----
        model (nn.Module): The PyTorch model to be wrapped for distributed training.
        rank (int): The rank of the current process in distributed training.
        world_size (int): The total number of processes.

    Returns
    -------
        nn.Module: The model wrapped in DistributedDataParallel and moved to the appropriate device.
    """
    if torch.cuda.is_available():
        logger.info(f"Running on {world_size} GPUs")
        device = torch.device(f"cuda:{rank}")
        model = model.to(device)
        # FIXME: Unet++ using timm-efficientnet-b1 makes find_unused_parameters=True necessary, but we should
        # remove it whenever we figure out why this is happening
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    else:
        logger.info("Running on CPU")
        device = torch.device("cpu")
        model = model.to(device)
        # FIXME: Unet++ using timm-efficientnet-b1 makes find_unused_parameters=True necessary, but we should
        # remove it whenever we figure out why this is happening
        model = nn.parallel.DistributedDataParallel(model, device_ids=None, find_unused_parameters=True)

    return model, device


####################################################
################## MODEL UTILITIES #################
####################################################


def load_model_and_concatenator(
    model_identifier: str, device: torch.device | str, satellite_id: SatelliteID
) -> tuple[nn.Module, BaseBandExtractor, dict[str, Any]]:
    """Load the model and extract the band concatenator and any other additional training parameters."""
    # load the model from an experiment
    # https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models-mlflow?view=azureml-api-2#loading-models-from-registry
    model = mlflow.pytorch.load_model(model_identifier, map_location=device)

    if isinstance(model, nn.DataParallel | nn.parallel.DistributedDataParallel):
        # Strip out the DataParallel wrapper.
        # More recent models should have it stripped before saving
        # so this is for backwards compatibility.
        model = model.module

    # Send the model to the specified device (GPU or CPU)
    model.to(device)

    try:
        band_concat_params = model.band_concat_params
        # TODO: is there a better solution to handle changed key names in band_concatenator?
        if "s2_bands" in band_concat_params:
            band_concat_params["all_available_bands"] = band_concat_params.pop("s2_bands")
    except AttributeError:
        # If model doesn't have band concatenator params, recreate with default values
        logger.info(
            "NOTE: Model was not saved with its ConcatenateSnapshots object, so we are using the defaults that were "
            "most likely used."
        )
        band_concat_params = {}
    band_concatenator = recreate_band_extractor_for_satellite(satellite_id, **band_concat_params)

    try:
        training_params = model.training_params
    except AttributeError:
        # If model doesn't have training params, recreate with default values
        logger.info(
            "NOTE: Model was not saved with its training parameters, so we are using the defaults that were most "
            "likely used."
        )
        training_params = recreating_training_params()

    return model, band_concatenator, training_params


def recreate_band_extractor_for_satellite(
    satellite_id: SatelliteID,
    **kwargs: Any,
) -> BaseBandExtractor:
    """
    Backwards compatability: Recreate a ConcatenateSnapshots object from the attributes saved alongside the model.

    Defaults are set so we can still use models trained before we started saving those parameters
    """
    if satellite_id in (SatelliteID.S2, SatelliteID.LANDSAT):
        defaults = get_satellite_concatenator_defaults(satellite_id)
        band_extractor = _recreate_snapshot_concatenator_band_extractor(defaults, **kwargs)
    elif satellite_id == SatelliteID.EMIT:
        band_extractor = _recreate_monotemporal_band_extractor(**kwargs)
    else:
        raise ValueError(f"Unhandled satellite {satellite_id}.")

    return band_extractor


def get_satellite_concatenator_defaults(satellite_id: SatelliteID) -> dict[str, Any]:
    """Get the default values for the concatenator parameters for a given satellite."""
    defaults = {
        SatelliteID.S2: {
            "snapshots": ("crop_before", "crop_after"),
            "all_available_bands": S2_BANDS,
            "temporal_bands": ("B11", "B12", "B8A", "B07"),
            "main_bands": ("B11", "B12", "B8A", "B07", "B05"),
            "scaling_factor": 1 / 10_000,
        },
        SatelliteID.LANDSAT: {
            "snapshots": ("crop_earlier", "crop_before"),
            "all_available_bands": LANDSAT_BANDS,
            "temporal_bands": ("swir16", "swir22", "nir08", "red"),
            "main_bands": ("swir16", "swir22", "nir08", "red", "green"),
            "scaling_factor": 1 / 10_000,
        },
    }
    return defaults[satellite_id]


def _recreate_snapshot_concatenator_band_extractor(
    defaults: dict[str, Any],
    all_available_bands: list[str] | None = None,
    snapshots: tuple[str, ...] | None = None,
    temporal_bands: tuple[str, ...] | None = None,
    main_bands: tuple[str, ...] | None = None,
    scaling_factor: float | None = None,
    # used to capture old keyword args that would otherwise cause a TypeError: unexpected keyword argument
    # e.g. s2_bands was renamed to all_available_bands and gets passed in from an old saved model
    **kwargs: dict,
) -> ConcatenateSnapshots:
    # Use provided values if given, otherwise use defaults
    return ConcatenateSnapshots(
        snapshots=list(snapshots if snapshots is not None else defaults["snapshots"]),
        all_available_bands=all_available_bands if all_available_bands is not None else defaults["all_available_bands"],
        temporal_bands=list(temporal_bands if temporal_bands is not None else defaults["temporal_bands"]),
        main_bands=list(main_bands if main_bands is not None else defaults["main_bands"]),
        scaling_factor=scaling_factor if scaling_factor is not None else defaults["scaling_factor"],
    )


def _recreate_monotemporal_band_extractor(band_indices: list[int], scaling_factor: float) -> MonotemporalBandExtractor:
    return MonotemporalBandExtractor(band_indices, scaling_factor)


def recreating_training_params() -> dict[str, Any]:
    """
    Backwards compatability: Recreate the training parameters from the attributes saved alongside the model.

    Defaults are set so we can still use models trained before we started saving those parameters.
    """
    training_params = {
        "MSE_multiplier": 1000.0,
        "binary_threshold": 0.001,
        "probability_threshold": 0.5,
        # ADD MORE PARAMS AS WE SAVE THEM
    }
    return training_params


####################################################
################ INITIALIZE CLIENTS ################
####################################################


def initialize_clients(
    force_msi: bool,
) -> tuple[MLClient, dict, AzureMachineLearningFileSystem, BlobServiceClient, S3Client]:
    """
    Initialize clients for Azure ML, storage, and S3.

    This is a convenience method to load all necessary clients at once.
    """
    # Azure
    ml_client = initialize_ml_client(force_msi=force_msi)
    storage_options = get_storage_options(ml_client)
    fs = get_azure_ml_file_system(ml_client)
    generic_blob_service_client = initialize_blob_service_client(ml_client)

    # AWS
    s3_client = initialize_s3_client(ml_client)
    return ml_client, storage_options, fs, generic_blob_service_client, s3_client


def initialize_s3_client(ml_client: MLClient, region_name: str = "eu-central-1") -> S3Client:
    """Initialize an S3 client."""
    credential = ml_client._credential

    secret_client = SecretClient(vault_url="", credential=credential)
    aws_access_key_id = secret_client.get_secret("").value
    aws_secret_access_key = secret_client.get_secret("").value
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )
    return s3_client


def earthaccess_login(ml_client: MLClient) -> None:
    """Initialize a client for NASA Earth Access."""
    credential = ml_client._credential

    secret_client = SecretClient(vault_url="", credential=credential)
    earthaccess_username = secret_client.get_secret("").value
    earthaccess_password = secret_client.get_secret("").value

    assert isinstance(earthaccess_username, str)
    assert isinstance(earthaccess_password, str)

    os.environ["EARTHDATA_USERNAME"] = earthaccess_username
    os.environ["EARTHDATA_PASSWORD"] = earthaccess_password

    earthaccess.login(strategy="environment", persist=True)


####################################################
#################### LOGGING #######################
####################################################


def setup_logging() -> logging.Logger:
    """Configure root logger and with minimal Azure logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    botocore_logger = logging.getLogger("botocore")
    botocore_logger.setLevel(logging.WARNING)

    azure_logger = logging.getLogger("azure")
    azure_logger.setLevel(logging.ERROR)
    # Return a logger for the calling module
    return logging.getLogger(__name__)
