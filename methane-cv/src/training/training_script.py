r"""Trains a neural network.

You can run it locally to test it or have it run on a GPU compute cluster on Azure ML.

Example:

Sentinel-2:
```sh
python src/training/training_script.py \
    --traindata_uri_glob "data/aviris/S2/training_L1C_2025-01-27_v2/modulate_1.0_resize_1.0/cloud_bucket_30/09UXA_2022-10-31.parquet" \
    --validation_uri_glob "data/aviris/S2/validation_L1C_2025-01-26-plume-object/modulate_1.0_resize_1.0" \
    --traindata_rowgroup_path "does_not_exist.json" \
    --valdata_rowgroup_path "does_not_exist.json" \
    --ground_truth_dataset "src/data/ancillary/ground_truth_plumes.csv" \
    --epochs 1 --lr 0.01 --random_state 42  \
    --num_workers 8 --beta1 0.9 --beta2 0.99 --eps 0.00001 \
    --min_batch_size_per_proc 8 --max_batch_size_per_proc 32  \
    --MSE_multiplier 1000.0 --probability_threshold 0.5 --binary_threshold 0.05 \
    --registered_model_name test_model \
    --satellite-id S2 \
    --model unetplusplus --encoder timm-efficientnet-b1 \
    --max_train_files 50 --early_patience 3 --epochs_warmup 1 --validate_every_x 1 \
    --remote-data
```

EMIT:
```sh
python src/training/training_script.py \
    --traindata_uri_glob "data/carbonmapper/EMIT/training_20250328-new-tile-selection" \
    --validation_uri_glob "data/carbonmapper/EMIT/validation_20250328-new-tile-selection" \
    --traindata_rowgroup_path "data/carbonmapper/EMIT/training_carbonmapper_aviris_and_emit_n55_20250317/modulate_1.0_resize_1.0"  --valdata_rowgroup_path  "data/carbonmapper/EMIT/validation_carbonmapper_aviris_and_emit_n55_20250317/modulate_1.0_resize_1.0"\
    --ground_truth_dataset "src/data/ancillary/EMIT_ground_truth_plumes.csv" \
    --epochs 1 --lr 0.01 --random_state 42 \
    --num_workers 2 --beta1 0.9 --beta2 0.99 --eps 0.00001 \
    --min_batch_size_per_proc 8 --max_batch_size_per_proc 32 \
    --MSE_multiplier 1.00 --probability_threshold 0.5 --binary_threshold 0.05 \
    --registered_model_name test_model \
    --satellite-id EMIT \
    --model unetplusplus  --encoder timm-efficientnet-b1 \
    --bands "ALL" --snapshots "only_maincrop" \
    --max_train_files 50 --early_patience 3 --epochs_warmup 1 --validate_every_x 1 \
    --remote-data
```

LANDSAT:
```sh
python src/training/training_script.py \
    --traindata_uri_glob "data/carbonmapper/LANDSAT/training_2025_03_25/LC08_L1TP_022028_20230112_20230125_02_T2.parquet" \
    --validation_uri_glob "data/carbonmapper/LANDSAT/validation_2023_05_25" \
    --traindata_rowgroup_path "does_not_exist.json" \
    --valdata_rowgroup_path "does_not_exist.json" \
    --ground_truth_dataset "src/data/ancillary/ground_truth_plumes.csv" \
    --epochs 1 --lr 0.01 --random_state 42 \
    --num_workers 8 --beta1 0.9 --beta2 0.99 --eps 0.00001 \
    --min_batch_size_per_proc 8 --max_batch_size_per_proc 32 \
    --MSE_multiplier 1000.0 --probability_threshold 0.5 --binary_threshold 0.05 \
    --registered_model_name test_model \
    --satellite-id LANDSAT \
    --model unetplusplus  --encoder timm-efficientnet-b1 \
    --max_train_files 50 --early_patience 3 --epochs_warmup 1 --validate_every_x 1 \
    --remote-data

python src/training/training_script.py \
    --MSE_multiplier 1000.0 \
    --bands swir16,swir22,nir08,red,green \
    --beta1 0.9 \
    --beta2 0.99 \
    --binary_threshold 0.05 \
    --early_patience 3 \
    --encoder timm-efficientnet-b1 \
    --epochs 0 \
    --epochs_warmup 1 \
    --eps 0.00001 \
    --ground_truth_dataset "src/data/ancillary/landsat_ground_truth_plumes_single.csv" \
    --lr 0.01 \
    --max_batch_size 32 \
    --max_train_files 50 \
    --min_batch_size 8 \
    --model unetplusplus \
    --num_workers 8 \
    --pretrained_model_identifier "models:/landsat/14" \
    --probability_threshold 0.5 \
    --random_state 42 \
    --registered_model_name test_model \
    --remote-data \
    --satellite-id LANDSAT \
    --snapshots crop_earlier,crop_before \
    --traindata_rowgroup_path "data/carbonmapper/LANDSAT/training_2025_03_29_psf_fix/rowgroup_mapping.json" \
    --traindata_uri_glob "data/carbonmapper/LANDSAT/training_2025_03_29_psf_fix/LC08_L1TP_022028_20230112_20230125_02_T2.parquet" \
    --valdata_rowgroup_path "data/carbonmapper/LANDSAT/validation_2025_03_29_psf_fix/rowgroup_mapping.json" \
    --validate_every_x 1 \
    --validation_uri_glob "data/carbonmapper/LANDSAT/validation_2025_03_29_psf_fix" \
```

"""  # noqa: E501 (line-too-long)

import argparse
import copy
import datetime
import functools
import glob
import logging
import operator
import os
import random
import time
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pyarrow as pa
import segmentation_models_pytorch as smp
import torch
import torch.distributed as dist
import torchvision
from azure.ai.ml import MLClient
from azureml.fsspec import AzureMachineLearningFileSystem
from radtran import get_gamma
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.azure_wrap.blob_storage_sdk_v2 import DATASTORE_URI
from src.azure_wrap.ml_client_utils import (
    create_ml_client_config,
    get_default_blob_storage,
)
from src.data.common.utils import tensor_to_dataarray
from src.data.dataset import ArrowDataset, ValidationDataset, collate_rowgroups, collate_rowgroups_val
from src.inference.inference_functions import predict, prepare_model_input
from src.inference.inference_target_location import (
    plot_ground_truth_plots_landsat,
    plot_ground_truth_plots_S2,
    quantify_plume_for_lat_lon,
)
from src.plotting.plotting_functions import (
    get_rgb_from_xarray,
    get_swir_ratio_from_xarray,
    grid16,
    plot_frac,
)
from src.training.loss_functions import TwoPartLoss
from src.training.models import ModelType
from src.training.models.spectral_unet import SpectralUNet, SpectralUNetPlusPlus
from src.training.transformations import (
    BaseBandExtractor,
    ConcatenateSnapshots,
    CustomHorizontalFlip,
    MethaneModulator,
    MonotemporalBandExtractor,
    Rotate90,
)
from src.utils.exceptions import InsufficientImageryException
from src.utils.parameters import (
    EMIT_BANDS,
    LANDSAT_BANDS,
    NUM_LANDSAT_BANDS,
    NUM_S2_BANDS,
    S2_BANDS,
    SATELLITE_COLUMN_CONFIGS,
    TARGET_COLUMN,
    SatelliteID,
)
from src.utils.utils import (
    earthaccess_login,
    initialize_clients,
    load_model_and_concatenator,
    setup_device_and_distributed_model,
    setup_distributed_processing,
)
from src.validation.metrics import Metrics

# Suppress the specific warning
warnings.filterwarnings(
    "ignore",
    message="Length of split at index 0 is 0. This might result in an empty dataset.",
    category=UserWarning,
    module="torch.utils.data.dataset",
)

PARAMS_MAX_STR_LENGTH = 500
NUM_DIMENSIONS = 4

logger = logging.getLogger(__name__)

####################################################
################## MAIN FUNCTION ###################
####################################################


def main(  # noqa: PLR0912, PLR0913, PLR0915 (too-many-arguments, too-many-statements, too-many-branches)
    rank: int,  # needs to be first parameter for distributed process spawning
    world_size: int,
    traindata_uri_glob: str,
    validation_uri_glob: str,
    traindata_rowgroup_path: str,
    valdata_rowgroup_path: str,
    max_train_files: int,
    ground_truth_dataset: str,
    random_state: int,
    model_type: ModelType,
    model_encoder: str,
    bands: list[str],
    snapshots: list[str],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    MSE_multiplier: float,
    binary_threshold: float,
    probability_threshold: float,
    epochs: int,
    early_patience: int,
    epochs_warmup: int,
    validate_every_x: int,
    registered_model_name: str,
    num_workers: int,
    min_batch_size_per_proc: int,
    max_batch_size_per_proc: int,
    satellite_id: SatelliteID,
    azure_cluster: bool = True,
    train_shrinkage: float = 1.0,
    validation_shrinkage: float = 1.0,
    train_monitoring_ratio: float = 0.1,
    modulation_start: float = 1.0,
    modulation_end: float = 0.05,
    pretrained_model_identifier: str | None = None,
    remote_data: bool = False,
) -> None:
    """
    Train the model.

    Arguments
    ---------
    rank : int
        Process rank for distributed training. In distributed training, each process handles a portion
        of the data. Rank identifies which process this is - the main process has rank 0, other
        processes have ranks 1 to (world_size-1). Used for coordinating between processes and ensuring
        metrics are properly aggregated. Example: 0 for main process
    world_size : int
        Total number of processes for distributed training. Example: 4
    traindata_uri_glob : str
        URI pattern for training data. Example: "data/aviris/S2/training_2025_02_21_hassi_marc_perm_splits"
    validation_uri_glob : str
        URI pattern for validation data. Example: "data/aviris/S2/validation_2025_02_21_hassi_marc_perm_splits"
    traindata_rowgroup_path: str
        Local path to json that contains a mapping from train parquet files to num_rows for faster dataset creation.
    valdata_rowgroup_path: str
        Local path to json that contains a mapping from val parquet files to num_rows for faster dataset creation.
    max_train_files: int
        Maximum number of random training parquet files to use every epoch. Example: 500
    ground_truth_dataset : str
        Path to ground truth CSV file. Example: "src/data/ancillary/ground_truth_plumes.csv"
    random_state : int
        Random seed for reproducibility. Example: 42
    model_type : str
        Model architecture to use.
    model_encoder : str
        Pretrained encoder to use. Example: timm-efficientnet-b1
    lr : float
        Learning rate for optimizer. Example: 0.001
    beta1 : float
        First beta parameter for Adam optimizer. Example: 0.9
    beta2 : float
        Second beta parameter for Adam optimizer. Example: 0.99
    eps : float
        Epsilon parameter for Adam optimizer. Example: 0.00001
    MSE_multiplier : float
        Multiplier for MSE loss component. Example: 1000.0
    binary_threshold : float
        Threshold for binary classification, in units of absolute frac. Example: 0.001
    probability_threshold : float
        Threshold for prediction probabilities. Example: 0.5
    epochs : int
        Number of training epochs. Example: 1000
    early_patience: int
        Number of validation epochs without metric improvement to wait before early stopping. Example: 3
    epochs_warmup: int
        Number of epochs to slowly increase modulation and batch size. Example: 20
    validate_every_x: int
        Number of epochs between validation runs to not use too much time validating. Example: 3
    registered_model_name : str
        Name to register trained model as. Example: "torchgeo_pwr_unet"
    num_workers : int
        Number of data loading workers. Example: 40
    min_batch_size_per_proc : int
        Starting batch size per process (GPU). Example: 16
    max_batch_size_per_proc : int
        Final batch size per process (GPU). Example: 96
    azure_cluster : bool, default=True
        Whether running on Azure ML compute cluster.
    train_shrinkage : float, default=1.0
        Proportion of training data to use (between 0 and 1)
    validation_shrinkage : float, default=1.0
        Proportion of validation data to use (between 0 and 1)
    train_monitoring_ratio: float , default=0.1
        Proportion of training data to use for monitoring non-modulated examples (between 0 and 1)
    modulation_start : float, default=1.0
        Starting value for modulation schedule
    modulation_end : float, default=0.05
        Ending value for modulation schedule
    pretrained_model_identifier : str | None, default=None
        Identifier of pretrained model to use. Example: "models:/torchgeo_pwr_unet/548".
        See README for more details.
    remote_data : bool, default=False
        Indicates to load the data from Azure Blob Storage.
    """
    # Distributed setup
    setup_distributed_processing(rank=rank, world_size=world_size, random_state=random_state)
    cpu_group = dist.new_group(backend="gloo")

    # We incrementally increase the batch size each epoch. This is inspired by the paper
    # "Don't Decay the Learning Rate, Increase the Batch Size https://arxiv.org/abs/1711.00489
    batchsize_schedule = np.round(
        np.logspace(np.log2(min_batch_size_per_proc), np.log2(max_batch_size_per_proc), epochs_warmup, base=2)
    ).astype(int)
    betas = (beta1, beta2)

    if satellite_id == SatelliteID.S2:
        all_available_bands = S2_BANDS
        band_extractor: BaseBandExtractor = ConcatenateSnapshots(
            snapshots=snapshots,
            all_available_bands=all_available_bands,
            temporal_bands=bands,
            main_bands=bands,
            scaling_factor=1 / 10_000,
        )
        swir16_band_name = "B11"
        swir22_band_name = "B12"
        region_to_idx = {"Hassi": 0, "Marcellus": 1, "Permian": 2, "Other": 0}
        fp_px_per_crop = 10
    elif satellite_id == SatelliteID.EMIT:
        band_extractor = MonotemporalBandExtractor(
            # FIXME get bands to be used from config
            band_indices=EMIT_BANDS,
            scaling_factor=0.1,
        )
        # The plotting functions expect a SWIR16 and SWIR22 band name, but EMIT is hyperspectral and has multiple
        # bands for both so we just give empty strings so the plotting function does not fail
        swir16_band_name = ""
        swir22_band_name = ""
        region_to_idx = {"Hassi": 0, "Colorado": 1, "Permian": 2, "Other": 0}
        fp_px_per_crop = 5  # EMIT is capable of reaching less FPs with still high recall
    elif satellite_id == SatelliteID.LANDSAT:
        all_available_bands = LANDSAT_BANDS
        band_extractor = ConcatenateSnapshots(
            snapshots=snapshots,
            all_available_bands=all_available_bands,
            temporal_bands=bands,
            main_bands=bands,
            scaling_factor=1 / 10_000,
        )
        swir16_band_name = "swir16"
        swir22_band_name = "swir22"
        region_to_idx = {"Hassi": 0, "Marcellus": 1, "Permian": 2, "Other": 0}
        fp_px_per_crop = 10
    else:
        raise ValueError(f"Satellite type {satellite_id.value} not handled.")

    fs = AzureMachineLearningFileSystem(DATASTORE_URI) if remote_data else None

    # We are calling data_preparation here only to get crop_size/in_channels.
    # We will re-initialise the datasets with a random subset of parquet files inside the epoch loop.
    datasets, crop_size, in_channels, _ = data_preparation(
        traindata_uri_glob,
        band_extractor,
        satellite_id,
        parquet_to_rowgroups_mapping_path=traindata_rowgroup_path,
        filesystem=fs,
        max_train_files=10,
        rank=rank,
    )
    train_dataset = datasets["train"]

    ######################### Transformations #################################

    # Define the additional transformations for the training dataset
    random_rotate = Rotate90()
    custom_flip_transform = CustomHorizontalFlip()
    modulation_schedule = np.linspace(modulation_start, modulation_end, epochs_warmup)

    if satellite_id == SatelliteID.EMIT:
        # FIXME: consider how to modulate EMIT data
        logger.warning("Not applying modulations to EMIT data.")
        train_transform = torchvision.transforms.Compose(
            [
                band_extractor,
                random_rotate,
                custom_flip_transform,
            ]
        )
    else:
        # We initialise the modulation transform with float("nan") because it will be modified inside of the epoch loop
        # according to the modulation_schedule. By setting it to NaN first, it means if for some reason the modulation
        # schedule doesn't work, we will notice straight away.
        modulate_transform = MethaneModulator(
            modulate=float("nan"),
            all_available_bands=all_available_bands,
            swir16_band_name=swir16_band_name,
            swir22_band_name=swir22_band_name,
            orig_swir16_band_name=train_dataset.dataset.orig_swir16_band_name,
            orig_swir22_band_name=train_dataset.dataset.orig_swir22_band_name,
        )
        train_transform = torchvision.transforms.Compose(
            [
                modulate_transform,
                band_extractor,
                random_rotate,
                custom_flip_transform,
            ]
        )

    ######################### Log Inputs to MLFlow ############################
    if rank == 0:
        params = {
            "world_size": world_size,
            "max_train_files": max_train_files,
            "model_type": model_type.value,
            "model_encoder": model_encoder,
            "learning_rate": lr,
            "MSE_multiplier": MSE_multiplier,
            "binary_threshold": binary_threshold,
            "probability_threshold": probability_threshold,
            "epochs": epochs,
            "epochs_warmup": epochs_warmup,
            "early_patience": early_patience,
            "validate_every_x": validate_every_x,
            "num_workers": num_workers,
            "min_batch_size_per_proc": min_batch_size_per_proc,
            "max_batch_size_per_proc": max_batch_size_per_proc,
            "modulation_start": modulation_start,
            "modulation_end": modulation_end,
            "snapshots": snapshots,
            "bands": bands,
            "satellite_id": satellite_id.value,
            "betas": (beta1, beta2),
            "eps": eps,
            "train_shrinkage": train_shrinkage,
            "validation_shrinkage": validation_shrinkage,
            "train_monitoring_ratio": train_monitoring_ratio,
            "ground_truth_dataset": ground_truth_dataset,
            "pretrained_model_identifier": pretrained_model_identifier,
            **band_extractor.asdict(),
        }

        for key, value in params.items():
            if len(str(value)) > PARAMS_MAX_STR_LENGTH:
                logger.error(f"Parameter {key} is too long: {str(value)[:PARAMS_MAX_STR_LENGTH]}...")
                params[key] = str(value)[:PARAMS_MAX_STR_LENGTH]

        mlflow.log_params(params)

    ######################### Set up neural network ############################
    if pretrained_model_identifier is not None:
        if rank == 0:
            logger.info(f"Loading pretrained model from {pretrained_model_identifier}")
        model, band_extractor_loaded, _ = load_model_and_concatenator(
            pretrained_model_identifier,
            device="cpu",  # Will be moved to correct device later
            satellite_id=satellite_id,
        )
    else:
        # FIXME: The weights should be part of the config and passed here as a CLI argument
        weights = "noisy-student" if model_encoder.startswith("timm-efficientnet-b") else "imagenet"
        if rank == 0:
            logger.info(f"Initializing new {model_type} model")
            logger.info(f"Using encoder weights {weights}")

        if model_type in [ModelType.UNET, ModelType.UNETPLUSPLUS]:
            UnetType = smp.UnetPlusPlus if model_type == ModelType.UNETPLUSPLUS else smp.Unet
            model = UnetType(
                encoder_name=model_encoder,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=2,
            )
        elif model_type in [ModelType.SPECTRALUNET, ModelType.SPECTRALUNETPLUSPLUS]:
            SpectralUnetType = SpectralUNetPlusPlus if model_type == ModelType.SPECTRALUNETPLUSPLUS else SpectralUNet
            model = SpectralUnetType(
                in_channels=in_channels,
                spectral_hidden_dims=[128, 64],
                encoder_name=model_encoder,
                encoder_weights=weights,
            )
        else:
            raise ValueError(f"Unhandled model type {model_type}")

    best_model = None

    # Set up model for distributed processing
    model, device = setup_device_and_distributed_model(model, rank, world_size)

    #################3 initialize our optimizer and loss function  #############

    # We use the AdamW optimiser, which is Adam with weight decay.
    # I haven't really explored the weight decay, so this is something we could revist.
    # Currently, it's set to the default decay parameter (0.01) and haven't tried other settings.
    # I've also seen some suggestions online to only apply decay to weights parameters
    # (as opposed to biases), which is easy to implement with a params dict.
    opt = AdamW(model.parameters(), lr=lr, betas=betas, eps=eps)
    lossFn = TwoPartLoss(binary_threshold=binary_threshold, MSE_multiplier=MSE_multiplier)

    ######################### Train neural network #############################
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of free parameters: {num_params / 1e6:.1f} million")

    best_metric = -np.inf
    best_early_stopping_metric = -np.inf
    early_stop_counter, early_stop_flag = 0, False
    start_time = time.time()  # measure how long training is going to take
    start_epoch_time = time.time()

    for i, epoch in enumerate(range(1, epochs + 1)):
        # set the modulation and batch size for this epoch
        if i < len(batchsize_schedule) + 1:
            modulation = modulation_schedule[i - 1]
            batch_size = int(batchsize_schedule[i - 1])
        else:
            modulation = modulation_schedule[-1]
            batch_size = int(batchsize_schedule[-1])
        if satellite_id != SatelliteID.EMIT:
            modulate_transform.modulate = modulation

        # Re-initialise the datasets with a random subset of `max_train_files` parquet files
        datasets, _, _, _ = data_preparation(
            traindata_uri_glob,
            band_extractor,
            satellite_id,
            validation_set=False,
            parquet_to_rowgroups_mapping_path=traindata_rowgroup_path,
            dataset_shrinkages={"train": train_shrinkage},
            train_monitoring_ratio=train_monitoring_ratio,
            filesystem=fs,
            max_train_files=max_train_files,
            random_state=random_state + i,
            rank=rank,
        )

        # Setup train monitoring dataset and dataloader
        train_monitoring_dataset = datasets["train_monitoring"]
        do_monitoring = len(train_monitoring_dataset) > 0
        if do_monitoring:
            train_monitoring_sampler: DistributedSampler = DistributedSampler(
                train_monitoring_dataset, shuffle=False, drop_last=False
            )
            train_monitoring_dataloader = DataLoader(
                train_monitoring_dataset,
                shuffle=False,
                sampler=train_monitoring_sampler,
                num_workers=num_workers,
                persistent_workers=False,  # will OOM if workers are persisted
                batch_size=max_batch_size_per_proc,
                timeout=(200 if num_workers > 0 else 0),
                pin_memory=True,
                collate_fn=collate_rowgroups,
                multiprocessing_context="forkserver",
                prefetch_factor=1,
            )

        # Setup training dataset and dataloader
        train_dataset = datasets["train"]
        if rank == 0:
            logger.info(
                f"[Ep{epoch}] --- modulate={modulation:.2f} --- batchsize={batch_size}"
                f" --- train samples={len(train_dataset)}"
            )
        # Apply the additional transformations to the training dataset
        # Note: this doesn't affect the transforms for the train_monitoring dataset
        train_dataset_dataset = train_dataset.dataset  # get inside the Subset wrapper
        assert isinstance(train_dataset_dataset, ArrowDataset)
        train_dataset_dataset.transform = train_transform
        ######################### Initialise DataLoader ############################
        # DistributedSampler ensures that training data is chunked across GPUs without overlapping samples
        # if the dataset isn't evenly divisible then `drop_last` will either drop the last batch from being
        # included in the evaluation (`drop_last=True`) or pad the last batch with duplicated samples
        # (`drop_last=False`). We have set `drop_last` to False for now to allow using very small toy datasets
        # of EMIT data (but not in the train_sampler as even our toy datasets should have sufficient samples to drop).
        train_sampler: DistributedSampler = DistributedSampler(train_dataset, seed=random_state + i)

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=False,
            sampler=train_sampler,
            num_workers=num_workers,
            persistent_workers=False,  # will OOM if workers are persisted
            batch_size=batch_size,
            timeout=(200 if num_workers > 0 else 0),  # allows enough time to download files
            pin_memory=True,
            collate_fn=collate_rowgroups,
            multiprocessing_context="forkserver",
            prefetch_factor=1,
        )

        train_metrics = single_epoch_train(model, opt, lossFn, train_dataloader, device)

        if do_monitoring:
            monitoring_losses = single_epoch_train_monitoring(model, lossFn, train_monitoring_dataloader, device)

        # Gather loss metrics from the different processes
        # We track the number of samples separately and divide once metrics are gathered because
        # each process can have a variable number of samples and we want to average by all samples
        train_metrics = gather_losses(train_metrics, world_size)
        if do_monitoring:
            monitoring_losses = gather_losses(monitoring_losses, world_size)

        if rank == 0:
            train_loss = train_metrics["loss"] / train_metrics["num_samples"]
            train_bce = train_metrics["bce"] / train_metrics["num_samples"]
            train_cond_mse = train_metrics["cond_mse"] / train_metrics["num_samples"]
            data_time = train_metrics["data_time"] / train_metrics["num_batches"]
            data_time_per_sample = train_metrics["data_time"] / train_metrics["num_samples"]
            batch_time = train_metrics["batch_time"] / train_metrics["num_batches"]
            log_metrics = {
                "train loss": train_loss,
                "train BCE": train_bce,
                "train Cond MSE": train_cond_mse,
                "train data_time": data_time,
                "train data_time per sample": data_time_per_sample,
                "train batch_time": batch_time,
            }
            logger.info(
                f"[Ep{epoch}] Train loss: {train_loss:.1f} (BCE: {train_bce:.1f}, Cond MSE: "
                f"{train_cond_mse:.3f}), BatchTime: {batch_time:.3f}s, DataTime: {data_time:.3f}s, "
                f"took {(time.time() - start_epoch_time) / 60:.1f} mins"
            )
            if do_monitoring:
                monitoring_loss = monitoring_losses["loss"] / monitoring_losses["num_samples"]
                monitoring_bce = monitoring_losses["bce"] / monitoring_losses["num_samples"]
                monitoring_cond_mse = monitoring_losses["cond_mse"] / monitoring_losses["num_samples"]
                monitoring_mse = monitoring_losses["mse"] / monitoring_losses["num_samples"]
                log_metrics.update(
                    {
                        "monitoring loss": monitoring_loss,
                        "monitoring Cond MSE": monitoring_cond_mse,
                        "monitoring BCE": monitoring_bce,
                        "monitoring MSE": monitoring_mse,
                    },
                )
                logger.info(
                    f"[Ep{epoch}] Monitoring loss: {monitoring_loss:.1f}, (BCE: {monitoring_bce:.1f}, "
                    f"Cond MSE: {monitoring_cond_mse:.3f})"
                )

        validate_this_epoch = epoch % validate_every_x == 0
        if validate_this_epoch:
            start_val_time = time.time()
            mRecall_global, mRecall_regions, threshold_found, metrics = validation_metrics(
                validation_uri_glob=validation_uri_glob,
                valdata_rowgroups=valdata_rowgroup_path,
                probability_threshold=probability_threshold,
                model=model,
                lossFn=lossFn,
                band_extractor=band_extractor,
                device=device,
                num_workers=num_workers * 2,
                batch_size=max_batch_size_per_proc,
                epoch=epoch,
                rank=rank,
                world_size=world_size,
                satellite_id=satellite_id,
                azure_cluster=azure_cluster,
                cpu_group=cpu_group,
                region_to_idx=region_to_idx,
                fp_px_per_crop=fp_px_per_crop,
                validation_shrinkage=validation_shrinkage,
                filesystem=fs,
            )
            dist.barrier()  # to have all GPUs wait for the slowest one
            if rank == 0:
                metrics = metrics.as_dict()
                logger.info(
                    f"[Ep{epoch}] Val - F1: {metrics['f1_score']:.4f}, Rec: {metrics['recall']:.4f}, Prec: "
                    f"{metrics['precision']:.4f} Loss: {metrics['average_combined_loss']:.1f}, "
                    f"(BCE: {metrics['average_binary_loss']:.1f}, Cond MSE: "
                    f"{metrics['average_conditional_loss']:.3f}), took "
                    f"{(time.time() - start_val_time) / 60:.2f} mins"
                )
                for region, idx in region_to_idx.items():
                    if region != "Other":
                        continue
                    log_metrics[f"val mRecall {region}"] = mRecall_regions[idx]
                log_metrics.update(
                    {
                        "val mRecall global": mRecall_global,
                        "val threshold": threshold_found,
                        "val f1_score": metrics["f1_score"],
                        "val precision": metrics["precision"],
                        "val recall": metrics["recall"],
                        "val signal2noise_ratio": metrics["signal2noise_ratio"],
                        "val Loss": metrics["average_combined_loss"],
                        "val BCE": metrics["average_binary_loss"],
                        "val Cond MSE": metrics["average_conditional_loss"],
                        "val MSE": metrics["mean_squared_error"],
                    }
                )
                mlflow.log_metrics(log_metrics, step=epoch)

                #### Early Stopping and model saving ####
                early_stopping_metric = mRecall_global
                if early_stopping_metric > best_metric:
                    best_metric = early_stopping_metric
                    best_model = copy.deepcopy(model)

                    if epochs > 0:
                        start_model_time = time.time()
                        artifact_path = f"{registered_model_name}_epoch_{epoch:03}_metric_{best_metric:.3f}"
                        # Strip out the DataParallel wrapper before saving, so we don't have to deal with it when
                        # loading the model. When reading the model, we will choose between Parallelism and CPU/GPU
                        model_artifact = (
                            model.module
                            if isinstance(model, nn.DataParallel | nn.parallel.DistributedDataParallel)
                            else model
                        )
                        save_and_register(model_artifact, band_extractor, params, registered_model_name, artifact_path)
                        logger.info(f"Saving model took {(time.time() - start_model_time) / 60:.2f} mins")

                # if the model has not improved, check for early stopping
                # it must improve by a significant bit, we dont care about 0.000001 improvements
                model_improved_for_stopping = early_stopping_metric > best_early_stopping_metric * 1.0005

                if model_improved_for_stopping:
                    early_stop_counter = 0  # reset counter
                    best_early_stopping_metric = early_stopping_metric
                elif epoch > epochs_warmup:
                    early_stop_counter += 1
                    if early_stop_counter > early_patience:
                        logging.info("Early Stopping")
                        early_stop_flag = True

        elif rank == 0:
            mlflow.log_metrics(log_metrics, step=epoch)

        if world_size > 1:
            # Broadcast the early_stop signal from rank 0 to all other processes
            early_stop_flag = torch.tensor(early_stop_flag, dtype=torch.int).cuda()
            torch.distributed.broadcast(early_stop_flag, src=0)  # src=0 means rank 0 is the source

            # Convert the signal to Python boolean
            early_stop_flag = early_stop_flag.item() == 1  # type: ignore

        if early_stop_flag:
            if rank == 0:
                print(f"Early stopping triggered at epoch {epoch}")
            break  # Exit the loop for all processes

        start_epoch_time = time.time()

    ########################### SBR Metrics & Visualizations ####################################
    if rank == 0:
        if best_model is None:  # for test runs that did not validate
            best_model = copy.deepcopy(model)
        # If we are only predicting for SBRs, use the loaded model
        if pretrained_model_identifier is not None and epochs == 0:
            logger.info("SBR mode: Using loaded model for predictions")
            band_extractor = band_extractor_loaded
            best_model = model

        # Unwrap model from DistributedDataParallel before inference
        if isinstance(best_model, nn.DataParallel | nn.parallel.DistributedDataParallel):
            best_model = best_model.module
        best_model.eval()

        logger.info(f"Total time taken to train the model: {(time.time() - start_time) / 3600:.2f} hours")

        # If running on Azure, create config.json
        if azure_cluster:
            create_ml_client_config()

        ml_client, _, _, _, s3_client = initialize_clients(azure_cluster)

        if satellite_id == SatelliteID.EMIT:
            # FIXME: implement metrics calculation for EMIT
            # NOTE: this would require accurate masks, which we do not currently have
            logger.warning("Skipping ground truth metrics for satellite EMIT.")

            earthaccess_login(ml_client)
        elif satellite_id == SatelliteID.LANDSAT:
            # FIXME: implement metrics calculation for LANDSAT
            # NOTE: this would require accurate masks, which we do not currently have
            logger.warning("Skipping ground truth metrics for satellite LANDSAT. Will be revisted post SBRs.")
        else:
            # FIXME: implement metrics calculation for S2 using new chip selection
            logger.warning("Skipping ground truth metrics for satellite S2.")
            # ground_truth_metrics_df = compute_metrics_for_ground_truth_data(
            #     ml_client=ml_client,
            #     s3_client=s3_client,
            #     ground_truth_dataset=ground_truth_dataset,
            #     model=best_model,
            #     band_extractor=band_extractor,
            #     model_training_params=params,
            #     satellite_id=satellite_id,
            #     crop_size=crop_size,
            #     lossFn=lossFn,
            #     device=device,
            # )
            # ground_truth_metrics_dict = ground_truth_metrics_df.to_dict(orient="records")
            # mlflow.log_dict(dictionary=ground_truth_metrics_dict, artifact_file="ground_truth_metrics.json")

        ground_truth_plumes = pd.read_csv(ground_truth_dataset)
        if satellite_id == SatelliteID.EMIT:
            cache_container_name = get_default_blob_storage(ml_client).container_name
            plot_ground_truth_plots_EMIT(
                ground_truth_dataset,
                best_model,
                device,
                crop_size,
                binary_threshold=binary_threshold,
                band_extractor=band_extractor,
                lossFn=lossFn,
                ml_client=ml_client,
                cache_container_name=cache_container_name,
                azure_cluster=azure_cluster,
            )
        elif satellite_id == SatelliteID.LANDSAT:
            plot_ground_truth_plots_landsat(
                ground_truth_plumes,
                best_model,
                device,
                crop_size,
                binary_threshold=binary_threshold,
                band_extractor=band_extractor,
                lossFn=lossFn,
                ml_client=ml_client,
                s3_client=s3_client,
                azure_cluster=azure_cluster,
            )
        elif satellite_id == SatelliteID.S2:
            plot_ground_truth_plots_S2(
                ground_truth_plumes,
                best_model,
                device,
                crop_size,
                binary_threshold=binary_threshold,
                band_extractor=band_extractor,
                lossFn=lossFn,
                ml_client=ml_client,
                s3_client=s3_client,
                azure_cluster=azure_cluster,
            )

    # cleanly shutdown distributed processes
    torch.distributed.destroy_process_group()


####################################################
################# CORE FUNCTIONS ###################
####################################################
def gather_metrics(metrics: Metrics, world_size: int) -> Metrics:
    """Gathers metrics from all processes."""
    gather_list = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gather_list, metrics)
    return functools.reduce(operator.add, gather_list)  # type:ignore[arg-type]


def gather_losses(losses: Counter, world_size: int) -> Counter:
    """Gathers loss Counters from all processes."""
    gather_list = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gather_list, losses)
    return functools.reduce(operator.add, gather_list)  # type:ignore[arg-type]


def gather_lists_cpu_only(lists: list, world_size: int, group: dist.ProcessGroup) -> list:
    """Gathers loss lists from all processes and concatenates them."""
    gather_list = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gather_list, lists, group=group)

    # Flatten the gathered lists into a single list
    return functools.reduce(operator.iadd, gather_list, [])  # Concatenate lists


def calculate_metrics(
    model: nn.Module,
    validation_dataloader: DataLoader,
    device: int | torch.device,
    lossFn: TwoPartLoss,
    probability_threshold: float,
) -> Metrics:
    """Calculate validation metrics after training."""
    metrics = Metrics(lossFn.binary_threshold, probability_threshold)

    model.eval()
    with torch.no_grad():
        for _, (x, y) in enumerate(tqdm(validation_dataloader, desc="Metrics batch -", disable=None)):
            if len(x.shape) != NUM_DIMENSIONS:
                raise ValueError(f"x is expected to be {NUM_DIMENSIONS}-dimensional, instead got {len(x.shape)}")

            (x, y) = (x.to(device), y.to(device))  # noqa: PLW2901 (redefined-loop-name)
            pred = model(x)

            # Returns average loss for the batch
            combined_loss, bce, cond_mse = lossFn(pred, y)

            pred_parts = lossFn.get_prediction_parts_as_dict(pred)
            metrics.update(
                pred_parts["marginal_pred"],
                pred_parts["binary_probability"],
                y,
                combined_loss.item(),
                bce,
                cond_mse,
            )
    return metrics


def single_epoch_train_monitoring(
    model: nn.Module, lossFn: TwoPartLoss, train_monitoring_dataloader: DataLoader, device: int | torch.device
) -> Counter:
    """Calculate train_monitoring metrics for current epoch."""
    monitoring_loss = 0.0
    monitoring_bce = 0.0
    monitoring_cond_mse = 0.0
    monitoring_mse = 0.0
    num_samples = 0

    model.eval()
    with torch.no_grad():
        for _, (x, y) in enumerate(tqdm(train_monitoring_dataloader, desc="Monitoring batch -", disable=None)):
            if len(x.shape) != NUM_DIMENSIONS:
                raise ValueError(f"x is expected to be {NUM_DIMENSIONS}-dimensional, instead got {len(x.shape)}")

            (x, y) = (x.to(device), y.to(device))  # noqa: PLW2901 (redefined-loop-name)
            pred = model(x)

            # Returns average loss for the batch
            loss, bce, cond_mse = lossFn(pred, y)
            mse = lossFn.calculate_mse_on_marginal(pred, y)  # average marginal mse for the batch

            monitoring_loss += loss.item()
            monitoring_bce += bce
            monitoring_cond_mse += cond_mse
            monitoring_mse += mse.item()
            num_samples += x.shape[0]  # Accumulate the total number of samples

    return Counter(
        {
            "num_samples": num_samples,
            "loss": monitoring_loss,
            "bce": monitoring_bce,
            "cond_mse": monitoring_cond_mse,
            "mse": monitoring_mse,
        }
    )


def single_epoch_train(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    lossFn: TwoPartLoss,
    train_dataloader: DataLoader,
    device: int | torch.device,
) -> Counter:
    """Trains the model for a single epoch."""
    model.train()
    train_loss = 0.0
    train_bce, train_cond_mse = 0.0, 0.0
    num_samples, num_batches = 0, 0
    data_time, batch_time = 0.0, 0.0
    end = time.time()

    for _, (x, y) in enumerate(tqdm(train_dataloader, desc="Training batch -", disable=None)):
        data_time += time.time() - end
        if len(x.shape) != NUM_DIMENSIONS:
            raise ValueError(f"x is expected to be {NUM_DIMENSIONS}-dimensional, instead got {len(x.shape)}")

        (x, y) = (x.to(device), y.to(device))  # noqa: PLW2901 (redefined-loop-name)
        pred = model(x)
        loss, bce, cond_mse = lossFn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        num_samples += x.shape[0]  # Accumulate the total number of samples
        num_batches += 1
        train_loss += loss.item()
        train_bce += bce
        train_cond_mse += cond_mse

        batch_time += time.time() - end
        end = time.time()

    return Counter(
        {
            "num_samples": num_samples,
            "num_batches": num_batches,
            "loss": train_loss,
            "bce": train_bce,
            "cond_mse": train_cond_mse,
            "data_time": data_time,
            "batch_time": batch_time,
        }
    )


def data_preparation(  # noqa PLR0913
    parquet_glob_uri: str,
    band_extractor: BaseBandExtractor,
    satellite_id: SatelliteID,
    validation_set: bool = False,
    parquet_to_rowgroups_mapping_path: str = "parquet_rowgroups_data_aviris_X.json",
    dataset_shrinkages: dict[str, float] = {},  # noqa: B006
    train_monitoring_ratio: float = 0.1,
    filesystem: AzureMachineLearningFileSystem | None = None,
    max_train_files: int = 10000,
    random_state: int = 42,
    rank: int = 0,
) -> tuple[dict[str, Subset], int, int, list[str] | list[int]]:
    """
    Read in parquet(s) as ArrowDataset.

    Split into training and train monitoring set if validation_set is False.

    Returns
    -------
        A dictionary of datasets.
        The patch size in the datasets.
        The number of channels passed into the model (i.e. returned from the band exctractor).
        The list of available bands.
    """
    if parquet_glob_uri.endswith(".parquet"):  # In our small runs, we use a single .parquet file
        files = parquet_glob_uri
    else:
        if filesystem is None:
            # e.g. parquet_glob_uri='/mnt/azureml/cr/j/f23[...]ca9/cap/data-capability/wd/INPUT_traindata_uri_glob'
            # we want to get the downloaded *parquets in that folder
            files = sorted(glob.glob(f"{parquet_glob_uri}/**/*.parquet", recursive=True))  # type: ignore
        else:
            files = filesystem.glob(f"{parquet_glob_uri}/**/*.parquet")

        if not validation_set:  # sample files for training to reduce CPU memory usage
            random.seed(random_state)
            random.shuffle(files)  # type: ignore
            files = files[:max_train_files]
        if rank == 0:
            logger.info(f"{len(files)} parquet files={list(files)[:1]}")

    parquet_dataset = pa.dataset.dataset(
        files,
        format="parquet",
        exclude_invalid_files=False,
        filesystem=filesystem,
    )

    # Get parquet dimensions
    crop_size, bands = compute_parquet_dimensions(parquet_dataset)
    # The number of input channels for the neural network
    # is the number of output channels of the band extractor.
    in_channels = band_extractor.output_channels

    if satellite_id == SatelliteID.EMIT:
        columns_config = SATELLITE_COLUMN_CONFIGS[satellite_id]
    # For S2: Construct the satellite config based on how many snapshots we are loading
    elif satellite_id == SatelliteID.S2:
        columns_config = {
            "crop_main": {"shape": (NUM_S2_BANDS, crop_size, crop_size), "dtype": torch.int16},
            "orig_swir16": {"shape": (1, crop_size, crop_size), "dtype": torch.int16},
            "orig_swir22": {"shape": (1, crop_size, crop_size), "dtype": torch.int16},
            TARGET_COLUMN: {"shape": (1, crop_size, crop_size), "dtype": torch.float32},
        }
        for snapshot in band_extractor.snapshots:
            columns_config[snapshot] = {"shape": (NUM_S2_BANDS, crop_size, crop_size), "dtype": torch.int16}
    elif satellite_id == SatelliteID.LANDSAT:
        columns_config = {
            "crop_main": {"shape": (NUM_LANDSAT_BANDS, crop_size, crop_size), "dtype": torch.int16},
            "orig_swir16": {"shape": (1, crop_size, crop_size), "dtype": torch.int16},
            "orig_swir22": {"shape": (1, crop_size, crop_size), "dtype": torch.int16},
            TARGET_COLUMN: {"shape": (1, crop_size, crop_size), "dtype": torch.float32},
        }
        for snapshot in band_extractor.snapshots:
            columns_config[snapshot] = {"shape": (NUM_LANDSAT_BANDS, crop_size, crop_size), "dtype": torch.int16}
    else:
        raise ValueError(f"Satellite type {satellite_id.value} not handled.")

    # Create the base dataset with ConcatenateSnapshots as the base transformation
    if "validation" in dataset_shrinkages:
        # we don't need the original swir bands for validation
        columns_config.pop("orig_swir16", None)
        columns_config.pop("orig_swir22", None)

        base_dataset = ValidationDataset(
            parquet_dataset,
            columns_config=columns_config,
            target_column=TARGET_COLUMN,
            transform=band_extractor,
            filesystem=filesystem,
            parquet_to_rowgroups_mapping_path=parquet_to_rowgroups_mapping_path,
        )
    else:
        base_dataset = ArrowDataset(
            parquet_dataset,
            columns_config=columns_config,
            target_column=TARGET_COLUMN,
            transform=band_extractor,
            filesystem=filesystem,
            parquet_to_rowgroups_mapping_path=parquet_to_rowgroups_mapping_path,
        )
    if not validation_set:
        # Split the dataset into training and train monitoring (without modulation)
        train_monitoring_ratio = train_monitoring_ratio * dataset_shrinkages.get("train", 1.0)
        train_ratio = (1 - train_monitoring_ratio) * dataset_shrinkages.get("train", 1.0)
        train_monitoring_dataset, train_dataset = split_dataset(base_dataset, train_monitoring_ratio, train_ratio)

        datasets = {
            "train_monitoring": train_monitoring_dataset,
            "train": train_dataset,
        }
    else:
        if "validation" in dataset_shrinkages:
            r = dataset_shrinkages["validation"]
            validation_dataset, _discard = torch.utils.data.random_split(base_dataset, [r, 1 - r])
        else:
            validation_dataset = Subset(base_dataset, list(range(len(base_dataset))))
        datasets = {"validation": validation_dataset}
    return datasets, crop_size, in_channels, bands


def compute_parquet_dimensions(dataset: pa.dataset.FileSystemDataset) -> tuple[int, list[str] | list[int]]:
    """Determine the crop size and available bands from the parquet files."""
    # Get the first rows to inspect the shape
    for k in range(100):
        try:
            row = dataset.head(k)
            bands = row.column("bands")[0].as_py()
            crop_size = row.column("size")[0].as_py()
            return crop_size, bands
        except Exception:
            pass
    return crop_size, bands


def model_performance_plots(  # noqa PLR0915
    bands: list[str] | list[int],
    model: nn.Module,
    lossFn: TwoPartLoss,
    device: int | torch.device,
    validation_dataset: Subset,
    output_folder_name: str,
    satellite_id: SatelliteID,
) -> None:
    """Generate performance plots by evaluating the trained model on a random subset of the validation dataset."""
    # generate predictions for random row groups until we have visualized 7 examples
    show_plots = 0
    HOW_MANY_VAL_PLOTS = 7

    np.random.seed(2222)
    val_rand_indices = np.random.randint(len(validation_dataset), size=HOW_MANY_VAL_PLOTS)
    val_rand_indices = list(set(list(val_rand_indices)))  # type:ignore # remove duplicates
    HOW_MANY_VAL_PLOTS = min(len(val_rand_indices), HOW_MANY_VAL_PLOTS)
    while True:
        idx = val_rand_indices[show_plots]
        untransformed_inputs, untransformed_target, _, _ = validation_dataset.dataset.get_untransformed_data(idx)
        x, y = validation_dataset.dataset.transform((untransformed_inputs, untransformed_target))
        (x, y) = (x.to(device), y.to(device))

        with torch.no_grad():
            pred = model(x)
            marginal_pred, binary_probability, conditional_pred, _ = lossFn.get_prediction_parts(pred)
            marginal_pred = marginal_pred.squeeze()
            binary_probability = binary_probability.squeeze()
            conditional_pred = conditional_pred.squeeze()

            # Iterate over samples in row group
            for i in range(y.shape[0]):
                binary_loss, conditional_loss = lossFn.get_loss_parts(pred[i : i + 1, ...], y[i : i + 1, ...])
                crop_main = tensor_to_dataarray(untransformed_inputs["crop_main"][i], bands)

                marginal_pred_ = marginal_pred[i] if y.shape[0] > 1 else marginal_pred
                binary_probability_ = binary_probability[i].cpu() if y.shape[0] > 1 else binary_probability.cpu()
                conditional_pred_ = conditional_pred[i] if y.shape[0] > 1 else conditional_pred

                fig = plt.figure(figsize=(10, 15))

                # First row
                plt.subplot(3, 2, 1)
                plt.title("Ground truth", fontsize=15)
                _, vmax = plot_frac(y[i, 0])
                grid16()
                plt.colorbar()

                plt.subplot(3, 2, 2)
                plt.title("Prediction", fontsize=15)
                _, _ = plot_frac(marginal_pred_, vmax=vmax)
                grid16()
                plt.colorbar()

                # Second row
                plt.subplot(3, 2, 3)
                plt.title("RGB Image of Main", fontsize=15)
                rgb_image = get_rgb_from_xarray(crop_main, satellite_id)
                # divide by max to get better contrast
                plt.imshow(rgb_image / rgb_image.max(), vmin=0.0, vmax=1.0, interpolation="nearest")
                grid16()
                plt.colorbar()

                plt.subplot(3, 2, 4)
                plt.title("SWIR Ratio for Main", fontsize=15)
                swir_ratio = get_swir_ratio_from_xarray(crop_main, satellite_id)
                plt.imshow(swir_ratio, interpolation="nearest", cmap="pink")
                grid16()
                plt.colorbar()

                # Third row
                plt.subplot(3, 2, 5)
                plt.title(f"Binary pred (loss={binary_loss:.1f})", fontsize=15)
                plt.imshow(binary_probability_, vmin=0.0, vmax=1.0, cmap="pink_r", interpolation="nearest")
                grid16()
                plt.colorbar()

                plt.subplot(3, 2, 6)
                plt.title(f"Cond pred (loss={conditional_loss:.1f})", fontsize=15)
                plot_frac(conditional_pred_, vmax=vmax, colorbar_swapped=(satellite_id == SatelliteID.EMIT))
                grid16()
                plt.colorbar()
                plt.tight_layout()

                mlflow.log_figure(fig, f"{output_folder_name}/prediction_rowgroup_{idx}_idx_{i}.png")
                plt.close()

                show_plots += 1
        if show_plots >= HOW_MANY_VAL_PLOTS:
            return


def plot_ground_truth_plots_EMIT(  # noqa
    ground_truth_plume_csv_path: str,
    model: nn.Module,
    device: int | torch.device,
    crop_size: int,
    binary_threshold: float,
    band_extractor: BaseBandExtractor,
    lossFn: TwoPartLoss,
    ml_client: MLClient,
    cache_container_name: str,
    azure_cluster: bool = False,
) -> None:
    """Plot the band ratio, predicted frac, RGB, and conditional prediction for a list of known methane sites.

    Parameters
    ----------
        ground_truth_plume_csv_path: path to CSV with a list of known methane sites
        bands: bands to download
        model: NN model used to predict the frac for the plots

    """
    satellite_id = SatelliteID.EMIT
    ground_truth_plumes = pd.read_csv(ground_truth_plume_csv_path)

    for _, row in ground_truth_plumes.iterrows():
        logger.info(f"PLOT FOR {row['site']} on {row['date']} ({row['lat']}, {row['lon']}) - {row['source']}")
        try:
            cropped_data, data_item = prepare_model_input(
                row["lat"],
                row["lon"],
                datetime.datetime.fromisoformat(row["date"]),
                crop_size,
                ml_client=ml_client,
                cache_container_name=cache_container_name,
            )
        except InsufficientImageryException as e:
            print(f"{e}. Skipping")
            continue

        # Get relevant metadata
        obs_crop = cropped_data[0]["crop_arrays"]["mask"]
        sensing_time = cropped_data[0]["crop_arrays"]["mask"].attrs["time_coverage_start"]

        pred = predict(model, device, band_extractor, data_item, lossFn)
        crop_main = tensor_to_dataarray(pred["x_dict"]["crop_main"][0], data_item.bands)
        marginal_pred = pred["marginal_pred"].squeeze()
        binary_probability = pred["binary_probability"].squeeze()
        conditional_pred = pred["conditional_pred"].squeeze()

        # Convert tensors to CPU if they are on GPU. This is necessary for plotting, otherwise we will get an error
        if binary_probability.is_cuda:
            binary_probability = binary_probability.cpu()
        if marginal_pred.is_cuda:
            marginal_pred = marginal_pred.cpu()
        if conditional_pred.is_cuda:
            conditional_pred = conditional_pred.cpu()

        # Get gamma and correct conditional prediction (conditional pred is retrieval*gamma)
        gamma = get_gamma(obs_crop.sel(bands="solar_zenith"), obs_crop.sel(bands="sensor_zenith"))
        conditional_pred = conditional_pred / gamma
        marginal_pred = marginal_pred / gamma

        emission_rate_dict = quantify_plume_for_lat_lon(
            lat=row["lat"],
            lon=row["lon"],
            retrieval_array=conditional_pred.values,
            retrieval_binary_probability=binary_probability.numpy(),
            sensing_time=sensing_time,
            crop_size=crop_size,
        )

        if emission_rate_dict is None:
            emission_rate_str = "No wind data available for quantification"
        else:
            emission_rate_str = (
                f"{emission_rate_dict['emission_rate']:.2f} kg/h "
                f"({emission_rate_dict['emission_rate_low']:.2f} kg/hr - "
                f"{emission_rate_dict['emission_rate_high']:.2f} kg/hr)"
            )

        fig = plt.figure(figsize=(10, 15))

        plt.subplot(3, 2, 1)
        plt.title("SWIR ratio")
        swir_ratio = get_swir_ratio_from_xarray(crop_main, satellite_id)
        plt.imshow(swir_ratio, interpolation="nearest", cmap="pink")
        grid16()
        plt.colorbar()

        plt.subplot(3, 2, 2)
        plt.title("RGB")
        rgb_image = get_rgb_from_xarray(crop_main, satellite_id)
        plt.imshow(rgb_image / rgb_image.max(), interpolation="nearest")
        grid16()

        plt.subplot(3, 2, 3)
        plt.title(f"Cond. Prediction, Sum: {conditional_pred.sum():.1f}")
        plot_frac(conditional_pred)
        grid16()
        plt.colorbar(label="mol/m²")

        plt.subplot(3, 2, 4)
        plt.title(f"Marg. Prediction, Sum: {marginal_pred.sum():.1f}")
        plot_frac(marginal_pred)
        grid16()
        plt.colorbar(label="mol/m²")

        plt.subplot(3, 2, 5)
        plt.title(f"Binary prediction, Sum: {binary_probability.sum():.0f}")
        plt.imshow(
            binary_probability * 10,
            vmin=0.0,
            vmax=10.0,
            cmap="pink_r",
            interpolation="nearest",
        )
        grid16()
        plt.colorbar()

        plt.subplot(3, 2, 6)
        plt.title("RGB with source and prediction")
        plt.imshow(rgb_image / rgb_image.max(), interpolation="nearest")
        predictions = np.abs(marginal_pred) > binary_threshold
        ymasked = np.ma.masked_where(predictions == 0, predictions)
        plt.imshow(ymasked, vmin=0.0, vmax=1.0, interpolation="nearest")
        grid16()
        center_x = rgb_image.shape[1] // 2
        center_y = rgb_image.shape[0] // 2

        # Add a dot in the center of the plot
        plt.scatter(center_x, center_y, color="green", marker="x")
        fig.suptitle(
            f"{row['site']} ({row['lat']}, {row['lon']})\n"
            f"Q: {row['quantification_kg_h']:.2f} kg/h ({row['source']})\n"
            f"{row['date']}\n"
            f"{emission_rate_str}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        if azure_cluster:
            basename_prefix = row["site"].replace("/", "_") if isinstance(row["site"], str) else "known_methane_sites"
            basename = f"{basename_prefix:15}_{row['lat']:.4f}_{row['lon']:.4f}_{row['date']}"
            mlflow.log_figure(fig, f"{basename}.png")
            plt.close()
        else:
            plt.show()


def validation_metrics(  # noqa: PLR0913, PLR0915 (too-many-arguments)
    validation_uri_glob: str,
    valdata_rowgroups: str,
    probability_threshold: float,
    model: nn.Module,
    lossFn: TwoPartLoss,
    band_extractor: BaseBandExtractor,
    device: int | torch.device,
    num_workers: int,
    batch_size: int,
    epoch: int,
    rank: int,
    world_size: int,
    satellite_id: SatelliteID,
    azure_cluster: bool,
    cpu_group: dist.ProcessGroup,
    region_to_idx: dict[str, int],
    fp_px_per_crop: int,
    validation_shrinkage: float = 1.0,
    filesystem: AzureMachineLearningFileSystem | None = None,
) -> tuple[float, list, float, Metrics]:
    """Calculate Recall over validation regions and emission rates at a threshold of acceptable False Positives.

    1. Predict for all validation samples tracking prediction values, regions and emission rates
    2. Find the threshold that gives 10 average false positive pixels over all samples over all regions
    3. Calculate recall for each region and emission rate.

    Returns
    -------
    mRecall_global: Mean of the regional average recall values over all emission rates.
    mRecall_regions: List of the regional average recall values.
    threshold_found: Threshold used.
    metrics: Global metrics calculated over the full validation set.

    """
    datasets, _, _, available_bands = data_preparation(
        validation_uri_glob,
        band_extractor,
        satellite_id,
        validation_set=True,
        parquet_to_rowgroups_mapping_path=valdata_rowgroups,
        dataset_shrinkages={"validation": validation_shrinkage},
        filesystem=filesystem,
        rank=rank,
    )
    validation_dataset = datasets["validation"]
    if rank == 0:
        logger.info(f"{len(validation_dataset)=}")

    model.eval()

    if rank == 0:
        model_performance_plots(
            available_bands,
            model,
            lossFn,
            device,
            validation_dataset,
            satellite_id=satellite_id,
            output_folder_name=f"validation/Epoch{epoch:03}",
        )

    validation_sampler: DistributedSampler = DistributedSampler(validation_dataset, shuffle=False, drop_last=False)
    validation_dataloader = DataLoader(
        validation_dataset,
        shuffle=False,
        sampler=validation_sampler,
        num_workers=num_workers,
        persistent_workers=False,  # will OOM if workers are persisted
        batch_size=batch_size,
        timeout=(200 if num_workers > 0 else 0),
        pin_memory=True,
        collate_fn=collate_rowgroups_val,
        multiprocessing_context="forkserver",
        prefetch_factor=1,
    )

    # We are aggregating predictions, ground truth masks, regions and emissions over all GPUs
    metrics = Metrics(lossFn.binary_threshold, probability_threshold)
    probs, ys = [], []
    region_overlaps, emissions = [], []
    num_batches = 0
    data_time, batch_time = 0.0, 0.0
    total_len = len(validation_dataloader)
    end = time.time()
    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(validation_dataloader, desc="Metrics batch -", disable=None), start=1):
            data_time += time.time() - end
            region_overlaps.extend([region_to_idx[k] for k in data["region_overlap"]])
            emissions.extend(data["emission"])
            x = data["X"]
            y = data["y"]
            ys.append((y[:, 0].abs() > lossFn.binary_threshold).numpy())
            if len(x.shape) != NUM_DIMENSIONS:
                raise ValueError(f"x is expected to be {NUM_DIMENSIONS}-dimensional, instead got {len(x.shape)}")

            (x, y) = (x.to(device), y.to(device))  # (redefined-loop-name)
            pred = model(x)

            # Returns average loss for the batch
            combined_loss, bce, cond_mse = lossFn(pred, y)

            pred_parts = lossFn.get_prediction_parts_as_dict(pred)

            metrics.update(
                pred_parts["marginal_pred"],
                pred_parts["binary_probability"],
                y,
                combined_loss.item(),
                bce,
                cond_mse,
            )

            probs.append((pred_parts["binary_probability"][:, 0].cpu().numpy() * 255).round(0).astype(np.uint8))
            num_batches += 1
            batch_time += time.time() - end
            end = time.time()
            if batch_count == total_len:
                data_time_avg = data_time / num_batches
                batch_time_avg = batch_time / num_batches
                logger.info(f"GPU{rank}: Val: Datatime: {data_time_avg:.3f}s, Batchtime: {batch_time_avg:.3f}s")

    # Syncing the lists of predictions, ground truth masks, region_overlaps and emissions across all GPUs
    probs = gather_lists_cpu_only(probs, world_size, cpu_group)
    ys = gather_lists_cpu_only(ys, world_size, cpu_group)
    region_overlaps = gather_lists_cpu_only(region_overlaps, world_size, cpu_group)
    emissions = gather_lists_cpu_only(emissions, world_size, cpu_group)

    metrics = gather_metrics(metrics, world_size)

    if rank == 0:
        probs = np.concatenate(probs)
        ys = np.concatenate(ys)
        logger.info(f"Final shapes: {probs.shape=}, {ys.shape=}, {len(emissions)=}, {len(region_overlaps)=}")  # type: ignore

        df_target = pd.DataFrame({"region": region_overlaps, "emission": emissions})

        region_to_idx_filtered = {k: v for k, v in region_to_idx.items() if k != "Other"}

        thresholds = np.array([0.025] + [k * 0.05 for k in list(range(1, 20))])
        threshold_found = calculate_false_positives(
            probs, df_target, thresholds, fp_px_per_crop, region_to_idx_filtered, epoch, azure_cluster
        )
        mRecall_global, mRecall_regions = calculate_recall(
            probs, ys, df_target, threshold_found, epoch, region_to_idx_filtered, azure_cluster
        )

        return mRecall_global, mRecall_regions, threshold_found, metrics
    return 0.0, [], 0.0, metrics


def calculate_false_positives(
    probs: np.ndarray | list,
    df_target: pd.DataFrame,
    thresholds: np.ndarray,
    fp_px_per_crop: int,
    region_to_idx: dict[str, int],
    epoch: int,
    azure_cluster: bool,
) -> float:
    """Calculate false positive px rates and determine threshold to get 10 avg FP px."""
    f, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Get no-plume indices for each region
    region_indices = {
        region: df_target[(df_target["region"] == region_idx) & (df_target["emission"] == 0)].index.tolist()
        for region, region_idx in region_to_idx.items()
    }
    region_lens = {
        region: df_target[df_target["region"] == region_idx].shape[0] for region, region_idx in region_to_idx.items()
    }

    for region in region_to_idx:
        logger.info(f"{region:10}: {len(region_indices[region]):5}/{region_lens[region]:5} chips without plumes")

    # How many FPs per 128x128 crop on average for threshold t and region X?
    fp_means_global = {}
    for region in region_to_idx:
        fp_means = []
        no_plume_idx = np.array(region_indices[region])
        if len(no_plume_idx) == 0:
            continue
        for t in thresholds:
            prob_ = probs[no_plume_idx] > t * 255  # (BS, H, W)
            # sum up all FP counts per sample, then calc average FP count over all samples
            fp_mean = prob_.sum(axis=(1, 2)).mean()
            fp_means.append(fp_mean)

        ax.plot(
            thresholds * 10,
            fp_means,
            marker="o",
            linestyle="-",
            label=f"avg(FP_px): {region} (n_crops={len(no_plume_idx)})",
        )
        fp_means_global[region] = fp_means

    # Configure plot
    ax.set_xlabel("Likelihood Threshold")
    ax.set_ylabel("# of FP pxs per (128x128) Chips")
    ax.set_title("False Positive Px Rate Curves")
    ax.grid(True)
    ax.legend()

    # Find optimal threshold
    minimum_threshold = 0.025
    threshold_found: float = minimum_threshold  # minimum threshold
    for idx, t in enumerate(thresholds):
        mean_fp_px = np.mean([fp_means_global[r][idx] for r in region_to_idx])
        logger.info(
            f"t={t:.3f}: avg(FP_px): {mean_fp_px:6.1f} "
            f"({', '.join(f'{region}: {fp_means_global[region][idx]:6.1f}' for region in region_to_idx)})"
        )

        if threshold_found == minimum_threshold and mean_fp_px < fp_px_per_crop:
            mean_fp_px_before = np.mean([fp_means_global[r][idx - 1] for r in region_to_idx])
            interpolated_t = thresholds[idx - 1] + (fp_px_per_crop - mean_fp_px_before) / (
                mean_fp_px - mean_fp_px_before
            ) * (t - thresholds[idx - 1])
            threshold_found = interpolated_t if interpolated_t >= 0 else 0.025
            logger.info(
                f"***** PX Threshold interpolated to yield avg(FP_px)={fp_px_per_crop}: {threshold_found:.3f} ******"
            )
            break

    if threshold_found == minimum_threshold:
        threshold_found = thresholds[-1]  # type: ignore
        logger.info(
            f"***** No Threshold found that gets mean(fp_px) < {fp_px_per_crop}: "
            f"Use Max Threshold = {threshold_found:.3f} ******"
        )

    ax.axvline(threshold_found * 10, color="r", linestyle="--", linewidth=2)
    ax.set_ylim([0, 150])
    plt.tight_layout()

    if azure_cluster:
        mlflow.log_figure(f, f"validation/Epoch{epoch:03}_FPR_t={threshold_found:.3f}.png")

    plt.close()
    return threshold_found


def calculate_recall(
    probs: np.ndarray | list,
    ys: np.ndarray | list,
    df_target: pd.DataFrame,
    threshold: float,
    epoch: int,
    region_to_idx: dict[str, int],
    azure_cluster: bool,
) -> tuple[float, list[float]]:
    """Calculate recall metrics across regions and emission rates for `threshold`."""
    # Define bins and labels
    bins = [0, 100, 200, 300, 500, 750, 1250, float("inf")]  # float("inf") covers >1250
    labels = ["0-100", "100-200", "200-300", "300-500", "500-750", "750-1250", ">1250"]

    # Cut emission rates into categories
    df_target["emission_category"] = pd.cut(df_target["emission"], bins=bins, labels=labels, right=True)

    # Using the selected threshold calculate TPs, FNs over all chips, all regions and emission rates
    f, ax = plt.subplots(1, 1, figsize=(10, 6))
    emission_cats = df_target["emission_category"].value_counts().index.categories.tolist()
    emission_cats_max = [100, 200, 300, 500, 750, 1250, 6200]

    mRecall_regions = []

    for region in region_to_idx:
        recalls = []
        for emission_cat in emission_cats:
            plume_idx = np.array(
                df_target[
                    (df_target["region"] == region_to_idx[region]) & (df_target["emission_category"] == emission_cat)
                ].index.tolist()
            )
            if len(plume_idx) == 0:
                logger.info(f"{region:10} {emission_cat:10}: nan Recall (0) chips)")
                recalls.append(0.0)
            else:
                gt = ys[plume_idx]
                prob_ = probs[plume_idx] > threshold * 255
                tps = (prob_ & gt).sum(axis=(1, 2))
                fns = (~prob_ & gt).sum(axis=(1, 2))
                recall = 100 * tps / (tps + fns + 0.00001)
                recalls.append(np.mean(recall).item())
                logger.info(f"{region:10} {emission_cat:10}: {recall.mean():6.1f}% Recall ({len(gt):4} chips)")

        avg_recall = np.mean(recalls).item()
        ax.plot(emission_cats_max, recalls, marker="o", linestyle="-", label=f"Recall: {region}")
        mRecall_regions.append(avg_recall)

    mRecall_global = np.mean(mRecall_regions).item()
    logger.info("*" * 100)
    logger.info(
        f"[Ep{epoch}] mAvgRecall: {mRecall_global:.2f}% "
        f"({', '.join(f'{region}: {mRecall_regions[i]:4.1f}%' for i, region in enumerate(region_to_idx))})"
    )

    ax.set_xlabel("Emission Rate (kg/h)")
    ax.set_ylabel("Probability of Detection = Recall")
    ax.set_title("Detection Probability Curves")
    ax.set_xlim(100, 7000)
    ax.set_xscale("log")
    ax.set_yticks(range(0, 100, 10))
    ax.grid(True)
    ax.legend()
    ax.axhline(mRecall_global, color="r", linestyle="--", linewidth=2)
    plt.tight_layout()

    if azure_cluster:
        mlflow.log_figure(f, f"validation/Epoch{epoch:03}_mRecall_global={mRecall_global:.2f}%.png")
    plt.close()
    return mRecall_global, mRecall_regions


def save_and_register(
    model: nn.Module,
    band_extractor: BaseBandExtractor,
    additional_training_params: dict,
    registered_model_name: str,
    artifact_path: str,
) -> None:
    """Save model to Azure workspace."""
    # Registering the model to the workspace
    logger.info("Registering the model via MLFlow")
    # We provide the parameters on the transform along with the model to make using the model elsewhere easier.
    # This eases reproducibility. An alternative would be to provide the transform itself i.e.
    # `model.transform = band_concatenator`, but this would cause problems whenever the definition of
    # ConcatenateSnapshots changes, so pickling its attributes is safer.
    model.band_concat_params = band_extractor.asdict()  # type: ignore[assignment]
    mlflow.pytorch.log_model(
        model,
        registered_model_name=registered_model_name,
        artifact_path=artifact_path,
        await_registration_for=10,  # waits 6mins per default, not necessary
    )
    model.training_params = additional_training_params


####################################################
################ HELPER FUNCTIONS ##################
####################################################


def split_dataset(dataset: ArrowDataset, test_ratio: float, train_ratio: float) -> tuple[Subset, Subset]:
    """Split the dataset into testing and train sets.

    The `test_ratio` and `train_ratio` parameters set what fraction of the dataset to use for testing and training.
    They should add up to 1, unless we want to deliberately discard some of the training data,
    which we do for example in order to match the total dataset size to a previous experiment.
    Ensures that the test dataset is independent so separate transformations can be applied.
    """
    assert test_ratio + train_ratio <= 1.0

    tol = 0.001
    if abs(test_ratio + train_ratio - 1) < tol:
        test, train = torch.utils.data.random_split(dataset, [test_ratio, train_ratio])
    else:
        discard_ratio = 1 - test_ratio - train_ratio
        test, train, _discard = torch.utils.data.random_split(dataset, [test_ratio, train_ratio, discard_ratio])

    # substitute the dataset in the test Subset with a copy of itself
    # to allow separately setting the `transform` in the train and test datasets
    test.dataset = copy.copy(test.dataset)

    return test, train


def get_untransformed_data_from_subset(subset: Subset, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return the untransformed Xy from the original dataset based on the subset index."""
    # Translate subset index to the original dataset index
    original_idx = subset.indices[idx]
    # Call the get_untransformed_data method on the original dataset
    subset_dataset = subset.dataset
    assert isinstance(subset.dataset, ArrowDataset)  # for type checking
    return subset_dataset.get_untransformed_data(original_idx)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--traindata_uri_glob", required=True, type=str)
    parser.add_argument("--validation_uri_glob", required=True, type=str)
    parser.add_argument("--traindata_rowgroup_path", required=True, type=str)
    parser.add_argument("--valdata_rowgroup_path", required=True, type=str)
    parser.add_argument("--max_train_files", required=True, type=int, default=500)
    parser.add_argument("--ground_truth_dataset", required=True, type=str)

    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--epochs", required=True, type=int, default=100)
    parser.add_argument("--early_patience", required=True, type=int, default=10)
    parser.add_argument("--epochs_warmup", required=True, type=int, default=20)
    parser.add_argument("--validate_every_x", required=True, type=int, default=3)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--MSE_multiplier", required=True, type=float)
    parser.add_argument("--binary_threshold", required=True, type=float)
    parser.add_argument("--probability_threshold", required=True, type=float)
    parser.add_argument("--registered_model_name", required=True, type=str)
    parser.add_argument("--num_workers", required=True, type=int)
    parser.add_argument("--min_batch_size", required=True, type=int)
    parser.add_argument("--max_batch_size", required=True, type=int)
    parser.add_argument("--beta1", required=True, type=float)
    parser.add_argument("--beta2", required=True, type=float)
    parser.add_argument("--eps", required=True, type=float)
    parser.add_argument(
        "--satellite-id",
        required=True,
        type=SatelliteID,
        choices=SatelliteID.list(),
        help="The satellite data we are using to train.",
    )
    parser.add_argument("--azure_cluster", action="store_true", help="Is this running on an azure cluster?")
    parser.add_argument(
        "--train_shrinkage",
        required=False,
        type=float,
        default=1.0,
        help="proportion of the training data to actually use for training",
    )
    parser.add_argument(
        "--validation_shrinkage",
        required=False,
        type=float,
        default=1.0,
        help="proportion of the validation data to actually use",
    )
    parser.add_argument(
        "--train_monitoring_ratio",
        required=False,
        type=float,
        default=0.1,
        help="proportion of train data used for monitoring on unmodulated train data",
    )
    parser.add_argument(
        "--modulation_start",
        required=False,
        type=float,
        default=1.0,
        help="starting value for modulation schedule",
    )
    parser.add_argument(
        "--modulation_end",
        required=False,
        type=float,
        default=0.05,
        help="ending value for modulation schedule",
    )
    parser.add_argument(
        "--pretrained_model_identifier",
        type=str,
        help="MLflow model identifier to continue training from, for example models:/torchgeo_pwr_unet/548",
        default=None,
    )
    parser.add_argument(
        "--remote-data",
        action="store_true",
        help="Indicates to load the data from Azure Blob Storage",
    )
    parser.add_argument(
        "--model",
        dest="model_type",
        type=ModelType,
        choices=[model.value for model in ModelType],
        default=ModelType.UNET,
        help="The model architecture to use.",
    )
    parser.add_argument("--encoder", dest="model_encoder", required=True, type=str, help="The type of encoder to use.")
    parser.add_argument(
        "--bands",
        required=True,
        type=str,
        help="Bands used for main and snapshots, e.g. 'B11,B12,B8A,B07,B05,B04,B03,B02'",
    )
    parser.add_argument(
        "--snapshots", required=True, type=str, help="Keys of snapshots, e.g. 'crop_earlier,crop_before'"
    )
    args = parser.parse_args()
    return args


# run script
if __name__ == "__main__":
    # This makes MLFLOW not hang 2mins when trying to log artifacts
    os.environ["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "10"
    os.environ["MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT"] = "10"
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "10"

    # add space in logs
    print("\n\n")
    print("*" * 60)

    args = parse_args()

    # Run distributed training
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    world_size = (
        torch.cuda.device_count()  # number of GPUs
        if torch.cuda.is_available()
        else 4  # Use 4 processes for CPU
    )
    logger.info(f"Running with {world_size} GPUs (or CPUs)")
    torch.multiprocessing.spawn(
        main,
        args=(
            world_size,
            args.traindata_uri_glob,
            args.validation_uri_glob,
            args.traindata_rowgroup_path,
            args.valdata_rowgroup_path,
            args.max_train_files,
            args.ground_truth_dataset,
            args.random_state,
            args.model_type,
            args.model_encoder,
            args.bands.split(","),
            args.snapshots.split(","),
            args.lr,
            args.beta1,
            args.beta2,
            args.eps,
            args.MSE_multiplier,
            args.binary_threshold,
            args.probability_threshold,
            args.epochs,
            args.early_patience,
            args.epochs_warmup,
            args.validate_every_x,
            args.registered_model_name,
            args.num_workers,
            args.min_batch_size,
            args.max_batch_size,
            args.satellite_id,
            args.azure_cluster,
            args.train_shrinkage,
            args.validation_shrinkage,
            args.train_monitoring_ratio,
            args.modulation_start,
            args.modulation_end,
            args.pretrained_model_identifier,
            args.remote_data,
        ),
        nprocs=world_size,  # this is what creates the `rank` parameter.  It is passed as the first argument
    )

    # add space in logs
    print("*" * 60)
    print("\n\n")
