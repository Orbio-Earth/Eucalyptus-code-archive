"""Test metrics for training.

This is a set of functions and a script in order to obtain per-tile
diagnostic metrics from the test set.
It also includes some plotting functions which can be used separately,
for example in a notebook.
This script uses argparse, use `-h` to see documentation for arguments.

Example usage (note this takes about 7 hours to run on a CPU machine):
python -m src.validation.validation_metrics_emit \
    --model_name models:/torchgeo_pwr_unet_emit/11 \
    --validation_dataset_uri "azureml://subscriptions/6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab/resourceGroups/orbio-ml-rg/providers/Microsoft.MachineLearningServices/workspaces/orbio-ml-ml-workspace/datastores/workspaceblobstore/paths/data/recycled_plumes/validation_L1C_20240813/modulate_0.1_resize_0.1" \
    --num_workers 0 
"""  # noqa: E501

from __future__ import annotations  # needed to reference self as a type annotation

import argparse
import copy
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import omegaconf
import pandas as pd
import pyarrow as pa
import torch
import xarray as xr
from azureml.fsspec import AzureMachineLearningFileSystem
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from src.data.common.utils import tensor_to_dataarray
from src.data.dataset import ArrowDataset, DiagnosticsDataset
from src.plotting.plotting_functions import (
    get_rgb_from_xarray,
    get_swir_ratio_from_xarray,
)
from src.training.loss_functions import TwoPartLoss
from src.training.transformations import MonotemporalBandExtractor
from src.utils.parameters import CROP_SIZE, EMIT_BANDS, SATELLITE_COLUMN_CONFIGS, TARGET_COLUMN, SatelliteID
from src.utils.utils import load_model_and_concatenator
from src.validation.metrics import FalseMetrics, Metrics, TrueMetrics

logger = logging.getLogger(__name__)
CLOUD_BUCKETS = ["cloud_bucket_1", "cloud_bucket_20", "cloud_bucket_50"]
SORTING_METRICS = ["false_positives", "false_negatives", "mean_squared_error"]


def main(  # (too-many-arguments)
    model_name: str,
    validation_datasets: list[str],
    num_workers: int,
    num_crops: int,
    probability_threshold: float,
    sorting_metrics: list[str],
    config: omegaconf.dictconfig.DictConfig,
) -> None:
    """Evaluate model on test dataset and log the test metrics with MLFlow.

    Parameters
    ----------
    model_name: the name of the model stored on AML to use for predictions e.g. 'models:/torchgeo_pwr_unet/70'
    validation_datasets: paths to the validation datasets to use for predicting and calculating metrics.
    num_workers: the number of workers to use for the DataLoader.
    num_crops: number of crops to create plots for.
    probability_threshold: the threshold to use on prediction probabilities for binary metrics.
    sorting_metrics: List of metrics to use as sorting keys e.g. ["false_positives", "false_negatives"].
        See the src.validation.metrics.TrueMetrics and FalseMetrics classes for possible values.
    config: path to the configuration file with validation files to compute metrics on.
    """
    for metric in sorting_metrics:
        if metric not in set(FalseMetrics) and metric not in set(TrueMetrics):
            raise ValueError(f"Metrics must be in FalseMetrics or TrueMetrics. Metric {metric} was not found.")

    # set the device we will be using to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, band_extractor, training_params = load_model_and_concatenator(model_name, device, SatelliteID.EMIT)

    model = nn.parallel.DataParallel(model)

    mlflow.log_param("model_name", model_name)
    mlflow.log_param("MSE_multiplier", training_params["MSE_multiplier"])
    mlflow.log_param("binary_threshold", training_params["binary_threshold"])
    mlflow.log_param("probability_threshold", probability_threshold)
    mlflow.log_param("num_workers", num_workers)
    mlflow.log_param("num_crops", num_crops)
    mlflow.log_param("sorting_metrics", sorting_metrics)

    # Log validation files on remote store.  Azure hijacks and changes the path for inputs
    for i, dataset in enumerate(config.validation.validation_datasets):
        mlflow.log_param(f"validation_dataset_{i}", dataset)

    lossFn = TwoPartLoss(
        binary_threshold=training_params["binary_threshold"], MSE_multiplier=training_params["MSE_multiplier"]
    )

    validation_metrics = Metrics(training_params["binary_threshold"], probability_threshold)
    crop_metrics_list: list[Metrics] = []

    # all the validation datasets (currently) have the same structure
    # we glob all the file from the base prefix and iterate through all as one dataset
    validation_dataset = Path(validation_datasets[0]).parent.parent.as_posix()
    # since Azure renames the downloaded inputs, we grab the original dataset uri from the config
    # and use it for attributing each crop to it's file.
    dataset_uri = Path(config.validation.validation_datasets[0]).parent.as_posix()

    dataset = data_preparation(validation_dataset, band_extractor)

    dataloader = DataLoader(
        DiagnosticsDataset(dataset),
        shuffle=False,
        num_workers=num_workers,
        batch_size=None,
        persistent_workers=True,
        timeout=(200 if num_workers > 0 else 0),  # allows enough time to download files
    )

    metrics_per_crop, validation_metrics = compute_validation_metrics(
        model,
        device,
        dataloader,
        lossFn,
        training_params["binary_threshold"],
        probability_threshold,
        dataset_uri,
    )

    crop_metrics_list.extend(metrics_per_crop)

    # Save metrics per crop as a parquet file
    filepath = "metrics_per_crop.parquet"
    crop_metrics = [metric.as_dict() for metric in tqdm(crop_metrics_list, desc="converting metrics list to dataframe")]
    pd.DataFrame(crop_metrics).to_parquet(filepath)
    mlflow.log_artifact(filepath)

    # Save overall metrics
    filepath = "validation_metrics.json"
    with open(filepath, "w") as f:
        json.dump(validation_metrics.as_dict(json=True), f)
    mlflow.log_artifact(filepath)

    # TODO: make new dataset with all validation and be able to load
    # each file row from that

    ## Plot Best and Worst crops overall
    for metric in sorting_metrics:
        # if metric is a false metric (larger worse / smaller better) will sort in ascending order
        # if metric is a true metric (larger better / smaller worse) will sort in descending order
        ascending = metric not in set(FalseMetrics)
        plot_crops(
            model,
            dataset.dataset,
            lossFn,
            probability_threshold,
            crop_metrics,
            metric,
            num_crops,
            "worst_crops",
            ascending=ascending,
        )

        # if metric is a false metric (larger worse / smaller better) will sort in ascending order
        # if metric is a true metric (larger better / smaller worse) will sort in descending order
        ascending = metric in set(FalseMetrics)
        plot_crops(
            model,
            dataset.dataset,
            lossFn,
            probability_threshold,
            crop_metrics,
            metric,
            num_crops,
            "best_crops",
            ascending=ascending,
        )


def plot_crops(  # (too-many-arguments)
    model: nn.Module,
    dataset: Dataset,
    lossFn: TwoPartLoss,
    probability_threshold: float,
    crop_metrics: list[dict],
    sorting_metric: str,
    num_crops: int,
    out_dir: str,
    ascending: bool,
) -> None:
    """Sort crops by specified metric and create plots.

    Parameters
    ----------
    model: the trained model to use for predictions.
    dataset: the dataset to predict on.
    lossFn: the loss function for the model.
    probability_threshold: the threshold to use on prediction probabilities for binary metrics.
    crop_metrics: a list of Metrics in dict form.
    sorting_metric: the metric to use as sorting key e.g. "f1_score"
    num_crops: number of crops to create plots for
    out_dir: the directory to store the figures.
    ascending: if True sort by increasing order, if False sort by descreasing order.
    """
    reverse = not ascending  # invert boolean to align the wording with how the sorting works
    filter_iter = filter(lambda x: not math.isnan(x[sorting_metric]), crop_metrics)
    crop_metrics = sorted(filter_iter, key=lambda d: d[sorting_metric], reverse=reverse)
    for i in range(num_crops):
        crop = crop_metrics[i]
        partition = crop["partition"]
        row = crop["row"]
        index = (partition, row)
        pred = prep_predictions_for_plot(model, dataset, index, lossFn, probability_threshold)

        # save plots
        all_plots = all_error_analysis_plots(probability_threshold=probability_threshold, **pred)
        fps = crop["false_positives"]
        fns = crop["false_negatives"]
        tps = crop["true_positives"]
        mse = crop["mean_squared_error"]

        base_dir = Path(f"{sorting_metric}/{out_dir}")
        dir = base_dir / "plots"
        filename = f"{str(i).zfill(3)}_FP={fps}_FN={fns}_TP={tps}_MSE={mse:0.6f}_{partition}:{row}"  # (line-too-long)
        mlflow.log_figure(all_plots, (dir / filename).with_suffix(Path(filename).suffix + ".png").as_posix())

        # save metrics
        dir = base_dir / "metrics"
        dir.mkdir(parents=True, exist_ok=True)

        filepath = (dir / filename).with_suffix(Path(filename).suffix + ".json").as_posix()
        with open(filepath, "w") as f:
            json.dump(crop, f)
        mlflow.log_artifact(filepath, dir.as_posix())


def data_preparation(
    parquet_glob_uri: str,
    band_extractor: MonotemporalBandExtractor,
    filesystem: AzureMachineLearningFileSystem | None = None,
) -> Subset:
    """
    Read in parquet(s) as ArrowDataset.

    Split into training and test set if validation_set is False.
    """
    # tell arrow how to read the hive partitioning (modulation & cloud buckets)
    part = pa.dataset.partitioning(pa.schema([("transformation", pa.string()), ("cloud_bucket", pa.string())]))

    parquet_dataset = pa.dataset.dataset(
        parquet_glob_uri,  # replace with list of sub-sumples parquets
        format="parquet",
        partitioning=part,
        exclude_invalid_files=True,  # incurs IO but unclear if it's once when creating the dataset or everytime.
        filesystem=filesystem,
    )

    base_dataset = ArrowDataset(
        parquet_dataset,
        columns_config=SATELLITE_COLUMN_CONFIGS[SatelliteID.EMIT],
        target_column=TARGET_COLUMN,
        transform=band_extractor,
        filesystem=filesystem,
    )

    validation_dataset = Subset(base_dataset, list(range(len(base_dataset))))

    return validation_dataset


def compute_validation_metrics(
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    lossFn: TwoPartLoss,
    frac_threshold: float,
    probability_threshold: float,
    dataset_uri: str,
) -> tuple[list[Metrics], Metrics]:
    """Compute FPR and TPR for the model on the test dataset."""
    metrics_per_crop: list[Metrics] = []
    validation_metrics = Metrics(frac_threshold, probability_threshold)

    # generate predictions for first row group
    model.eval()
    for datum in tqdm(dataloader):
        partition_index = datum["partition"]
        X = datum["X"]
        y = datum["y"]
        with torch.no_grad():
            (X, y) = (X.to(device), y.to(device))
            pred = model(X)
            marginal_pred, binary_probability, conditional_pred, binary_logit = lossFn.get_prediction_parts(pred)

            combined_loss, bce, cond_mse = lossFn(pred, y)

            # Overall Metrics
            # we calculate these separately because we can't combine batches of 1 samples due because
            # the sample must have >2 samples to compute
            validation_metrics.update(
                marginal_pred,
                binary_probability,
                y,
                combined_loss.item(),
                bce,
                cond_mse,
            )

            # grab metadata for the partition
            metadata = dataloader.dataset.wrapped.get_metadata(
                partition_index, ["resize", "modulate", "cloud_bucket", "transformation"]
            )

            for i in range(y.shape[0]):
                target = y.select(0, i).reshape((1, 1, CROP_SIZE, CROP_SIZE))
                ypred = pred.select(0, i).reshape((1, 2, CROP_SIZE, CROP_SIZE))

                y_pred = marginal_pred.select(0, i).reshape((1, 1, CROP_SIZE, CROP_SIZE))
                y_probability = binary_probability.select(0, i).reshape((1, 1, CROP_SIZE, CROP_SIZE))

                combined_loss, bce, cond_mse = lossFn(ypred, target)

                # this is horribly gross but since Azure renames the inputs we have to recreate the filename
                # of the crop from the cloud_bucket (from the filename because it's not saved in the parquet)
                # and the modulate and resize from the parquet file.
                # TODO: save the cloud bucket in the file for each crop and even save the filename itself.
                filename = Path(metadata["file"][i]).name
                cloud_bucket = Path(metadata["file"][i]).parent.stem
                modulate = metadata["modulate"][i]
                resize = metadata["resize"][i]
                transformation = f"modulate_{modulate}_resize_{resize}"
                file = (Path(dataset_uri) / transformation / f"cloud_bucket_{cloud_bucket}" / filename).as_posix()
                extra_info = {
                    "partition": partition_index,
                    "row": i,
                    "total_true_frac": target.sum().item(),
                    "total_predicted_frac": y_pred.sum().item(),
                    "modulate": modulate,
                    "resize": resize,
                    "file": file,
                    # these should really just be additional columns in the file
                    "cloud_bucket": cloud_bucket,
                    "transformation": transformation,
                }

                metrics = Metrics(frac_threshold, probability_threshold)
                metrics.update(
                    y_pred,
                    y_probability,
                    target,
                    combined_loss.item(),
                    bce,
                    cond_mse,
                ).add_metadata(copy.copy(extra_info))

                metrics_per_crop.append(metrics)

    return metrics_per_crop, validation_metrics


###########################################
############## PLOTTING ###################
###########################################
def all_error_analysis_plots(  # (too-many-arguments, too-many-statements)
    rgb_main: xr.DataArray,
    ratio_main: xr.DataArray,
    target_retrieval: torch.Tensor,
    ground_truth_mask: torch.Tensor,
    predicted_frac: torch.Tensor,
    predicted_mask: torch.Tensor,
    conditional_pred: torch.Tensor,
    probability_threshold: float = 0.25,
    **kwargs: Any,
) -> plt:
    """Return matplotlib figure with RGB, frac, ratio, and ground truth."""
    fps_plot = ((predicted_mask == 1) & (ground_truth_mask == 0)).to(torch.uint8)
    fns_plot = ((predicted_mask == 0) & (ground_truth_mask == 1)).to(torch.uint8)

    alpha = 0.5
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(5, 2, figsize=(8, 20))  # Adjusted to 2 rows, 5 columns

    ###########################################################
    # Define Grid (2 rows, 5 columns)
    ###########################################################

    # Row 1
    axes[0, 0].set_title("RGB", fontsize=14)
    im = axes[0, 0].imshow(rgb_main / rgb_main.max(), vmin=0.0, vmax=1.0, interpolation="nearest")
    fig.colorbar(im, ax=axes[0, 0])

    axes[0, 1].set_title("B12/B11 Ratio", fontsize=14)
    im = axes[0, 1].imshow(ratio_main, vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im, ax=axes[0, 1])

    # Row 2
    axes[1, 0].set_title("Ground truth mask", fontsize=14)
    axes[1, 0].imshow(ground_truth_mask, cmap="pink_r", interpolation="nearest")

    axes[1, 1].set_title(f"Predicted Mask (threshold={probability_threshold})", fontsize=14)
    axes[1, 1].imshow(predicted_mask, cmap="pink_r", interpolation="nearest")

    # Row 3
    axes[2, 0].set_title(f"FPs = RED ({fps_plot.sum()}), FNs = BLACK ({fns_plot.sum()})", fontsize=14)
    axes[2, 0].imshow(np.ma.masked_where(fps_plot != 1, fps_plot), cmap="autumn", interpolation="none", alpha=1)
    axes[2, 0].imshow(np.ma.masked_where(fns_plot != 1, fns_plot), cmap="gray", interpolation="none", alpha=1)

    axes[2, 1].set_title("Conditional Prediction", fontsize=14)
    im = axes[2, 1].imshow(conditional_pred, vmin=-0.08, vmax=0.08, cmap="RdBu", interpolation="nearest")
    fig.colorbar(im, ax=axes[2, 1])

    axes[3, 0].set_title("Ground Truth Retrieval", fontsize=14)
    im = axes[3, 0].imshow(target_retrieval, vmin=-0.08, vmax=0.08, cmap="RdBu", interpolation="nearest")
    fig.colorbar(im, ax=axes[3, 0])

    # Row 4
    axes[3, 1].set_title("Predicted Retreival", fontsize=14)
    im = axes[3, 1].imshow(predicted_frac, vmin=-0.08, vmax=0.08, cmap="RdBu", interpolation="nearest")
    fig.colorbar(im, ax=axes[3, 1])

    # Row 5
    axes[4, 0].set_title("RGB (ground truth mask)", fontsize=14)
    axes[4, 0].imshow(rgb_main / rgb_main.max(), vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[4, 0].imshow(np.ma.masked_where(ground_truth_mask == 0, ground_truth_mask), cmap="autumn", alpha=alpha)

    axes[4, 1].set_title("RGB (prediction mask)", fontsize=14)
    axes[4, 1].imshow(rgb_main / rgb_main.max(), vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[4, 1].imshow(np.ma.masked_where(predicted_mask == 0, predicted_mask), cmap="autumn", alpha=alpha)

    # Remove axis ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    return fig


def prep_predictions_for_plot(
    model: nn.Module,
    dataset: Dataset,
    index: tuple[int, int],
    lossFn: TwoPartLoss,
    probability_threshold: float,
) -> dict[str, Any]:
    """Prepare data for plotting predictions."""
    retrieval_threshold = lossFn.binary_threshold

    dataset_index, item_index = index
    model.eval()

    metadata = dataset.get_metadata(dataset_index, ["main_cloud_ratio"])
    untransformed_inputs, untransformed_targets = dataset.get_untransformed_data(dataset_index)
    X, y = dataset.transform((untransformed_inputs, untransformed_targets))

    # move data to device of the model
    device = next(model.parameters()).device
    (X, y) = (X.to(device), y.to(device))

    with torch.no_grad():
        pred = model(X)
        marginal_pred, binary_probability, conditional_pred, _ = lossFn.get_prediction_parts(pred)

    # Get prediction
    predicted_frac = marginal_pred[item_index, ...].squeeze().cpu()
    binary_probability = binary_probability[item_index, ...].squeeze().cpu()
    prediction_mask = binary_probability >= probability_threshold
    conditional_pred = conditional_pred[item_index, ...].squeeze().cpu()

    # Get ground truth
    crop_main = tensor_to_dataarray(untransformed_inputs["crop_main"][item_index], EMIT_BANDS)
    target_retrieval = y[item_index, ...].squeeze().cpu()
    ground_truth_mask = (torch.abs(target_retrieval) > retrieval_threshold).cpu()

    main_cloud_ratio = metadata["main_cloud_ratio"][item_index]

    rgb_main = get_rgb_from_xarray(crop_main, SatelliteID.EMIT)

    ratio_main = get_swir_ratio_from_xarray(crop_main, SatelliteID.EMIT)

    return {
        "main_cloud_ratio": main_cloud_ratio,
        "rgb_main": rgb_main,
        "ratio_main": ratio_main,
        "target_retrieval": target_retrieval,
        "ground_truth_mask": ground_truth_mask,
        "predicted_frac": predicted_frac,
        "predicted_mask": prediction_mask,
        "conditional_pred": conditional_pred,
        "probability": binary_probability,
    }


###########################################
############## SCRIPT ###################
###########################################


def parse_args() -> argparse.Namespace:
    """Set up the CLI argument parser."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        help="The model with version to apply, for example 'models:/torchgeo_pwr_unet/566'.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--validation_dataset_uri",
        required=True,
        type=str,
        action="append",
        help="Validation dataset URI for a given transformation. Can be specified multiple times.",
    )
    parser.add_argument("--num_workers", required=False, default=4, type=int)
    parser.add_argument("--num_crops", required=False, default=50, type=int, help="Number of crops to create plots for")
    parser.add_argument(
        "--probability_threshold", required=False, default=0.25, type=float, help="probability threshold for masking."
    )

    # We need to pass in the config file to pull the unmodified validation files as Azure changes the Input paths
    parser.add_argument(
        "--config",
        "-f",
        required=True,
        help=(
            "Config file to use as base. The configs directory is 'src/training/configs' so"
            " any configs there should have that prefix e.g. 'src/training/configs/s2.yaml'"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = omegaconf.OmegaConf.load(args.config)

    main(
        model_name=args.model_name,
        validation_datasets=args.validation_dataset_uri,
        num_workers=args.num_workers,
        num_crops=args.num_crops,
        sorting_metrics=SORTING_METRICS,
        probability_threshold=args.probability_threshold,
        config=config,
    )
