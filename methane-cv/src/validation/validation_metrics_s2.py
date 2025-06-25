"""Test metrics for training.

This is a set of functions and a script in order to obtain per-tile diagnostic metrics from the validation set.
It also includes some plotting functions which can be used separately, for example in a notebook.
This script uses argparse, use `-h` to see documentation for arguments.

Example usage:
python -m src.validation.validation_metrics \
    --model_name models:/torchgeo_pwr_unet/566 \
    --validation_dataset "azureml://subscriptions/6e71ce37-b9fe-4c43-942b-cf0f7e78c8ab/resourceGroups/orbio-ml-rg/providers/Microsoft.MachineLearningServices/workspaces/orbio-ml-ml-workspace/datastores/workspaceblobstore/paths/data/recycled_plumes/validation_L1C_20240813/modulate_0.1_resize_0.1" \
    --num_workers 0 
"""  # noqa: E501

from __future__ import annotations  # needed to reference self as a type annotation

import argparse
import glob
import logging
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import omegaconf
import pandas as pd
import pyarrow as pa
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.data.dataset import ArrowDataset, DiagnosticsDataset
from src.plotting.plotting_functions import (
    CMAP,
    S2_LAND_COVER_CLASSIFICATIONS,
    get_band_ratio_from_tensor,
    get_rgb_from_tensor,
    grid16,
)
from src.training.loss_functions import TwoPartLoss
from src.training.training_script import compute_parquet_dimensions
from src.training.transformations import ConcatenateSnapshots
from src.utils.parameters import S2_BANDS, TARGET_COLUMN
from src.utils.utils import load_model_and_concatenator, setup_device_and_distributed_model
from src.validation.metrics import FalseMetrics, Metrics, TrueMetrics

# logger: logging.Logger = setup_logging()
logger = logging.getLogger(__name__)
SORTING_METRICS = ["false_positives", "false_negatives", "mean_squared_error"]
METADATA_COLUMNS = [
    "crop_x",
    "crop_y",
    "main_and_reference_ids",
    "plume_files",
    "chip_SCL_NO_DATA_perc_main",
    "chip_SCL_NO_DATA_perc_before",
    "chip_SCL_NO_DATA_perc_earlier",
    "chip_cloud_combined_perc_main",
    "chip_cloud_shadow_omni_perc_main",
    "chip_cloud_combined_perc_before",
    "chip_cloud_shadow_omni_perc_before",
    "chip_cloud_combined_perc_earlier",
    "chip_cloud_shadow_omni_perc_earlier",
    "tile_cloud_combined_perc_main",
    "tile_cloud_shadow_combined_perc_main",
    "tile_cloud_combined_perc_before",
    "tile_cloud_shadow_combined_perc_before",
    "tile_cloud_combined_perc_earlier",
    "tile_cloud_shadow_combined_perc_earlier",
    "how_many_plumes_we_wanted",
    "how_many_plumes_we_inserted",
    "plumes_inserted_idxs",
    "plume_sizes",
    "frac_abs_sum",
    "plume_category",
    "min_frac",
    "exclusion_perc",
    "region_overlap",
    "reference_indices_chosen",
    "main_and_reference_dates",
    "plume_emissions",
]


def main(  # noqa: PLR0913
    rank: int,  # needs to be first parameter for distributed process spawning
    world_size: int,
    model_name: str,
    validation_dataset: str,
    valdata_rowgroup_path: str,
    num_workers: int,
    num_crops: int,
    probability_threshold: float,
    sorting_metrics: list[str],
    config: omegaconf.dictconfig.DictConfig,
) -> None:
    """Evaluate model on validation dataset and log the metrics with MLFlow.

    Parameters
    ----------
    model_name: the name of the model stored on AML to use for predictions e.g. 'models:/torchgeo_pwr_unet/1226'
    validation_dataset: path to the validation datasets to use for predicting and calculating metrics.
    valdata_rowgroup_path: Local path to json that contains a mapping from val parquet files to num_rows for
        faster dataset creation.
    num_workers: the number of workers to use for the DataLoader.
    num_crops: number of crops to create plots for.
    probability_threshold: the threshold to use on prediction probabilities for binary metrics.
    sorting_metrics: List of metrics to use as sorting keys e.g. ["false_positives", "false_negatives"].
        See the src.validation.metrics.TrueMetrics and FalseMetrics classes for possible values.
    config: configuration file.
    """
    for metric in sorting_metrics:
        if metric not in set(FalseMetrics) and metric not in set(TrueMetrics):
            raise ValueError(f"Metrics must be in FalseMetrics or TrueMetrics. Metric {metric} was not found.")

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

    model, band_concatenator, training_params = load_model_and_concatenator(
        model_name, "cpu", config.train.satellite_id
    )
    # Set up model for distributed processing
    model, device = setup_device_and_distributed_model(model, rank, world_size)

    if rank == 0:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("MSE_multiplier", training_params["MSE_multiplier"])
        mlflow.log_param("binary_threshold", training_params["binary_threshold"])
        mlflow.log_param("probability_threshold", probability_threshold)
        mlflow.log_param("num_workers", num_workers)
        mlflow.log_param("num_crops", num_crops)
        mlflow.log_param("sorting_metrics", sorting_metrics)
        # Log validation files on remote store.  Azure hijacks and changes the path for inputs
        mlflow.log_param("validation_dataset", validation_dataset)

    lossFn = TwoPartLoss(training_params["binary_threshold"], training_params["MSE_multiplier"])

    # we glob all the file from the base prefix and iterate through all as one dataset
    validation_dataset = Path(validation_dataset).as_posix()
    logger.error(f"{validation_dataset=}")

    # since Azure renames the downloaded inputs, we grab the original dataset uri from the config
    # and use it for attributing each crop to it's file.
    dataset_folder = Path(config.validation.data.uri).as_posix()
    logger.error(f"{dataset_folder=}")

    dataset = data_preparation(validation_dataset, band_concatenator, valdata_rowgroup_path)
    validation_sampler: DistributedSampler = DistributedSampler(dataset, shuffle=False, drop_last=False)

    dataloader = DataLoader(
        DiagnosticsDataset(dataset),
        shuffle=False,
        sampler=validation_sampler,
        num_workers=num_workers,
        batch_size=None,
        persistent_workers=True,
        timeout=(200 if num_workers > 0 else 0),  # allows enough time to download files
        multiprocessing_context="forkserver",
    )

    compute_validation_metrics(
        rank,
        model,
        device,
        dataloader,
        lossFn,
        training_params["binary_threshold"],
        probability_threshold,
        dataset_folder,
    )

    if rank == 0:
        # Open metrics per crop from multiple GPUs and resave as one big parquet file
        metrics_per_crop = pd.concat(
            [pd.read_parquet(path) for path in glob.glob("metrics_per_crop*.parquet")], axis=0, ignore_index=True
        )
        logger.error(f"{metrics_per_crop.shape=}")
        filepath = "metrics_per_crop.parquet"
        logger.error(f"Saving to {filepath}")
        metrics_per_crop.to_parquet(filepath)
        mlflow.log_artifact(filepath)

        metrics_per_crop = metrics_per_crop.to_dict(orient="records")

        ## Plot worst crops
        for metric in sorting_metrics:
            # if metric is a false metric (larger worse / smaller better) will sort in ascending order
            # if metric is a true metric (larger better / smaller worse) will sort in descending order
            ascending = metric not in set(FalseMetrics)
            plot_crops(
                model,
                dataset.dataset,
                lossFn,
                probability_threshold,
                metrics_per_crop,
                metric,
                num_crops,
                ascending=ascending,
            )

        ## Plot best crops = most True Positives
        metric = "true_positives"
        ascending = False
        plot_crops(
            model,
            dataset.dataset,
            lossFn,
            probability_threshold,
            metrics_per_crop,
            metric,
            num_crops,
            ascending=ascending,
        )


def plot_crops(
    model: nn.Module,
    dataset: Dataset,
    lossFn: TwoPartLoss,
    probability_threshold: float,
    crop_metrics: list[dict],
    sorting_metric: str,
    num_crops: int,
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
        simple_plots = diff_plots(probability_threshold=probability_threshold, **pred)

        fps = crop["false_positives"]
        fns = crop["false_negatives"]
        tps = crop["true_positives"]
        mse = crop["mean_squared_error"]

        base_dir = Path(f"{sorting_metric}")
        dir = base_dir / "plots"
        filename = f"{str(i).zfill(3)}_FP={fps}_FN={fns}_TP={tps}_MSE={mse:0.6f}_{partition}:{row}"
        mlflow.log_figure(all_plots, (dir / filename).with_suffix(Path(filename).suffix + ".png").as_posix())

        dir = Path(f"{sorting_metric}/plots_simple/")
        mlflow.log_figure(simple_plots, (dir / filename).with_suffix(Path(filename).suffix + ".png").as_posix())

        # save metrics
        dir = base_dir / "metrics"
        dir.mkdir(parents=True, exist_ok=True)


def data_preparation(
    parquet_glob_uri: str,
    band_concatenator: ConcatenateSnapshots,
    valdata_rowgroup_path: str,
) -> Subset:
    """Read in parquet(s) as ArrowDataset."""
    exclude_invalid_files = True
    if len(glob.glob(f"{parquet_glob_uri}/*.parquet")) > 0:
        parquet_glob_uri = sorted(glob.glob(f"{parquet_glob_uri}/*.parquet"))  # type:ignore
        exclude_invalid_files = False

    parquet_dataset = pa.dataset.dataset(
        parquet_glob_uri,
        format="parquet",
        exclude_invalid_files=exclude_invalid_files,
    )

    # Create the base dataset with ConcatenateSnapshots as the base transformation
    crop_size, bands = compute_parquet_dimensions(parquet_dataset)
    num_s2_bands = len(bands)

    columns_config = {
        "crop_main": {"shape": (num_s2_bands, crop_size, crop_size), "dtype": torch.int16},
        TARGET_COLUMN: {"shape": (1, crop_size, crop_size), "dtype": torch.float32},
    }
    for snapshot in band_concatenator.snapshots:
        columns_config[snapshot] = {"shape": (num_s2_bands, crop_size, crop_size), "dtype": torch.int16}

    base_dataset = ArrowDataset(
        parquet_dataset,
        columns_config=columns_config,
        target_column=TARGET_COLUMN,
        transform=band_concatenator,
        parquet_to_rowgroups_mapping_path=valdata_rowgroup_path,
    )

    validation_dataset = Subset(base_dataset, list(range(len(base_dataset))))
    logger.error(f"{len(validation_dataset)=}")

    return validation_dataset


def compute_validation_metrics(
    rank: int,
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    lossFn: TwoPartLoss,
    frac_threshold: float,
    probability_threshold: float,
    dataset_folder: str,
) -> None:
    """Predict and save metrics for each chip in the valset."""
    metrics_per_crop: list[dict] = []

    model.eval()
    for count, datum in tqdm(enumerate(dataloader), total=len(dataloader), disable=None):
        partition_index = datum["partition"]
        X = datum["X"]
        y = datum["y"]
        with torch.no_grad():
            (X, y) = (X.to(device), y.to(device))
            pred = model(X)
            marginal_pred, binary_probability, _, _ = lossFn.get_prediction_parts(pred)
            combined_loss, bce, cond_mse = lossFn(pred, y)

            # grab metadata for the partition
            metadata = dataloader.dataset.wrapped.get_metadata(partition_index, METADATA_COLUMNS)

            filename = Path(metadata["file"][0]).name
            file = (Path(dataset_folder) / filename).as_posix()
            extra_info = {
                "partition": partition_index,
                "row": 0,
                "total_true_frac": y.sum().item(),
                "total_predicted_frac": marginal_pred.sum().item(),
                "file": file,
            }
            for col in METADATA_COLUMNS:
                extra_info[col] = metadata[col][0]

            metrics = Metrics(frac_threshold, probability_threshold)
            metrics.update(
                marginal_pred,
                binary_probability,
                y,
                combined_loss.item(),
                bce,
                cond_mse,
            ).add_metadata(extra_info)
            metrics = metrics.as_dict()  # type:ignore
            metrics_per_crop.append(metrics)  # type:ignore
        if count % (len(dataloader) // 20) == 0:
            logger.error(f"Processed {count}/{len(dataloader)} crops")
    filepath = f"metrics_per_crop_GPU{rank}.parquet"
    logger.error(f"Saving to {filepath}")
    pd.DataFrame(metrics_per_crop).to_parquet(filepath)
    mlflow.log_artifact(filepath)


###########################################
############## PLOTTING ###################
###########################################
def all_error_analysis_plots(  # noqa (too-many-arguments, too-many-statements)
    rgb_main: torch.Tensor,
    rgb_earlier: torch.Tensor,
    rgb_before: torch.Tensor,
    ratio_main: torch.Tensor,
    ratio_before: torch.Tensor,
    ratio_earlier: torch.Tensor,
    lcc_main: torch.Tensor,
    lcc_before: torch.Tensor,
    lcc_earlier: torch.Tensor,
    s2_ids: list,
    target_frac: torch.Tensor,
    ground_truth_mask: torch.Tensor,
    predicted_frac: torch.Tensor,
    predicted_mask: torch.Tensor,
    conditional_pred: torch.Tensor,
    probability_threshold: float = 0.25,
    **kwargs: Any,
) -> plt:
    """Return matplotlib figure with RGB, frac, ratio, and ground truth."""
    fps_plot = ((predicted_mask == 1) & (ground_truth_mask == 0)).to(torch.uint8)
    fp = fps_plot.sum()
    fns_plot = ((predicted_mask == 0) & (ground_truth_mask == 1)).to(torch.uint8)
    fn = fns_plot.sum()
    tps_plot = ((predicted_mask == 1) & (ground_truth_mask == 1)).to(torch.uint8)
    tp = tps_plot.sum()

    s2_date_earlier = s2_ids[2].split("_")[2]
    s2_date_earlier = f"{s2_date_earlier[:4]}-{s2_date_earlier[4:6]}-{s2_date_earlier[6:8]}"
    s2_date_before = s2_ids[1].split("_")[2]
    s2_date_before = f"{s2_date_before[:4]}-{s2_date_before[4:6]}-{s2_date_before[6:8]}"
    s2_date_main = s2_ids[0].split("_")[2]
    s2_date_main = f"{s2_date_main[:4]}-{s2_date_main[4:6]}-{s2_date_main[6:8]}"

    alpha = 0.5
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(42, 15))

    #########################
    # Column 1: Plot RGB
    #########################
    plt.subplot(3, 7, 1)
    plt.title(
        f"""RGB (t=earlier)
              Min {rgb_earlier.min():.3f}, Max {rgb_earlier.max():.3f}, Mean {rgb_earlier.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(
        (rgb_earlier.numpy() / 0.35 * 255).clip(0, 255).astype(np.uint8), vmin=0.0, vmax=1.0, interpolation="nearest"
    )
    grid16()

    plt.subplot(3, 7, 8)
    plt.title(
        f"""RGB (t=before)
              Min {rgb_before.min():.3f}, Max {rgb_before.max():.3f}, Mean {rgb_before.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(
        (rgb_before.numpy() / 0.35 * 255).clip(0, 255).astype(np.uint8), vmin=0.0, vmax=1.0, interpolation="nearest"
    )
    grid16()

    plt.subplot(3, 7, 15)
    plt.title(
        f"""RGB (t=main)
              Min {rgb_main.min():.3f}, Max {rgb_main.max():.3f}, Mean {rgb_main.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(
        (rgb_main.numpy() / 0.35 * 255).clip(0, 255).astype(np.uint8), vmin=0.0, vmax=1.0, interpolation="nearest"
    )
    grid16()

    #########################
    # Column 2: Plot Ratios
    #########################
    # Plot ratio earlier
    plt.subplot(3, 7, 2)
    plt.title(
        f"""B12/B11 Ratio (t-2=earlier)
              Min {ratio_earlier.min():.3f}, Max {ratio_earlier.max():.3f}, Mean {ratio_earlier.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(ratio_earlier, vmin=0, vmax=1, interpolation="nearest")
    grid16()
    plt.colorbar()

    # Plot ratio before
    plt.subplot(3, 7, 9)
    plt.title(
        f"""B12/B11 Ratio (t-1=before)
              Min {ratio_before.min():.3f}, Max {ratio_before.max():.3f}, Mean {ratio_before.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(ratio_before, vmin=0, vmax=1, interpolation="nearest")
    grid16()
    plt.colorbar()

    # Plot ratio main
    plt.subplot(3, 7, 16)
    plt.title(
        f"""B12/B11 Ratio (t=main)
              Min {ratio_main.min():.3f}, Max {ratio_main.max():.3f}, Mean {ratio_main.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(ratio_main, vmin=0, vmax=1, interpolation="nearest")
    grid16()
    plt.colorbar()

    #########################
    # Column 3: Plot Ratio Diffs
    #########################
    # Plot ratio diff
    ratio_diff = ratio_main - (ratio_before + ratio_earlier) / 2
    plt.subplot(3, 7, 3)
    plt.title(
        f"""Ratio Diff (main - reference) 
        Min {ratio_diff.min():.3f}, Max {ratio_diff.max():.3f}, Mean {ratio_diff.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(ratio_diff, vmin=-0.2, vmax=0.2, interpolation="nearest")
    plt.colorbar()
    grid16()

    # Plot ratio diff with ground truth mask
    plt.subplot(3, 7, 10)
    plt.title("Ratio Diff (ground truth mask)", fontsize=15)
    plt.imshow(ratio_diff, vmin=-0.2, vmax=0.2, interpolation="nearest")
    plt.colorbar()
    plt.imshow(np.ma.masked_where(ground_truth_mask == 0, ground_truth_mask), cmap="autumn", alpha=alpha)
    grid16()

    # Plot ratio ratio diff with prediction mask
    plt.subplot(3, 7, 17)
    plt.title(
        f"""Ratio Diff (prediction mask)
              FPs: {fp:.0f}, FNs: {fn:.0f}, TPs: {tp:.0f}""",
        fontsize=15,
    )
    plt.imshow(ratio_diff, vmin=-0.2, vmax=0.2, interpolation="nearest")
    plt.colorbar()
    plt.imshow(np.ma.masked_where(predicted_mask == 0, predicted_mask), cmap="autumn", alpha=alpha)
    grid16()

    #########################
    # Column 4: Plot Masks
    #########################
    # Plot ground truth mask
    plt.subplot(3, 7, 4)
    plt.title("Ground truth mask", fontsize=15)
    plt.imshow(ground_truth_mask, cmap="pink_r", interpolation="nearest")
    grid16()

    # Plot predicted mask
    plt.subplot(3, 7, 11)
    plt.title(f"Predicted Mask (threshold={probability_threshold})", fontsize=15)
    plt.imshow(predicted_mask, cmap="pink_r", interpolation="nearest")
    grid16()

    # Plot FP and FN
    plt.subplot(3, 7, 18)
    plt.title(f"FPs = RED ({fps_plot.sum()}), FNs = BLACK ({fns_plot.sum()})", fontsize=15)
    plt.imshow(np.ma.masked_where(fps_plot != 1, fps_plot), cmap="autumn", interpolation="none", alpha=1)
    plt.imshow(np.ma.masked_where(fns_plot != 1, fns_plot), cmap="gray", interpolation="none", alpha=1)
    grid16()

    # # Plot Probability
    # plt.subplot(3, 7, 12)
    # plt.title(f"Predicted Probability ({probability_threshold:0.2f})")
    # plt.imshow(probability.cpu(), cmap="pink_r", interpolation="nearest")
    # grid16()
    # plt.colorbar()

    #########################
    # Column 5: Plot Frac
    #########################
    # Plot ground truth frac
    plt.subplot(3, 7, 5)
    plt.title("Ground truth frac", fontsize=15)
    plt.imshow(target_frac, vmin=-0.08, vmax=0.08, cmap="RdBu", interpolation="nearest")
    grid16()
    plt.colorbar()

    # Plot predicted frac
    plt.subplot(3, 7, 12)
    plt.title("Predicted Frac", fontsize=15)
    plt.imshow(predicted_frac, vmin=-0.08, vmax=0.08, cmap="RdBu", interpolation="nearest")
    grid16()
    plt.colorbar()

    # Plot Conditional Pred
    plt.subplot(3, 7, 19)
    plt.title("Conditional Prediction", fontsize=15)
    plt.imshow(conditional_pred, vmin=-0.08, vmax=0.08, cmap="RdBu", interpolation="nearest")
    grid16()
    plt.colorbar()

    #########################
    # Column 6: RGB preds
    #########################
    # Plot RBG main
    plt.subplot(3, 7, 6)
    plt.title("RGB (t=main)", fontsize=15)
    plt.imshow(
        (rgb_main.numpy() / 0.35 * 255).clip(0, 255).astype(np.uint8), vmin=0.0, vmax=1.0, interpolation="nearest"
    )
    grid16()

    # Plot RGB with ground truth mask
    plt.subplot(3, 7, 13)
    plt.title("RGB (ground truth mask)", fontsize=15)
    plt.imshow(
        (rgb_main.numpy() / 0.35 * 255).clip(0, 255).astype(np.uint8), vmin=0.0, vmax=1.0, interpolation="nearest"
    )
    plt.imshow(np.ma.masked_where(ground_truth_mask == 0, ground_truth_mask), cmap="autumn", alpha=alpha)
    grid16()

    # Plot RGB with prediction mask
    plt.subplot(3, 7, 20)
    plt.title("RGB (prediction mask)", fontsize=15)
    plt.imshow(
        (rgb_main.numpy() / 0.35 * 255).clip(0, 255).astype(np.uint8), vmin=0.0, vmax=1.0, interpolation="nearest"
    )
    plt.imshow(np.ma.masked_where(predicted_mask == 0, predicted_mask), cmap="autumn", alpha=alpha)
    grid16()

    #########################
    # Column 7: Land Cover Classification
    #########################
    # Plot LCC earlier
    plt.subplot(3, 7, 7)
    plt.title(f"LCC (t-2=earlier)\n{s2_date_earlier}", fontsize=15)
    plt.imshow(lcc_earlier, vmin=0, vmax=len(S2_LAND_COVER_CLASSIFICATIONS), cmap=CMAP, interpolation="nearest")
    grid16()
    cbar = plt.colorbar(ticks=np.arange(0.5, len(S2_LAND_COVER_CLASSIFICATIONS), 1))
    cbar.set_label("Land Cover Classification", fontsize=15)
    cbar.set_ticklabels(S2_LAND_COVER_CLASSIFICATIONS, fontsize=15)

    # Plot LCC before
    plt.subplot(3, 7, 14)
    plt.title(f"LCC (t-1=before)\n{s2_date_before}", fontsize=15)
    plt.imshow(lcc_before, vmin=0, vmax=len(S2_LAND_COVER_CLASSIFICATIONS), cmap=CMAP, interpolation="nearest")
    grid16()
    cbar = plt.colorbar(ticks=np.arange(0.5, len(S2_LAND_COVER_CLASSIFICATIONS), 1))
    cbar.set_label("Land Cover Classification", fontsize=15)
    cbar.set_ticklabels(S2_LAND_COVER_CLASSIFICATIONS, fontsize=15)

    # Plot LCC main
    plt.subplot(3, 7, 21)
    plt.title(f"LCC (t=main)\n{s2_date_main}", fontsize=15)
    plt.imshow(lcc_main, vmin=0, vmax=len(S2_LAND_COVER_CLASSIFICATIONS), cmap=CMAP, interpolation="nearest")
    grid16()
    cbar = plt.colorbar(ticks=np.arange(0.5, len(S2_LAND_COVER_CLASSIFICATIONS), 1))
    cbar.set_label("Land Cover Classification")
    cbar.set_ticklabels(S2_LAND_COVER_CLASSIFICATIONS, fontsize=15)

    plt.tight_layout()

    return fig


def diff_plots(  # (too-many-arguments, too-many-statements)
    rgb_main: torch.Tensor,
    ratio_main: torch.Tensor,
    ratio_before: torch.Tensor,
    ratio_earlier: torch.Tensor,
    ground_truth_mask: torch.Tensor,
    predicted_mask: torch.Tensor,
    **kwargs: Any,
) -> plt:
    """Return matplotlib figure with RGB, frac, ratio, and ground truth."""
    fps_plot = ((predicted_mask == 1) & (ground_truth_mask == 0)).to(torch.uint8)
    fp = fps_plot.sum()
    fns_plot = ((predicted_mask == 0) & (ground_truth_mask == 1)).to(torch.uint8)
    fn = fns_plot.sum()
    tps_plot = ((predicted_mask == 1) & (ground_truth_mask == 1)).to(torch.uint8)
    tp = tps_plot.sum()

    alpha = 0.5
    plt.rcParams["figure.constrained_layout.use"] = True
    fig = plt.figure(figsize=(20, 5))

    #########################
    # Column 1: Plot RGB
    #########################
    # Plot RGB with ground truth mask
    plt.subplot(1, 4, 1)
    plt.title("RGB (ground truth mask)")
    plt.imshow(
        (rgb_main.numpy() / 0.35 * 255).clip(0, 255).astype(np.uint8), vmin=0.0, vmax=1.0, interpolation="nearest"
    )
    plt.imshow(np.ma.masked_where(ground_truth_mask == 0, ground_truth_mask), cmap="autumn", alpha=alpha)
    grid16()

    # Plot ratio diff
    ratio_diff = ratio_main - (ratio_before + ratio_earlier) / 2
    plt.subplot(1, 4, 2)
    plt.title(
        f"""Ratio Diff (main - reference) 
        Min {ratio_diff.min():.3f}, Max {ratio_diff.max():.3f}, Mean {ratio_diff.mean():.3f}""",
    )
    plt.imshow(ratio_diff, vmin=-0.2, vmax=0.2, interpolation="nearest")
    plt.colorbar()
    grid16()

    # Plot ratio diff with ground truth mask
    plt.subplot(1, 4, 3)
    plt.title("Ratio Diff (ground truth mask)")
    plt.imshow(ratio_diff, vmin=-0.2, vmax=0.2, interpolation="nearest")
    plt.colorbar()
    plt.imshow(np.ma.masked_where(ground_truth_mask == 0, ground_truth_mask), cmap="autumn", alpha=alpha)
    grid16()

    # Plot ratio ratio diff with prediction mask
    plt.subplot(1, 4, 4)
    plt.title(
        f"""Ratio Diff (prediction mask)
              FPs: {fp:.0f}, FNs: {fn:.0f}, TPs: {tp:.0f}""",
    )
    plt.imshow(ratio_diff, vmin=-0.2, vmax=0.2, interpolation="nearest")
    plt.colorbar()
    plt.imshow(np.ma.masked_where(predicted_mask == 0, predicted_mask), cmap="autumn", alpha=alpha)
    grid16()

    return fig


def prep_predictions_for_plot(
    model: nn.Module,
    dataset: Dataset,
    index: tuple[int, int],
    lossFn: TwoPartLoss,
    probability_threshold: float,
) -> dict[str, Any]:
    """Prepare data for plotting predictions."""
    frac_threshold = lossFn.binary_threshold

    dataset_index, item_index = index
    metadata = dataset.get_metadata(dataset_index, ["main_and_reference_ids"])
    untransformed_inputs, untransformed_targets = dataset.get_untransformed_data(dataset_index)
    X, y = dataset.transform((untransformed_inputs, untransformed_targets))

    # move data to device of the model
    device = next(model.parameters()).device
    X = X.to(device)

    with torch.no_grad():
        pred = model(X)
        marginal_pred, binary_probability, conditional_pred, _ = lossFn.get_prediction_parts(pred)

    # Get prediction
    predicted_frac = marginal_pred[item_index, ...].squeeze().cpu()
    binary_probability = binary_probability[item_index, ...].squeeze().cpu()
    prediction_mask = binary_probability >= probability_threshold
    conditional_pred = conditional_pred[item_index, ...].squeeze().cpu()

    # Get ground truth
    target_frac = y[item_index, ...].squeeze()
    ground_truth_mask = torch.abs(target_frac) > frac_threshold

    s2_ids = metadata["main_and_reference_ids"][item_index]

    lcc_earlier = untransformed_inputs["crop_earlier"][item_index, -1]
    lcc_before = untransformed_inputs["crop_before"][item_index, -1]
    lcc_main = untransformed_inputs["crop_main"][item_index, -1]

    rgb_earlier = get_rgb_from_tensor(untransformed_inputs["crop_earlier"], S2_BANDS, item_index)
    rgb_before = get_rgb_from_tensor(untransformed_inputs["crop_before"], S2_BANDS, item_index)
    rgb_main = get_rgb_from_tensor(untransformed_inputs["crop_main"], S2_BANDS, item_index)

    ratio_earlier = get_band_ratio_from_tensor(untransformed_inputs["crop_earlier"], S2_BANDS, item_index)
    ratio_before = get_band_ratio_from_tensor(untransformed_inputs["crop_before"], S2_BANDS, item_index)
    ratio_main = get_band_ratio_from_tensor(untransformed_inputs["crop_main"], S2_BANDS, item_index)

    return {
        "s2_ids": s2_ids,
        "lcc_earlier": lcc_earlier,
        "lcc_before": lcc_before,
        "lcc_main": lcc_main,
        "rgb_earlier": rgb_earlier,
        "rgb_before": rgb_before,
        "rgb_main": rgb_main,
        "ratio_earlier": ratio_earlier,
        "ratio_before": ratio_before,
        "ratio_main": ratio_main,
        "target_frac": target_frac,
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
        help="The model with version to apply, for example 'models:/torchgeo_pwr_unet/1226'.",
        required=True,
        type=str,
    )
    parser.add_argument("--validation_dataset_uri", required=True, type=str)
    parser.add_argument("--valdata_rowgroup_path", required=True, type=str)
    parser.add_argument("--num_workers", required=False, default=1, type=int)
    parser.add_argument("--num_crops", required=False, default=15, type=int, help="Number of crops to create plots for")
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
    # This makes MLFLOW not hang 2mins when trying to log artifacts
    os.environ["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "10"
    os.environ["MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT"] = "10"
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "10"

    args = parse_args()

    config = omegaconf.OmegaConf.load(args.config)

    # Run distributed validation
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
            args.model_name,
            args.validation_dataset_uri,
            args.valdata_rowgroup_path,
            args.num_workers,
            args.num_crops,
            args.probability_threshold,
            SORTING_METRICS,
            config,
        ),
        nprocs=world_size,  # this is what creates the `rank` parameter.  It is passed as the first argument
    )
