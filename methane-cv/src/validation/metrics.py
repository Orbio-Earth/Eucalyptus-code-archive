"""Validation metrics for training.

The Metrics() class is designed for accumulating metrics
as crops / tiles are processed.
"""

from __future__ import annotations  # needed to reference self as a type annotation

import logging
import math
from enum import Enum
from typing import Any, cast

import numpy as np
import torch

NUM_DIMENSIONS = 4

logger = logging.getLogger(__name__)


class FalseMetrics(str, Enum):
    """Metrics where smaller is better."""

    FALSE_NEGATIVES = "false_negatives"
    FALSE_POSITIVES = "false_positives"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    FALSE_NEGATIVE_RATE = "false_negative_rate"
    AVERAGE_COMBINED_LOSS = "average_combined_loss"
    AVERAGE_BINARY_LOSS = "average_binary_loss"
    AVERAGE_CONDITIONAL_LOSS = "average_conditional_loss"
    MEAN_SQUARED_ERROR = "mean_squared_error"
    SIGNAL2NOISE_RATIO = "signal2noise_ratio"  # signal (frac) is typically negative


class TrueMetrics(str, Enum):
    """Metrics where larger is better."""

    TRUE_NEGATIVES = "true_negatives"
    TRUE_POSITIVES = "true_positives"
    TRUE_POSITIVE_RATE = "true_positive_rate"
    TRUE_NEGATIVE_RATE = "true_negative_rate"
    RECALL = "recall"
    PRECISION = "precision"
    F1_SCORE = "f1_score"


class Metrics:
    """Calculate and store validation metrics.

    Metrics for chunked or large datasets can be incrementally calculated by calling `update(...)`
    before calling final metrics.

    Example
    -------
    ```python
    predictions = [
        torch.tensor([[1.0, 0.0], [0.5, 0.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    ]
    probabilities = [
        torch.tensor([[0.9, 0.1], [0.5, 0.2]]),
        torch.tensor([[0.9, 0.1], [0.8, 0.2]]),
    ]
    targets = [
        torch.tensor([[1.0, 0.0], [0.0, 1.5]]),
        torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
    ]
    metrics = Metrics(
        frac_threshold=0.001, binary_probability=0.5
    )

    for prediction, probability, target in list(
        zip(predictions, probabilities, targets)
    ):
        metrics.update(prediction, probability, target)

    false_positive_rate = metrics.fpr
    true_positive_rate = metrics.tpr
    f1_score = metrics.f1_score
    ```

    ```python
    from src.training.loss_functions import TwoPartLoss

    dataloader = <DataLoader>
    probability_threshold = 0.5
    lossFn = TwoPartLoss(binary_threshold=0.001, MSE_multiplier=1000.0)
    metrics = Metrics(lossFn.binary_threshold, probability_threshold)

    for x, y in dataloader:
        pred = model(x)
        pred_parts = lossFn.get_prediction_parts_as_dict(pred)
        metrics.update(pred_parts["marginal_pred"], pred_parts["binary_probability"], y)

    metrics.as_dict()
    ```
    """

    def __init__(
        self,
        frac_threshold: float,
        probability_threshold: float,
        manual_target_mask: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize metrics.

        Parameters
        ----------
            frac_threshold: the background frac threshold a pixel needs to be greater than to be considered methane.
            probability_threshold: the probability threshold a pixel needs to be greater than to be considered methane.
            manual_target_mask: if True, frac_threshold is not applied to the target mask.
        """
        self.frac_threshold = frac_threshold
        self.probability_threshold = probability_threshold
        self.manual_target_mask = manual_target_mask
        self.verbose = verbose
        self.metadata: dict = {}

        self._total_pixels = 0
        self._total_samples = 0
        self._total_negatives = 0
        self._total_positives = 0
        self.tn = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.squared_error: float | None = 0.0

        # Initialize losses to None to explicitly indicate "not calculated"
        self.total_combined_loss: float | None = None
        self.total_binary_loss: float | None = None
        self.total_conditional_loss: float | None = None

        # S/N metric
        self.methane_pixels_count = 0
        self.non_methane_pixels_count = 0
        self.methane_mean = torch.nan
        self.non_methane_mean = torch.nan
        self.methane_var = torch.nan
        self.non_methane_var = torch.nan

    def __add__(self, obj: Metrics) -> Metrics:
        """Add two Metrics together."""
        assert self.frac_threshold == obj.frac_threshold, f"{self.frac_threshold} != {obj.frac_threshold}"
        assert (
            self.probability_threshold == obj.probability_threshold
        ), f"{self.probability_threshold} != {obj.probability_threshold}"

        metrics = Metrics(self.frac_threshold, self.probability_threshold)
        metrics._total_pixels = self._total_pixels + obj._total_pixels
        metrics._total_samples = self._total_samples + obj._total_samples
        metrics._total_negatives = self._total_negatives + obj._total_negatives
        metrics._total_positives = self._total_positives + obj._total_positives
        metrics.tn = self.tn + obj.tn
        metrics.tp = self.tp + obj.tp
        metrics.fp = self.fp + obj.fp
        metrics.fn = self.fn + obj.fn
        if self.squared_error is None or obj.squared_error is None:
            metrics.squared_error = None
        else:
            metrics.squared_error = self.squared_error + obj.squared_error

        # Signal2Noise
        metrics.methane_pixels_count = self.methane_pixels_count + obj.methane_pixels_count
        metrics.non_methane_pixels_count = self.non_methane_pixels_count + obj.non_methane_pixels_count

        metrics.methane_mean = self._combine_batch_means(
            self.methane_pixels_count, obj.methane_pixels_count, self.methane_mean, obj.methane_mean
        )
        metrics.non_methane_mean = self._combine_batch_means(
            self.non_methane_pixels_count,
            obj.non_methane_pixels_count,
            self.non_methane_mean,
            obj.non_methane_mean,
        )

        metrics.non_methane_var = self._combine_batch_variances(
            self.non_methane_pixels_count,
            obj.non_methane_pixels_count,
            self.non_methane_mean,
            obj.non_methane_mean,
            self.non_methane_var,
            obj.non_methane_var,
        )
        metrics.methane_var = self._combine_batch_variances(
            self.methane_pixels_count,
            obj.methane_pixels_count,
            self.methane_mean,
            obj.methane_mean,
            self.methane_var,
            obj.methane_var,
        )

        # Handle None in loss sums
        metrics.total_combined_loss = self._add_loss_values(self.total_combined_loss, obj.total_combined_loss)
        metrics.total_binary_loss = self._add_loss_values(self.total_binary_loss, obj.total_binary_loss)
        metrics.total_conditional_loss = self._add_loss_values(self.total_conditional_loss, obj.total_conditional_loss)

        return metrics

    def update(
        self,
        prediction: torch.Tensor | np.ndarray,
        binary_probability: torch.Tensor | np.ndarray,
        target: torch.Tensor | np.ndarray,
        combined_loss: float | None = None,
        binary_loss: float | None = None,
        conditional_loss: float | None = None,
    ) -> Metrics:
        """Intake more data, calculate TP / FP / TN / FN, squared errors and update metrics with the new batch.

        Tensors are expected to have NUM_DIMENSIONS (4) where: (batch_size, classes / values, height, width)

        Parameters
        ----------
            prediction: array of predicted values (marginal prediction).
            binary_probability: array of binary probabilities that each pixel is methane.
            target: array of ground truth values.
            combined_loss
            binary_loss
            conditional_loss
        """
        if isinstance(prediction, np.ndarray):
            prediction = torch.from_numpy(prediction)

        if isinstance(binary_probability, np.ndarray):
            binary_probability = torch.from_numpy(binary_probability)

        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)

        assert (
            prediction.ndim == NUM_DIMENSIONS
        ), f"Dimension mismatch: prediction has {prediction.ndim} dimensions, expect {NUM_DIMENSIONS} dimensions."

        assert (
            binary_probability.ndim == NUM_DIMENSIONS
        ), f"Dimension mismatch: binary_probability has {binary_probability.ndim} dimensions, expect {NUM_DIMENSIONS} dimensions."  # noqa: E501 (line-too-long)

        assert (
            target.ndim == NUM_DIMENSIONS
        ), f"Dimension mismatch: target has {target.ndim} dimensions, expect {NUM_DIMENSIONS} dimensions."

        assert (
            prediction.shape == binary_probability.shape == target.shape
        ), f"Dimension mismatch: prediction {prediction.shape}, binary_probability {binary_probability.shape}, target {target.ndim}."  # noqa: E501 (line-too-long)

        self._total_pixels += self._num_pixels(prediction)
        self._total_samples += prediction.shape[0]  # shape is (batch_size, classes, height, width)

        # Binary Metrics
        pred_mask = binary_probability >= self.probability_threshold
        target_mask = target.bool() if self.manual_target_mask else target.abs() > self.frac_threshold

        self._total_negatives += self._negatives(target_mask)
        self._total_positives += self._positives(target_mask)
        self.tn += self._true_negatives(pred_mask, target_mask)
        self.tp += self._true_positives(pred_mask, target_mask)
        self.fp += self._false_positives(pred_mask, target_mask)
        self.fn += self._false_negatives(pred_mask, target_mask)
        # Update loss values, handling None
        if combined_loss is not None:
            self.total_combined_loss = (self.total_combined_loss or 0) + combined_loss
        if binary_loss is not None:
            self.total_binary_loss = (self.total_binary_loss or 0) + binary_loss
        if conditional_loss is not None:
            self.total_conditional_loss = (self.total_conditional_loss or 0) + conditional_loss

        assert self._total_negatives == (self.tn + self.fp)
        assert self._total_positives == (self.tp + self.fn)

        # Continuous Metrics
        if (
            not self.manual_target_mask and self.squared_error is not None
        ):  # NOTE: both checks to avoid type hint error, 2nd check can be thought as redundant
            self.squared_error += self._squared_error(prediction, target)
        else:
            self.squared_error = None

        # Signal2Noise
        (
            self.methane_pixels_count,
            self.non_methane_pixels_count,
            self.methane_mean,
            self.non_methane_mean,
            self.methane_var,
            self.non_methane_var,
        ) = self._update_signal2noise_ratio_statistics(prediction, target_mask)

        return self

    def add_metadata(self, metadata: dict) -> Metrics:
        """Add extra information to the metrics.

        When tracking metrics per crop, we want to be able to track additional information about
        what the metric was calculated for, e.g. the specific crop or dataset uri.  This additional
        information dictionary is included in the dictionary when calling `metrics.as_dict()`
        """
        self.metadata = metadata
        return self

    @staticmethod
    def _add_loss_values(a: float | None, b: float | None) -> float | None:
        """Add loss values, handling None."""
        if a is None and b is None:
            return None
        elif a is None:
            return b
        elif b is None:
            return a
        else:
            return a + b

    @property
    def average_combined_loss(self) -> float:
        """Calculate and return the average combined loss per pixel for the entire dataset."""
        if self.total_combined_loss is None:
            if self.verbose:
                logger.warning(
                    "Average combined loss cannot be calculated — no combined loss values have been recorded yet."
                )
            return torch.nan
        return self.total_combined_loss / self._total_samples

    @property
    def average_binary_loss(self) -> float:
        """Calculate and return the average binary loss per pixel for the entire dataset."""
        if self.total_binary_loss is None:
            if self.verbose:
                logger.warning(
                    "Average binary loss cannot be calculated — no binary loss values have been recorded yet."
                )
            return torch.nan
        return self.total_binary_loss / self._total_samples

    @property
    def average_conditional_loss(self) -> float:
        """Calculate and return the average binary loss per pixel for the entire dataset."""
        if self.total_conditional_loss is None:
            if self.verbose:
                logger.warning(
                    "Average conditional loss cannot be calculated — no conditional loss values have been recorded yet."
                )
            return torch.nan
        return self.total_conditional_loss / self._total_samples

    def _num_pixels(self, crop: torch.Tensor) -> int:
        """Return the number of pixels in the crop."""
        assert crop.ndim == NUM_DIMENSIONS, f"Expected a {NUM_DIMENSIONS}D tensor, but got shape {crop.shape}"
        batch_size, _, height, width = crop.shape
        return batch_size * height * width

    @property
    def total_samples(self) -> int:
        """Return the total number of samples in the dataset."""
        return self._total_samples

    @property
    def total_pixels(self) -> int:
        """Return the total number of pixels in the dataset."""
        return self._total_pixels

    def _squared_error(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate the suqared error."""
        return ((target - prediction) ** 2).sum().item()

    @property
    def mean_squared_error(self) -> float | None:
        """Calculate Mean Squared Error for the dataset."""
        if self.squared_error is None:
            return None
        return self.squared_error / self._total_samples

    @property
    def total_positives(self) -> int:
        """Return total number of positives in target."""
        return self._total_positives

    @property
    def total_negatives(self) -> int:
        """Return total number of negatives in target."""
        return self._total_negatives

    def _positives(self, target: torch.Tensor) -> int:
        """Return the number of pixels with methane.

        Parameters
        ----------
            target: 4D array of ground truth values (boolean).
        """
        positives = torch.sum(target).item()
        return cast(int, positives)

    def _negatives(self, target: torch.Tensor) -> int:
        """Return the number of pixels with no methane.

        Parameters
        ----------
            target: 4D array of ground truth values (boolean).
        """
        negatives = torch.sum(~target).item()
        return cast(int, negatives)

    def _true_positives(self, prediction: torch.Tensor, target: torch.Tensor) -> int:
        """Return the number of predicted pixels to have methane, that are targeted as having methane.

        Parameters
        ----------
            prediction: 4D array of predicted values (boolean).
            target: 4D array of ground truth values (boolean).
        """
        tp = torch.sum((prediction) & (target)).item()
        return cast(int, tp)

    def _true_negatives(self, prediction: torch.Tensor, target: torch.Tensor) -> int:
        """Return the number of predicted pixels to not have methane, that are targeted as not having methane.

        Parameters
        ----------
            prediction: 4D array of predicted values (boolean).
            target: 4D array of ground truth values (boolean).
        """
        tn = torch.sum((~prediction) & (~target)).item()
        return cast(int, tn)

    def _false_positives(self, prediction: torch.Tensor, target: torch.Tensor) -> int:
        """Return the number of predicted pixels to have methane, that are targeted as not having methane.

        Parameters
        ----------
            prediction: 4D array of predicted values (boolean).
            target: 4D array of ground truth values (boolean).
        """
        fp = torch.sum((prediction) & (~target)).item()
        return cast(int, fp)

    def _false_negatives(self, prediction: torch.Tensor, target: torch.Tensor) -> int:
        """Return the number of predicted pixels to not have methane, that are targeted as having methane.

        Parameters
        ----------
            prediction: 4D array of predicted values (boolean).
            target: 4D array of ground truth values (boolean).
        """
        fn = torch.sum((~prediction) & (target)).item()
        return cast(int, fn)

    @property
    def true_positives(self) -> int:
        """Return True Positives."""
        return self.tp

    @property
    def true_negatives(self) -> int:
        """Return True Negatives."""
        return self.tn

    @property
    def false_positives(self) -> int:
        """Return False Positives."""
        return self.fp

    @property
    def false_negatives(self) -> int:
        """Return False Negatives."""
        return self.fn

    @property
    def false_positive_rate(self) -> float:
        """Calculate the false positive rate (Type I Error)."""
        if self._total_negatives == 0:
            if self.verbose:
                logger.warning("False Positive Rate cannot be calculated - cannot divide by 0 total_negatives.")
            return torch.nan

        return self.fp / self._total_negatives

    @property
    def false_negative_rate(self) -> float:
        """Calculate the false negative rate (Type II Error)."""
        if self._total_positives == 0:
            if self.verbose:
                logger.warning("False Negative Rate cannot be calculated - cannot divide by 0 total_positives.")
            return torch.nan

        return self.fn / self._total_positives

    @property
    def true_negative_rate(self) -> float:
        """Calculate the true negative rate (Specificity).

        How many negative selected elements are truly negative?
        e.g. How many non-methane pixels are identified as not having methane?
        """
        if self._total_negatives == 0:
            if self.verbose:
                logger.warning("True Negative Rate cannot be calculated - cannot divide by 0 total_negatives.")
            return torch.nan

        return self.tn / self._total_negatives

    @property
    def true_positive_rate(self) -> float:
        """Calculate the true positive rate (Sensitivity).

        How many relevent items are selected?
        e.g. how many methane pixels are correctly identified as having methane?
        """
        if self._total_positives == 0:
            if self.verbose:
                logger.warning("True Positive Rate cannot be calculated - cannot divide by 0 total_positives.")
            return torch.nan

        return self.tp / self._total_positives

    @property
    def precision(self) -> float:
        """Calculate the precision, the fraction of relevant pixels among the retrieved pixels."""
        if (self.tp == 0) and (self.fp == 0):
            if self.verbose:
                logger.warning("Precision cannot be calculated - no pixels predicted positive.")
            return torch.nan

        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        """Calculate the recall, the fraction of relevant pixels that were retrieved."""
        return self.true_positive_rate

    @property
    def f1_score(self) -> float:
        """Calculate the F1 Score (harmonic mean of precision and recall)."""
        if (self.tp == 0) and (self.fp == 0) and (self.fn == 0):
            if self.verbose:
                logger.warning(
                    "F1 Score cannot be calculated - all pixels are negative and all pixels are predicted negative."
                )
            return torch.nan

        return (2 * self.tp) / (2 * self.tp + self.fp + self.fn)

    def as_dict(self, json: bool = False) -> dict[str, Any]:
        """Return all metrics for the dataset.

        json: if True will convert NaN values to None so the dict is JSON serializable
        """
        metrics: dict[str, Any] = {
            "total_samples": self.total_samples,
            "total_pixels": self.total_pixels,
            "total_positives": self.total_positives,
            "total_negatives": self.total_negatives,
            "mean_squared_error": self.mean_squared_error,
            "true_positives": self.tp,
            "true_negatives": self.tn,
            "false_positives": self.fp,
            "false_negatives": self.fn,
            "true_positive_rate": self.true_positive_rate,
            "true_negative_rate": self.true_negative_rate,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "recall": self.recall,
            "precision": self.precision,
            "f1_score": self.f1_score,
            "average_combined_loss": self.average_combined_loss,
            "average_binary_loss": self.average_binary_loss,
            "average_conditional_loss": self.average_conditional_loss,
            "signal2noise_ratio": self.signal2noise_ratio,
        }

        # serialize NaNs into None for JSON
        metrics.update(self.metadata)

        if json:
            metrics = {k: v if not math.isnan(v) else None for k, v in metrics.items()}

        return metrics

    @staticmethod
    def _combine_batch_means(a: int, b: int, mean_a: float, mean_b: float) -> float:
        """Combine two batch means (batch A and batch B).

        The updated mean is a linear combination of the mean over the two batches.
        - https://git.orbio.earth/orbio/orbio/-/merge_requests/211

        Parameters
        ----------
            a (int): number of samples in batch A
            b (int): number of samples in batch B
            mean_a (float): mean of batch A
            mean_b (float): mean of batch B
        """
        if a == 0:
            assert math.isnan(
                mean_a
            ), f"If batch samples are 0 then sample mean should be NaN: mean_a={mean_a} ({type(mean_a)})"
        if b == 0:
            assert math.isnan(
                mean_b
            ), f"If batch samples are 0 then sample mean should be NaN: mean_b={mean_b} ({type(mean_b)})"

        if math.isnan(mean_a):
            return mean_b
        if math.isnan(mean_b):
            return mean_a

        return ((a / (a + b)) * mean_a) + ((b / (a + b)) * mean_b)

    @staticmethod
    def _combine_batch_variances(a: int, b: int, mean_a: float, mean_b: float, var_a: float, var_b: float) -> float:
        """Combine two batch means (batch A and batch B).

        The update is a linear combination of the observed variances plus a correction by the means.
        - https://git.orbio.earth/orbio/orbio/-/merge_requests/211

        !!! note "On variance"
            We calculate the _sample_ variance because we are only evaluating metrics on a sample of the population.

        Parameters
        ----------
            a (int): number of samples in batch A
            b (int): number of samples in batch B
            mean_a (float): mean of batch A
            mean_b (float): mean of batch B
            var_a (float): variance of batch A
            var_b (float): variance of batch B
        """
        if a == 0:
            assert math.isnan(
                mean_a
            ), f"If batch samples are 0 then sample mean should be NaN: mean_a={mean_a} ({type(mean_a)})"
        if b == 0:
            assert math.isnan(
                mean_b
            ), f"If batch samples are 0 then sample mean should be NaN: mean_b={mean_b} ({type(mean_b)})"

        # min_samples_for_sample_variance = 2
        min_samples_for_population_variance = 1
        if a < min_samples_for_population_variance:
            assert math.isnan(
                var_a
            ), f"If batch samples are 0 then sample variance should be NaN: var_a={var_a} ({type(var_a)})"
        if b < min_samples_for_population_variance:
            assert math.isnan(
                var_b
            ), f"If batch samples are 0 then sample variance should be NaN: var_b={var_b} ({type(var_b)})"

        if (a + b) < min_samples_for_population_variance:
            return torch.nan

        if math.isnan(var_a):
            var_a = 0.0
        if math.isnan(var_b):
            var_b = 0.0

        if math.isnan(mean_a):
            mean_a = 0.0
        if math.isnan(mean_b):
            mean_b = 0.0

        within_group_contribution = (a / (a + b)) * var_a + (b / (b + a)) * var_b
        between_group_contribution = (a * b) / (a + b) ** 2 * (mean_a - mean_b) ** 2
        return within_group_contribution + between_group_contribution

    def _methane_pixels(self, prediction: torch.Tensor, methane_mask: torch.Tensor) -> torch.Tensor:
        """Get the pixels from the prediction masked by the methane pixels according to the target."""
        mask = methane_mask.int() == 1
        return prediction[mask.nonzero(as_tuple=True)]

    def _non_methane_pixels(self, prediction: torch.Tensor, methane_mask: torch.Tensor) -> torch.Tensor:
        """Get the pixels from the prediction masked by the non-methane pixels according to the target."""
        mask = methane_mask.int() == 0
        return prediction[mask.nonzero(as_tuple=True)]

    def _update_signal2noise_ratio_statistics(
        self, prediction: torch.Tensor, methane_mask: torch.Tensor
    ) -> tuple[int, int, float, float, float, float]:
        methane_pixels = self._methane_pixels(prediction, methane_mask)
        non_methane_pixels = self._non_methane_pixels(prediction, methane_mask)

        methane_mean = self._combine_batch_means(
            self.methane_pixels_count,
            methane_pixels.numel(),
            self.methane_mean,
            methane_pixels.mean().item(),
        )

        non_methane_mean = self._combine_batch_means(
            self.non_methane_pixels_count,
            non_methane_pixels.numel(),
            self.non_methane_mean,
            non_methane_pixels.mean().item(),
        )

        methane_var = self._combine_batch_variances(
            self.methane_pixels_count,
            methane_pixels.numel(),
            self.methane_mean,
            methane_pixels.mean().item(),
            self.methane_var,
            methane_pixels.var(correction=0).item(),
        )

        non_methane_var = self._combine_batch_variances(
            self.non_methane_pixels_count,
            non_methane_pixels.numel(),
            self.non_methane_mean,
            non_methane_pixels.mean().item(),
            self.non_methane_var,
            non_methane_pixels.var(correction=0).item(),
        )

        return (
            self.methane_pixels_count + methane_pixels.numel(),
            self.non_methane_pixels_count + non_methane_pixels.numel(),
            methane_mean,
            non_methane_mean,
            methane_var,
            non_methane_var,
        )

    @property
    def signal2noise_ratio(self) -> float:
        """Return the signal to noise metric of frac in pixels, (mean of methane pixels / std dev non methane pixels).

        Methane and non-methane pixels are determined from the target mask which is applied to the prediction
        to find the average predicted frac of true methane pixels to std dev frac of true non-methane pixels.

        If variance of non-methane frac in pixels is 0 then return NaN, because we can't divide by zero
        """
        if self.non_methane_var == 0:
            if self.verbose:
                logger.warning("Variance of non-methane frac in pixels in 0, cannot divide by zero.")
            return torch.nan
        return self.methane_mean / self.non_methane_var**0.5
