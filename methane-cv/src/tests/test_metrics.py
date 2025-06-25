"""Unit tests for Metrics()."""

import logging

import numpy as np
import pytest
import torch
from _pytest.python_api import RaisesContext

from src.validation.metrics import Metrics

logger = logging.getLogger(__name__)
FRAC_THRESHOLD = 0.001
PROBABILITY_THRESHOLD = 0.5


@pytest.fixture(scope="module")
def frac_threshold() -> float:
    """Fixture for the frac_threshold."""
    return FRAC_THRESHOLD


@pytest.fixture(scope="module")
def probability_threshold() -> float:
    """Fixture for the probability_threshold."""
    return PROBABILITY_THRESHOLD


@pytest.mark.parametrize(
    "prediction,probability,label,expectation",
    [
        pytest.param(
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            (pytest.raises(AssertionError)),
            id="Prediction has incorrect dimensions",
        ),
        pytest.param(
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            (pytest.raises(AssertionError)),
            id="Probability has incorrect dimensions",
        ),
        pytest.param(
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            (pytest.raises(AssertionError)),
            id="Label has incorrect dimensions",
        ),
        pytest.param(
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            (pytest.raises(AssertionError)),
            id="Different shapes",
        ),
    ],
)
def test_shape_mismatch_error(
    prediction: torch.Tensor | np.ndarray,
    probability: torch.Tensor | np.ndarray,
    label: torch.Tensor | np.ndarray,
    expectation: RaisesContext[AssertionError],
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that AssertionErrors are raised if dimensions or shapes don't match."""
    with expectation:
        _ = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)


@pytest.mark.parametrize(
    "prediction,probability,label,total_pixels,should_fail",
    [
        pytest.param(
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            8,
            False,
            id="torch 4D correct",
        ),
        pytest.param(
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            4,
            True,
            id="torch 4D incorrect",
        ),
        pytest.param(
            np.array([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            np.array([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            np.array([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            8,
            False,
            id="numpy 4D correct",
        ),
        pytest.param(
            np.array([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            np.array([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            np.array([[[[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            4,
            True,
            id="numpy 4D incorrect",
        ),
    ],
)
def test_total_pixels(
    prediction: torch.Tensor | np.ndarray,
    probability: torch.Tensor | np.ndarray,
    label: torch.Tensor | np.ndarray,
    total_pixels: int,
    should_fail: bool,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the total pixels are calculated correctly."""
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    if should_fail:
        assert metrics.total_pixels != total_pixels, "Expected mismatch in total pixels"
    else:
        assert metrics.total_pixels == total_pixels


@pytest.mark.parametrize(
    "prediction,probability,label,mean_squared_error",
    [
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            0.0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            0.0,
        ),
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            4.0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            4.0,
        ),
        (
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]]),
            2.5,
        ),
    ],
)
def test_mean_squared_error(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
    mean_squared_error: float,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the mean squared error is calculated correctly."""
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    assert metrics.mean_squared_error == mean_squared_error


@pytest.mark.parametrize(
    "prediction,probability,label,total_positives",
    [
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            4,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            0,
        ),
    ],
)
def test_total_positives(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
    total_positives: int,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the total positives are calculated correctly."""
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    assert metrics.total_positives == total_positives


@pytest.mark.parametrize(
    "prediction,probability,label,total_negatives",
    [
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            4,
        ),
    ],
)
def test_total_negatives(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
    total_negatives: int,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the total negatives are calculated correctly."""
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    assert metrics.total_negatives == total_negatives


@pytest.mark.parametrize(
    "prediction,probability,label,true_positives",
    [
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            4,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            0,
        ),
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            0,
        ),
        (
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]]),
            1,
        ),
    ],
)
def test_true_positives(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
    true_positives: int,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the true positives are calculated correctly."""
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    assert metrics.true_positives == true_positives


@pytest.mark.parametrize(
    "prediction,probability,label,true_negatives",
    [
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            4,
        ),
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            0,
        ),
        (
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]]),
            1,
        ),
    ],
)
def test_true_negatives(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
    true_negatives: int,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the true negatives are calculated correctly."""
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    assert metrics.true_negatives == true_negatives


@pytest.mark.parametrize(
    "prediction,probability,label,false_positives",
    [
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            0,
        ),
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            4,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            0,
        ),
        (
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]]),
            1,
        ),
    ],
)
def test_false_positives(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
    false_positives: int,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the false positives are calculated correctly."""
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    assert metrics.false_positives == false_positives


@pytest.mark.parametrize(
    "prediction,probability,label,false_negatives",
    [
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            0,
        ),
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            4,
        ),
        (
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]]),
            1,
        ),
    ],
)
def test_false_negatives(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
    false_negatives: int,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the false negatives are calculated correctly."""
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    assert metrics.false_negatives == false_negatives


@pytest.mark.parametrize(
    "prediction,probability,label,true_positive_rate",
    [
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            1.0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            0.0,
        ),
        (
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]]),
            0.5,
        ),
    ],
)
def test_true_positive_rate(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
    true_positive_rate: float,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the true positive rate is calculated correctly."""
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    torch.testing.assert_close(metrics.true_positive_rate, true_positive_rate, equal_nan=True, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "prediction,probability,label,true_negative_rate",
    [
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            1.0,
        ),
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            0.0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]]),
            0.5,
        ),
    ],
)
def test_true_negative_rate(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
    true_negative_rate: float,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the true negative rate is calculated correctly."""
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    torch.testing.assert_close(metrics.true_negative_rate, true_negative_rate, equal_nan=True, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "prediction,probability,label,false_positive_rate",
    [
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            0.0,
        ),
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            1.0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]]),
            0.5,
        ),
    ],
)
def test_false_positive_rate(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
    false_positive_rate: float,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the false positive rate is calculated correctly."""
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    torch.testing.assert_close(metrics.false_positive_rate, false_positive_rate, equal_nan=True, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "prediction,probability,label,false_negative_rate",
    [
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            0.0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            1.0,
        ),
        (
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]]),
            0.5,
        ),
    ],
)
def test_false_negative_rate(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
    false_negative_rate: float,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the false negative rate is calculated correctly."""
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    torch.testing.assert_close(metrics.false_negative_rate, false_negative_rate, equal_nan=True, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "prediction,probability,label,recall",
    [
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            1.0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            0.0,
        ),
        (
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]]),
            0.5,
        ),
    ],
)
def test_recall(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
    recall: float,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the recall is calculated correctly."""
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    torch.testing.assert_close(metrics.recall, recall, equal_nan=True, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "prediction,probability,label,precision",
    [
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            1.0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            0.0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]]),
            0.5,
        ),
    ],
)
def test_precision(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
    precision: float,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the precision is calculated correctly."""
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    torch.testing.assert_close(metrics.precision, precision, equal_nan=True, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "prediction,probability,label,f1_score",
    [
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            1.0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            0.0,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            0.0,
        ),
        (
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]]),
            0.5,
        ),
    ],
)
def test_f1_score(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
    f1_score: float,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the F1 score is calculated correctly."""
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    torch.testing.assert_close(metrics.f1_score, f1_score, equal_nan=True, rtol=0.0, atol=0.0)


def test_update(
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test metrics are updated when new data added."""
    prediction = torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]])
    probability = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    label = torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]])
    metrics = (
        Metrics(frac_threshold, probability_threshold)
        .update(prediction, probability, label)
        .update(prediction, probability, label)
    )

    total_pixels = metrics.total_pixels
    total_positives = metrics.total_positives
    total_negatives = metrics.total_negatives
    tp = metrics.true_positives
    tn = metrics.true_negatives
    fp = metrics.false_positives
    fn = metrics.false_negatives
    mse = metrics.mean_squared_error
    assert (total_pixels, total_positives, total_negatives, mse, tp, tn, fp, fn) == (8, 4, 4, 2.5, 2, 2, 2, 2)


def test_add(
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test metrics are updated when Metrics objects are added."""
    prediction = torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]])
    probability = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    label = torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]])
    metrics_a = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)
    metrics_b = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)
    metrics = metrics_a + metrics_b

    expected = (
        Metrics(frac_threshold, probability_threshold)
        .update(prediction, probability, label)
        .update(prediction, probability, label)
    )

    assert (
        expected.total_pixels == metrics.total_pixels
    ), f"total_pixels: {expected.total_pixels} == {metrics.total_pixels}"
    assert (
        expected.total_positives == metrics.total_positives
    ), f"total_positives: {expected.total_positives} == {metrics.total_positives}"
    assert (
        expected.total_negatives == metrics.total_negatives
    ), f"total_negatives: {expected.total_negatives} == {metrics.total_negatives}"
    assert expected.tp == metrics.tp, f"tp: {expected.tp} == {metrics.tp}"
    assert expected.tn == metrics.tn, f"tn: {expected.tn} == {metrics.tn}"
    assert expected.fp == metrics.fp, f"fp: {expected.fp} == {metrics.fp}"
    assert expected.fn == metrics.fn, f"fn: {expected.fn} == {metrics.fn}"
    assert (
        expected.squared_error == metrics.squared_error
    ), f"squared_error: {expected.squared_error} == {metrics.squared_error}"


def test_ladd(
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test metrics are updated when Metrics objects are added."""
    prediction = torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]])
    probability = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    label = torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]])
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)
    metrics += metrics

    expected = (
        Metrics(frac_threshold, probability_threshold)
        .update(prediction, probability, label)
        .update(prediction, probability, label)
    )

    assert (
        expected.total_pixels == metrics.total_pixels
    ), f"total_pixels: {expected.total_pixels} == {metrics.total_pixels}"
    assert (
        expected.total_positives == metrics.total_positives
    ), f"total_positives: {expected.total_positives} == {metrics.total_positives}"
    assert (
        expected.total_negatives == metrics.total_negatives
    ), f"total_negatives: {expected.total_negatives} == {metrics.total_negatives}"
    assert expected.tp == metrics.tp, f"tp: {expected.tp} == {metrics.tp}"
    assert expected.tn == metrics.tn, f"tn: {expected.tn} == {metrics.tn}"
    assert expected.fp == metrics.fp, f"fp: {expected.fp} == {metrics.fp}"
    assert expected.fn == metrics.fn, f"fn: {expected.fn} == {metrics.fn}"
    assert (
        expected.squared_error == metrics.squared_error
    ), f"squared_error: {expected.squared_error} == {metrics.squared_error}"


def test_as_dict(
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the metrics are returned correctly."""
    prediction = torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]])
    probability = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    label = torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]])
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    true_metrics = {
        "total_samples": 1,
        "total_pixels": 4,
        "total_positives": 2,
        "total_negatives": 2,
        "mean_squared_error": 2.5,
        "true_positives": 1,
        "true_negatives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "true_positive_rate": 0.5,
        "true_negative_rate": 0.5,
        "false_positive_rate": 0.5,
        "false_negative_rate": 0.5,
        "recall": 0.5,
        "precision": 0.5,
        "f1_score": 0.5,
        "signal2noise_ratio": 2.0,
        "average_combined_loss": torch.nan,
        "average_binary_loss": torch.nan,
        "average_conditional_loss": torch.nan,
    }

    assert metrics.as_dict() == true_metrics


def test_as_dict_json(
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the metrics are converted to a dict that is JSON serializable."""
    prediction = torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]])
    probability = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    label = torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]])
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    true_metrics = {
        "total_samples": 1,
        "total_pixels": 4,
        "total_positives": 2,
        "total_negatives": 2,
        "mean_squared_error": 2.5,
        "true_positives": 1,
        "true_negatives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "true_positive_rate": 0.5,
        "true_negative_rate": 0.5,
        "false_positive_rate": 0.5,
        "false_negative_rate": 0.5,
        "recall": 0.5,
        "precision": 0.5,
        "f1_score": 0.5,
        "signal2noise_ratio": 2.0,
        "average_combined_loss": None,
        "average_binary_loss": None,
        "average_conditional_loss": None,
    }

    assert metrics.as_dict(json=True) == true_metrics


@pytest.mark.parametrize(
    "batch_a,batch_b",
    [
        (torch.tensor([1.0, 1.0, 1.0, 1.0]), torch.tensor([1.0, 1.0, 1.0, 1.0])),
        (torch.tensor([0.0, 0.0, 0.0, 0.0]), torch.tensor([2.0, 2.0, 2.0, 2.0])),
        (torch.rand(100), torch.rand(10)),
        (torch.tensor([]), torch.tensor([])),
        (torch.tensor([0.0]), torch.tensor([])),
        (torch.tensor([]), torch.tensor([0.0])),
    ],
)
def test_combine_batch_means(batch_a: torch.Tensor, batch_b: torch.Tensor) -> None:
    """Test mean is correctly updated with a new batch."""
    mean = Metrics._combine_batch_means(batch_a.numel(), batch_b.numel(), batch_a.mean().item(), batch_b.mean().item())
    expected = torch.cat((batch_a, batch_b)).mean().item()

    torch.testing.assert_close(mean, expected, equal_nan=True, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize(
    "batch_a,batch_b",
    [
        (torch.tensor([1.0, 1.0, 1.0, 1.0]), torch.tensor([1.0, 1.0, 1.0, 1.0])),
        (torch.tensor([0.0, 0.0, 0.0, 0.0]), torch.tensor([2.0, 2.0, 2.0, 2.0])),
        (torch.tensor([]), torch.tensor([])),
        (torch.tensor([0.0]), torch.tensor([])),
        (torch.tensor([]), torch.tensor([0.0])),
        (torch.tensor([0.0, 0.0]), torch.tensor([])),
        (torch.tensor([1.0, 0.0]), torch.tensor([])),
        (torch.tensor([]), torch.tensor([0.0, 0.0])),
        (torch.tensor([]), torch.tensor([1.0, 0.0])),
        (torch.rand(1), torch.rand(1)),
        (torch.rand(10), torch.rand(10)),
        (torch.rand(100), torch.rand(100)),
        (torch.rand(1_000), torch.rand(1_000)),
        (torch.rand(10_000), torch.rand(10_000)),
        (torch.rand(100_000), torch.rand(100_000)),
        (torch.rand(1_000_000), torch.rand(1_000_000)),
        (torch.rand(100), torch.rand(10)),
    ],
)
def test_combine_batch_variances(batch_a: torch.Tensor, batch_b: torch.Tensor) -> None:
    """Test variance is correctly updated with a new batch."""
    var = Metrics._combine_batch_variances(
        batch_a.numel(),
        batch_b.numel(),
        batch_a.mean().item(),
        batch_b.mean().item(),
        batch_a.var(correction=0).item(),
        batch_b.var(correction=0).item(),
    )
    expected = torch.cat((batch_a, batch_b)).var(correction=0).item()

    torch.testing.assert_close(var, expected, equal_nan=True, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize(
    "prediction,probability,label,signal2noise_ratio",
    [
        (torch.tensor([[[[1.0, 1.0]]]]), torch.tensor([[[[1.0, 1.0]]]]), torch.tensor([[[[1.0, 1.0]]]]), torch.nan),
        (
            torch.tensor([[[[1.0, 0.0, 1.0]]]]),
            torch.tensor([[[[1.0, 0.0, 1.0]]]]),
            torch.tensor([[[[1.0, 0.0, 0.0]]]]),
            2.0,
        ),
        (
            torch.tensor([[[[1.0, 1.0, 0.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0, 0.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]]),
            2.0,
        ),
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.5, 1.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.5, 1.0]]]]),
            torch.tensor([[[[1.0, 0.5], [0.0, 0.0]]]]),
            0.0,
        ),
        (
            torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]),
            1.0,
        ),
    ],
)
def test_signal2noise_ratio(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
    signal2noise_ratio: float,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the signal2noise_ratio is calculated correctly."""
    metrics = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    torch.testing.assert_close(metrics.signal2noise_ratio, signal2noise_ratio, equal_nan=True, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "prediction,probability,label,signal2noise_ratio",
    [
        (torch.tensor([[[[1.0, 1.0]]]]), torch.tensor([[[[1.0, 1.0]]]]), torch.tensor([[[[1.0, 1.0]]]]), torch.nan),
        (
            torch.tensor([[[[1.0, 0.0, 1.0]]]]),
            torch.tensor([[[[1.0, 0.0, 1.0]]]]),
            torch.tensor([[[[1.0, 0.0, 0.0]]]]),
            2.0,
        ),
        (
            torch.tensor([[[[1.0, 1.0, 0.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0, 0.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]]),
            2.0,
        ),
        (
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
            torch.nan,
        ),
        (
            torch.tensor([[[[0.0, 0.0], [0.5, 1.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.5, 1.0]]]]),
            torch.tensor([[[[1.0, 0.5], [0.0, 0.0]]]]),
            0.0,
        ),
        (
            torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
            torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
            torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]),
            1.0,
        ),
    ],
)
def test_signal2noise_ratio_combination(
    prediction: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
    signal2noise_ratio: float,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that the signal2noise_ratio is calculated correctly."""
    metrics_a = Metrics(frac_threshold, probability_threshold).update(prediction, probability, label)

    metrics_b = (
        Metrics(frac_threshold, probability_threshold)
        .update(prediction, probability, label)
        .update(prediction, probability, label)
    )

    torch.testing.assert_close(
        metrics_a.signal2noise_ratio, metrics_b.signal2noise_ratio, equal_nan=True, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(metrics_a.signal2noise_ratio, signal2noise_ratio, equal_nan=True, rtol=0.0, atol=0.0)
    torch.testing.assert_close(metrics_b.signal2noise_ratio, signal2noise_ratio, equal_nan=True, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (None, None, None),
        (None, 5.0, 5.0),
        (3.0, None, 3.0),
        (3.0, 5.0, 8.0),
    ],
)
def test_add_loss_values(
    a: float | None,
    b: float | None,
    expected: float | None,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that loss values are added correctly, handling None."""
    result = Metrics(frac_threshold, probability_threshold)._add_loss_values(a, b)
    assert result == expected, f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "input_loss, expected_loss",
    [
        (3.0, 3.0),  # Test case where the loss is a valid float
        (None, torch.nan),  # Test case where the loss is None
    ],
)
@pytest.mark.parametrize(
    "loss_type, loss_name",
    [
        ("combined_loss", "average_combined_loss"),
        ("binary_loss", "average_binary_loss"),
        ("conditional_loss", "average_conditional_loss"),
    ],
)
def test_average_loss(
    loss_type: str,
    loss_name: str,
    input_loss: float | None,
    expected_loss: float | torch.Tensor,
    frac_threshold: float,
    probability_threshold: float,
) -> None:
    """Test that average losses are calculated correctly and handle None values correctly."""
    metrics = Metrics(frac_threshold, probability_threshold)

    # Update the metrics object with the appropriate loss type
    update_kwargs = {loss_type: input_loss}
    metrics.update(
        torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
        torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]]),
        torch.tensor([[[[1.0, 0.0], [0.0, 1.5]]]]),
        **update_kwargs,
    )

    result = getattr(metrics, loss_name)
    torch.testing.assert_close(result, expected_loss, equal_nan=True, rtol=0.0, atol=0.0)
