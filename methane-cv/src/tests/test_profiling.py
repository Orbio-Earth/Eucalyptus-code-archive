"""Tests for profiling code."""

from collections.abc import Generator
from typing import Any

import pytest

from src.utils.profiling import MEASUREMENTS, timer
from src.utils.utils import setup_logging

logger = setup_logging()


@pytest.fixture(scope="function", autouse=True)
def setup_teardown() -> Generator[Any, Any, Any]:
    """Set up and tear down code before and after each test."""
    # SETUP
    # Because it's used as a global variable, we need to clear the
    # accumulator before and after each test to be sure.
    MEASUREMENTS.clear()

    # TEST CODE
    yield

    # TEARDOWN

    MEASUREMENTS.clear()


@timer(phase="test", accumulator=MEASUREMENTS, verbose=True, logger=logger)
def example_func(num_times: int, *, rank: int | None = None) -> None:
    """Show example function with decorator."""
    for _ in range(num_times):
        sum([number**2 for number in range(10_000)])


@timer("test", MEASUREMENTS)
def main(rank: int, num_times: int) -> None:
    """Show example function with decorator."""
    for _ in range(num_times):
        sum([number**2 for number in range(10_000)])


def test_timer_accumulator() -> None:
    """Test that timer decorator is accumulated."""
    example_func(2)
    example_func(num_times=2)
    num_timings = 2
    assert len(MEASUREMENTS) == num_timings


def test_rank_kwarg_for_func() -> None:
    """Test that timer grabs the rank as a kwarg for functions."""
    MEASUREMENTS.clear()
    rank = 0
    example_func(num_times=2, rank=rank)
    assert MEASUREMENTS[0].device == rank


def test_rank_arg_for_main() -> None:
    """Test that timer grabs the rank as the first arg."""
    MEASUREMENTS.clear()
    rank = 0
    main(rank, 2)
    assert len(MEASUREMENTS) == 1
    assert MEASUREMENTS[0].device == rank
