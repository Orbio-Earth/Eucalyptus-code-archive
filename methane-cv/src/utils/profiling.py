"""Utilities for timing / profiling training code.

```python
from src.utils.profiling import MEASUREMENTS, timer
from src.utils.utils import setup_logging

logger = setup_logging()


@timer(
    phase="test",
    accumulator=MEASUREMENTS,
    verbose=True,
    logger=logger,
)
def func():
    time.sleep(1)


func()

df = pd.DataFrame([m.as_dict() for m in MEASUREMENTS])
print(df)
```
"""

import functools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psutil
from torch import device

from src.utils.utils import setup_logging

LOGGER = setup_logging()


def log_memory(stage: str) -> None:
    """Log current memory usage.

    Args:
        stage (str): Description of the current processing stage.
    """
    memory_gb = psutil.Process().memory_info().rss / (1024**3)
    LOGGER.info(f"{stage}: Memory usage: {memory_gb:.2f} GB")


@dataclass
class TimeMeasurement:
    """A time measurement for a phase in the training script.

    time.perf_counter is used to measure execution time because it is more accurate than datetime.now() and
    datetime.now() is used to record the exact time of the measurement, useful for matching measurements to logs.

    Parameters
    ----------
    phase: a user-provided name for where this timing is happening
    func: the name of the function being timed
    device: the device on which the function is being executed
    end: the end time of the measurement, tracks CPU time
    begin: the beginning time of the measurement, tracks CPU time
    end_datetime: the end time of the measurement, tracks wall time
    begin_datetime: the beginning time of the measurement, tracks wall time
    """

    phase: str
    func: str
    device: int | str | device | None
    end: float
    begin: float
    end_datetime: datetime
    begin_datetime: datetime

    @property
    def duration(self) -> float:
        """Duration of the measurement in seconds."""
        return self.end - self.begin

    def __repr__(self) -> str:
        """Respresent as string."""
        return f"TimeMeasurement({self.phase}, {self.duration}, {self.device})"

    def as_dict(self) -> dict:
        """Format as dictionary."""
        return {
            "phase": self.phase,
            "function": self.func,
            "begin": self.begin,
            "end": self.end,
            "begin_datetime": self.begin_datetime,
            "end_datetime": self.end_datetime,
            "duration_s": self.duration,
            "device": self.device,
        }


MEASUREMENTS: list[TimeMeasurement] = []  # for accumulating timings


def timer(
    phase: str, accumulator: list[TimeMeasurement], verbose: bool = False, logger: logging.Logger | None = None
) -> Callable:
    """Decorate functions for timing / profiling."""
    logger = logger if logger is not None else LOGGER

    def decorator_timer(func: Callable) -> Callable:
        """Additional wrapping function to enable passing arguments to the timing code."""

        @functools.wraps(func)
        def wrapper_timer(*args: Any, **kwargs: Any) -> Callable:
            """Actual timing / profiling code."""
            begin_datetime = datetime.now()
            begin = time.perf_counter()

            value = func(*args, **kwargs)
            end = time.perf_counter()
            end_datetime = datetime.now()

            if verbose:
                logger.info(f"{phase} took {end - begin:0.4f} seconds")

            # we force the 'rank' parameter to be a kwarg in all functions so we can easily access
            # with the exception of the 'main' function because of how torch.multiprocessing works
            rank = args[0] if func.__name__ == "main" else kwargs.get("rank")
            assert isinstance(rank, device | int | str | None)
            measurement = TimeMeasurement(
                begin=begin,
                end=end,
                begin_datetime=begin_datetime,
                end_datetime=end_datetime,
                phase=phase,
                func=func.__name__,
                device=rank,
            )

            accumulator.append(measurement)

            return value

        return wrapper_timer

    return decorator_timer
