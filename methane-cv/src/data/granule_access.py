"""Base class for satellite data access."""

import abc
from datetime import datetime


class BaseGranuleAccess(abc.ABC):
    """
    Abstract base class for accessing satellite data granules/tiles.

    This class defines the common interface that should be implemented by
    specific satellite data access classes (e.g., Sentinel-2, EMIT).
    """

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """Get the unique identifier for this granule/tile."""
        pass

    @property
    @abc.abstractmethod
    def instrument(self) -> str:
        """Get the instrument ID."""
        pass

    @property
    @abc.abstractmethod
    def instrument_name(self) -> str:
        """Get the instrument name."""
        pass

    @property
    @abc.abstractmethod
    def time(self) -> datetime:
        """Get the acquisition timestamp."""
        pass

    @property
    @abc.abstractmethod
    def datetime_(self) -> datetime:
        """Get the acquisition timestamp."""
        pass

    @property
    @abc.abstractmethod
    def timestamp(self) -> str:
        """Get the acquisition timestamp."""
        pass

    @property
    @abc.abstractmethod
    def date(self) -> str:
        """Get the acquisition timestamp."""
        pass

    @property
    @abc.abstractmethod
    def acquisition_start_time(self) -> str:
        """Get the start time of the acquisition."""
        pass

    @property
    @abc.abstractmethod
    def acquisition_end_time(self) -> str:
        """Get the end time of the acquisition."""
        pass

    @property
    @abc.abstractmethod
    def imaging_mode(self) -> str | None:
        """Get the imaging mode."""
        pass

    @property
    @abc.abstractmethod
    def off_nadir_angle(self) -> float:
        """Get the off-nadir angle."""
        pass

    @property
    @abc.abstractmethod
    def viewing_azimuth(self) -> float | None:
        """Get the viewing azimuth."""
        pass

    @property
    @abc.abstractmethod
    def solar_zenith(self) -> float:
        """Get the solar zenith."""
        pass

    @property
    @abc.abstractmethod
    def solar_azimuth(self) -> float:
        """Get the solar azimuth."""
        pass

    @property
    @abc.abstractmethod
    def orbit_state(self) -> str:
        """Get the orbit state (ascending/descending)."""
        pass
