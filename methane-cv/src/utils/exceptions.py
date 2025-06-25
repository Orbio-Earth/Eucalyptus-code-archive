"""Bespoke exception types."""


class InsufficientImageryException(Exception):
    """Base exception for insufficient imagery.

    Imagery may be missing or have insufficient spatial or temporal coverage.
    """

    pass


class MissingImageException(InsufficientImageryException):
    """Requested image is missing."""

    pass


class InsufficientTemporalImageryException(InsufficientImageryException):
    """Imagery is present but we can't build a sufficient time series."""

    pass


class InsufficientCoverageException(InsufficientImageryException):
    """The spatial coverage of our image is insufficient, e.g. or crop excedes image bounds."""

    pass
