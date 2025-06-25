"""Model architectures for training."""

from enum import Enum

from src.training.models.spectral_unet import SpectralUNet, SpectralUNetPlusPlus

__all__ = ["ModelType", "SpectralUNet", "SpectralUNetPlusPlus"]


class ModelType(str, Enum):
    """Enumerate allowed model types."""

    UNET = "unet"
    UNETPLUSPLUS = "unetplusplus"
    SPECTRALUNET = "spectralunet"
    SPECTRALUNETPLUSPLUS = "spectralunetplusplus"

    def __str__(self) -> str:
        """Give string representation."""
        return self.value
