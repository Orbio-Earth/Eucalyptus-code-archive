"""Implementations of spectral layers preceding U-Net and U-Net++.

U-Net implementations preceded by 1-D convolutions to compress spectral information.
Inspired by [HyperspectralViTs](https://ar5iv.org/html/2410.17248).
"""

from abc import ABC, abstractmethod

import segmentation_models_pytorch as smp
from torch import nn

__all__ = ["SpectralUNet", "SpectralUNetPlusPlus"]


class BaseSpectralUNet(nn.Module, ABC):
    """Base class for U-Nets with spectral layers."""

    def __init__(
        self, in_channels: int, spectral_hidden_dims: list[int], encoder_name: str, encoder_weights: str | None = None
    ):
        """Instantiate the model with spectral layers.

        Args:
            in_channels: Number of input spectral bands
            spectral_hidden_dims: List of hidden layer dimensions. The last of these will be the number of channels
              passed to the U-Net.
            encoder_name: The name of the encoder to use in the U-Net.
        """
        super().__init__()

        spectral_layers = []
        current_dim = in_channels

        # Build hidden layers using 1x1 convolutions
        for hidden_dim in spectral_hidden_dims:
            spectral_layers.extend(
                [nn.Conv2d(current_dim, hidden_dim, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(hidden_dim)]
            )
            current_dim = hidden_dim

        unet = self.UNetClass(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=spectral_hidden_dims[-1],
            classes=2,
        )

        self.network = nn.Sequential(*spectral_layers, unet)

    @property
    @abstractmethod
    def UNetClass(self) -> nn.Module:
        """The U-Net type to use."""
        ...

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns
        -------
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width)
        """
        return self.network(x)


class SpectralUNet(BaseSpectralUNet):
    """U-Net with a preceding spectral layer."""

    @property
    def UNetClass(self) -> nn.Module:
        """The U-Net type to use."""
        return smp.Unet


class SpectralUNetPlusPlus(BaseSpectralUNet):
    """U-Net++ with a preceding spectral layer."""

    @property
    def UNetClass(self) -> nn.Module:
        """The U-Net type to use."""
        return smp.UnetPlusPlus
