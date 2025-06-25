"""Transformations used in training."""

import abc
import math
from typing import Any, cast

import torch
import torchvision.transforms.functional as TF

from src.utils.parameters import CROP_SIZE, S2_BANDS

NUM_DIMENSIONS = 4
NUM_SNAPSHOTS = 2


class CustomHorizontalFlip:
    """Applies random horizontal flips to both input and target tensors.

    Flips are controlled by a random state to ensure the same flip is consistently applied
    across inputs and targets within the same batch.

    Adds an attribute to track whether a flip was applied.

    NOTE: This assumes X is already transformed by ConcatenateSnapshots
    """

    def __init__(self) -> None:
        self.flip_probability = 0.5

    def __call__(self, Xy: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """When called, flip horizontally."""
        X, y = Xy
        numitems = X.shape[0]

        flipped_X, flipped_y = X, y  # no need to copy
        numitems = X.shape[0]
        for i in range(numitems):
            flip = torch.rand(1).item() < self.flip_probability
            if flip:
                flipped_X[i, :, :, :] = TF.hflip(X[i, :, :, :])
                flipped_y[i, :, :, :] = TF.hflip(y[i, :, :, :])

        return flipped_X, flipped_y


class Rotate90:
    """Rotates an input tensor and its corresponding target by 0°, 90°, 180°, or 270°.

    Roation is determined randomly using a provided generator to ensure consistency
    across inputs and targets within the same batch.

    Adds an attribute to track the last applied rotation degree.

    NOTE: This assumes X is already transformed by ConcatenateSnapshots
    """

    def __call__(self, Xy: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """When called, randomly rotate either 0°, 90°, 180°, 270°."""
        X, y = Xy

        # Assert that the last two dimensions of the tensor match the expected size for image height and width.
        # Currently, we expect this to be 128x128, but this may change if a different image size is used in the future.
        assert X.shape[-2] == CROP_SIZE and X.shape[-1] == CROP_SIZE, (
            f"Expected the last two dimensions of X to represent image height and width as 128x128, "
            f"but got dimensions {X.shape[-2]}x{X.shape[-1]} at indices -2 and -1."
        )
        assert y.shape[-2] == CROP_SIZE and y.shape[-1] == CROP_SIZE, (
            f"Expected the last two dimensions of y to represent image height and width as 128x128, "
            f"but got dimensions {y.shape[-2]}x{y.shape[-1]} at indices -2 and -1."
        )

        numitems = X.shape[0]
        # Apply rotation to both X and y
        rotated_X, rotated_y = X, y  # no need to copy
        for i in range(numitems):
            # Choose a random rotation degree: 0, 1, 2, or 3, corresponding to 0°, 90°, 180°, 270°
            rotation = cast(int, torch.randint(0, 4, (1,)).item())
            if rotation == 0:
                continue
            rotated_X[i, :, :, :] = torch.rot90(X[i, :, :, :], rotation, [-2, -1])
            rotated_y[i, :, :, :] = torch.rot90(y[i, :, :, :], rotation, [-2, -1])

        return rotated_X, rotated_y


class BaseBandExtractor(abc.ABC):
    """Abstract base class for band selection and optionally snapshot concatenation."""

    def __init__(self, snapshots: list[str] | None = None) -> None:
        self.snapshots = snapshots or []

    @abc.abstractmethod
    def __call__(self, Xy: tuple[dict[str, torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """When called, extract and scale the chosen bands."""
        pass

    @abc.abstractmethod
    def asdict(self) -> dict:
        """Return the snapshots in a dict."""
        pass

    @property
    @abc.abstractmethod
    def output_channels(self) -> int:
        """The number of channels in the data returned by the extractor.

        Note: This is the number of *input* channels to the model.
        """
        pass


class MonotemporalBandExtractor(BaseBandExtractor):
    """Extracts and rescales the desired bands from the `crop_main` column, returning a single tensor.

    Args:
        band_indices: Integer indices of the bands we want to use for training.
        scaling_factor: Multiplicative scaling factor for the band values.

    Note: This is used for EMIT.
    """

    def __init__(
        self,
        band_indices: list[int],
        scaling_factor: float,
    ) -> None:
        self.band_indices = band_indices
        self.scaling_factor = scaling_factor
        super().__init__()

    def __call__(self, Xy: tuple[dict[str, torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """When called, extract and scale the chosen bands."""
        X, y = Xy

        assert (
            X["crop_main"].ndim == NUM_DIMENSIONS
        ), f"Expected {NUM_DIMENSIONS} dimensions (batch, bands, height, width)"

        # take our subset of bands
        transformed_X = X["crop_main"][:, self.band_indices, :, :] * self.scaling_factor

        return transformed_X, y

    def asdict(self) -> dict:
        """Return the snapshots in a dict."""
        return {
            "band_indices": self.band_indices,
            "scaling_factor": self.scaling_factor,
        }

    @property
    def output_channels(self) -> int:
        """The number of channels in the data returned by the extractor.

        Note: This is the number of *input* channels to the model.
        """
        return len(self.band_indices)


class ConcatenateSnapshots(BaseBandExtractor):
    """Concatenate temporal snapshots with main bands into a single tensor.

    This class combines temporal bands from two snapshots with main bands from the target date.
    The output tensor has bands arranged as: [temporal_bands_t1, main_bands, temporal_bands_t2].
    All bands are rescaled to reflectance values between 0 and 1.

    Args:
        snapshots: List of two snapshot keys to use for temporal bands
        all_available_bands: List of all available band names in the satellite dataset
        temporal_bands: List of band names to use from temporal snapshots
        main_bands: List of band names to use from main reference image
        satellite_id: Satellite ID
    """

    def __init__(
        self,
        snapshots: list[str],
        all_available_bands: list[str],
        temporal_bands: list[str],
        main_bands: list[str],
        scaling_factor: float,
    ) -> None:
        if len(snapshots) != NUM_SNAPSHOTS:
            raise ValueError(f"Expected exactly {NUM_SNAPSHOTS} snapshots, got {len(snapshots)}")

        self.all_available_bands = all_available_bands
        self.temporal_bands = temporal_bands
        self.main_bands = main_bands
        self.scaling_factor = float(scaling_factor)
        super().__init__(snapshots)

    def __call__(self, Xy: tuple[dict[str, torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """When called, concatenate snapshots."""
        X, y = Xy

        assert (
            X["crop_main"].ndim == NUM_DIMENSIONS
        ), f"Expected {NUM_DIMENSIONS} dimensions (batch, bands, height, width)"

        for crop_key in self.snapshots:
            assert (
                X[crop_key].ndim == NUM_DIMENSIONS
            ), f"Expected {NUM_DIMENSIONS} dimensions (batch, bands, height, width)"

        # we choose some bands so the total number of bands still adds up to 13
        band_indices_temporal = [self.all_available_bands.index(b) for b in self.temporal_bands]
        band_indices_main = [self.all_available_bands.index(b) for b in self.main_bands]

        transformed_X = torch.cat(
            (
                X[self.snapshots[0]][:, band_indices_temporal, :, :],
                X["crop_main"][:, band_indices_main, :, :],
                X[self.snapshots[1]][:, band_indices_temporal, :, :],
            ),
            dim=1,  # concatenate bands (dimension 1)
        )
        # rescale bands to reflectance (between 0 and 1)
        transformed_X = transformed_X * self.scaling_factor  # implicitly converts to float32

        return transformed_X, y

    def asdict(self) -> dict[str, Any]:
        """Return the snapshots in a dict."""
        return {
            "snapshots": self.snapshots,
            "all_available_bands": self.all_available_bands,
            "temporal_bands": self.temporal_bands,
            "main_bands": self.main_bands,
            "scaling_factor": self.scaling_factor,
        }

    @property
    def output_channels(self) -> int:
        """The number of channels in the data returned by the extractor.

        Note: This is the number of *input* channels to the model.
        """
        return len(self.main_bands) + len(self.snapshots) * len(self.temporal_bands)


def modulate_signal(
    B11: torch.Tensor,
    B11o: torch.Tensor,
    B12: torch.Tensor,
    B12o: torch.Tensor,
    frac: torch.Tensor,
    modulate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given the original and modified bands B11 and B12, modulates the signal by a factor of `modulate`.

    For a modulation factor `m`, this is implemented by modifying the reflectances B with
        B_mod = B^m * B_o^(1-m)
    where B is the reflectance with the full signal, B_o is the methane-free reflectance
    (we know this since it's from a simulation), and B_mod is the modified reflectance.

    The reason to do it this way instead of the more obvious B_mod = m*B + (1-m)*B_o
    is that the modulation of the frac is then more easy to calculate as
        frac_mod = (1+frac)^m - 1
    If we used the linear transformation, we would need to recaculate frac,
    but this actually causes problems when one of the reflectances is zero.
    We added and subtracted an offset when we calculated the frac in the original
    simulation to get around this problem, but we no longer have the retrieval
    so can't do this here again. Hence being able to modify the frac is
    the only option, which motivated the multiplicative form.

    Arguments
    ---------
    B11
        The band 11 reflectance with the full signal absorption
    B11_o
        The methane free (background) band 11 reflectance
    B12
        The band 11 reflectance with the full signal absorption
    B12_o
        The methane free (background) band 12 reflectance
    modulate: float
        The modulation factor, a number between 0 and 1.

    Note: the implementation is completely generic, so the reflectances
          could be floats, numpy array or pytorch tensors.
    """
    assert (modulate >= 0).all()
    assert (modulate <= 1).all()  # would have to think about this
    B11_mod = B11**modulate * B11o ** (1 - modulate)
    B12_mod = B12**modulate * B12o ** (1 - modulate)

    frac_mod = (1 + frac) ** modulate - 1
    return B11_mod, B12_mod, frac_mod


# FIXME: This will need to be updated to work with EMIT.
class MethaneModulator:
    """Transform class that modulates the strength of the methane signal.

    This can be used to provide more weak sources in the training data
    without having to re-generate a new dataset.
    In principle, the modulation could also be adapted from one epoch to the next.

    See the docstring for `modulate_signal` for more details on how
    the modulation is implemented mathematically.

    Arguments:
    ----------
    modulate: float
        Between 0 and 1, by how much should we modulate the methane signal?
    """

    def __init__(
        self,
        modulate: float,
        all_available_bands: list[str],
        swir16_band_name: str,
        swir22_band_name: str,
        orig_swir16_band_name: str = "orig_swir16",
        orig_swir22_band_name: str = "orig_swir22",
    ) -> None:
        if not (0 <= modulate <= 1 or math.isnan(modulate)):
            raise ValueError(f"Modulation factor must be between 0 and 1, got {modulate}")
        self.modulate = modulate
        self.all_available_bands = all_available_bands
        self.swir16_band_name = swir16_band_name
        self.swir22_band_name = swir22_band_name
        self.orig_swir16_band_name = orig_swir16_band_name
        self.orig_swir22_band_name = orig_swir22_band_name

    def random_mod(self, num: int) -> torch.Tensor:
        """Sample the modulation factor uniformly between `self.modulate` and 1."""
        return (1 - self.modulate) * torch.rand(num) + self.modulate

    def __call__(
        self, Xy: tuple[dict[str, torch.Tensor], torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Modulate the strength of the methane signal.

        Args:
            Xy: Tuple of (X, y) where X is a dict containing input tensors and y contains target values

        Returns
        -------
            Tuple of (X_mod, y_mod) with modulated signal strengths
        """
        X, y = Xy
        # all tensors have indices (sample, band, x, y)
        # Get indices for SWIR bands (i.e for Sentinel2 they are B11 and B12)
        swir16_idx = self.all_available_bands.index(self.swir16_band_name)
        swir22_idx = self.all_available_bands.index(self.swir22_band_name)

        # Extract SWIR bands from main crop
        swir16 = X["crop_main"][:, swir16_idx : swir16_idx + 1, :, :]
        swir22 = X["crop_main"][:, swir22_idx : swir22_idx + 1, :, :]

        # Get original (unmodified) SWIR bands. The band name for this depends on the col name in the generated parquet
        # files. Ex: historically it was orig_band_11 and orig_band_12 when only processing Sentinel2, but now it's
        # orig_swir16 and orig_swir22
        swir16_orig = X[self.orig_swir16_band_name]
        swir22_orig = X[self.orig_swir22_band_name]

        # Generate modulation factors
        numitems = X["crop_main"].shape[0]
        m = self.random_mod(numitems)[:, None, None, None]
        swir16_mod, swir22_mod, frac_mod = modulate_signal(swir16, swir16_orig, swir22, swir22_orig, y, m)
        # round back to the nearest integer,
        # so the neural network can't use non-integer values
        # as a way to detect methane
        torch.round(swir16_mod, decimals=0, out=swir16_mod)
        torch.round(swir22_mod, decimals=0, out=swir22_mod)

        # now swap the modulated bands into the input tensors
        # (modify in place)
        X["crop_main"][:, swir16_idx : swir16_idx + 1, :, :] = swir16_mod
        X["crop_main"][:, swir22_idx : swir22_idx + 1, :, :] = swir22_mod

        assert frac_mod.shape == y.shape

        return X, frac_mod


class UseOriginalB11B12_X:
    """Exchange synthetic B11/B12 with original bands in input data."""

    def __init__(self) -> None:
        pass

    def __call__(self, X: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Transform input by replacing synthetic B11/B12 with originals.

        Args:
            X (Dict[str, torch.Tensor]): Input tensor dictionary.

        Returns
        -------
            Dict[str, torch.Tensor]: Transformed input dictionary.
        """
        ib11 = S2_BANDS.index("B11")
        ib12 = S2_BANDS.index("B12")
        X["crop_main"][:, ib11 : ib11 + 1, :, :] = X["orig_band_11"]
        X["crop_main"][:, ib12 : ib12 + 1, :, :] = X["orig_band_12"]
        return X


class UseOriginalB11B12:
    """Exchange syntheticly changed B11B12 with original B11B12 in main chip."""

    def __init__(self) -> None:
        pass

    def __call__(
        self, Xy: tuple[dict[str, torch.Tensor], torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Exchange syntheticly changed B11B12 with original B11B12 in main chips.

        Args:
            Xy: Tuple of (X, y) where X is a dict containing input tensors and y contains target values

        Returns
        -------
            Tuple of (X, y) with original B11, B12 in main chips
        """
        X, y = Xy
        # all tensors have indices (sample, band, x, y)
        ib11 = S2_BANDS.index("B11")
        ib12 = S2_BANDS.index("B12")
        # now swap the modulated bands into the input tensors
        X["crop_main"][:, ib11 : ib11 + 1, :, :] = X["orig_band_11"]
        X["crop_main"][:, ib12 : ib12 + 1, :, :] = X["orig_band_12"]

        return X, y
