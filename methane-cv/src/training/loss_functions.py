"""Loss functions for the regression task."""

import torch
from torch import nn


class AsymmetricMSELoss(nn.Module):
    """Custom MSE loss function that applies different weights to errors based on whether the target is zero."""

    FRAC_THRESHOLD = 0.001  # the background frac threshold a pixel needs to be greater than to be considered methane

    def __init__(self, weight_for_zeros: float):
        """
        Initialize the asymmetric MSE loss function.

        Args:
            weight_for_zeros (float): Weight to apply to errors where target is zero.
        """
        super().__init__()
        self.weight_for_zeros = torch.tensor(weight_for_zeros)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the asymmetric MSE loss.

        Args:
            prediction (torch.Tensor): Predicted values.
            target (torch.Tensor): Target values.

        Returns
        -------
            torch.Tensor: Computed loss.
        """
        # Calculate squared errors
        squared_errors = (prediction - target) ** 2

        # Obtain the weight. This is very simple, but looks a bit
        # more complex because we're handling dtypes and which device
        # the data sits on.
        target_is_zero = target.abs() < self.FRAC_THRESHOLD
        weights = torch.where(target_is_zero, self.weight_for_zeros, 1)

        # Apply weights
        weighted_squared_errors = squared_errors * weights

        # Compute the mean loss within each image
        avg_image_loss = torch.mean(weighted_squared_errors, dim=(-1, -2))

        # Sum the loss over the images to handle batches of different sizes
        loss = torch.sum(avg_image_loss)

        return loss


class TwoPartLoss(nn.Module):
    """
    The TwoPartLoss for a regression task expects two output channels and separates the loss into two parts.

    1. The first channel is interpreted as the logit of the probability that a pixel
       exceeds a specified threshold for detecting methane.
       Here we use a binary cross entropy loss.
    2. The second channel is interpreted as the conditional prediction IF the threshold is exceeded.
       So, if methane is detected from the first channel,
       then we use the second channel to estimate how much methane is present.
       Here we use a mean squared error loss.

    In our case, this separates the loss into two separate questions:
    1. Is there methane?
    2. If yes, then how much methane?
    """

    NUM_PREDICTION_CHANNELS = 2
    binary_loss = torch.nn.BCEWithLogitsLoss(reduction="sum")

    def __init__(self, binary_threshold: float, MSE_multiplier: float):
        """
        Initialize the two-part loss function.

        We need to set a threshold for the binary outcome,and a multiplier which weights the MSE relative to the
        cross-entropy loss.

        Args:
            binary_threshold (positive float): Threshold on the ground truth determining whether there is methane
            MSE_multiplier (float): multiplier which weights the MSE relative to the cross-entropy loss
        """
        super().__init__()
        assert 0 < binary_threshold < 1
        self.binary_threshold = binary_threshold
        assert MSE_multiplier >= 0
        self.MSE_multiplier = MSE_multiplier

    def get_prediction_parts_as_dict(self, prediction: torch.Tensor) -> dict[str, torch.Tensor]:
        """Store the predictions from the prediction tensor (binary, marginal, conditional) as a dictionary."""
        marginal_pred, binary_probability, conditional_pred, binary_logit = self.get_prediction_parts(prediction)
        return {
            "binary_probability": binary_probability,
            "conditional_pred": conditional_pred,
            "marginal_pred": marginal_pred,
            "binary_logit": binary_logit,
        }

    def get_loss_parts(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the average binary and conditional loss for all images in the batch."""
        # This is factored out of self.forward() into its own method
        # so we can look at the two losses separately (for diagnostic or tuning purposes)

        pred_dict = self.get_prediction_parts_as_dict(prediction)
        squared_errors = (pred_dict["conditional_pred"] - target) ** 2

        target_is_plume = target.abs() > self.binary_threshold

        # Compute the binary cross-entropy loss on the plume classification
        binary_target = torch.where(target_is_plume, 1.0, 0.0)
        binary_loss_per_batch = self.binary_loss(pred_dict["binary_logit"], binary_target)

        # Compute the conditional loss: the sum of squared errors on plume pixels
        conditional_square_errors = torch.where(target_is_plume, squared_errors, 0)
        conditional_loss_per_batch = torch.sum(conditional_square_errors)

        return binary_loss_per_batch, conditional_loss_per_batch

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the asymmetric MSE loss. This computes the average combined loss for all images in the batch.

        Args:
            prediction (torch.Tensor): Predicted values.
            target (torch.Tensor): Target values.

        Returns
        -------
            torch.Tensor: Computed loss.
        """
        binary_loss_per_batch, conditional_loss_per_batch = self.get_loss_parts(prediction, target)
        # we now combine the two losses with the MSE_multiplier used to modulate their relative weights
        combined_loss_per_batch = binary_loss_per_batch + self.MSE_multiplier * conditional_loss_per_batch

        return (
            combined_loss_per_batch,
            binary_loss_per_batch.item(),
            self.MSE_multiplier * conditional_loss_per_batch.item(),
        )

    def get_prediction_parts(
        self, prediction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract the different predictions from the prediction tensor: binary, marginal, conditional.

        Our model (Unet) output consists of two layers:
        1. Binary Prediction Layer: This layer outputs a binary classification
        prediction for each pixel, indicating whether its methane or not.
        2. Conditional Prediction Layer: This layer outputs a conditional
        prediction on how much methane is present in the pixel.

        The marginal prediction combines these two predictions to give a single
        prediction per pixel. This is done by applying a sigmoid function to the
        binary prediction to get a probability, and then multiplying this probability
        by the conditional prediction.
        """
        assert (
            prediction.shape[-3] == self.NUM_PREDICTION_CHANNELS
        ), f"Predictions must have two channels, got {prediction.shape} shape."
        binary_logit = prediction[:, 0:1, :, :]
        conditional_pred = prediction[:, 1:2, :, :]

        sigmoid = nn.Sigmoid()  # inverse logit to get probabilities between 0 and 1
        binary_probability = sigmoid(binary_logit)
        marginal_pred = binary_probability * conditional_pred

        return marginal_pred, binary_probability, conditional_pred, binary_logit

    def calculate_mse_on_marginal(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the average MSE for all images in the batch."""
        ypred = self.get_prediction_parts_as_dict(prediction)["marginal_pred"]
        assert target.shape == ypred.shape, "Shapes of prediction and target must match."

        mse_per_pixel = ((target - ypred) ** 2).sum()
        return mse_per_pixel
