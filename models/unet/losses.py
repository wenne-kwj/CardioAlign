"""Regression losses for normalized fluorescence prediction."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class StructuralSimilarity(nn.Module):
    """Differentiable channel-wise SSIM for images scaled to [0, 1]."""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer.")
        coordinates = torch.arange(window_size, dtype=torch.float32)
        coordinates -= (window_size - 1) / 2
        gaussian = torch.exp(-(coordinates**2) / (2 * sigma**2))
        gaussian /= gaussian.sum()
        kernel = gaussian[:, None] * gaussian[None, :]
        self.register_buffer("kernel", kernel[None, None], persistent=False)
        self.window_size = window_size
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2

    def per_sample_channel(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Return one SSIM value for every sample and channel as a [B, C] tensor."""
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes differ: {prediction.shape} != {target.shape}"
            )
        if prediction.ndim != 4:
            raise ValueError(f"Expected BCHW tensors, got shape {prediction.shape}")

        prediction = prediction.float()
        target = target.float()
        channels = prediction.shape[1]
        kernel = self.kernel.to(device=prediction.device, dtype=prediction.dtype)
        kernel = kernel.expand(channels, 1, self.window_size, self.window_size)
        padding = self.window_size // 2

        mu_prediction = F.conv2d(
            prediction, kernel, padding=padding, groups=channels
        )
        mu_target = F.conv2d(target, kernel, padding=padding, groups=channels)
        mu_prediction_sq = mu_prediction.square()
        mu_target_sq = mu_target.square()
        mu_product = mu_prediction * mu_target

        variance_prediction = (
            F.conv2d(prediction.square(), kernel, padding=padding, groups=channels)
            - mu_prediction_sq
        ).clamp_min(0.0)
        variance_target = (
            F.conv2d(target.square(), kernel, padding=padding, groups=channels)
            - mu_target_sq
        ).clamp_min(0.0)
        covariance = (
            F.conv2d(prediction * target, kernel, padding=padding, groups=channels)
            - mu_product
        )

        numerator = (2 * mu_product + self.c1) * (
            2 * covariance + self.c2
        )
        denominator = (mu_prediction_sq + mu_target_sq + self.c1) * (
            variance_prediction + variance_target + self.c2
        )
        # C1*C2 is 9e-8 at data_range=1, below float32 eps (1.19e-7).
        # Clamping to eps changes valid dark-image denominators, breaks SSIM(x,x)=1,
        # and suppresses gradients for low-intensity false positives. Only guard
        # against representational underflow; the SSIM constants ensure positivity.
        ssim_map = numerator / denominator.clamp_min(torch.finfo(prediction.dtype).tiny)
        return ssim_map.mean(dim=(2, 3)).clamp(-1.0, 1.0)

    def per_channel(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return batch-averaged SSIM for every channel as a [C] tensor."""
        return self.per_sample_channel(prediction, target).mean(dim=0)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.per_channel(prediction, target).mean()


class CharbonnierSSIMLoss(nn.Module):
    """Equal-channel weighted Charbonnier plus SSIM loss."""

    def __init__(
        self,
        charbonnier_weight: float = 0.8,
        ssim_weight: float = 0.2,
        epsilon: float = 1e-3,
        ssim_window_size: int = 11,
        ssim_sigma: float = 1.5,
    ) -> None:
        super().__init__()
        if charbonnier_weight < 0 or ssim_weight < 0:
            raise ValueError("Loss weights must be non-negative.")
        if not math.isclose(charbonnier_weight + ssim_weight, 1.0):
            raise ValueError("Charbonnier and SSIM weights must sum to one.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        self.charbonnier_weight = charbonnier_weight
        self.ssim_weight = ssim_weight
        self.epsilon = epsilon
        self.ssim = StructuralSimilarity(
            window_size=ssim_window_size,
            sigma=ssim_sigma,
        )

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = prediction.float()
        target = target.float()
        difference = prediction - target
        charbonnier_per_channel = torch.sqrt(
            difference.square() + self.epsilon**2
        ).mean(dim=(0, 2, 3))
        ssim_per_channel = self.ssim.per_channel(prediction, target)
        combined_per_channel = (
            self.charbonnier_weight * charbonnier_per_channel
            + self.ssim_weight * (1.0 - ssim_per_channel)
        )
        loss = combined_per_channel.mean()
        return loss, {
            "charbonnier": charbonnier_per_channel.mean(),
            "ssim": ssim_per_channel.mean(),
            "charbonnier_per_channel": charbonnier_per_channel,
            "ssim_per_channel": ssim_per_channel,
            "combined_per_channel": combined_per_channel,
        }
