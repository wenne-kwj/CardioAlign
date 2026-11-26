"""Image-regression metrics for normalized fluorescence predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from models.unet.losses import StructuralSimilarity


METRIC_NAMES = ("mae", "nrmse", "ssim", "pcc", "psnr")
METRIC_DIRECTIONS = {
    "mae": "lower",
    "nrmse": "lower",
    "ssim": "higher",
    "pcc": "higher",
    "psnr": "higher",
}
METRIC_DEFINITIONS = {
    "mae": "mean(abs(prediction - target))",
    "nrmse": "L2(prediction - target) / L2(target)",
    "ssim": (
        "Gaussian-window SSIM with data_range=1, window_size=11, sigma=1.5 "
        "unless overridden by the checkpoint loss configuration"
    ),
    "pcc": "pixel-wise Pearson correlation coefficient",
    "psnr": "10 * log10(1 / MSE), with data_range=1",
}


def _validate_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    if prediction.ndim != 4:
        raise ValueError(f"Expected BCHW tensors, got shape {prediction.shape}")
    if prediction.shape[0] <= 0 or prediction.shape[1] <= 0:
        raise ValueError(f"Batch and channel dimensions must be positive: {prediction.shape}")


def per_sample_channel_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    ssim: StructuralSimilarity | None = None,
    data_range: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Calculate five metrics as [B, C] tensors.

    NRMSE and PCC are undefined when the target energy or centered variance is
    zero. Those entries are returned as NaN and excluded by the accumulator.
    """
    _validate_pair(prediction, target)
    if data_range <= 0:
        raise ValueError("data_range must be positive")

    prediction = prediction.float()
    target = target.float()
    spatial_dimensions = tuple(range(2, prediction.ndim))
    difference = prediction - target
    squared_error = difference.square()
    mae = difference.abs().mean(dim=spatial_dimensions)
    mse = squared_error.mean(dim=spatial_dimensions)

    error_energy = squared_error.sum(dim=spatial_dimensions)
    target_energy = target.square().sum(dim=spatial_dimensions)
    nan = torch.full_like(error_energy, float("nan"))
    nrmse = torch.where(
        target_energy > 0,
        torch.sqrt(error_energy / target_energy),
        nan,
    )

    prediction_centered = prediction - prediction.mean(
        dim=spatial_dimensions,
        keepdim=True,
    )
    target_centered = target - target.mean(
        dim=spatial_dimensions,
        keepdim=True,
    )
    covariance_sum = (prediction_centered * target_centered).sum(
        dim=spatial_dimensions
    )
    pcc_denominator = torch.sqrt(
        prediction_centered.square().sum(dim=spatial_dimensions)
        * target_centered.square().sum(dim=spatial_dimensions)
    )
    pcc = torch.where(
        pcc_denominator > 0,
        covariance_sum / pcc_denominator,
        nan,
    ).clamp(-1.0, 1.0)

    infinite = torch.full_like(mse, float("inf"))
    psnr = torch.where(
        mse > 0,
        10.0 * torch.log10((data_range**2) / mse),
        infinite,
    )
    ssim_module = ssim if ssim is not None else StructuralSimilarity(data_range=data_range)
    ssim_values = ssim_module.per_sample_channel(prediction, target)
    return {
        "mae": mae,
        "nrmse": nrmse,
        "ssim": ssim_values,
        "pcc": pcc,
        "psnr": psnr,
    }


class RegressionMetricAccumulator:
    """Streaming macro-average accumulator for per-sample/channel metrics."""

    def __init__(self, channel_names: Sequence[str]) -> None:
        self.channel_names = tuple(channel_names)
        if not self.channel_names or len(set(self.channel_names)) != len(
            self.channel_names
        ):
            raise ValueError("channel_names must be non-empty and unique")
        channel_count = len(self.channel_names)
        self._sums = {
            metric: torch.zeros(channel_count, dtype=torch.float64)
            for metric in METRIC_NAMES
        }
        self._counts = {
            metric: torch.zeros(channel_count, dtype=torch.int64)
            for metric in METRIC_NAMES
        }
        self.sample_count = 0

    def update(self, metrics: Mapping[str, torch.Tensor]) -> None:
        missing = set(METRIC_NAMES) - set(metrics)
        if missing:
            raise ValueError(f"Missing metrics: {sorted(missing)}")
        expected_shape: tuple[int, int] | None = None
        for metric in METRIC_NAMES:
            values = metrics[metric]
            if values.ndim != 2 or values.shape[1] != len(self.channel_names):
                raise ValueError(
                    f"{metric} must have shape [B, {len(self.channel_names)}], "
                    f"got {tuple(values.shape)}"
                )
            if expected_shape is None:
                expected_shape = tuple(values.shape)
            elif tuple(values.shape) != expected_shape:
                raise ValueError("All metric tensors must have equal shapes")
            values = values.detach().to(device="cpu", dtype=torch.float64)
            valid = torch.isfinite(values)
            self._sums[metric] += torch.where(valid, values, 0.0).sum(dim=0)
            self._counts[metric] += valid.sum(dim=0)
        assert expected_shape is not None
        self.sample_count += expected_shape[0]

    @staticmethod
    def _mean(total: torch.Tensor, count: torch.Tensor) -> float | None:
        count_value = int(count)
        if count_value == 0:
            return None
        return float(total / count)

    def summary(self) -> dict[str, Any]:
        per_channel: dict[str, dict[str, float | None]] = {}
        per_channel_counts: dict[str, dict[str, int]] = {}
        for channel_index, channel in enumerate(self.channel_names):
            per_channel[channel] = {
                metric: self._mean(
                    self._sums[metric][channel_index],
                    self._counts[metric][channel_index],
                )
                for metric in METRIC_NAMES
            }
            per_channel_counts[channel] = {
                metric: int(self._counts[metric][channel_index])
                for metric in METRIC_NAMES
            }

        overall = {
            metric: self._mean(
                self._sums[metric].sum(),
                self._counts[metric].sum(),
            )
            for metric in METRIC_NAMES
        }
        overall_counts = {
            metric: int(self._counts[metric].sum())
            for metric in METRIC_NAMES
        }
        return {
            "sample_count": self.sample_count,
            "overall": overall,
            "per_channel": per_channel,
            "valid_counts": {
                "overall": overall_counts,
                "per_channel": per_channel_counts,
            },
        }
