"""A compact 2D U-Net for one brightfield input and three fluorescence outputs."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest valid GroupNorm group count up to ``maximum``."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    raise AssertionError("Every positive channel count is divisible by one.")


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with GroupNorm and SiLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.block(image)


class UNet(nn.Module):
    """Five-level U-Net with a shared decoder and three-channel output.

    Inputs have shape ``[B, 1, H, W]`` and outputs have shape
    ``[B, 3, H, W]``. When ``auto_pad`` is enabled, arbitrary spatial sizes are
    reflect-padded to a multiple of 16 and cropped back after prediction.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        base_channels: int = 32,
        bottleneck_dropout: float = 0.1,
        output_activation: Literal["sigmoid", "identity"] = "sigmoid",
        auto_pad: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or base_channels <= 0:
            raise ValueError("Channel counts must be positive.")
        if not 0.0 <= bottleneck_dropout < 1.0:
            raise ValueError("bottleneck_dropout must be in [0, 1).")
        if output_activation not in {"sigmoid", "identity"}:
            raise ValueError(f"Unsupported output activation: {output_activation}")

        channels = [base_channels * (2**level) for level in range(5)]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.auto_pad = auto_pad
        self.output_activation = output_activation

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoders = nn.ModuleList(
            [
                DoubleConv(in_channels, channels[0]),
                DoubleConv(channels[0], channels[1]),
                DoubleConv(channels[1], channels[2]),
                DoubleConv(channels[2], channels[3]),
                DoubleConv(channels[3], channels[4]),
            ]
        )
        self.bottleneck_dropout = nn.Dropout2d(p=bottleneck_dropout)
        self.decoders = nn.ModuleList(
            [
                DoubleConv(channels[4] + channels[3], channels[3]),
                DoubleConv(channels[3] + channels[2], channels[2]),
                DoubleConv(channels[2] + channels[1], channels[1]),
                DoubleConv(channels[1] + channels[0], channels[0]),
            ]
        )
        self.output_head = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    @staticmethod
    def _pad_to_multiple(image: torch.Tensor, multiple: int = 16) -> tuple[torch.Tensor, int, int]:
        height, width = image.shape[-2:]
        pad_height = (-height) % multiple
        pad_width = (-width) % multiple
        if pad_height == 0 and pad_width == 0:
            return image, 0, 0

        mode = "reflect"
        if pad_height >= height or pad_width >= width:
            mode = "replicate"
        padded = F.pad(image, (0, pad_width, 0, pad_height), mode=mode)
        return padded, pad_height, pad_width

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected a 4D BCHW tensor, got shape {tuple(image.shape)}")
        if image.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channel(s), got {image.shape[1]}"
            )

        original_height, original_width = image.shape[-2:]
        if self.auto_pad:
            image, _, _ = self._pad_to_multiple(image)
        elif original_height % 16 or original_width % 16:
            raise ValueError("Input height and width must be divisible by 16 when auto_pad=False.")

        skips: list[torch.Tensor] = []
        features = image
        for encoder in self.encoders[:-1]:
            features = encoder(features)
            skips.append(features)
            features = self.pool(features)

        features = self.bottleneck_dropout(self.encoders[-1](features))
        for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
            features = F.interpolate(
                features,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            features = decoder(torch.cat((skip, features), dim=1))

        output = self.output_head(features)
        if self.output_activation == "sigmoid":
            output = torch.sigmoid(output)
        return output[..., :original_height, :original_width]


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable scalar parameters in ``model``."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
