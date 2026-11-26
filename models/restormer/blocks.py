"""Restormer blocks using native PyTorch instead of einops.

Adapted from swz30/Restormer, basicsr/models/archs/restormer_arch.py.
Copyright (c) 2022 Syed Waqas Zamir and contributors. MIT license; see
LICENSE.Restormer in this directory. All attention is channel-transposed.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


LayerNormType = Literal["WithBias", "BiasFree"]


def positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


class BiasFreeLayerNorm(nn.Module):
    """Official bias-free norm: divide by channel SD without subtracting mean."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(positive_integer(dim, "dim")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(variance + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    """Per-pixel channel LayerNorm, not normalization over spatial dimensions."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        dim = positive_integer(dim, "dim")
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Apply Restormer channel normalization to BCHW features."""

    def __init__(self, dim: int, layer_norm_type: LayerNormType = "WithBias") -> None:
        super().__init__()
        if layer_norm_type not in {"WithBias", "BiasFree"}:
            raise ValueError(f"Unsupported layer norm type: {layer_norm_type}")
        norm_class = WithBiasLayerNorm if layer_norm_type == "WithBias" else BiasFreeLayerNorm
        self.body = norm_class(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BCHW -> BHWC -> BCHW; normalize only the last/channel dimension.
        return self.body(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class FeedForward(nn.Module):
    """Gated-Dconv Feed-Forward Network (GDFN)."""

    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        dim = positive_integer(dim, "dim")
        if not math.isfinite(ffn_expansion_factor) or ffn_expansion_factor <= 0:
            raise ValueError("ffn_expansion_factor must be finite and positive.")
        hidden_features = int(dim * ffn_expansion_factor)
        if hidden_features < 1:
            raise ValueError("ffn_expansion_factor produces zero hidden channels.")
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    """MDTA with per-head [C/head, C/head] attention, never [HW, HW]."""

    def __init__(self, dim: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()
        dim = positive_integer(dim, "dim")
        num_heads = positive_integer(num_heads, "num_heads")
        if dim % num_heads:
            raise ValueError("Attention channels must be divisible by num_heads.")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        head_shape = (batch, self.num_heads, channels // self.num_heads, height * width)
        q = F.normalize(q.reshape(head_shape), dim=-1)
        k = F.normalize(k.reshape(head_shape), dim=-1)
        v = v.reshape(head_shape)
        attention = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        features = (attention @ v).reshape(batch, channels, height, width)
        return self.project_out(features)


class TransformerBlock(nn.Module):
    """Pre-norm MDTA and GDFN, each with its own feature residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: LayerNormType = "WithBias",
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class OverlapPatchEmbed(nn.Module):
    """Overlapping stride-one 3x3 embedding without spatial token reduction."""

    def __init__(self, in_channels: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    """B,C,H,W -> B,2C,H/2,W/2, preserving official pixel ordering."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """B,C,H,W -> B,C/2,2H,2W with learned convolution + PixelShuffle."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
