"""Four-level Restormer for one BF channel and three fluorescence channels.

Adapted from swz30/Restormer, basicsr/models/archs/restormer_arch.py.
Copyright (c) 2022 Syed Waqas Zamir and contributors. MIT license; see
LICENSE.Restormer. The image-space input residual and dual-pixel branch have
deliberately been removed: brightfield and fluorescence are different domains.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import (
    Downsample,
    LayerNormType,
    OverlapPatchEmbed,
    TransformerBlock,
    Upsample,
    positive_integer,
)


class Restormer(nn.Module):
    """Restormer-BF-32: MDTA + GDFN with a shared fluorescence output head.

    Default inputs are [0, 1] BCHW tensors with one channel.
    Outputs are [B, 3, H, W], in caller-defined channel order and bounded by sigmoid.
    Features use widths [32, 64, 128, 256], 40 backbone Transformer blocks
    plus 4 refinement blocks. The full-resolution decoder/refinement width
    is 2*dim, matching the official architecture.

    Reflect-padding extends the right/bottom to a multiple of eight; only
    tiny inputs for which reflection is impossible use replication. The
    prediction is always cropped back. No image-space input residual exists.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        dim: int = 32,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: LayerNormType = "WithBias",
        output_activation: Literal["sigmoid", "identity"] = "sigmoid",
        auto_pad: bool = True,
    ) -> None:
        super().__init__()
        in_channels = positive_integer(in_channels, "in_channels")
        out_channels = positive_integer(out_channels, "out_channels")
        dim = positive_integer(dim, "dim")
        if dim % 2:
            raise ValueError("dim must be even for convolution + PixelUnshuffle.")
        if len(num_blocks) != 4 or len(heads) != 4:
            raise ValueError("num_blocks and heads must each contain four integers.")
        num_blocks = tuple(positive_integer(value, "num_blocks") for value in num_blocks)
        heads = tuple(positive_integer(value, "heads") for value in heads)
        num_refinement_blocks = positive_integer(num_refinement_blocks, "num_refinement_blocks")
        widths = tuple(dim * 2**level for level in range(4))
        if any(width % head for width, head in zip(widths, heads, strict=True)):
            raise ValueError("Each level's width must be divisible by its head count.")
        if output_activation not in {"sigmoid", "identity"}:
            raise ValueError(f"Unsupported output activation: {output_activation}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.channels = widths
        self.num_blocks = num_blocks
        self.heads = heads
        self.output_activation = output_activation
        self.auto_pad = auto_pad

        def stage(channels: int, head_count: int, depth: int) -> nn.Sequential:
            return nn.Sequential(
                *(
                    TransformerBlock(
                        dim=channels,
                        num_heads=head_count,
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias,
                        layer_norm_type=layer_norm_type,
                    )
                    for _ in range(depth)
                )
            )

        self.patch_embed = OverlapPatchEmbed(in_channels, dim)
        self.encoder_level1 = stage(dim, heads[0], num_blocks[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = stage(dim * 2, heads[1], num_blocks[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = stage(dim * 4, heads[2], num_blocks[2])
        self.down3_4 = Downsample(dim * 4)
        self.latent = stage(dim * 8, heads[3], num_blocks[3])

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = stage(dim * 4, heads[2], num_blocks[2])
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = stage(dim * 2, heads[1], num_blocks[1])
        self.up2_1 = Upsample(dim * 2)
        # Do not reduce channels here: retain the official 2*dim high-res path.
        self.decoder_level1 = stage(dim * 2, heads[0], num_blocks[0])
        self.refinement = stage(dim * 2, heads[0], num_refinement_blocks)
        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, padding=1, bias=bias)

    @staticmethod
    def _pad_to_multiple(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        pad_height, pad_width = (-height) % 8, (-width) % 8
        if not pad_height and not pad_width:
            return image
        mode = "replicate" if pad_height >= height or pad_width >= width else "reflect"
        return F.pad(image, (0, pad_width, 0, pad_height), mode=mode)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected a 4D BCHW tensor, got shape {tuple(image.shape)}")
        if image.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channel(s), got {image.shape[1]}")
        if image.shape[0] <= 0 or min(image.shape[-2:]) <= 0:
            raise ValueError("Batch size, height and width must be positive.")
        if not image.is_floating_point():
            raise TypeError("Restormer requires floating-point normalized images.")
        height, width = image.shape[-2:]
        if self.auto_pad:
            image = self._pad_to_multiple(image)
        elif height % 8 or width % 8:
            raise ValueError("Height and width must be divisible by 8 when auto_pad=False.")

        enc1 = self.encoder_level1(self.patch_embed(image))
        enc2 = self.encoder_level2(self.down1_2(enc1))
        enc3 = self.encoder_level3(self.down2_3(enc2))
        latent = self.latent(self.down3_4(enc3))
        dec3 = self.reduce_chan_level3(torch.cat((self.up4_3(latent), enc3), dim=1))
        dec3 = self.decoder_level3(dec3)
        dec2 = self.reduce_chan_level2(torch.cat((self.up3_2(dec3), enc2), dim=1))
        dec2 = self.decoder_level2(dec2)
        dec1 = self.decoder_level1(torch.cat((self.up2_1(dec2), enc1), dim=1))
        output = self.output(self.refinement(dec1))
        if self.output_activation == "sigmoid":
            output = torch.sigmoid(output)
        return output[..., :height, :width]


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable scalar parameters, not buffers."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
