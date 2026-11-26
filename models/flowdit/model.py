"""A self-contained small pixel DiT, informed by LightningDiT/PixRestore/JiT.

This is a task adaptation, not a byte-for-byte reproduction. It has a learned
BF encoder instead of DINO, optional dense conditioning, and a direct patch
projection. It neither adds BF to the output nor uses BF as the noise anchor.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * self.weight.float()).to(x.dtype)


class Rotary2D(nn.Module):
    """Adjacent-pair RoPE: half of each attention head for each spatial axis."""
    def __init__(self, grid, head_dim):
        super().__init__()
        if head_dim % 4:
            raise ValueError("2D rotary attention requires head dimension divisible by four")
        y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        inverse = 10000.0 ** (-torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2))
        angles = torch.cat((x.flatten()[:, None] * inverse, y.flatten()[:, None] * inverse), -1)
        angles = angles.repeat_interleave(2, dim=-1)[None, None]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, value):
        pairs = value.reshape(*value.shape[:-1], -1, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), -1).flatten(-2)
        return value * self.cos.to(value.dtype) + rotated * self.sin.to(value.dtype)


class Attention(nn.Module):
    def __init__(self, dim, heads, cross=False, dropout=0.0):
        super().__init__()
        self.heads, self.head_dim, self.dropout = heads, dim // heads, dropout
        self.cross = cross
        if cross:
            self.query = nn.Linear(dim, dim)
            self.key_value = nn.Linear(dim, dim * 2)
        else:
            self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x, rope, context=None):
        b, n, dim = x.shape
        if self.cross:
            q = self.query(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
            kv = self.key_value(context).view(b, context.shape[1], 2, self.heads, self.head_dim)
            k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        else:
            qkv = self.qkv(x).view(b, n, 3, self.heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = rope(self.q_norm(q)), rope(self.k_norm(k))
        value = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                                             dropout_p=0.0)
        value = self.projection(value.transpose(1, 2).reshape(b, n, dim))
        return F.dropout(value, self.dropout, self.training)


class SwiGLU(nn.Module):
    def __init__(self, dim, ratio):
        super().__init__()
        inner = math.ceil((dim * ratio * 2 / 3) / 64) * 64
        self.input = nn.Linear(dim, inner * 2)
        self.output = nn.Linear(inner, dim)

    def forward(self, x):
        gate, value = self.input(x).chunk(2, -1)
        return self.output(F.silu(gate) * value)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, ratio, dense, dropout):
        super().__init__()
        self.dense = dense
        self.table = nn.Parameter(torch.zeros(9 if dense else 6, dim))
        self.self_norm = RMSNorm(dim)
        self.self_attention = Attention(dim, heads, dropout=dropout)
        if dense:
            self.cross_norm = RMSNorm(dim)
            self.cross_attention = Attention(dim, heads, cross=True, dropout=dropout)
        self.ff_norm, self.ff = RMSNorm(dim), SwiGLU(dim, ratio)

    def forward(self, x, time_modulation, context, rope):
        parts = (time_modulation + self.table[None]).unbind(1)
        shift, scale, gate = parts[:3]
        x = x + gate[:, None] * self.self_attention(modulate(self.self_norm(x), shift, scale), rope)
        if self.dense:
            shift, scale, gate = parts[3:6]
            x = x + gate[:, None] * self.cross_attention(
                modulate(self.cross_norm(x), shift, scale), rope, context)
        shift, scale, gate = parts[-3:]
        return x + gate[:, None] * self.ff(modulate(self.ff_norm(x), shift, scale))


class BrightfieldEncoder(nn.Module):
    """Learned spatial BF tokens on the same patch grid; computed once in sampling."""
    def __init__(self, dim, patch_size, width):
        super().__init__()
        layers, channels = [], 1
        for stage in range(int(math.log2(patch_size))):
            output = min(dim, width * (2 ** stage))
            layers += [nn.Conv2d(channels, output, 3, stride=2, padding=1),
                       nn.GroupNorm(math.gcd(output, 8), output), nn.SiLU()]
            channels = output
        layers += [nn.Conv2d(channels, dim, 1)]
        self.network, self.norm = nn.Sequential(*layers), RMSNorm(dim)

    def forward(self, bf):
        return self.norm(self.network(bf).flatten(2).transpose(1, 2))


class ConditionalPixelDiT(nn.Module):
    def __init__(self, image_size=256, patch_size=8, hidden_size=384, depth=12,
                 num_heads=6, mlp_ratio=4.0, condition_mode="dense", condition_width=32,
                 dropout=0.0, gradient_checkpointing=False, aligned_condition=False):
        super().__init__()
        if (image_size < patch_size or image_size % patch_size or patch_size < 2
                or patch_size & (patch_size - 1)):
            raise ValueError("Image size must be divisible by a power-of-two patch size >=2")
        if hidden_size % num_heads or (hidden_size // num_heads) % 4 or depth < 1:
            raise ValueError("Invalid DiT attention dimensions/depth")
        if condition_mode not in ("concat", "dense") or not 0 <= dropout < 1:
            raise ValueError("Invalid conditioning mode/dropout")
        if aligned_condition and condition_mode != 'dense':
            raise ValueError('Aligned BF features require dense conditioning')
        self.image_size, self.patch_size = image_size, patch_size
        self.condition_mode, self.gradient_checkpointing = condition_mode, gradient_checkpointing
        self.patch_embed = nn.Conv2d(4, hidden_size, patch_size, stride=patch_size)
        self.bf_encoder = BrightfieldEncoder(hidden_size, patch_size, condition_width) if condition_mode == "dense" else None
        self.condition_projection = nn.Linear(hidden_size, hidden_size, bias=False) if aligned_condition else None
        self.time_mlp = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        count = 9 if condition_mode == "dense" else 6
        self.shared_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, count * hidden_size))
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio,
                                            condition_mode == "dense", dropout) for _ in range(depth)])
        self.rope = Rotary2D(image_size // patch_size, hidden_size // num_heads)
        self.final_norm = RMSNorm(hidden_size)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))
        self.output = nn.Linear(hidden_size, patch_size * patch_size * 3)
        self.apply(self._initialize)
        nn.init.zeros_(self.shared_modulation[-1].weight)
        nn.init.zeros_(self.shared_modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        # Small nonzero image head lets conditioning/patch gradients be checked at initialization.
        nn.init.normal_(self.output.weight, std=0.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode_condition(self, bf):
        return self.bf_encoder(bf) if self.bf_encoder is not None else None

    def forward(self, state, time, bf, context=None):
        if state.ndim != 4 or state.shape[1] != 3 or bf.shape != (state.shape[0], 1, self.image_size, self.image_size):
            raise ValueError("Expected BF [B,1,H,W] and independent IF state [B,3,H,W]")
        if state.shape[-2:] != (self.image_size, self.image_size) or time.shape != (state.shape[0],):
            raise ValueError("State/time shape differs from model configuration")
        if self.bf_encoder is not None and context is None:
            context = self.encode_condition(bf)
        frequency = torch.exp(-math.log(10000) * torch.arange(128, device=time.device).float() / 128)
        phase = time.float()[:, None] * frequency[None] * 1000
        embedding = self.time_mlp(torch.cat((phase.cos(), phase.sin()), -1))
        modulation = self.shared_modulation(embedding).view(state.shape[0], -1, embedding.shape[-1])
        x = self.patch_embed(torch.cat((bf, state), 1)).flatten(2).transpose(1, 2)
        if self.condition_projection is not None:
            # Preserve the registered BF pixel grid before any global attention.
            # This is a feature condition, never an image-space anchor/residual.
            x = x + self.condition_projection(context)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, modulation, context, self.rope, use_reentrant=False)
            else:
                x = block(x, modulation, context, self.rope)
        shift, scale = self.final_modulation(embedding).chunk(2, -1)
        pixels = self.output(modulate(self.final_norm(x), shift, scale))
        b, _, _ = pixels.shape
        grid, patch = self.image_size // self.patch_size, self.patch_size
        return pixels.view(b, grid, grid, patch, patch, 3).permute(0, 5, 1, 3, 2, 4).reshape(b, 3, self.image_size, self.image_size)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
