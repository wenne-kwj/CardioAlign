"""Equivalent SDPA attention adapter for the retained Palette backbone."""
import types
import torch
from torch import nn
from torch.nn import functional as F

class LegacySDPA(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.n_heads = heads

    def forward(self, qkv):
        b, width, length = qkv.shape
        d = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(b, self.n_heads, 3 * d, length).split(d, dim=2)
        output = F.scaled_dot_product_attention(q.transpose(-1, -2).contiguous(), k.transpose(-1, -2).contiguous(), v.transpose(-1, -2).contiguous(), dropout_p=0.)
        return output.transpose(-1, -2).reshape(b, self.n_heads * d, length)


def direct_attention(self, x):
    # Original custom checkpoint does not restore autocast during backward.
    # Attention has no dropout: computing it directly preserves its function.
    return self._forward(x)


def accelerate(network, sdpa=True):
    changed = 0
    for module in network.denoise_fn.modules():
        if type(module).__name__ == "AttentionBlock":
            module.forward = types.MethodType(direct_attention, module)
            if sdpa:
                if type(module.attention).__name__ != "QKVAttentionLegacy":
                    raise ValueError("Unsupported upstream QKV order")
                module.attention = LegacySDPA(module.num_heads)
            changed += 1
    return dict(direct_attention_blocks=changed, sdpa=sdpa)
