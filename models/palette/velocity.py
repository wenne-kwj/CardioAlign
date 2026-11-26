"""Production v prediction with the original, unscaled gamma conditioning."""
import torch
from torch import nn
from torch.nn import functional as F


class GammaVelocityDenoiser(nn.Module):
    """Train the base U-Net on v; expose epsilon to existing DDIM/DDPM samplers."""
    def __init__(self, base):
        super().__init__()
        self.base = base

    @property
    def out(self):
        return self.base.out

    def velocity(self, packed, gamma):
        return self.base(packed, gamma)

    def forward(self, packed, gamma):
        velocity = self.velocity(packed, gamma).float()
        g = gamma.reshape(-1, 1, 1, 1).float()
        return (1-g).sqrt()*packed[:, 1:].float()+g.sqrt()*velocity


def velocity_loss(network, clean, bf, generator=None):
    """FP32 q(x_t|x0), continuous gamma and unweighted v-MSE."""
    if not isinstance(network.denoise_fn, GammaVelocityDenoiser):
        raise ValueError('v-MSE requires the gamma/v denoiser')
    batch = len(clean)
    t = torch.randint(1, network.num_timesteps, (batch,), device=clean.device, generator=generator)
    fraction = torch.rand((batch, 1), device=clean.device, generator=generator)
    previous = network.gammas[t-1, None]
    current = network.gammas[t, None]
    gamma = previous+(current-previous)*fraction
    g = gamma[:, :, None, None]
    noise = torch.randn(clean.shape, dtype=torch.float32, device=clean.device, generator=generator)
    state = g.sqrt()*clean+(1-g).sqrt()*noise
    target = g.sqrt()*noise-(1-g).sqrt()*clean
    predicted = network.denoise_fn.velocity(torch.cat([bf, state], 1), gamma).float()
    return F.mse_loss(predicted, target)
