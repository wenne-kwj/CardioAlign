"""Continuous-gamma epsilon-prediction objective."""
import torch
from torch.nn import functional as F

def training_loss(network, clean, bf, predictor=None):
    """Upstream continuous-gamma draw order, FP32 diffusion and epsilon MSE."""
    b = len(clean)
    t = torch.randint(1, network.num_timesteps, (b,), device=clean.device)
    previous = network.gammas[t - 1].reshape(b, 1)
    current = network.gammas[t].reshape(b, 1)
    gamma = (current - previous) * torch.rand((b, 1), device=clean.device) + previous
    noise = torch.randn_like(clean)
    noisy = network.q_sample(clean, gamma[:, :, None, None], noise)
    denoiser = network.denoise_fn if predictor is None else predictor
    predicted = denoiser(torch.cat([bf, noisy], 1), gamma)
    return F.mse_loss(predicted.float(), noise.float())
