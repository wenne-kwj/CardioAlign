"""DDIM sampling from independent Gaussian noise and a BF condition."""
import torch


@torch.inference_mode()
def sample(network, condition, *, steps=100, eta=0.0, generator=None):
    if not 1 <= steps <= network.num_timesteps or not 0 <= eta <= 1:
        raise ValueError("Invalid DDIM step count or eta")
    batch = len(condition)
    shape = (batch, network.x0_low.shape[1], *condition.shape[-2:])
    state = torch.randn(shape, device=condition.device, generator=generator)
    times = torch.linspace(network.num_timesteps - 1, 0, steps).round().long().tolist()
    for index, timestep in enumerate(times):
        alpha = network.gammas[timestep]
        gamma = alpha.expand(batch, 1)
        epsilon = network.denoise_fn(torch.cat((condition, state), 1), gamma).float()
        clean = (state - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()
        clean = torch.maximum(torch.minimum(clean, network.x0_high), network.x0_low)
        if index == len(times) - 1:
            return clean
        epsilon = (state - alpha.sqrt() * clean) / (1 - alpha).sqrt()
        previous = network.gammas[times[index + 1]]
        variance = ((1 - previous) / (1 - alpha) * (1 - alpha / previous)).clamp_min(0)
        sigma = eta * variance.sqrt()
        state = previous.sqrt() * clean + (1 - previous - sigma.square()).clamp_min(0).sqrt() * epsilon
        if eta:
            state += sigma * torch.randn(shape, device=condition.device, generator=generator)
    raise AssertionError("At least one sampling step is required")
