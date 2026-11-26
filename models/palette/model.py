"""Conditional Palette network with gamma conditioning and optional v prediction."""
import torch

from .network import Network, extract
from .attention import accelerate
from .velocity import GammaVelocityDenoiser


class PaletteNetwork(Network):
    def __init__(self, unet, beta_schedule):
        super().__init__(unet=unet, beta_schedule=beta_schedule, init_type="kaiming")
        channels = unet["out_channel"]
        self.register_buffer("x0_low", torch.full((1, channels, 1, 1), -1.0))
        self.register_buffer("x0_high", torch.full((1, channels, 1, 1), 1.0))

    def p_mean_variance(self, y_t, t, clip_denoised=True, y_cond=None):
        gamma = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        epsilon = self.denoise_fn(torch.cat((y_cond, y_t), dim=1), gamma)
        clean = self.predict_start_from_noise(y_t, t, epsilon)
        if clip_denoised:
            clean = torch.maximum(torch.minimum(clean, self.x0_high), self.x0_low)
        return self.q_posterior(clean, y_t, t)


def build_network(config, device="cpu"):
    if config.get("noise_conditioning", "gamma") != "gamma":
        raise ValueError("This implementation uses continuous gamma conditioning")
    network = PaletteNetwork(config["unet"], config["beta_schedule"])
    network.init_weights()
    network.set_loss(torch.nn.MSELoss())
    network.set_new_noise_schedule(device=torch.device(device), phase="train")
    accelerate(network, sdpa=config.get("sdpa_attention", True))
    prediction = config.get("prediction_type", "v")
    if prediction == "v":
        network.denoise_fn = GammaVelocityDenoiser(network.denoise_fn)
    elif prediction != "epsilon":
        raise ValueError("prediction_type must be v or epsilon")
    return network.to(device)
