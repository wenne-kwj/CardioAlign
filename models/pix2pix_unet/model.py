"""Canonical U-Net generator and conditional PatchGAN pair."""
from contextlib import redirect_stdout
import io

from .networks import UnetGenerator, NLayerDiscriminator, get_norm_layer, init_weights


def build_models(config):
    norm = get_norm_layer(config.get("norm", "batch"))
    generator = UnetGenerator(
        config["input_nc"], config["output_nc"], config.get("num_downs", 8),
        ngf=config["ngf"], norm_layer=norm, use_dropout=config.get("dropout", True),
    )
    discriminator = NLayerDiscriminator(
        config["input_nc"] + config["output_nc"], ndf=config["ndf"],
        n_layers=3, norm_layer=norm,
    )
    with redirect_stdout(io.StringIO()):
        init_weights(generator, "normal", 0.02)
        init_weights(discriminator, "normal", 0.02)
    return generator, discriminator
