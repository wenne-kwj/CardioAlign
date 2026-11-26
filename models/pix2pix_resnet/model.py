"""ResNet-50 generator and conditional discriminator with supported-size guards."""
from .networks import AdaptiveGenerator, AdaptiveDiscriminator, weights_init_normal


class Generator(AdaptiveGenerator):
    def forward(self, image):
        if image.ndim != 4 or min(image.shape[-2:]) < 144:
            raise ValueError("Expected BCHW inputs with both spatial sides >= 144")
        return super().forward(image)


class Discriminator(AdaptiveDiscriminator):
    def forward(self, image, target):
        if (image.ndim != 4 or target.ndim != 4
                or image.shape[0] != target.shape[0]
                or image.shape[-2:] != target.shape[-2:]):
            raise ValueError("Conditional pairs must have matching batch/spatial dimensions")
        if min(image.shape[-2:]) < 144:
            raise ValueError("Spatial sides >= 144 preserve all discriminator blocks")
        return super().forward(image, target)


def build_models(config):
    generator = Generator(config["input_nc"], config["output_nc"], config["ngf"])
    generator.apply(weights_init_normal)
    discriminator = Discriminator(config["input_nc"] + config["output_nc"], config["ndf"])
    discriminator.apply(weights_init_normal)
    return generator, discriminator
