"""Model-specific inference behind a shared unit-interval tensor interface."""
import torch

from .io import load_checkpoint
from .registry import build_components


@torch.inference_mode()
def predict_batch(components, config, image, *, seed=0):
    components.model.eval()
    name = config["name"]
    if name in ("unet", "restormer"):
        result = components.model(image)
    elif name == "pix2pix_unet":
        from models.pix2pix_unet.inference import inference_policy
        mode = config.get("sampling", {}).get("batchnorm_mode", "current")
        with inference_policy(components.model, mode):
            if mode == "current" and len(image) > 1:
                result = torch.cat([components.model(item * 2 - 1) for item in image.split(1)])
            else:
                result = components.model(image * 2 - 1)
        result = (result + 1) / 2
    elif name == "pix2pix_resnet":
        result = (components.model(image * 2 - 1) + 1) / 2
    else:
        condition = image * 2 - 1
        generator = torch.Generator(device=image.device).manual_seed(seed)
        if name == "cellresdm":
            shape = (len(image), 3, *image.shape[-2:])

            def noise_source(_step):
                return torch.randn(shape, device=image.device, generator=generator)

            result = components.process.sample(components.model, condition, noise_source)
        elif name == "palette":
            from models.palette.sampling import sample
            settings = config["sampling"]
            if settings.get("method", "ddim").lower() != "ddim":
                raise ValueError("The compact Palette inference flow uses DDIM")
            result = sample(components.model, condition, steps=settings["steps"], eta=settings.get("eta", 0.0), generator=generator)
        elif name == "flowdit":
            result = components.process.sample(components.model, condition, generator=generator)
        else:
            raise ValueError(f"Unknown model: {name}")
        result = (result + 1) / 2
    if result.shape != (len(image), 3, *image.shape[-2:]) or not torch.isfinite(result).all():
        raise ValueError("Prediction must be finite and have shape [N,3,H,W]")
    return result.float().clamp(0, 1)


def predict(checkpoint_path, image, *, device="cpu", weights="raw", seed=0):
    if weights not in ("raw", "ema"):
        raise ValueError("weights must be raw or ema")
    checkpoint = load_checkpoint(checkpoint_path)
    key = "model" if weights == "raw" else "ema"
    if key not in checkpoint:
        raise ValueError(f"Checkpoint does not contain {weights} weights")
    config = checkpoint["config"]
    components = build_components(config, device)
    components.model.load_state_dict(checkpoint[key], strict=True)
    predictions = [predict_batch(components, config, item.to(device), seed=seed + index).cpu()
                   for index, item in enumerate(image.split(1))]
    return torch.cat(predictions)
