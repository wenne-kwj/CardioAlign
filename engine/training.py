"""Compact training flow over externally supplied paired tensors."""
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .io import CHECKPOINT_FORMAT
from .registry import build_components


def _optimizer(module, config, *, discriminator=False):
    classes = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}
    cls = classes[config.get("optimizer", "Adam")]
    rate = config.get("discriminator_learning_rate", config["learning_rate"]) if discriminator else config["learning_rate"]
    return cls(
        module.parameters(), lr=rate, betas=tuple(config.get("betas", (0.9, 0.999))),
        weight_decay=config.get("weight_decay", 0.0),
    )


def _backward_step(loss, module, optimizer, clip):
    if not torch.isfinite(loss):
        raise FloatingPointError("Training loss is non-finite")
    loss.backward()
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(module.parameters(), clip, error_if_nonfinite=True)
    elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in module.parameters()):
        raise FloatingPointError("Training gradients are non-finite")
    optimizer.step()


def _gan_step(components, config, image, target, optimizer_g, optimizer_d):
    name = config["name"]
    if name == "pix2pix_unet":
        from models.pix2pix_unet.losses import generator_losses, discriminator_loss
    else:
        from models.pix2pix_resnet.losses import generator_losses, discriminator_loss
    generator, discriminator = components.model, components.discriminator
    real_a, real_b = image * 2 - 1, target * 2 - 1
    fake = generator(real_a)
    real_label = config["loss"].get("real_label", 1.0)
    clip = config["training"].get("gradient_clip")
    result = {}

    def logits(value):
        if name == "pix2pix_unet":
            return discriminator(torch.cat((real_a, value), dim=1))
        return discriminator(real_a, value)

    def update_discriminator():
        discriminator.requires_grad_(True)
        optimizer_d.zero_grad(set_to_none=True)
        loss = discriminator_loss(logits(real_b), logits(fake.detach()), real_label=real_label)
        _backward_step(loss, discriminator, optimizer_d, clip)
        result["discriminator"] = float(loss.detach())

    def update_generator():
        discriminator.requires_grad_(False)
        optimizer_g.zero_grad(set_to_none=True)
        loss, details = generator_losses(
            logits(fake), fake, real_b, real_label=real_label,
            lambda_L1=config["loss"].get("lambda_L1", 100.0),
        )
        _backward_step(loss, generator, optimizer_g, clip)
        result.update(generator=float(loss.detach()), **{key: float(value.detach()) for key, value in details.items()})

    order = config["training"].get("update_order", "D_then_G")
    if order == "D_then_G":
        update_discriminator()
        update_generator()
    elif order == "G_then_D":
        update_generator()
        update_discriminator()
    else:
        raise ValueError("Unknown adversarial update order")
    discriminator.requires_grad_(True)
    return result


def objective(components, config, image, target):
    name = config["name"]
    if name in ("unet", "restormer"):
        return components.criterion(components.model(image), target)
    condition, clean = image * 2 - 1, target * 2 - 1
    if name == "cellresdm":
        process = components.process
        times = torch.randint(process.steps, (len(clean),), device=clean.device)
        noisy = process.q_sample(clean, condition, times, torch.randn_like(clean))
        prediction = process.predict_x0(components.model, noisy, condition, times)
        return components.criterion(prediction, clean)
    if name == "palette":
        if config.get("prediction_type", "v") == "v":
            from models.palette.velocity import velocity_loss
            loss = velocity_loss(components.model, clean, condition)
        else:
            from models.palette.losses import training_loss
            loss = training_loss(components.model, clean, condition)
        return loss, {}
    if name == "flowdit":
        return components.process.loss(components.model, condition, clean)
    raise ValueError("Adversarial models use their alternating-update flow")


@torch.no_grad()
def _update_ema(ema, model, decay):
    for averaged, current in zip(ema.parameters(), model.parameters(), strict=True):
        averaged.lerp_(current, 1 - decay)
    for averaged, current in zip(ema.buffers(), model.buffers(), strict=True):
        averaged.copy_(current)


def train(config, image, target, output, *, device="cpu", steps=None, seed=0):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if (image.ndim != 4 or image.shape[1] != 1
            or target.shape != (len(image), 3, *image.shape[-2:]) or len(image) < 1):
        raise ValueError("Expected paired input [N,1,H,W] and target [N,3,H,W]")
    torch.manual_seed(seed)
    components = build_components(config, device, training=True)
    components.model.train()
    if components.discriminator is not None:
        components.discriminator.train()
    settings = config["training"]
    count = settings["steps"] if steps is None else steps
    if count < 1 or settings.get("batch_size", 1) < 1:
        raise ValueError("Steps and batch size must be positive")
    optimizer = _optimizer(components.model, settings)
    optimizer_d = _optimizer(components.discriminator, settings, discriminator=True) if components.discriminator is not None else None
    decay = settings.get("ema_decay")
    if decay is not None and not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0,1)")
    ema = deepcopy(components.model).eval().requires_grad_(False) if decay is not None else None
    loader = DataLoader(TensorDataset(image, target), batch_size=settings.get("batch_size", 1), shuffle=True)
    iterator = iter(loader)
    metrics = {}
    for step in range(1, count + 1):
        try:
            batch_image, batch_target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch_image, batch_target = next(iterator)
        batch_image, batch_target = batch_image.to(device), batch_target.to(device)
        if optimizer_d is not None:
            metrics = _gan_step(components, config, batch_image, batch_target, optimizer, optimizer_d)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss, details = objective(components, config, batch_image, batch_target)
            _backward_step(loss, components.model, optimizer, settings.get("gradient_clip"))
            metrics = {"loss": float(loss.detach())}
            metrics.update({key: float(value.detach()) for key, value in details.items() if isinstance(value, torch.Tensor) and value.numel() == 1})
        if ema is not None:
            _update_ema(ema, components.model, decay)
    checkpoint = {
        "format": CHECKPOINT_FORMAT, "config": config, "step": count, "seed": seed,
        "model": components.model.state_dict(), "optimizer": optimizer.state_dict(),
    }
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
    if optimizer_d is not None:
        checkpoint["discriminator"] = components.discriminator.state_dict()
        checkpoint["discriminator_optimizer"] = optimizer_d.state_dict()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {"steps": count, "last_step": metrics}
