"""Lazy construction of the seven model implementations and their objectives."""
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


MODEL_NAMES = (
    "unet", "restormer", "pix2pix_unet", "pix2pix_resnet", "cellresdm", "palette", "flowdit",
)


@dataclass
class Components:
    model: nn.Module
    discriminator: nn.Module | None = None
    process: Any = None
    criterion: nn.Module | None = None


def build_components(config, device="cpu", *, training=False):
    name = config["name"]
    device = torch.device(device)
    if name in ("unet", "restormer"):
        from models.unet.losses import CharbonnierSSIMLoss
        if name == "unet":
            from models.unet.model import UNet
            model = UNet(**config["model"])
        else:
            from models.restormer.model import Restormer
            model = Restormer(**config["model"])
        components = Components(model, criterion=CharbonnierSSIMLoss(**config["loss"]))
    elif name in ("pix2pix_unet", "pix2pix_resnet"):
        if name == "pix2pix_unet":
            from models.pix2pix_unet.model import build_models
        else:
            from models.pix2pix_resnet.model import build_models
        model, discriminator = build_models(config["model"])
        components = Components(model, discriminator=discriminator)
    elif name == "cellresdm":
        from models.cellresdm.model import CellResDM
        from models.cellresdm.diffusion import ResidualDiffusion
        model = CellResDM(**config["model"])
        criterion = None
        if training:
            from models.cellresdm.losses import DiffusionLoss
            criterion = DiffusionLoss(
                lpips_weight=config["loss"]["lpips_weight"],
                torch_hub_dir=config["loss"].get("pretrained_cache"),
            )
        components = Components(model, process=ResidualDiffusion(**config["diffusion"]), criterion=criterion)
    elif name == "palette":
        from models.palette.model import build_network
        components = Components(build_network(config, device))
    elif name == "flowdit":
        from models.flowdit.model import ConditionalPixelDiT
        from models.flowdit.flow import ConditionalFlow
        components = Components(
            ConditionalPixelDiT(**config["model"]),
            process=ConditionalFlow(**config["flow"], sampling=config["sampling"]),
        )
    else:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    for value in (components.model, components.discriminator, components.process, components.criterion):
        if isinstance(value, nn.Module):
            value.to(device)
    return components
