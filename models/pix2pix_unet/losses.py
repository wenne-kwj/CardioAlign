"""Conditional adversarial BCE and L1 reconstruction objectives."""
import torch
from torch.nn import functional as F


def generator_losses(fake_logits, prediction, target, *, real_label=1.0, lambda_L1=100.0):
    adversarial = F.binary_cross_entropy_with_logits(
        fake_logits.float(), torch.full_like(fake_logits.float(), real_label),
    )
    l1 = F.l1_loss(prediction.float(), target.float())
    return adversarial + lambda_L1 * l1, {"gan": adversarial, "l1": l1}


def discriminator_loss(real_logits, fake_logits, *, real_label=1.0):
    real = F.binary_cross_entropy_with_logits(
        real_logits.float(), torch.full_like(real_logits.float(), real_label),
    )
    fake = F.binary_cross_entropy_with_logits(
        fake_logits.float(), torch.zeros_like(fake_logits.float()),
    )
    return (real + fake) * 0.5
