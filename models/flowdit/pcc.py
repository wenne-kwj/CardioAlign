"""Stable auxiliary correlation on clean IF, averaged per image and channel."""
import math

import torch


def clean_image_pcc_loss(prediction, target, prediction_variance_floor=1e-6,
                         target_variance_min=1e-12):
    """FP32 PCC with a prediction variance floor and target-only validity mask.

    The floor uses mean squared variation, so its meaning is independent of
    image size. Constant predictions remain penalized and get finite gradients;
    only targets without meaningful numerical variation are excluded. This
    stabilized training objective does not replace the unfloored evaluation PCC.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError('Expected matching BCHW image tensors')
    for value in (prediction_variance_floor, target_variance_min):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('PCC variance safeguards must be finite and positive')
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        x = prediction.float().flatten(2)
        y = target.float().flatten(2)
        x = x - x.mean(-1, keepdim=True)
        y = y - y.mean(-1, keepdim=True)
        prediction_variance = x.square().mean(-1)
        target_variance = y.square().mean(-1)
        covariance = (x * y).mean(-1)
        denominator = (prediction_variance.clamp_min(prediction_variance_floor)
                       * target_variance.clamp_min(target_variance_min)).sqrt()
        correlation = (covariance / denominator).clamp(-1, 1)
        valid = target_variance > target_variance_min
        count = valid.sum()
        loss = torch.where(valid, 1 - correlation, 0).sum() / count.clamp_min(1)
        mean_correlation = torch.where(valid, correlation, 0).sum() / count.clamp_min(1)
    return loss, dict(pcc_stabilized=mean_correlation.detach(), pcc_valid_pairs=count.detach(),
                      pcc_total_pairs=valid.numel())
