"""Channel-wise reconstruction and threshold-based foreground metrics."""
import torch

from models.unet.losses import StructuralSimilarity
from .image_metrics import per_sample_channel_metrics


def _otsu_threshold(image, bins):
    if bins < 2:
        raise ValueError("Otsu needs at least two histogram bins")
    if image.min() == image.max():
        return image.min()
    histogram = torch.histc(image.float(), bins=bins, min=0.0, max=1.0)
    probability = histogram / histogram.sum()
    centers = (torch.arange(bins, device=image.device) + 0.5) / bins
    weight = probability.cumsum(0)
    mean = (probability * centers).cumsum(0)
    denominator = weight * (1 - weight)
    score = (mean[-1] * weight - mean).square() / denominator.clamp_min(1e-12)
    score[denominator <= 0] = -1
    return (score.argmax().float() + 1) / bins


def _foreground_metrics(prediction, target, config):
    masks = []
    mode = config.get("method", "fixed")
    for images in (prediction, target):
        if mode == "fixed":
            threshold = config.get("threshold", 0.5)
            if not 0 <= threshold <= 1:
                raise ValueError("Fixed threshold must be in [0,1]")
            masks.append(images > threshold)
        elif mode == "otsu":
            masks.append(torch.stack([
                torch.stack([channel > _otsu_threshold(channel, config.get("bins", 256)) for channel in image])
                for image in images
            ]))
        else:
            raise ValueError("Foreground method must be fixed or otsu")
    predicted_mask, target_mask = masks
    intersection = (predicted_mask & target_mask).sum((-2, -1)).float()
    predicted_area = predicted_mask.sum((-2, -1)).float()
    target_area = target_mask.sum((-2, -1)).float()
    area_sum = predicted_area + target_area
    union = area_sum - intersection
    return {
        "dice": torch.where(area_sum > 0, 2 * intersection / area_sum.clamp_min(1), 1.0),
        "iou": torch.where(union > 0, intersection / union.clamp_min(1), 1.0),
    }


def _mean(values):
    defined = values[~torch.isnan(values)]
    return float(defined.mean()) if defined.numel() else None


def evaluate(prediction, target, config):
    ssim = StructuralSimilarity(
        window_size=config.get("ssim_window_size", 11),
        sigma=config.get("ssim_sigma", 1.5), data_range=1.0,
    )
    values = per_sample_channel_metrics(prediction, target, ssim=ssim, data_range=1.0)
    foreground = config.get("foreground")
    if foreground is not None:
        values.update(_foreground_metrics(prediction, target, foreground))
    requested = config.get("metrics", list(values))
    missing = set(requested) - set(values)
    if missing:
        raise ValueError(f"Unsupported metrics or missing foreground configuration: {sorted(missing)}")
    values = {name: values[name] for name in requested}
    return {
        "samples": len(prediction), "channels": prediction.shape[1], "data_range": 1.0,
        "overall": {name: _mean(value) for name, value in values.items()},
        "per_channel": {
            f"channel_{index}": {name: _mean(value[:, index]) for name, value in values.items()}
            for index in range(prediction.shape[1])
        },
        "defined_counts": {name: int((~torch.isnan(value)).sum()) for name, value in values.items()},
        "per_sample": [{"index": index, **{name: value[index].tolist() for name, value in values.items()}}
                       for index in range(len(prediction))],
        "configuration": config,
    }
