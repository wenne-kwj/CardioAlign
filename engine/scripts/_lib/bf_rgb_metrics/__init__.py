"""Counting and object/foreground evaluation for physical R, G, B channels."""

from .metrics import (
    DetectionConfig, ImageAnalysis, MetricAccumulator, analyze_array,
    compare_analyses, evaluate_arrays, foreground_dice, match_instances,
    prepare_rgb, read_image, foreground_masks,
)

__all__ = [
    'DetectionConfig', 'ImageAnalysis', 'MetricAccumulator', 'analyze_array',
    'compare_analyses', 'evaluate_arrays', 'foreground_dice', 'match_instances',
    'prepare_rgb', 'read_image', 'foreground_masks',
]
