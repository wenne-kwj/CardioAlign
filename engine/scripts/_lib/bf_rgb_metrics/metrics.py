"""Intensity-preserving RGB analysis using the supplied watershed algorithm.

Physical output channels are always R,G,B. The array API defaults to the BF
models' CHW B,G,R float32 [0,1] tensors after detach().cpu().numpy(). File IO
preserves RGB16 via OpenCV, and never applies per-image intensity stretching.
"""

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from . import _watershed as watershed


CHANNELS = ('R', 'G', 'B')
FILE_EXTENSIONS = {'.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg', '.webp', '.npy'}
VALUE_RANGES = {'unit': (0., 1.), 'uint8': (0., 255.),
                'uint16': (0., 65535.), 'signed_unit': (-1., 1.)}


@dataclass(frozen=True)
class DetectionConfig:
    red_threshold16: float = 58 * 257
    blue_threshold16: float = 25 * 257
    green_threshold16: float = 20 * 257
    red_min_area: int = 15
    blue_min_area: int = 50
    iou_threshold: float = .5
    blue_threshold_mode: str = 'otsu'

    def __post_init__(self):
        if self.blue_threshold_mode not in ('otsu', 'fixed'):
            raise ValueError('blue_threshold_mode must be otsu or fixed')
        for name in ('red_threshold16', 'blue_threshold16', 'green_threshold16'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 65535:
                raise ValueError(f'{name} must be finite and within 0..65535')
            object.__setattr__(self, name, float(value))
        for name in ('red_min_area', 'blue_min_area'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer in original pixels')
            object.__setattr__(self, name, int(value))
        if not math.isfinite(self.iou_threshold) or not 0 < self.iou_threshold <= 1:
            raise ValueError('iou_threshold must be in (0,1]')
        object.__setattr__(self, 'iou_threshold', float(self.iou_threshold))

    def protocol(self):
        data = asdict(self)
        data.update(
            schema_version=2, output_channel_order=list(CHANNELS),
            thresholds_unit={'R': self.red_threshold16 / 65535,
                             'G': self.green_threshold16 / 65535,
                             'B': self.blue_threshold16 / 65535 if self.blue_threshold_mode == 'fixed' else None},
            comparisons={'R': '>=', 'G': '>', 'B': '>=' if self.blue_threshold_mode == 'fixed' else '>'},
            blue_threshold_rule=('Fixed configured threshold.' if self.blue_threshold_mode == 'fixed' else
                'Per-image OpenCV Otsu on full-FOV blue rounded to native uint16 levels; no stretch or background subtraction. Constant channels are empty.'),
            threshold_scope='Complete original FOV, before training crop or augmentation; prediction and GT evaluated independently.',
            foreground='Raw threshold mask, before watershed, hole filling or area filtering.',
            instances='Supplied EDM/MaximumFinder watershed, binary fill holes, 8-connectivity, minimum area, consecutive IDs.',
            geometry='No resizing; minimum areas are measured on original image pixels.',
            border_policy='Include border-touching objects.',
            fixes=['EDM maximum<=1 skips watershed; internal EDM conversion avoids division by zero.',
                   'All-foreground masks are kept as a single component.'],
            matching='Maximum-cardinality one-to-one bipartite matching among pairs with IoU >= threshold; deterministic CSR ordering.',
            dice_empty='Both empty=1; exactly one empty=0. Nonempty-GT means are reported separately.',
            object_empty='Undefined precision/recall/F1 denominators produce null. Main object metrics pool TP/FP/FN.',
            source_algorithm_sha256=watershed.UPSTREAM_SHA256,
        )
        return data

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.protocol(), sort_keys=True).encode()).hexdigest()


def prepare_rgb(image, *, channel_order, layout, value_range):
    """Convert an explicitly described array to unit-range HWC RGB.

    No float-to-uint8 conversion, auto-contrast, resizing or channel inference.
    uint16 levels remain distinguishable in float32; float64 inputs retain their
    precision. Invalid ranges, nonfinite values and ambiguous shapes fail.
    """
    array = np.asarray(image)
    if channel_order not in ('RGB', 'BGR') or layout not in ('HWC', 'CHW'):
        raise ValueError('Specify channel_order RGB/BGR and layout HWC/CHW')
    if value_range not in VALUE_RANGES:
        raise ValueError(f'value_range must be one of {tuple(VALUE_RANGES)}')
    if array.ndim != 3 or array.dtype.kind not in 'uif':
        raise ValueError('Expected a finite numeric three-dimensional array')
    if layout == 'CHW':
        array = array.transpose(1, 2, 0)
    if array.shape[2] != 3 or min(array.shape[:2]) < 1:
        raise ValueError(f'Expected exactly three nonempty channels: {array.shape}')
    lower, upper = VALUE_RANGES[value_range]
    if not np.isfinite(array).all() or float(array.min()) < lower or float(array.max()) > upper:
        raise ValueError(f'Pixels must be finite and in {value_range} range {lower}..{upper}; no automatic clipping')
    dtype = np.float64 if array.dtype.kind == 'f' and array.dtype.itemsize >= 8 else np.float32
    array = array.astype(dtype, copy=False)
    if value_range == 'signed_unit':
        array = (array + dtype(1)) / dtype(2)
    elif value_range != 'unit':
        array = array / dtype(upper)
    if channel_order == 'BGR':
        array = array[..., ::-1]
    return np.ascontiguousarray(array)


def read_image(path, *, npy_order='BGR', npy_layout='CHW', npy_range='unit'):
    """Return (unit HWC RGB, input provenance). PNG/TIFF require uint8/uint16 RGB.

    npy defaults match the saved unquantized CHW BGR [0,1] model predictions.
    File hashes identify the exact bytes read, including for read-only datasets.
    """
    path = Path(path)
    encoded = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix not in FILE_EXTENSIONS:
        raise ValueError(f'Unsupported image extension: {path}')
    if suffix == '.npy':
        array = np.load(io.BytesIO(encoded), allow_pickle=False)
        order, layout, value_range = npy_order, npy_layout, npy_range
    else:
        array = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)
        if array is None or array.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
            raise ValueError(f'Expected uint8/uint16 RGB raster: {path}')
        order, layout = 'BGR', 'HWC'
        value_range = 'uint16' if array.dtype == np.uint16 else 'uint8'
    rgb = prepare_rgb(array, channel_order=order, layout=layout, value_range=value_range)
    return rgb, {'path': str(path.absolute()), 'file_sha256': hashlib.sha256(encoded).hexdigest(),
                 'dtype': str(array.dtype), 'shape': list(array.shape),
                 'channel_order': order, 'layout': layout, 'value_range': value_range}


def foreground_masks(rgb, config=None):
    """Return physical RGB masks and actual thresholds on the complete unit FOV.

    Only the derived Otsu histogram is placed on the native16-bit grid. Original
    float predictions/GT remain unchanged. A constant image has no separable
    blue foreground; explicitly keep it empty instead of making a giant object.
    """
    config = config or DetectionConfig()
    rgb = prepare_rgb(rgb, channel_order='RGB', layout='HWC', value_range='unit')
    rt = np.asarray(config.red_threshold16 / 65535, dtype=rgb.dtype)
    gt = np.asarray(config.green_threshold16 / 65535, dtype=rgb.dtype)
    thresholds = {'R': config.red_threshold16, 'G': config.green_threshold16}
    masks = {'R': rgb[..., 0] >= rt, 'G': rgb[..., 1] > gt}
    if config.blue_threshold_mode == 'fixed':
        masks['B'] = rgb[..., 2] >= np.asarray(config.blue_threshold16 / 65535, dtype=rgb.dtype)
        thresholds['B'] = config.blue_threshold16
    else:
        blue16 = np.rint(rgb[..., 2] * np.asarray(65535, dtype=rgb.dtype)).astype(np.uint16)
        if blue16.min() == blue16.max():
            thresholds['B'] = float(blue16.flat[0])
            masks['B'] = np.zeros(blue16.shape, dtype=bool)
        else:
            threshold, _ = cv2.threshold(blue16, 0, 65535, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            thresholds['B'] = float(threshold)
            masks['B'] = blue16 > threshold
    return masks, thresholds


def _instance_labels(mask, minimum):
    """Apply the supplied method, returning area-filtered consecutive labels."""
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)
    h, w = mask.shape
    edm = watershed.make_edm(mask)
    if mask.all() or float(edm.max()) <= 1:
        segmented_mask = mask
    else:
        types, maxima = watershed.find_maxima_types(edm, w, h)
        if len(maxima) == 0:
            segmented_mask = mask
        else:
            types = watershed.analyze_and_mark_maxima(edm, types, maxima, w, h)
            encoded = watershed.make8bit_edm(edm, types, w, h)
            encoded, types = watershed.cleanup_maxima(encoded, types, maxima, w, h)
            segmented = watershed.watershed_segment(encoded, w, h)
            segmented_mask = segmented.reshape(h, w) == 255
    filled = ndimage.binary_fill_holes(segmented_mask)
    labels, _ = ndimage.label(filled, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())
    accepted = np.flatnonzero(sizes >= minimum)
    accepted = accepted[accepted != 0]
    lookup = np.zeros(len(sizes), dtype=np.int32)
    lookup[accepted] = np.arange(1, len(accepted) + 1, dtype=np.int32)
    return lookup[labels]


@dataclass
class ImageAnalysis:
    config: DetectionConfig
    foreground: dict
    labels: dict
    thresholds16: dict | None = None

    @property
    def shape(self):
        return self.foreground['R'].shape

    def summary(self):
        height, width = self.shape
        channels = {}
        for channel in CHANNELS:
            pixels = int(self.foreground[channel].sum())
            channels[channel] = {'foreground_pixels': pixels,
                                 'foreground_fraction': pixels / (height * width)}
            if self.thresholds16 is not None:
                channels[channel]['threshold16'] = self.thresholds16[channel]
                channels[channel]['threshold_8bit_equivalent'] = self.thresholds16[channel] / 257
            if channel in self.labels:
                channels[channel]['count'] = int(self.labels[channel].max())
        blue_count = channels['B']['count']
        return {'height': height, 'width': width, 'per_channel': channels,
                'blue_threshold_mode': self.config.blue_threshold_mode,
                'red_blue_count_ratio': channels['R']['count'] / blue_count if blue_count else None}

    def save_labels(self, path):
        metadata = {} if self.thresholds16 is None else {
            'thresholds16_RGB': np.array([self.thresholds16[c] for c in CHANNELS], dtype=np.float64),
            'blue_threshold_mode': np.array(self.config.blue_threshold_mode),
        }
        np.savez_compressed(path, **{'foreground_' + c: self.foreground[c] for c in CHANNELS},
                            labels_R=self.labels['R'], labels_B=self.labels['B'], **metadata)


def analyze_array(image, config=None, *, channel_order='BGR', layout='CHW', value_range='unit'):
    """Analyze a model array; pass HWC/RGB/uint16 for a native RGB16 array."""
    config = config or DetectionConfig()
    rgb = prepare_rgb(image, channel_order=channel_order, layout=layout, value_range=value_range)
    foreground, thresholds = foreground_masks(rgb, config)
    labels = {'R': _instance_labels(foreground['R'], config.red_min_area),
              'B': _instance_labels(foreground['B'], config.blue_min_area)}
    return ImageAnalysis(config, foreground, labels, thresholds)


def foreground_dice(prediction, target):
    prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.size == 0:
        raise ValueError('Foreground masks must have the same nonempty 2D shape')
    total = int(prediction.sum()) + int(target.sum())
    return 2 * int(np.count_nonzero(prediction & target)) / total if total else 1.


def _object_scores(tp, fp, fn):
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def _areas(labels):
    if labels.ndim != 2 or labels.size == 0 or labels.dtype.kind not in 'iu' or labels.min() < 0:
        raise ValueError('Instance labels must be nonnegative 2D integer arrays')
    maximum = int(labels.max())
    if maximum > labels.size:
        raise ValueError('Instance labels must use consecutive IDs 1..N')
    areas = np.bincount(labels.ravel().astype(np.int64), minlength=maximum + 1)
    if not np.all(areas[1:] > 0):
        raise ValueError('Instance labels must use consecutive IDs 1..N')
    return areas


def match_instances(prediction, target, iou_threshold=.5, *, include_matches=False):
    """Maximum-cardinality one-to-one matching on threshold-valid IoU edges.

    Sparse intersections avoid dense N_gt x N_pred allocations. Importantly,
    thresholding precedes assignment so invalid edges cannot steal valid TPs.
    """
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError('Predicted and GT instance label shapes differ')
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError('iou_threshold must be in (0,1]')
    pa, ga = _areas(prediction), _areas(target)
    npred, ngt = len(pa) - 1, len(ga) - 1
    pairs = []
    if npred and ngt:
        both = (prediction > 0) & (target > 0)
        keys = target[both].astype(np.int64) * (npred + 1) + prediction[both]
        unique, intersections = np.unique(keys, return_counts=True)
        gids, pids = unique // (npred + 1), unique % (npred + 1)
        iou = intersections / (ga[gids] + pa[pids] - intersections)
        valid = iou >= iou_threshold
        graph = csr_matrix((np.ones(int(valid.sum()), dtype=np.int8), (gids[valid] - 1, pids[valid] - 1)),
                           shape=(ngt, npred))
        graph.sort_indices()
        matching = maximum_bipartite_matching(graph, perm_type='column')
        gt_indices = np.flatnonzero(matching >= 0)
        tp = len(gt_indices)
        if include_matches:
            lookup = {int(k): float(v) for k, v in zip(unique[valid], iou[valid])}
            pairs = [{'gt_id': int(g + 1), 'prediction_id': int(matching[g] + 1),
                      'iou': lookup[int((g + 1) * (npred + 1) + matching[g] + 1)]} for g in gt_indices]
    else:
        tp = 0
    result = _object_scores(int(tp), int(npred - tp), int(ngt - tp))
    result.update(target_count=ngt, prediction_count=npred)
    if include_matches:
        result['matches'] = pairs
    return result


def compare_analyses(prediction, target, *, include_matches=False):
    """Compare reusable analyses, e.g. one cached GT against multiple models."""
    if prediction.shape != target.shape:
        raise ValueError('Prediction and GT dimensions differ; no implicit resizing')
    if prediction.config != target.config:
        raise ValueError('Prediction and GT must use the same fixed detection configuration')
    channels = {}
    for channel in CHANNELS:
        pm, gm = prediction.foreground[channel], target.foreground[channel]
        p, g, intersection = int(pm.sum()), int(gm.sum()), int((pm & gm).sum())
        record = {'foreground_dice': foreground_dice(pm, gm), 'prediction_foreground_pixels': p,
                  'target_foreground_pixels': g, 'intersection_pixels': intersection,
                  'foreground_fraction_abs_error': abs(p - g) / pm.size,
                  'target_empty': g == 0, 'prediction_empty': p == 0}
        if prediction.thresholds16 is not None:
            record['prediction_threshold16'] = prediction.thresholds16[channel]
        if target.thresholds16 is not None:
            record['target_threshold16'] = target.thresholds16[channel]
        if channel in ('R', 'B'):
            objects = match_instances(prediction.labels[channel], target.labels[channel],
                                      prediction.config.iou_threshold, include_matches=include_matches)
            record.update(objects)
            record['count_abs_error'] = abs(objects['prediction_count'] - objects['target_count'])
        channels[channel] = record
    return {'height': prediction.shape[0], 'width': prediction.shape[1],
            'protocol_sha256': prediction.config.fingerprint(), 'per_channel': channels}


def evaluate_arrays(prediction, target, config=None, *, channel_order='BGR', layout='CHW', value_range='unit'):
    """Convenience API for equally encoded model predictions and GT arrays."""
    config = config or DetectionConfig()
    options = dict(channel_order=channel_order, layout=layout, value_range=value_range)
    return compare_analyses(analyze_array(prediction, config, **options), analyze_array(target, config, **options))


class MetricAccumulator:
    """Per-FOV macro count/Dice metrics plus pooled object TP/FP/FN metrics."""

    def __init__(self):
        self.sample_count = 0
        self.protocol_sha256 = None
        self.totals = {c: dict(dice_sum=0., dice_nonempty_sum=0., nonempty_gt=0, gt_empty=0, both_empty=0,
                              pred_pixels=0, gt_pixels=0, intersection=0, coverage_error_sum=0.,
                              count_error_sum=0, predicted_count=0, target_count=0, tp=0, fp=0, fn=0)
                       for c in CHANNELS}

    def update(self, record):
        protocol = record['protocol_sha256']
        if self.protocol_sha256 is not None and protocol != self.protocol_sha256:
            raise ValueError('Cannot aggregate results from different detection protocols')
        self.protocol_sha256 = protocol
        self.sample_count += 1
        for c in CHANNELS:
            r, t = record['per_channel'][c], self.totals[c]
            t['dice_sum'] += r['foreground_dice']
            t['gt_empty'] += r['target_empty']
            t['both_empty'] += r['target_empty'] and r['prediction_empty']
            if not r['target_empty']:
                t['dice_nonempty_sum'] += r['foreground_dice']
                t['nonempty_gt'] += 1
            t['pred_pixels'] += r['prediction_foreground_pixels']
            t['gt_pixels'] += r['target_foreground_pixels']
            t['intersection'] += r['intersection_pixels']
            t['coverage_error_sum'] += r['foreground_fraction_abs_error']
            if c in ('R', 'B'):
                t['count_error_sum'] += r['count_abs_error']
                t['predicted_count'] += r['prediction_count']
                t['target_count'] += r['target_count']
                for name in ('tp', 'fp', 'fn'):
                    t[name] += r[name]

    def summary(self):
        channels = {}
        n = self.sample_count
        for c, t in self.totals.items():
            denom = t['pred_pixels'] + t['gt_pixels']
            values = {'foreground_dice_macro': t['dice_sum'] / n if n else None,
                      'foreground_dice_micro': 2 * t['intersection'] / denom if denom else (1. if n else None),
                      'foreground_dice_nonempty_gt_macro': t['dice_nonempty_sum'] / t['nonempty_gt'] if t['nonempty_gt'] else None,
                      'gt_empty_images': t['gt_empty'], 'both_empty_images': t['both_empty'],
                      'nonempty_gt_images': t['nonempty_gt'],
                      'foreground_fraction_mae': t['coverage_error_sum'] / n if n else None}
            if c in ('R', 'B'):
                scores = _object_scores(t['tp'], t['fp'], t['fn'])
                values.update(count_mae=t['count_error_sum'] / n if n else None,
                              target_count_total=t['target_count'], prediction_count_total=t['predicted_count'],
                              object_precision_micro=scores['precision'], object_recall_micro=scores['recall'],
                              object_f1_micro=scores['f1'], tp=t['tp'], fp=t['fp'], fn=t['fn'])
            channels[c] = values
        return {'sample_count': n, 'protocol_sha256': self.protocol_sha256, 'per_channel': channels}
