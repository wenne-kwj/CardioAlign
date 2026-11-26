"""Native RGB16 rendering functions from the validated data_red exporter."""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import struct
import cv2
import numpy as np
from PIL import Image

RGB_INDEX = {'ch1': 2, 'ch3': 1, 'ch4': 0}

def auto_window_uint16(array: np.ndarray) -> dict:
    """Port the historical ImageJ Auto window calculation without an 8-bit step."""
    if array.ndim != 2 or array.size == 0 or array.dtype != np.uint16:
        raise ValueError(f'Expected nonempty 2D uint16 image: {array.shape}, {array.dtype}')
    histogram16 = np.bincount(array.ravel(), minlength=65536)
    levels = np.flatnonzero(histogram16)
    raw_min, raw_max = int(levels[0]), int(levels[-1])
    bin_size = (raw_max - raw_min + 1) / 256.0
    indexes = np.minimum(((levels - raw_min) * (1.0 / bin_size)).astype(np.int64), 255)
    histogram = np.bincount(indexes, weights=histogram16[levels], minlength=256).astype(np.int64)
    count = int(array.size)
    threshold, peak_limit = count // 5000, count // 10
    valid = np.flatnonzero((histogram > threshold) & (histogram <= peak_limit))
    fallback = 'none'
    hmin = hmax = None
    if valid.size:
        hmin, hmax = int(valid[0]), int(valid[-1])
        minimum = raw_min + hmin * bin_size
        maximum = raw_min + hmax * bin_size
        if minimum == maximum:
            minimum, maximum = float(raw_min), float(raw_max)
            fallback = 'equal_auto_endpoints_use_raw_range'
    else:
        minimum, maximum = float(raw_min), float(raw_max)
        fallback = 'no_valid_bins_reset_raw_range'
    # ImageJ ShortProcessor.setMinAndMax uses Java Math.round.
    display_min = int(np.floor(minimum + 0.5))
    display_max = int(np.floor(maximum + 0.5))
    return {
        'raw_min': raw_min, 'raw_max': raw_max, 'pixel_count': count,
        'histogram_bin_size': bin_size, 'hmin': hmin, 'hmax': hmax,
        'auto_threshold': threshold, 'peak_limit': peak_limit,
        'ignored_peak_bins': np.flatnonzero(histogram > peak_limit).tolist(),
        'auto_min': minimum, 'auto_max': maximum,
        'display_min': display_min, 'display_max': display_max, 'fallback': fallback,
    }

def apply_window_uint16(array: np.ndarray, minimum: int, maximum: int) -> np.ndarray:
    """Map the Auto window to 0..65535, rounding half up in exact arithmetic."""
    if array.dtype != np.uint16 or not 0 <= minimum <= maximum <= 65535:
        raise ValueError('Expected uint16 input and a valid uint16 window')
    span = maximum - minimum
    if span == 0:
        return np.where(array > minimum, 65535, 0).astype(np.uint16)
    delta = np.clip(array.astype(np.int64) - minimum, 0, span)
    return ((delta * 65535 + span // 2) // span).astype(np.uint16)

def shifted_window_min(minimum: int, maximum: int, fraction: float) -> int:
    """Use the same rounded Auto-span min shift as the reviewed previews."""
    if not 0 <= minimum <= maximum <= 65535 or not 0 <= fraction < 1:
        raise ValueError('Expected a uint16 window and a finite fraction in [0, 1)')
    if minimum == maximum:
        return minimum
    return min(int(np.floor(minimum + fraction * (maximum - minimum) + 0.5)), maximum - 1)

def png_header(path: Path) -> dict:
    with path.open('rb') as stream:
        header = stream.read(33)
    if header[:8] != b'\x89PNG\r\n\x1a\n' or header[8:16] != b'\x00\x00\x00\rIHDR':
        raise ValueError(f'Invalid PNG header: {path}')
    width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', header[16:29])
    if (depth, color, compression, filtering, interlace) != (16, 2, 0, 0, 0):
        raise ValueError(f'Expected noninterlaced truecolor PNG16: {path}')
    return {'width': width, 'height': height, 'bit_depth': depth, 'color_type': color}

def export_sample(sample: dict, output_dir: str, compression: int, previous: dict | None,
                  red_range_fraction: float = 0.0) -> dict:
    cv2.setNumThreads(1)
    rgb = np.empty((1080, 1080, 3), dtype=np.uint16)
    channels = {}
    source_hashes = {}
    for channel, rgb_index in RGB_INDEX.items():
        source_bytes = Path(sample['sources'][channel]).read_bytes()
        source_hashes[channel] = hashlib.sha256(source_bytes).hexdigest()
        with Image.open(io.BytesIO(source_bytes)) as image:
            if image.format != 'TIFF' or getattr(image, 'n_frames', 1) != 1:
                raise ValueError(f'Expected single-plane TIFF: {sample["sources"][channel]}')
            array = np.array(image)
        if array.shape != (1080, 1080) or array.dtype != np.uint16:
            raise ValueError(f'Expected 1080x1080 uint16 TIFF: {sample["sources"][channel]}')
        stats = auto_window_uint16(array)
        if previous is not None:
            for key in ('raw_min', 'raw_max', 'display_min', 'display_max'):
                if stats[key] != previous[channel][key]:
                    raise ValueError(f'Historical Auto mismatch: {sample["sample_id"]} {channel} {key}')
        fraction = red_range_fraction if channel == 'ch4' else 0.0
        low = shifted_window_min(stats['display_min'], stats['display_max'], fraction)
        high = stats['display_max']
        rendered = apply_window_uint16(array, low, high)
        rgb[..., rgb_index] = rendered
        stats.update({
            'render_min': low, 'render_max': high, 'min_shift_fraction': fraction,
            'below_window_pixels': int(np.count_nonzero(array < low)),
            'above_window_pixels': int(np.count_nonzero(array > high)),
            'output_black_pixels': int(np.count_nonzero(rendered == 0)),
            'output_white_pixels': int(np.count_nonzero(rendered == 65535)),
            'non_8bit_grid_pixels': int(np.count_nonzero(rendered % 257)),
        })
        channels[channel] = stats
    # OpenCV's in-memory order is BGR; the PNG file itself has standard RGB order.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    destination = Path(output_dir) / (sample['sample_id'] + '_merge.png')
    temporary = destination.with_name('.' + destination.stem + '.part.png')
    if not cv2.imwrite(str(temporary), bgr, [cv2.IMWRITE_PNG_COMPRESSION, compression]):
        raise RuntimeError(f'Failed to write PNG: {temporary}')
    header = png_header(temporary)
    check = cv2.imread(str(temporary), cv2.IMREAD_UNCHANGED)
    if header['width'] != 1080 or header['height'] != 1080 or check is None:
        raise ValueError(f'PNG dimensions/decode check failed: {temporary}')
    if check.dtype != np.uint16 or check.shape != bgr.shape or not np.array_equal(check, bgr):
        raise ValueError(f'Lossless uint16 PNG round-trip failed: {temporary}')
    temporary.rename(destination)
    return {
        **sample, 'source_sha256': source_hashes, 'output': str(destination),
        'width': 1080, 'height': 1080, 'bit_depth_per_channel': 16,
        'red_range_fraction': red_range_fraction,
        'bytes': destination.stat().st_size, 'channels': channels,
        'historical_auto_windows_match': previous is not None,
        'lossless_round_trip': True,
        'png_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
