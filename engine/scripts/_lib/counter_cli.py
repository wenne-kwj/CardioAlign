#!/usr/bin/env python3
"""Count/segment RGB targets or evaluate pairs with per-FOV blue Otsu.

Use count/evaluate --help. Default .npy inputs are CHW BGR float [0,1], matching
the BF models' saved predictions. Raster images are read at their native bit
depth. No model inference, intensity stretching or resizing. Otsu is computed
independently for each complete GT/prediction; its algorithm is fixed.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import csv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import scipy

from bf_rgb_metrics import DetectionConfig, MetricAccumulator, analyze_array, compare_analyses, read_image
from bf_rgb_metrics.metrics import CHANNELS, FILE_EXTENSIONS


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    if not records:
        raise ValueError('Cannot write empty results')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def safe_id(value):
    if not value or value in ('.', '..') or Path(value).name != value or '\\' in value or '\x00' in value:
        raise ValueError(f'Unsafe or empty sample ID: {value!r}')
    return value


def resolve_input(value, parent):
    if not value:
        raise ValueError('Empty input path in manifest')
    path = Path(value)
    path = path if path.is_absolute() else parent / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path.absolute())


def image_rows(path, recursive, pattern=None):
    path = Path(path).absolute()
    if path.is_file():
        files = [path]
        parent = path.parent
    elif path.is_dir():
        files = sorted((p for p in (path.rglob('*') if recursive else path.iterdir())
                        if p.is_file() and p.suffix.lower() in FILE_EXTENSIONS
                        and (pattern is None or p.match(pattern))), key=str)
        parent = path
    else:
        raise FileNotFoundError(path)
    return [{'sample_id': '__'.join(p.relative_to(parent).with_suffix('').parts),
             'source_dataset': 'unspecified', 'split': '', 'image_path': str(p)} for p in files]


def collect_jobs(args):
    if args.manifest:
        manifest = args.manifest.absolute()
        originals = read_csv(manifest)
        rows = []
        for row in originals:
            selected_column = args.image_column if args.mode == 'count' else args.target_column
            if selected_column is None:
                selected_column = next((c for c in ('target_rgb', 'target_png', 'image_path') if row.get(c)), None)
            if selected_column is None or not row.get(selected_column):
                raise ValueError('Manifest requires a target_rgb/target_png path or an explicit image/target column')
            item = {'sample_id': safe_id(row['sample_id']),
                    'source_dataset': row.get('source_dataset') or row.get('measurement_id') or 'unspecified',
                    'split': row.get('split', ''),
                    'image_path': resolve_input(row[selected_column], manifest.parent)}
            if args.mode == 'evaluate' and not args.prediction_dir:
                item['prediction_path'] = resolve_input(row.get(args.prediction_column), manifest.parent)
            rows.append(item)
    else:
        rows = image_rows(args.input if args.mode == 'count' else args.target_dir, args.recursive, args.input_pattern)
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('No input images or duplicate sample IDs')
    for row in rows:
        safe_id(row['sample_id'])
        if args.mode == 'evaluate' and args.prediction_dir:
            relative = Path(args.prediction_template.format(sample_id=row['sample_id']))
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Prediction template must resolve inside prediction-dir')
            row['prediction_path'] = resolve_input(str(relative), args.prediction_dir.absolute())
    population = len(rows)
    if args.limit is not None:
        rows = rows[:args.limit]
    return rows, population


def run_item(job):
    """CPU-only, picklable worker; each output label path belongs to one FOV."""
    row, config_values, options = job
    cv2.setNumThreads(1)
    config = DetectionConfig(**config_values)
    reader = dict(npy_order=options['npy_order'], npy_layout=options['npy_layout'], npy_range=options['npy_range'])
    try:
        rgb, provenance = read_image(row['image_path'], **reader)
        expected = options['expected_size']
        if expected and rgb.shape[:2] != (expected, expected):
            raise ValueError(f'Expected full {expected}x{expected} FOV, got {rgb.shape[:2]}; no implicit resize')
        target = analyze_array(rgb, config, channel_order='RGB', layout='HWC', value_range='unit')
        record = {key: row[key] for key in ('sample_id', 'source_dataset', 'split')}
        label_root = Path(options['output_dir']) / 'labels'
        if options['mode'] == 'count':
            record.update(input=provenance, analysis=target.summary())
            if options['save_labels']:
                path = label_root / (row['sample_id'] + '.npz')
                target.save_labels(path)
                record['labels'] = str(path.relative_to(options['output_dir']))
        else:
            pred_rgb, prediction_provenance = read_image(row['prediction_path'], **reader)
            if pred_rgb.shape != rgb.shape:
                raise ValueError(f'Prediction shape {pred_rgb.shape} != GT {rgb.shape}; no implicit resize')
            prediction = analyze_array(pred_rgb, config, channel_order='RGB', layout='HWC', value_range='unit')
            record.update(target=provenance, prediction=prediction_provenance)
            record.update(compare_analyses(prediction, target, include_matches=options['save_labels']))
            if options['save_labels']:
                paths = {}
                for name, analysis in (('gt', target), ('prediction', prediction)):
                    path = label_root / (row['sample_id'] + '__' + name + '.npz')
                    analysis.save_labels(path)
                    paths[name] = str(path.relative_to(options['output_dir']))
                record['labels'] = paths
        return record
    except Exception as error:
        raise RuntimeError(f"{row['sample_id']}: {error}") from error


def flatten_count(record):
    analysis = record['analysis']
    result = {key: record[key] for key in ('sample_id', 'source_dataset', 'split')}
    result.update(image_path=record['input']['path'], height=analysis['height'], width=analysis['width'],
                  R_count=analysis['per_channel']['R']['count'], B_count=analysis['per_channel']['B']['count'],
                  R_B_ratio=analysis['red_blue_count_ratio'])
    for c in CHANNELS:
        values = analysis['per_channel'][c]
        result[c + '_foreground_pixels'] = values['foreground_pixels']
        result[c + '_foreground_percent'] = 100 * values['foreground_fraction']
        result[c + '_threshold16'] = values['threshold16']
        result[c + '_threshold_8bit_equivalent'] = values['threshold_8bit_equivalent']
    result['B_threshold_mode'] = analysis['blue_threshold_mode']
    return result


def flatten_metrics(record):
    result = {key: record[key] for key in ('sample_id', 'source_dataset', 'split')}
    result.update(target_path=record['target']['path'], prediction_path=record['prediction']['path'])
    for c in CHANNELS:
        for name, value in record['per_channel'][c].items():
            if name != 'matches':
                result[c + '_' + name] = value
    return result


def counts_summary(records):
    return {'sample_count': len(records),
            'R_count_total': sum(r['analysis']['per_channel']['R']['count'] for r in records),
            'B_count_total': sum(r['analysis']['per_channel']['B']['count'] for r in records),
            'G_foreground_pixels_total': sum(r['analysis']['per_channel']['G']['foreground_pixels'] for r in records),
            'G_foreground_fraction_macro': sum(r['analysis']['per_channel']['G']['foreground_fraction'] for r in records) / len(records)}


def execute(args):
    output = args.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / '_RUNNING', {'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()})
    logger = logging.getLogger('rgb_metrics')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output / 'run.log')):
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
    started = time.monotonic()
    try:
        values = asdict(DetectionConfig())
        if args.config:
            supplied = json.loads(args.config.read_text())
            if not isinstance(supplied, dict) or set(supplied) - set(values):
                raise ValueError('Config must contain only DetectionConfig fields')
            values.update(supplied)
        values.update({name: getattr(args, name) for name in values if getattr(args, name, None) is not None})
        config = DetectionConfig(**values)
        write_json(output / 'detector_config.json', asdict(config))
        write_json(output / 'protocol.json', dict(config.protocol(), protocol_sha256=config.fingerprint()))
        rows, population = collect_jobs(args)
        options = {name: getattr(args, name) for name in ('mode', 'npy_order', 'npy_layout', 'npy_range', 'expected_size', 'save_labels')}
        options['output_dir'] = str(output)
        if args.save_labels:
            (output / 'labels').mkdir()
        snapshot = output / 'source'
        snapshot.mkdir()
        shutil.copy2(__file__, snapshot / Path(__file__).name)
        shutil.copytree(ROOT / 'bf_rgb_metrics', snapshot / 'bf_rgb_metrics', ignore=shutil.ignore_patterns('__pycache__'))
        records, all_metrics, grouped = [], MetricAccumulator(), {}
        pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
        jobs = [(row, asdict(config), options) for row in rows]
        logger.info('%s: %d/%d FOVs, protocol=%s, workers=%d', args.mode, len(rows), population, config.fingerprint(), args.workers)
        try:
            results = pool.map(run_item, jobs) if pool else map(run_item, jobs)
            with (output / ('counts.jsonl' if args.mode == 'count' else 'metrics_fov.jsonl')).open('w') as stream:
                for record in results:
                    records.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    if args.mode == 'evaluate':
                        all_metrics.update(record)
                        grouped.setdefault(record['source_dataset'], MetricAccumulator()).update(record)
                    logger.info('%d/%d %s complete', len(records), len(rows), record['sample_id'])
        finally:
            if pool:
                pool.shutdown(wait=True, cancel_futures=True)
        report = {'status': 'complete', 'mode': args.mode, 'population_count': population,
                  'processed_count': len(records), 'complete_population': len(records) == population,
                  'protocol_sha256': config.fingerprint(), 'protocol': config.protocol(),
                  'input_settings': options, 'detection_parameters_fitted': False,
                  'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  'input_manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest() if args.manifest else None,
                  'definition': 'Agreement of the fixed detector applied independently to prediction and GT; not a human-annotated instance benchmark.',
                  'runtime': {'python': sys.version, 'executable': sys.executable,
                              'numpy': np.__version__, 'scipy': scipy.__version__, 'opencv': cv2.__version__},
                  'source_sha256': {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(snapshot.rglob('*.py'))},
                  'seconds': time.monotonic() - started}
        if args.mode == 'evaluate':
            write_csv(output / 'metrics.csv', [flatten_metrics(r) for r in records])
            report.update(metrics=all_metrics.summary(), per_source={k: v.summary() for k, v in grouped.items()})
        else:
            write_csv(output / 'counts.csv', [flatten_count(r) for r in records])
            report.update(counts=counts_summary(records), per_source={source: counts_summary([r for r in records if r['source_dataset'] == source])
                         for source in sorted({r['source_dataset'] for r in records})})
        write_json(output / 'summary.json', report)
        write_json(output / '_SUCCESS', {'processed_count': len(records), 'complete_population': report['complete_population'],
                                        'protocol_sha256': config.fingerprint()})
        logger.info('SUCCESS %s', output)
        return report
    except BaseException:
        failure = traceback.format_exc()
        (output / '_FAILED').write_text(failure)
        logger.error(failure)
        raise
    finally:
        (output / '_RUNNING').unlink(missing_ok=True)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--output-dir', type=Path, required=True)
    common.add_argument('--config', type=Path, help='Detector config JSON, e.g. configs/rgb_object_metrics.json')
    for name, default in (('red_threshold16', 14906), ('blue_threshold16', 6425), ('green_threshold16', 5140)):
        common.add_argument('--' + name.replace('_', '-'), type=float, help=f'0..65535 intensity; default {default}')
    common.add_argument('--red-min-area', type=int, help='Minimum R instance area, default 15 pixels')
    common.add_argument('--blue-min-area', type=int, help='Minimum B instance area, default 50 pixels')
    common.add_argument('--blue-threshold-mode', choices=('otsu', 'fixed'),
                        help='Blue threshold: per-FOV otsu (default), or explicit legacy fixed mode')
    common.add_argument('--iou-threshold', type=float, help='Minimum valid instance IoU, default 0.5')
    common.add_argument('--npy-order', choices=('RGB', 'BGR'), default='BGR')
    common.add_argument('--npy-layout', choices=('CHW', 'HWC'), default='CHW')
    common.add_argument('--npy-range', choices=('unit', 'uint8', 'uint16', 'signed_unit'), default='unit')
    common.add_argument('--expected-size', type=int, default=1080, help='Required square full-FOV size; 0 disables the size gate')
    common.add_argument('--workers', type=int, default=1)
    common.add_argument('--limit', type=int, help='Explicit partial smoke run; summary marks incomplete population')
    common.add_argument('--save-labels', action='store_true', help='Save foreground masks and R/B instance labels as NPZ')
    common.add_argument('--recursive', action='store_true', help='Recurse when reading an input/target directory')
    common.add_argument('--input-pattern', help='Directory input filter, e.g. *.npy when prediction folders contain both NPY and PNG exports')
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest='mode', required=True)
    count = sub.add_parser('count', parents=[common], help='Analyze one image, directory or manifest')
    source = count.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path)
    source.add_argument('--manifest', type=Path)
    count.add_argument('--image-column', help='Manifest image path column; auto target_rgb/target_png/image_path')
    evaluate = sub.add_parser('evaluate', parents=[common], help='Evaluate prediction/GT pairs; preserve existing pixel metrics')
    target = evaluate.add_mutually_exclusive_group(required=True)
    target.add_argument('--manifest', type=Path)
    target.add_argument('--target-dir', type=Path)
    evaluate.add_argument('--target-column', help='Manifest GT column; auto target_rgb/target_png/image_path')
    evaluate.add_argument('--prediction-dir', type=Path)
    evaluate.add_argument('--prediction-template', default='{sample_id}.npy', help='Relative filename template; use {sample_id}.png for PNG16')
    evaluate.add_argument('--prediction-column', default='prediction_path', help='Manifest prediction path if prediction-dir is omitted')
    return result


def main():
    args = parser().parse_args()
    if args.workers < 1 or args.expected_size < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError('workers/limit must be positive and expected-size nonnegative')
    if args.mode == 'evaluate' and args.target_dir and not args.prediction_dir:
        raise ValueError('target-dir mode requires prediction-dir')
    execute(args)


if __name__ == '__main__':
    main()
