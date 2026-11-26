#!/usr/bin/env python3
"""Run the existing watershed counter, then add exact PNG16 intensity statistics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _lib.png_render import png_header


SCRIPTS = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPTS.parent
DATA = DEFAULT_DATASET
CHANNELS = ('R', 'G', 'B')
QC_IDS = {'049c3f86_B02_F01_P01_T01', '049c3f86_B05_F09_P01_T01',
          '077b1a14_B02_F01_P01_T01', '077b1a14_B05_F09_P01_T01',
          'ab11d61a_B02_F01_P01_T01', 'ab11d61a_D06_F01_P01_T01'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def intensity_statistics(values, foreground):
    if values.dtype != np.uint16 or foreground.dtype != np.bool_ or values.shape != foreground.shape:
        raise ValueError('Expected native uint16 intensities and a matching boolean mask')
    total = int(values.sum(dtype=np.uint64))
    foreground_pixels = int(np.count_nonzero(foreground))
    foreground_sum = int(values[foreground].sum(dtype=np.uint64))
    return {'foreground_pixels': foreground_pixels,
            'coverage_percent': 100 * foreground_pixels / values.size,
            'mean_intensity16': total / values.size,
            'foreground_mean_intensity16': foreground_sum / foreground_pixels if foreground_pixels else None,
            'integrated_intensity16': total,
            'foreground_integrated_intensity16': foreground_sum}


def self_check():
    pixels = np.array([[0, 10], [20, 30]], dtype=np.uint16)
    s = intensity_statistics(pixels, pixels >= 20)
    assert s == {'foreground_pixels': 2, 'coverage_percent': 50.0, 'mean_intensity16': 15.0,
                 'foreground_mean_intensity16': 25.0, 'integrated_intensity16': 60,
                 'foreground_integrated_intensity16': 50}
    empty = intensity_statistics(pixels, np.zeros_like(pixels, dtype=bool))
    assert empty['foreground_mean_intensity16'] is None and empty['foreground_integrated_intensity16'] == 0
    maximum = intensity_statistics(np.full((2, 2), 65535, np.uint16), np.ones((2, 2), bool))
    assert maximum['integrated_intensity16'] == 262140 and maximum['mean_intensity16'] == 65535


def make_qc(row, bgr, labels, output):
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 22)
        small = ImageFont.truetype('DejaVuSans.ttf', 18)
    except OSError:
        font = ImageFont.load_default(size=22)
        small = ImageFont.load_default(size=18)
    canvas = Image.new('RGB', (1140, 490), '#f4f5f7')
    draw = ImageDraw.Draw(canvas)
    dose = f" {float(row['dose_value']):g}" if row['dose_value'] else ''
    title = f"{row['acquisition']} | {row['compound_label']}{dose} | {row['well']} F{int(row['field']):02d}"
    draw.text((15, 10), title, fill='#121820', font=small)
    rgb8 = ((bgr[..., ::-1].astype(np.uint32) + 128) // 257).astype(np.uint8)
    panels = [('PNG composite', rgb8)]
    for c in ('B', 'R'):
        lab = labels['labels_' + c]
        rng = np.random.default_rng(12345)
        colors = rng.integers(60, 256, size=(int(lab.max()) + 1, 3), dtype=np.uint16).astype(np.uint8)
        colors[0] = 0
        panels.append((f"{c} watershed count: {row[c + '_count']}", colors[lab]))
    for i, (title, image) in enumerate(panels):
        draw.text((15 + 375 * i, 45), title, fill='#121820', font=font)
        canvas.paste(Image.fromarray(image).resize((360, 360), Image.Resampling.NEAREST if i else Image.Resampling.LANCZOS),
                     (15 + 375 * i, 82))
    draw.text((15, 457), f"R/B={row['R_B_ratio']:.4f}" if row['R_B_ratio'] is not None else 'R/B undefined (B=0)',
              fill='#404956', font=small)
    canvas.save(Path(output) / 'qc' / (row['sample_id'] + '.png'))


def enrich_one(job):
    cv2.setNumThreads(1)
    source_row, count_row, output = job
    path = Path(source_row['target_png'])
    encoded = path.read_bytes()
    image_sha = hashlib.sha256(encoded).hexdigest()
    if image_sha != count_row['input_sha256']:
        raise ValueError(f'PNG changed since counting: {path}')
    bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.dtype != np.uint16 or bgr.shape != (1080, 1080, 3):
        raise ValueError(f'Expected native 1080x1080 RGB16 PNG: {path}')
    label_path = Path(output) / 'counts' / 'labels' / (source_row['sample_id'] + '.npz')
    with np.load(label_path, allow_pickle=False) as npz:
        labels = {key: npz[key] for key in ('labels_R', 'labels_B', 'foreground_R', 'foreground_G', 'foreground_B')}
        thresholds = npz['thresholds16_RGB']
    row = {key: source_row[key] for key in ('sample_id', 'acquisition', 'measurement_id', 'compound_label',
                                          'dose_value', 'dose_unit', 'well', 'field', 'split_group_id')}
    row.update(image_path=str(path), image_sha256=image_sha, pixel_count=1080 * 1080,
               B_count=int(count_row['B_count']), R_count=int(count_row['R_count']))
    for c in ('R', 'B'):
        lab = labels['labels_' + c]
        if int(lab.max()) != row[c + '_count']:
            raise ValueError(f'Instance labels/count mismatch: {source_row["sample_id"]}')
    row['R_B_ratio'] = row['R_count'] / row['B_count'] if row['B_count'] else None
    row['R_B_ratio_percent'] = row['R_B_ratio'] * 100 if row['R_B_ratio'] is not None else None
    for i, c in enumerate(CHANNELS):
        values = bgr[..., 2 - i]
        mask = labels['foreground_' + c]
        threshold = float(thresholds[i])
        native_mask = values >= threshold if c == 'R' else values > threshold
        if c == 'B' and values.min() == values.max():
            native_mask[:] = False
        if not np.array_equal(mask, native_mask):
            raise ValueError(f'Native PNG threshold/mask mismatch: {source_row["sample_id"]} {c}')
        stats = intensity_statistics(values, mask)
        if stats['foreground_pixels'] != int(count_row[c + '_foreground_pixels']):
            raise ValueError('Counter coverage and saved foreground mask differ')
        row[c + '_threshold16'] = threshold
        row.update({c + '_' + key: value for key, value in stats.items()})
    row['labels_npz'] = str(label_path)
    if row['sample_id'] in QC_IDS:
        make_qc(row, bgr, labels, output)
    return row


def pool_rows(rows, identity):
    result = dict(identity)
    result.update(n_fovs=sum(r.get('n_fovs', 1) for r in rows),
                  n_wells=len({(r['acquisition'], r['well']) for r in rows}) if 'well' in rows[0] else len(rows),
                  pixel_count=sum(r['pixel_count'] for r in rows),
                  B_count=sum(r['B_count'] for r in rows), R_count=sum(r['R_count'] for r in rows))
    result['B_count_mean_per_fov'] = result['B_count'] / result['n_fovs']
    result['R_count_mean_per_fov'] = result['R_count'] / result['n_fovs']
    result['R_B_ratio'] = result['R_count'] / result['B_count'] if result['B_count'] else None
    result['R_B_ratio_percent'] = result['R_B_ratio'] * 100 if result['R_B_ratio'] is not None else None
    for c in CHANNELS:
        fg = sum(r[c + '_foreground_pixels'] for r in rows)
        total = sum(r[c + '_integrated_intensity16'] for r in rows)
        fg_total = sum(r[c + '_foreground_integrated_intensity16'] for r in rows)
        result.update({c + '_foreground_pixels': fg,
                       c + '_coverage_percent': 100 * fg / result['pixel_count'],
                       c + '_mean_intensity16': total / result['pixel_count'],
                       c + '_foreground_mean_intensity16': fg_total / fg if fg else None,
                       c + '_integrated_intensity16': total,
                       c + '_foreground_integrated_intensity16': fg_total})
    return result


def grouped_report(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [pool_rows(groups[key], dict(zip(keys, key))) for key in sorted(groups)]


def main():
    global DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--input-dir', type=Path, help='RGB16 PNG root; default: DATASET/previews.')
    parser.add_argument('--output-dir', type=Path, help='New result directory; default: DATASET/analysis/rgb_metrics_TIMESTAMP.')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    self_check()
    DATA = args.dataset_root.absolute()
    png_root = args.input_dir.absolute() if args.input_dir else DATA / 'previews'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output_dir.absolute() if args.output_dir else DATA / 'analysis' / ('rgb_metrics_' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('inputs', 'source', 'qc'):
        (output / name).mkdir()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output / 'pipeline.log')])
    started = time.monotonic()
    counter = None
    (output / '_RUNNING').touch()
    try:
        metadata_files = sorted((DATA / 'metadata').glob('*.csv'))
        metadata_sha = {p.name: sha(p) for p in metadata_files}
        for path in metadata_files:
            shutil.copy2(path, output / 'inputs' / path.name)
        shutil.copy2(__file__, output / 'source' / Path(__file__).name)
        shutil.copytree(SCRIPTS / '_lib', output / 'source/_lib', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(SCRIPTS / 'rgb_object_metrics.json', output / 'source/rgb_object_metrics.json')
        shutil.copy2(SCRIPTS / 'rgb_object_metrics.json', output / 'source/detector_config.json')
        fovs = read_csv(DATA / 'metadata/fovs.csv')
        excluded = [r for r in fovs if r['eligible_four_channel'].lower() != 'true']
        selected = [r for r in fovs if r['eligible_four_channel'].lower() == 'true']
        manifest = []
        for row in selected:
            parent = Path(row['ch1_path']).parent.relative_to('raw_data')
            path = png_root / parent / (row['fov_id'] + '_merge.png')
            if not path.is_file():
                raise FileNotFoundError(path)
            header = png_header(path)
            if (header['width'], header['height']) != (1080, 1080):
                raise ValueError(f'Expected full-resolution 1080x1080 RGB16 PNG: {path}')
            manifest.append({**{k: row[k] for k in ('acquisition', 'measurement_id', 'compound_label', 'dose_value',
                                                    'dose_unit', 'well', 'field', 'split_group_id')},
                             'sample_id': row['fov_id'], 'source_dataset': row['acquisition'], 'target_png': str(path)})
        if len({r['sample_id'] for r in manifest}) != len(manifest):
            raise ValueError('Duplicate FOV IDs')
        actual = {p for acq in {r['acquisition'] for r in manifest} for p in (png_root / acq).rglob('*.png')}
        if actual != {Path(r['target_png']) for r in manifest}:
            raise ValueError('PNG coverage differs from metadata')
        write_csv(output / 'inputs/manifest.csv', manifest)
        write_csv(output / 'excluded_fovs.csv', excluded)
        total = len(manifest)
        logging.info('Starting unchanged watershed counter on %d PNG16 FOVs; %d metadata exclusions', total, len(excluded))
        command = [sys.executable, str(SCRIPTS / '_lib/counter_cli.py'), 'count',
                   '--manifest', str(output / 'inputs/manifest.csv'), '--image-column', 'target_png',
                   '--config', str(SCRIPTS / 'rgb_object_metrics.json'), '--output-dir', str(output / 'counts'),
                   '--workers', str(args.workers), '--save-labels']
        write_json(output / 'command.json', {'argv': command})
        environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (output / 'counter_console.log').open('w') as console:
            counter = subprocess.Popen(command, cwd=SCRIPTS, env=environment, stdout=console, stderr=subprocess.STDOUT)
            while True:
                try:
                    code = counter.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    path = output / 'counts/counts.jsonl'
                    count = sum(1 for _ in path.open()) if path.exists() else 0
                    elapsed = time.monotonic() - started
                    write_json(output / 'progress.json', {'stage': 'watershed_counting', 'completed': count,
                                                          'total': total, 'elapsed_seconds': elapsed})
                    logging.info('Watershed count: %d/%d; elapsed %.1fs', count, total, elapsed)
        if code:
            raise RuntimeError(f'Existing watershed CLI failed with exit code {code}; see counter_console.log')
        count_report = json.loads((output / 'counts/summary.json').read_text())
        if count_report['processed_count'] != total or not count_report['complete_population']:
            raise ValueError('Counter did not complete the complete PNG population')
        counts = {r['sample_id']: r for r in read_csv(output / 'counts/counts.csv')}
        with (output / 'counts/counts.jsonl').open() as stream:
            for line in stream:
                record = json.loads(line)
                counts[record['sample_id']]['input_sha256'] = record['input']['file_sha256']
        logging.info('Watershed complete; calculating exact intensities and verifying masks for all %d FOVs', total)
        enriched = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(enrich_one, (row, counts[row['sample_id']], str(output))): row['sample_id'] for row in manifest}
            for future in as_completed(futures):
                row = future.result()
                enriched[row['sample_id']] = row
                if len(enriched) % 50 == 0 or len(enriched) == total:
                    write_json(output / 'progress.json', {'stage': 'intensity_and_mask_verification',
                                                          'completed': len(enriched), 'total': total,
                                                          'elapsed_seconds': time.monotonic() - started})
                    logging.info('Intensity/mask verification: %d/%d', len(enriched), total)
        records = [enriched[r['sample_id']] for r in manifest]
        write_csv(output / 'per_fov.csv', records)
        well_keys = ['acquisition', 'measurement_id', 'compound_label', 'dose_value', 'dose_unit', 'well']
        wells = grouped_report(records, well_keys)
        write_csv(output / 'per_well.csv', wells)
        condition_keys = ['acquisition', 'compound_label', 'dose_value', 'dose_unit']
        conditions = grouped_report(records, condition_keys)
        well_groups = defaultdict(list)
        for row in wells:
            well_groups[tuple(row[k] for k in condition_keys)].append(row)
        metrics = ['B_count_mean_per_fov', 'R_count_mean_per_fov', 'R_B_ratio']
        metrics += [c + '_' + metric for c in CHANNELS for metric in
                    ('coverage_percent', 'mean_intensity16', 'foreground_mean_intensity16')]
        for condition in conditions:
            wr = well_groups[tuple(condition[k] for k in condition_keys)]
            for metric in metrics:
                values = [w[metric] for w in wr if w[metric] is not None]
                condition[metric + '_well_mean'] = statistics.mean(values) if values else None
                condition[metric + '_well_sd'] = statistics.stdev(values) if len(values) > 1 else None
                condition[metric + '_n_valid_wells'] = len(values)
        write_csv(output / 'per_condition.csv', conditions)
        acquisition = grouped_report(records, ['acquisition', 'measurement_id'])
        write_csv(output / 'per_acquisition.csv', acquisition)
        if len(records) != total or len(list((output / 'counts/labels').glob('*.npz'))) != total:
            raise ValueError('Output record or segmentation-label coverage mismatch')
        for path in metadata_files:
            if sha(path) != metadata_sha[path.name]:
                raise ValueError('Source metadata changed during analysis')
        definitions = {'input_domain': 'Converted data_red RGB16 PNG intensity, not raw TIFF fluorescence intensity.',
                       'B_count_R_count': 'Watershed-separated, hole-filled, minimum-area-filtered blue/red instance counts.',
                       'R_B_ratio': 'R_count/B_count; null/empty CSV when B_count=0. Ratios are not clipped to [0,1].',
                       'coverage_percent': '100 * raw threshold foreground pixels / all image pixels, before watershed and area filtering.',
                       'mean_intensity16': 'Mean of all pixels in the selected physical channel, on the PNG 0..65535 scale.',
                       'foreground_mean_intensity16': 'Mean over the same threshold foreground used for coverage; null if empty.',
                       'integrated_intensity16': 'Exact uint64 sum of all native uint16 channel values.',
                       'foreground_integrated_intensity16': 'Exact uint64 sum over threshold foreground pixels.',
                       'thresholds': 'Bundled rgb_object_metrics.json: blue native16 Otsu per full FOV; R>=14906; G>5140; R min area15, B min area50.',
                       'aggregation': 'Per-well and pooled condition values use count/pixel/intensity sums. Condition *_well_mean and *_well_sd give equal-well mean and sample SD (ddof=1) across valid well records.',
                       'replication': 'n_wells counts acquisition/well records; independent biological replication is not established.',
                       'interpretation': 'No additional stretch, resizing or background subtraction. Existing PNG Auto/red0.9 preprocessing remains baked into these measurements. Nuclear/viability identities are not independently validated.'}
        write_json(output / 'metric_definitions.json', definitions)
        summary = {'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(),
                   'output_dir': str(output), 'dataset': str(DATA), 'input_domain': definitions['input_domain'],
                   'processed_fovs': total, 'excluded_fovs': len(excluded), 'well_records': len(wells),
                   'condition_records': len(conditions), 'by_acquisition': dict(Counter(r['acquisition'] for r in records)),
                   'segmentation_label_files': total, 'qc_images': len(list((output / 'qc').glob('*.png'))),
                   'PNG_hashes_unchanged_after_counting': total, 'native_threshold_masks_verified': total * 3,
                   'B_count_total': sum(r['B_count'] for r in records), 'R_count_total': sum(r['R_count'] for r in records),
                   'zero_blue_count_fovs': sum(r['B_count'] == 0 for r in records),
                   'empty_foreground_fovs': {c: sum(r[c + '_foreground_pixels'] == 0 for r in records) for c in CHANNELS},
                   'protocol_sha256': count_report['protocol_sha256'], 'source_metadata_unchanged': True,
                   'source_metadata_sha256': metadata_sha, 'elapsed_seconds': time.monotonic() - started}
        write_json(output / 'summary.json', summary)
        (output / 'README.txt').write_text(
            '本结果基于已转换的 data_red RGB16 PNG（含逐图Auto和红通道0.9处理），不表示原始TIFF荧光强度。\n'
            'per_fov.csv：逐视野计数、红蓝比、RGB覆盖率、整图平均/前景平均/整图积分/前景积分强度及阈值。\n'
            'per_well.csv：按采集批次及孔汇总；计数为视野计数之和，同时给出每视野均值。\n'
            'per_condition.csv：按批次、药物和浓度汇总，附各孔指标的等权均值和样本标准差。\n'
            'per_acquisition.csv：按采集批次汇总。不同倍率和孔中的计数不代表去重后的独立细胞总数。\n'
            'counts/：原计数脚本的完整输出、协议、源码快照和所有分水岭实例/前景NPZ。\n'
            'qc/：六个代表视野的原合成图、蓝色实例和红色实例可视化。\n'
            '红蓝比=红色实例数/蓝色实例数；蓝色数为0时留空。覆盖率使用分水岭前的阈值掩膜。\n'
            '强度单位为PNG的0–65535灰度值；前景为空时前景平均值留空、前景积分值为0。\n'
            '字段定义详见metric_definitions.json。原始图像和元数据均只读。\n')
        (output / '_RUNNING').unlink()
        (output / '_SUCCESS').touch()
        write_json(output / 'progress.json', {'stage': 'complete', 'completed': total, 'total': total})
        logging.info('SUCCESS: %d FOVs, %d well records, %d condition records; %s', total, len(wells), len(conditions), output)
    except BaseException:
        if counter is not None and counter.poll() is None:
            counter.terminate()
            try:
                counter.wait(timeout=10)
            except subprocess.TimeoutExpired:
                counter.kill()
        (output / '_FAILED').write_text(traceback.format_exc())
        (output / '_RUNNING').unlink(missing_ok=True)
        logging.exception('Pipeline failed')
        raise


if __name__ == '__main__':
    main()
