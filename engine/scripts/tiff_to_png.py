#!/usr/bin/env python3
"""Render all metadata-indexed drug-response FOVs with the historical data_red exporter.

RGB16 PNGs mirror the raw_data paths recorded by metadata/fovs.csv directly under
previews/. Browser RGB8 PNGs mirror the same paths under previews/_rgb8/.
Source TIFFs and source metadata are opened read-only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image

from _lib import png_render as renderer


DEFAULT_DATASET = Path(__file__).resolve().parent.parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, records: list[dict], fields: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def relative_folder(row: dict) -> Path:
    parents = {Path(row[f'ch{channel}_path']).parent for channel in (1, 2, 3, 4)}
    if len(parents) != 1:
        raise ValueError(f"FOV channels use different folders: {row['fov_id']}")
    parent = parents.pop()
    if parent.is_absolute() or '..' in parent.parts:
        raise ValueError(f'Unsafe source relative path: {parent}')
    relative = parent.relative_to('raw_data')
    if not relative.parts or relative.parts[0] != row['acquisition']:
        raise ValueError(f"Acquisition/path mismatch: {row['fov_id']}")
    return relative


def export_one(job: dict) -> dict:
    row, root = job['row'], Path(job['dataset'])
    sources = {f'ch{channel}': str(root / row[f'ch{channel}_path']) for channel in (1, 3, 4)}
    reference = job['reference']
    before = {channel: (Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
              for channel, path in sources.items()}
    record = renderer.export_sample(
        {'sample_id': row['fov_id'], 'sources': sources}, job['output_folder'], 6,
        reference['channels'] if reference else None, 0.9)
    if job['expected_source_sha256'] is not None and record['source_sha256'] != job['expected_source_sha256']:
        raise ValueError(f"Source TIFF SHA-256 differs from source_manifest: {row['fov_id']}")
    if reference and record['png_sha256'] != reference['png_sha256']:
        raise ValueError(f"Rendered PNG differs from preceding data_red PNG: {row['fov_id']}")
    bgr16 = cv2.imread(record['output'], cv2.IMREAD_UNCHANGED)
    rgb8 = ((bgr16[..., ::-1].astype(np.uint32) + 128) // 257).astype(np.uint8)
    browser_path = Path(job['browser_folder']) / (row['fov_id'] + '_merge.png')
    temporary = browser_path.with_name('.' + browser_path.stem + '.part.png')
    Image.fromarray(rgb8).save(temporary)
    with Image.open(temporary) as image:
        if image.mode != 'RGB' or not np.array_equal(np.asarray(image), rgb8):
            raise ValueError(f"RGB8 file round-trip mismatch: {row['fov_id']}")
    temporary.replace(browser_path)
    after = {channel: (Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
             for channel, path in sources.items()}
    if before != after:
        raise ValueError(f"Source TIFF changed during conversion: {row['fov_id']}")
    record.update({key: row[key] for key in ('acquisition', 'measurement_id', 'well', 'field',
                                             'compound_label', 'dose_value', 'dose_unit', 'split_group_id')})
    record.update(browser_png=str(browser_path), browser_sha256=sha(browser_path),
                  browser_bytes=browser_path.stat().st_size, browser_exact_round_trip=True,
                  reference_png_exact_match=bool(reference),
                  source_folder=str(relative_folder(row)))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output-dir', type=Path, help='Output directory; default: DATASET/previews. Existing PNGs are preserved.')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    root = args.dataset_root.absolute()
    output = args.output_dir.absolute() if args.output_dir else root / 'previews'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = output / ('_run_data_red_' + stamp)
    input_paths = [root / 'metadata' / name for name in
                   ('fovs.csv', 'acquisitions.csv', 'well_treatments.csv', 'original_folders.csv', 'H2O2_column_doses.csv')]
    input_paths += [path for path in (root / 'raw_data/source_manifest.json', root / 'raw_data/channel_mapping.json')
                    if path.is_file()]
    input_hashes = {str(path.relative_to(root)): sha(path) for path in input_paths}
    rows = read_csv(root / 'metadata/fovs.csv')
    if not rows or len({r['fov_id'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate metadata FOV IDs')
    indexed = {r[f'ch{channel}_path'] for r in rows for channel in (1, 2, 3, 4)}
    actual = {str((Path(base) / name).relative_to(root))
              for base, _, names in os.walk(root / 'raw_data') for name in names
              if name.lower().endswith(('.tif', '.tiff'))}
    if len(indexed) != len(rows) * 4 or actual != indexed:
        raise ValueError(f'Disk/metadata TIFF coverage mismatch: missing={len(indexed-actual)}, unlisted={len(actual-indexed)}')
    eligible = [r for r in rows if r['eligible_four_channel'].lower() == 'true']
    excluded = [r for r in rows if r['eligible_four_channel'].lower() != 'true']
    if any(any(r[f'ch{channel}_readable'].lower() != 'true' for channel in (1, 2, 3, 4)) for r in eligible):
        raise ValueError('Eligible FOV has an unreadable channel')
    source_hashes = None
    manifest_path = root / 'raw_data/source_manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        source_hashes = {'raw_data/' + r['target_relative_to_raw_data']: r['target_sha256']
                         for r in manifest['files'] if r['kind'] == 'tiff'}
        if set(source_hashes) != indexed:
            raise ValueError('Source manifest does not exactly cover metadata TIFF paths')
    folders = {relative_folder(row) for row in rows}
    for row in read_csv(root / 'metadata/original_folders.csv'):
        folder = Path(row['folder_path']).relative_to('raw_data')
        if folder.is_absolute() or '..' in folder.parts:
            raise ValueError('Unsafe original folder path')
        folders.add(folder)
    jobs = []
    for row in eligible:
        folder = relative_folder(row)
        filename = row['fov_id'] + '_merge.png'
        for destination in (output / folder / filename, output / '_rgb8' / folder / filename):
            if destination.exists():
                raise FileExistsError(f'Preserving existing output; destination exists: {destination}')
        expected = ({f'ch{ch}': source_hashes[row[f'ch{ch}_path']] for ch in (1, 3, 4)}
                    if source_hashes is not None else None)
        jobs.append({'row': row, 'dataset': str(root), 'output_folder': str(output / folder),
                     'browser_folder': str(output / '_rgb8' / folder),
                     'expected_source_sha256': expected, 'reference': None})
    output.mkdir(exist_ok=True)
    run.mkdir(exist_ok=False)
    for folder in folders:
        (output / folder).mkdir(parents=True, exist_ok=True)
        (output / '_rgb8' / folder).mkdir(parents=True, exist_ok=True)
    (run / 'input_metadata').mkdir()
    for path in input_paths[:5]:
        shutil.copy2(path, run / 'input_metadata' / path.name)
    shutil.copy2(__file__, run / Path(__file__).name)
    shutil.copytree(Path(__file__).resolve().parent / '_lib', run / '_lib', ignore=shutil.ignore_patterns('__pycache__'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run / 'run.log')])
    started = time.monotonic()
    completed = {}
    (run / '_RUNNING').touch()
    config = {'run_dir': str(run), 'dataset_root': str(root), 'output_dir': str(output),
              'metadata_fovs': len(rows), 'eligible_fovs': len(eligible), 'excluded_fovs': len(excluded),
              'source_tiffs': len(actual), 'workers': args.workers, 'red_range_fraction': 0.9,
              'expected_reference_matches': sum(bool(j['reference']) for j in jobs),
              'mirrored_condition_directories': len(folders), 'input_sha256': input_hashes,
              'renderer_sha256': sha(Path(renderer.__file__)),
              'source_manifest_available': source_hashes is not None}
    write_json(run / 'config.json', config)
    write_json(output / 'latest_export.json', {'status': 'running', **config})

    def progress(status: str) -> None:
        elapsed = time.monotonic() - started
        count = len(completed)
        write_json(run / 'progress.json', {'status': status, 'completed': count, 'total': len(jobs),
                                          'elapsed_seconds': elapsed,
                                          'estimated_remaining_seconds': elapsed * (len(jobs)-count) / count if count else None,
                                          'completed_by_acquisition': dict(Counter(r['acquisition'] for r in completed.values()))})

    try:
        logging.info('START: %d indexed FOVs; exporting %d, excluding %d; %d reference PNGs; %d condition directories',
                     len(rows), len(jobs), len(excluded), config['expected_reference_matches'], len(folders))
        write_csv(run / 'skipped_fovs.csv', excluded, list(rows[0]))
        shutil.copy2(run / 'skipped_fovs.csv', output / 'skipped_fovs.csv')
        progress('running')
        with (run / 'channels.jsonl').open('w') as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(export_one, job): job['row']['fov_id'] for job in jobs}
            for future in as_completed(futures):
                record = future.result()
                completed[record['sample_id']] = record
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                if len(completed) == 1 or len(completed) % 25 == 0 or len(completed) == len(jobs):
                    progress('running')
                    logging.info('Completed %d/%d (%.1f%%), elapsed %.1fs', len(completed), len(jobs),
                                 100 * len(completed) / len(jobs), time.monotonic() - started)
        records = [completed[r['fov_id']] for r in eligible]
        expected_rgb16 = {Path(r['output']) for r in records}
        expected_rgb8 = {Path(r['browser_png']) for r in records}
        acquisitions = sorted({r['acquisition'] for r in rows})
        actual_rgb16 = {p for acq in acquisitions for p in (output / acq).rglob('*_merge.png')}
        actual_rgb8 = {p for acq in acquisitions for p in (output / '_rgb8' / acq).rglob('*_merge.png')}
        if actual_rgb16 != expected_rgb16 or actual_rgb8 != expected_rgb8:
            raise ValueError('Final PNG filename coverage differs from the metadata')
        for path in input_paths:
            if sha(path) != input_hashes[str(path.relative_to(root))]:
                raise ValueError(f'Source metadata changed during export: {path}')
        fields = ['sample_id', 'acquisition', 'measurement_id', 'compound_label', 'dose_value', 'dose_unit',
                  'well', 'field', 'source_folder', 'output', 'browser_png', 'png_sha256', 'browser_sha256',
                  'reference_png_exact_match', 'lossless_round_trip']
        write_csv(run / 'outputs.csv', records, fields)
        shutil.copy2(run / 'outputs.csv', output / 'outputs.csv')
        summary = {**config, 'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(),
                   'exported_rgb16_pngs': len(records), 'exported_browser_rgb8_pngs': len(records),
                   'dimensions': [1080, 1080], 'rgb16_bits_per_channel': 16,
                   'channel_mapping': {'ch1': 'blue', 'ch3': 'green', 'ch4': 'red'},
                   'total_rgb16_bytes': sum(r['bytes'] for r in records),
                   'total_browser_bytes': sum(r['browser_bytes'] for r in records),
                   'source_tiff_sha256_matches': len(records) * 3 if source_hashes is not None else 0,
                   'source_tiff_sha256_recorded': len(records) * 3,
                   'verified_lossless_rgb16_round_trips': sum(r['lossless_round_trip'] for r in records),
                   'verified_browser_round_trips': sum(r['browser_exact_round_trip'] for r in records),
                   'reference_png_exact_matches': sum(r['reference_png_exact_match'] for r in records),
                   'by_acquisition': dict(Counter(r['acquisition'] for r in records)),
                   'by_compound': dict(Counter(r['compound_label'] for r in records)),
                   'input_metadata_unchanged': True, 'exact_metadata_path_coverage': True,
                   'elapsed_seconds': time.monotonic() - started}
        write_json(run / 'summary.json', summary)
        write_json(output / 'summary.json', summary)
        write_json(output / 'latest_export.json', summary)
        (output / 'README.txt').write_text(
            'Drug-response fluorescence visualizations indexed by metadata/fovs.csv.\n'
            'acq_*/: full-resolution RGB16 PNGs. Paths mirror raw_data acquisition/condition folders.\n'
            '_rgb8/acq_*/: the same composites as full-resolution RGB8 browser previews.\n'
            'Each FOV merges ch1=blue, ch3=green, ch4=red; brightfield ch2 is not merged.\n'
            'ImageJ Reset -> first Auto per fluorescence channel; red minimum shifted by 0.9 of its Auto span.\n'
            'Browser conversion: uint8((uint32(RGB16) + 128) // 257), without additional contrast or gamma.\n'
            'Each FOV uses independent Auto windows; brightness is not a shared raw-intensity scale.\n'
            'outputs.csv lists every exported FOV and both PNG paths; skipped_fovs.csv lists incomplete FOVs.\n'
            'Original empty condition folders are retained. Source TIFFs and metadata were opened read-only.\n'
            'Rendering windows, hashes and frozen scripts are saved under the _run_data_red_* directory.\n')
        progress('complete')
        (run / '_RUNNING').unlink()
        (run / '_SUCCESS').touch()
        logging.info('SUCCESS: %d RGB16 PNGs + %d browser PNGs; %d skipped FOVs; %.1fs',
                     len(records), len(records), len(excluded), summary['elapsed_seconds'])
    except BaseException as error:
        logging.exception('EXPORT FAILED')
        progress('failed')
        write_json(output / 'latest_export.json', {'status': 'failed', **config,
                                                  'completed': len(completed), 'error': repr(error)})
        (run / '_RUNNING').unlink(missing_ok=True)
        (run / '_FAILED').touch()
        raise


if __name__ == '__main__':
    main()
