"""Read a bounded, reproducible magnetic pilot without scanning 100+ GB of CSVs."""
from pathlib import Path
import argparse
import bisect
import csv
from datetime import datetime
import hashlib
import json
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOTS = {
    '3302': '/data/linqian/deviceManage/5-21DeviceLabels/3302',
    '3303': '/data/linqian/deviceManage/deviceLabel_CN_machine',
    '3304': '/data/linqian/deviceManage/5-21DeviceLabels/3304',
}
CLASSES = ['off', 'speed_up', 'speed_max', 'speed_down']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stamp(text):
    # The acquisition and run-log timestamps both denote local Shanghai time.
    # Retain that wall-clock convention explicitly, irrespective of host timezone.
    return (datetime.fromisoformat(text.strip()).replace(tzinfo=None)
            - datetime(1970, 1, 1)).total_seconds()


def row(line):
    a, b = line.decode('utf-8-sig').strip().split(',')
    return stamp(a), float(b)


def bounds(path):
    with path.open('rb') as f:
        f.readline()
        first = f.readline()
        size = path.stat().st_size
        f.seek(max(0, size - 4096))
        last = f.read().splitlines()[-1]
    return row(first)[0], row(last)[0], digest(first + b'\n' + last)


def read_interval(path, start, end):
    """Binary-search a timestamp-sorted CSV, then read just the requested interval."""
    size = path.stat().st_size
    with path.open('rb') as f:
        f.readline()
        header_end = f.tell()
        lo, hi = header_end, size
        while hi - lo > 32768:
            mid = (lo + hi) // 2
            f.seek(mid)
            f.readline()
            pos = f.tell()
            line = f.readline()
            if not line:
                hi = mid
                continue
            ts, _ = row(line)
            if ts < start:
                lo = pos
            else:
                hi = mid
        f.seek(lo)
        times, values, raw = [], [], []
        byte_start = None
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            ts, value = row(line)
            if ts < start:
                continue
            if ts >= end:
                break
            if byte_start is None:
                byte_start = pos
            times.append(ts)
            values.append(value)
            raw.append(line)
        return np.asarray(times), np.asarray(values), {
            'byte_start': byte_start, 'byte_end': pos,
            'raw_segment_sha256': digest(b''.join(raw)),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-days', type=int, default=12)
    ap.add_argument('--per-class-day', type=int, default=12)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    inventory, exclusions, label_receipts, windows, records, candidates = [], [], [], [], [], []
    selected_days = {}
    for dev, root_str in ROOTS.items():
        root = Path(root_str)
        files = []
        for rel in ('LabeledData/Magnetic', 'LabeledData/off/Magnetic'):
            for p in sorted((root / rel).glob('*.csv')):
                try:
                    a, b, h = bounds(p)
                    if not a < b:
                        raise ValueError('nonpositive span')
                    entry = {'device': dev, 'relative_path': str(p.relative_to(root)),
                             'start': a, 'end': b, 'bytes': p.stat().st_size,
                             'mtime_ns': p.stat().st_mtime_ns, 'head_tail_sha256': h}
                    inventory.append(entry)
                    files.append((a, b, p))
                except Exception as exc:
                    exclusions.append({'file': str(p), 'reason': f'inventory: {exc}'})
        labels = sorted((root / 'matchedLabels_v2').glob('*.csv'))
        available = []
        for p in labels:
            day = p.stem.split('-')[-1]
            if any(f'_{day}_' in q.name for _, _, q in files):
                available.append((day, p))
        indices = np.unique(np.linspace(0, len(available)-1,
                              min(args.max_days, len(available))).round().astype(int))
        selected_days[dev] = [available[j][0] for j in indices]
        for j in indices:
            day, label_file = available[j]
            label_receipts.append({'device': dev, 'day': day, 'file': str(label_file),
                                   'sha256': digest(label_file.read_bytes())})
            with label_file.open(encoding='utf-8-sig') as f:
                labels_day = list(csv.DictReader(f))
            labels_day.sort(key=lambda r: stamp(r['timestamp']))
            ts = [stamp(r['timestamp']) for r in labels_day]
            pools = {c: [] for c in CLASSES}
            for k, r in enumerate(labels_day):
                c = r['status']
                if c not in pools:
                    continue
                t = ts[k]
                # Require the nearest log label to agree across the 2-second
                # window plus a 1-second boundary guard on each side.
                def nearest(x):
                    ix = bisect.bisect_left(ts, x)
                    choices = [q for q in (ix-1, ix) if 0 <= q < len(ts)]
                    return min(choices, key=lambda q: abs(ts[q]-x))
                if any(labels_day[nearest(t+d)]['status'] != c for d in (-2, 2)):
                    continue
                matches = [(a, b, p) for a, b, p in files if a <= t-1.5 and b >= t+1.5]
                if not matches:
                    continue
                # Deterministic choice avoids duplicated overlapping CSV slices.
                a, b, p = min(matches, key=lambda q: (q[1]-q[0], str(q[2])))
                pools[c].append({'device': dev, 'day': day, 'label': CLASSES.index(c),
                                 'state': c, 'center': t, 'source': str(p),
                                 'relative_source': str(p.relative_to(root)),
                                 'log_rpm': float(r['rotation_speed']) if r['rotation_speed'] else None})
            for c, pool in pools.items():
                # Preselect at most 12 candidates per state/day, before quality
                # checking. Do not replace rejected windows with easier cases.
                ranked = sorted(pool, key=lambda r: digest(f"42:{dev}:{day}:{c}:{r['center']}".encode()))
                chosen = []
                for r in ranked:
                    if all(abs(r['center']-q['center']) >= 4 for q in chosen):
                        chosen.append(r)
                    if len(chosen) >= args.per_class_day:
                        break
                candidates.extend(chosen)
    (args.out / 'inventory.json').write_text(json.dumps({
        'roots': ROOTS, 'files': inventory, 'labels': label_receipts,
        'selected_days': selected_days, 'preselected': candidates,
    }, indent=2), encoding='utf-8')
    seen = set()
    for index, item in enumerate(candidates):
        try:
            t = item['center']
            times, values, receipt = read_interval(Path(item['source']), t-1.5, t+1.5)
            if len(times) < 1000 or not np.isfinite(values).all():
                raise ValueError('insufficient or nonfinite samples')
            if (np.diff(times) < 0).any():
                raise ValueError('nonmonotonic timestamps')
            raw_rate = (len(times)-1)/(times[-1]-times[0])
            # Some CSVs repeat the reception timestamp for a complete 150-sample
            # packet. Preserve sample order and values; averaging these readings
            # would destroy the within-packet waveform and its frequency content.
            unique, inv, counts = np.unique(times, return_inverse=True, return_counts=True)
            gaps = np.diff(unique)
            if unique[0] > t-1.40 or unique[-1] < t+1.40 or gaps.max() > .10:
                raise ValueError('packet-timestamp gap >100ms or truncated margin')
            fs = min((4000, 5000), key=lambda r: abs(r-raw_rate))
            if abs(raw_rate/fs-1) > .075:
                raise ValueError(f'raw sample rate {raw_rate} not within 7.5% of 4/5kHz')
            grid = times[0] + np.arange(len(values))/fs
            if abs(grid[-1]-times[-1]) > .10:
                raise ValueError('sample-clock and reception-clock span disagree >100ms')
            filtered = sosfiltfilt(butter(4, 100, fs=fs, output='sos'), values)
            sample = np.interp(t-1+np.arange(512)/256, grid, filtered).astype('<f4')
            if not np.isfinite(sample).all() or sample.std() < 1e-6:
                raise ValueError('nonfinite or constant input')
            h = digest(sample.tobytes())
            if h in seen:
                raise ValueError('duplicate processed input')
            seen.add(h)
            windows.append(sample)
            records.append({**item, **receipt, 'id': len(records), 'input_sha256': h,
                            'start': t-1, 'end': t+1, 'raw_samples_with_margin': len(times),
                            'estimated_raw_hz': raw_rate, 'inferred_sample_clock_hz': fs,
                            'timestamp_group_hz': float(1/np.median(gaps)),
                            'max_same_timestamp_samples': int(counts.max()),
                            'max_timestamp_gap_s': float(gaps.max())})
        except Exception as exc:
            exclusions.append({'candidate': item, 'reason': str(exc)})
        if index % 100 == 0:
            print(f'{index+1}/{len(candidates)} accepted={len(records)} elapsed={time.time()-started:.1f}s', flush=True)
    if not windows:
        raise ValueError('No usable magnetic windows')
    np.save(args.out / 'windows.npy', np.stack(windows))
    manifest = {'classes': CLASSES, 'records': records, 'exclusions': exclusions,
                'n_preselected': len(candidates), 'n_accepted': len(records),
                'window_seconds': 2, 'output_hz': 256, 'lowpass_hz': 100,
                'unit': 'uT (CSV header; sensor calibration not independently verified)',
                'channels': ['Magnetic_Field_uT'], 'elapsed_s': time.time()-started,
                'sample_clock': 'preserve CSV row order; choose 4/5kHz nearest local count/span, require <7.5% rate disagreement and <100ms span discrepancy; packet timestamp anchors first selected sample',
                'sampling': 'at most 12 log-anchored candidates per state/day, at most 12 evenly spaced available dates/device; seed 42',
                'windows_sha256': digest((args.out / 'windows.npy').read_bytes())}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'accepted': len(records), 'preselected': len(candidates),
                      'excluded': len(exclusions), 'selected_days': selected_days,
                      'elapsed_s': manifest['elapsed_s']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
