"""Create a pinned, conservatively thinned benchmark from immutable cache objects."""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--layout', choices=['mirror', 'dataset'], default='mirror')
    p.add_argument('--out', type=Path, default=Path('data'))
    p.add_argument('--protocol', type=Path, default=Path('code/protocol.json'))
    a = p.parse_args()
    start = time.time()
    cfg = json.loads(a.protocol.read_text())
    a.out.mkdir(exist_ok=True, parents=True)
    meta = a.source if a.layout == 'mirror' else a.source / 'metadata'
    catalog = json.loads((meta / 'FEATURES_STATUS.json').read_text())
    qfile = (a.source / 'results/QUALITY_STATUS.json' if a.layout == 'mirror'
             else meta / 'quality/QUALITY_STATUS.json')
    quality = json.loads(qfile.read_text())
    receipts, accounting, candidates = {}, {}, []
    cls = cfg['classes']
    for dev in cfg['devices']:
        info = quality['canonical'][dev]
        tag = info['dataset_tag']
        source = next(r for r in catalog['archives'] if r['dataset_tag'] == tag)
        members = {r['member']: r for r in source['members']}
        arrays = {}
        for name in ['Xm', 'y_status', 'day', 'wall_start', 'day_names', 'status_names']:
            item = members[name + '.npy']
            path = (a.source / 'objects' / (item['sha256'] + '.npy') if a.layout == 'mirror'
                    else a.source / item['saved'])
            actual = sha(path)
            if actual != item['sha256']:
                raise ValueError('Immutable object SHA mismatch: ' + name)
            arrays[name] = np.load(path, allow_pickle=False, mmap_mode='r')
            receipts[dev + '/' + name] = {'sha256': actual, 'bytes': path.stat().st_size,
                                          'shape': list(arrays[name].shape)}
        x, ys, day, wall, names, classes = [arrays[k] for k in
                                           ['Xm', 'y_status', 'day', 'wall_start', 'day_names', 'status_names']]
        assert x.shape[1:] == (48, 79)
        assert list(classes.astype(str)) == cls
        assert len(x) == len(ys) == len(day) == len(wall)
        valid = np.isfinite(x).all((1, 2)) & (x.std((1, 2)) > 1e-6)
        valid &= (ys.min(1) >= 0) & (ys.max(1) < 4) & (day >= 0) & (day < len(names))
        valid &= np.isfinite(wall) & (wall >= 0)
        pure = (ys == ys[:, :1]).all(1)
        # The merge appends off windows after all running windows. Keep the
        # append boundary separate even if the relative starts happen to rise.
        run = next(r for r in catalog['archives'] if r['dataset_tag'] == tag.replace('_off_all', '_all'))
        n_run = next(r for r in run['members'] if r['member'] == 'Xm.npy')['shape'][0]
        group = -1
        last_selected = -float('inf')
        spaced = []
        for i in range(len(x)):
            if i == 0 or i == n_run or day[i] != day[i - 1] or wall[i] <= wall[i - 1]:
                group += 1
                last_selected = -float('inf')
            if not (valid[i] and pure[i]):
                continue
            if float(wall[i]) < last_selected + cfg['minimum_spacing_seconds_within_inferred_cache_block'] - 1e-5:
                continue
            last_selected = float(wall[i])
            spaced.append((i, group))
        buckets = collections.defaultdict(list)
        for i, block in spaced:
            buckets[(str(names[day[i]]), int(ys[i, 0]))].append((i, block))
        rng = np.random.default_rng(cfg['selection_seed'] + int(dev))
        chosen = []
        for k, bucket in sorted(buckets.items()):
            limit = cfg['max_windows_per_device_day_class']
            idx = np.sort(rng.choice(len(bucket), size=min(limit, len(bucket)), replace=False))
            chosen.extend(bucket[j] for j in idx)
        for i, block in sorted(chosen):
            arr = np.array(x[i], dtype='<f4', copy=True)
            digest = hashlib.sha256(arr.tobytes()).hexdigest()
            candidates.append({'sample_id': tag + ':' + str(i), 'device': dev,
                               'day': str(names[day[i]]), 'class_id': int(ys[i, 0]),
                               'state': cls[int(ys[i, 0])], 'source_index': i, 'dataset_tag': tag,
                               'relative_start_s': float(wall[i]), 'inferred_cache_block': block,
                               'input_sha256': digest, '_x': arr})
        accounting[dev] = {'archive_windows': len(x), 'numeric_valid': int(valid.sum()),
                           'pure_valid': int((valid & pure).sum()), 'after_spacing': len(spaced),
                           'after_day_class_cap': len(chosen), 'active_days': len(set(names[day].astype(str)))}
        print('PREPARED', dev, accounting[dev], flush=True)
    groups = collections.defaultdict(list)
    for r in candidates:
        groups[r['input_sha256']].append(r)
    removed = []
    records, xs = [], []
    for r in candidates:
        same = groups[r['input_sha256']]
        # A shared exact input across dates/devices is excluded entirely. Within
        # the same group, keep only the first copy and drop label conflicts.
        cross_group = len({(s['device'], s['day']) for s in same}) > 1
        conflict = len({s['class_id'] for s in same}) > 1
        if cross_group or conflict or r is not same[0]:
            removed.append({k: v for k, v in r.items() if k != '_x'})
            continue
        xs.append(r.pop('_x'))
        records.append(r)
    x = np.stack(xs).astype('float32')
    assert len({r['input_sha256'] for r in records}) == len(records)
    for dev in cfg['devices']:
        rows = [r for r in records if r['device'] == dev]
        accounting[dev].update({'selected': len(rows), 'selected_days': len({r['day'] for r in rows}),
                                'classes': dict(collections.Counter(r['state'] for r in rows))})
        assert {r['class_id'] for r in rows} == {0, 1, 2, 3}
    np.save(a.out / 'spectrograms.npy', x)
    # Eight fixed contiguous frequency groups. Averaging dB gives a smooth
    # spectral envelope trajectory, not a reconstructed magnetic waveform.
    temporal = x.reshape(len(x), 8, 6, 79).mean(2)
    np.save(a.out / 'band_trajectories.npy', temporal)
    with (a.out / 'samples.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    (a.out / 'removed_duplicates.json').write_text(json.dumps(removed, indent=2))
    (a.out / 'manifest.json').write_text(json.dumps({'protocol_sha256': sha(a.protocol), 'records': records,
                                                    'source_objects': receipts}, indent=2))
    result = {'accounting': accounting, 'selected_windows': len(x),
              'device_days': len({(r['device'], r['day']) for r in records}),
              'removed_duplicates': len(removed), 'input_shape': list(x.shape),
              'elapsed_seconds': time.time() - start,
              'artifacts': {name: {'sha256': sha(a.out / name), 'bytes': (a.out / name).stat().st_size}
                            for name in ['spectrograms.npy', 'band_trajectories.npy', 'manifest.json', 'samples.csv']},
              'scope': 'Archived LP100 feature benchmark; numeric integrity does not certify acquisition clocks or log synchronization.'}
    (a.out / 'data_audit.json').write_text(json.dumps(result, indent=2))
    print('BENCHMARK_READY', json.dumps({k: v for k, v in result.items() if k not in ['artifacts']}), flush=True)


if __name__ == '__main__':
    main()
