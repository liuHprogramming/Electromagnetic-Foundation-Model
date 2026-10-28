"""Independently audit selected inputs, split membership and every saved prediction."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    root = Path.cwd()
    start = time.time()
    manifest = json.loads((root / 'data/manifest.json').read_text())
    cfg = json.loads((root / 'code/protocol.json').read_text())
    records = manifest['records']
    assert sha(root / 'code/protocol.json') == manifest['protocol_sha256']
    summary = json.loads((root / 'results/summary.json').read_text())
    y = np.array([r['class_id'] for r in records])
    device = np.array([r['device'] for r in records])
    day = np.array([r['day'] for r in records])
    ids = np.array([r['sample_id'] for r in records])
    x = np.load(root / 'data/spectrograms.npy', mmap_mode='r')
    assert len(records) == len(x) == summary['n']
    assert len(set(r['input_sha256'] for r in records)) == len(records)
    buckets = collections.defaultdict(list)
    classes_per_day = collections.Counter()
    # Re-read source cache members; confirm sample pointers, states and inputs.
    src = root.parent / 'magnetic_dataset_20261026'
    for dev in cfg['devices']:
        rec = manifest['source_objects']
        def original(name):
            return np.load(src / 'objects' / (rec[dev + '/' + name]['sha256'] + '.npy'), mmap_mode='r', allow_pickle=False)
        xx, yy, dd, names, starts = [original(name) for name in ['Xm', 'y_status', 'day', 'day_names', 'wall_start']]
        for i in np.flatnonzero(device == dev):
            r = records[i]
            j = r['source_index']
            assert np.array_equal(x[i], xx[j])
            assert np.all(yy[j] == r['class_id']) and str(names[dd[j]]) == r['day']
            assert float(starts[j]) == r['relative_start_s']
            assert hashlib.sha256(np.asarray(x[i], dtype='<f4').tobytes()).hexdigest() == r['input_sha256']
            buckets[(dev, r['day'], r['inferred_cache_block'])].append(r['relative_start_s'])
            classes_per_day[(dev, r['day'], r['class_id'])] += 1
    for starts in buckets.values():
        assert np.all(np.diff(sorted(starts)) >= 20 - 1e-5)
    assert max(classes_per_day.values()) <= cfg['max_windows_per_device_day_class']
    audit = json.loads((root / 'results/split_audit.json').read_text())
    splits = {}
    for r in audit:
        target = r['target']
        protocol = r['protocol']
        if protocol == 'leave_device':
            train, test = np.flatnonzero(device != target), np.flatnonzero(device == target)
        elif protocol == 'within_device_leave_day':
            date = r['fold'].split('/')[1]
            train = np.flatnonzero((device == target) & (day != date))
            test = np.flatnonzero((device == target) & (day == date))
        else:
            date = r['calibration_day']
            test = np.flatnonzero((device == target) & (day != date))
            train = np.flatnonzero((device != target) | ((device == target) & (day == date) & (protocol == 'calibration_one_day')))
        assert len(train) == r['n_train'] and len(test) == r['n_test']
        assert not {(device[i], day[i]) for i in train} & {(device[i], day[i]) for i in test}
        for indices, name in [(train, 'train'), (test, 'test')]:
            assert hashlib.sha256('\n'.join(ids[indices]).encode()).hexdigest() == r[name + '_ids_sha256']
        splits[(protocol, r['fold'])] = (train, test)
    saved = collections.defaultdict(dict)
    seen = set()
    row_count = 0
    with gzip.open(root / 'results/predictions.csv.gz', 'rt', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            method, protocol, fold = r['method'], r['protocol'], r['fold']
            ix, truth, pred = int(r['sample_index']), int(r['true']), int(r['prediction'])
            assert truth == y[ix]
            assert pred == int(np.argmax([float(r['s' + str(i)]) for i in range(4)]))
            key = (method, protocol, fold, ix)
            assert key not in seen
            seen.add(key)
            saved[(method, protocol, fold)][ix] = pred
            row_count += 1
    for method, info in summary['methods'].items():
        grouped = collections.defaultdict(dict)
        for m in info['fold_metrics']:
            k = (method, m['protocol'], m['fold'])
            prediction = saved[k]
            indices = np.array(sorted(prediction))
            assert set(indices) == set(splits[(m['protocol'], m['fold'])][1])
            pr = np.array([prediction[i] for i in indices])
            assert abs(f1_score(y[indices], pr, labels=[0, 1, 2, 3], average='macro', zero_division=0) - m['macro_f1']) < 1e-12
            assert abs(accuracy_score(y[indices], pr) - m['accuracy']) < 1e-12
            assert confusion_matrix(y[indices], pr, labels=[0, 1, 2, 3]).tolist() == m['confusion']
            assert not set(grouped[m['protocol']]) & set(prediction)
            grouped[m['protocol']].update(prediction)
        for protocol, preds in grouped.items():
            values = []
            for dev in cfg['devices']:
                ix = np.array([i for i in sorted(preds) if device[i] == dev])
                pr = np.array([preds[i] for i in ix])
                f1 = f1_score(y[ix], pr, labels=[0, 1, 2, 3], average='macro', zero_division=0)
                values.append(f1)
                assert abs(f1 - info['protocols'][protocol]['per_device'][dev]['macro_f1']) < 1e-12
            assert abs(np.mean(values) - info['protocols'][protocol]['device_mean_macro_f1']) < 1e-12
    assert sha(root / 'results/predictions.csv.gz') == summary['prediction_file_sha256']
    receipt = json.loads((root / 'features/model_receipt.json').read_text())
    for name, item in receipt['artifacts'].items():
        assert sha(root / 'features' / name) == item['sha256']
    result = {'state': 'passed', 'selected_windows': len(records), 'device_days': len(set(zip(device, day))),
              'methods': len(summary['methods']), 'splits_per_method': len(audit), 'prediction_rows': row_count,
              'source_pointers_labels_inputs': 'every selected window rechecked against immutable source objects',
              'all_saved_fold_metrics': 'independently recalculated', 'split_membership': 'independently reconstructed',
              'score_argmax': 'every saved prediction checked', 'selected_exact_hash_duplicates': 0,
              'cache_spacing_minimum_s': 20, 'day_class_cap': cfg['max_windows_per_device_day_class'],
              'bootstrap_scope': summary['interval_scope'],
              'elapsed_seconds': time.time() - start,
              'limitations': cfg['limitations']}
    (root / 'results/verification.json').write_text(json.dumps(result, indent=2))
    print('VERIFICATION_PASSED', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
