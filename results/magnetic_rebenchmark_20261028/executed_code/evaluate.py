"""Identical fixed probes; device/date holdouts and paired target calibration."""
import collections
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def metrics(y, pred):
    c = confusion_matrix(y, pred, labels=[0, 1, 2, 3])
    return {'macro_f1': float(f1_score(y, pred, labels=[0, 1, 2, 3], average='macro', zero_division=0)),
            'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'class_recall': np.divide(c.diagonal(), c.sum(1), out=np.zeros(4), where=c.sum(1) != 0).tolist(),
            'confusion': c.tolist(), 'n': len(y)}


def f1_cm(c):
    den = c.sum(-1) + c.sum(-2)
    return np.divide(2 * np.diagonal(c, axis1=-2, axis2=-1), den,
                     out=np.zeros_like(den, dtype='float64'), where=den != 0).mean(-1)


def main():
    root = Path.cwd()
    out = root / 'results'
    out.mkdir(exist_ok=True)
    cfg = json.loads((root / 'code/protocol.json').read_text())
    manifest = json.loads((root / 'data/manifest.json').read_text())
    assert manifest['protocol_sha256'] == sha(root / 'code/protocol.json')
    records = manifest['records']
    n = len(records)
    y = np.array([r['class_id'] for r in records])
    devices = np.array([r['device'] for r in records])
    days = np.array([r['day'] for r in records])
    ids = np.array([r['sample_id'] for r in records])
    hashes = np.array([r['input_sha256'] for r in records])
    assert len(np.unique(hashes)) == n
    receipt = json.loads((root / 'features/model_receipt.json').read_text())
    features = {}
    for name, info in receipt['artifacts'].items():
        p = root / 'features' / name
        assert sha(p) == info['sha256'], 'feature relay mismatch: ' + name
        features[p.stem] = np.load(p, mmap_mode='r')
        assert features[p.stem].shape[0] == n and np.isfinite(features[p.stem]).all()
    features['mantis_plus_spectral'] = np.column_stack([features['mantis_pretrained'], features['spectral_trends']])
    features['majority'] = np.zeros((n, 1), dtype='float32')
    splits = []
    for dev in cfg['devices']:
        splits.append({'protocol': 'leave_device', 'fold': dev, 'train': np.flatnonzero(devices != dev),
                       'test': np.flatnonzero(devices == dev), 'target': dev})
    for dev in cfg['devices']:
        for day in sorted(set(days[devices == dev])):
            splits.append({'protocol': 'within_device_leave_day', 'fold': dev + '/' + day,
                           'train': np.flatnonzero((devices == dev) & (days != day)),
                           'test': np.flatnonzero((devices == dev) & (days == day)), 'target': dev})
    calibration_days = {}
    for dev in cfg['devices']:
        complete = [day for day in sorted(set(days[devices == dev]))
                    if set(y[(devices == dev) & (days == day)]) == {0, 1, 2, 3}]
        assert complete, 'No complete calibration day on ' + dev
        day = complete[0]
        calibration_days[dev] = day
        test = np.flatnonzero((devices == dev) & (days != day))
        for calibrated in [False, True]:
            train = np.flatnonzero((devices != dev) | ((devices == dev) & (days == day) & calibrated))
            splits.append({'protocol': 'calibration_one_day' if calibrated else 'calibration_zero_day_paired',
                           'fold': dev, 'target': dev, 'calibration_day': day, 'train': train, 'test': test})
    audit = []
    for split in splits:
        tr, te = split['train'], split['test']
        assert len(tr) and len(te) and not set(tr) & set(te)
        assert {0, 1, 2, 3} == set(y[tr])
        trgroups = {(r['device'], r['day']) for r in (records[i] for i in tr)}
        tegroups = {(r['device'], r['day']) for r in (records[i] for i in te)}
        assert not trgroups & tegroups
        if split['protocol'] in ['leave_device', 'calibration_zero_day_paired']:
            assert not set(devices[tr]) & set(devices[te])
        audit.append({k: v for k, v in split.items() if k not in ['train', 'test']} |
                     {'n_train': len(tr), 'n_test': len(te), 'train_device_days': sorted(map(list, trgroups)),
                      'test_device_days': sorted(map(list, tegroups)),
                      'train_ids_sha256': hashlib.sha256('\n'.join(ids[tr]).encode()).hexdigest(),
                      'test_ids_sha256': hashlib.sha256('\n'.join(ids[te]).encode()).hexdigest()})
    (out / 'split_audit.json').write_text(json.dumps(audit, indent=2))
    result = {'task': cfg['task'], 'n': n, 'calibration_days': calibration_days,
              'readout': cfg['readout'], 'methods': {}, 'started_unix': time.time(),
              'primary_metric': 'unweighted mean of held-device macro-F1; fixed four-class label set',
              'interval_scope': 'Conditional device-day block bootstrap on these three devices and fixed fitted models; no population inference over unseen hardware.'}
    oof = {}
    with gzip.open(out / 'predictions.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'protocol', 'fold', 'sample_index', 'true', 'prediction', 's0', 's1', 's2', 's3'])
        for key, z in features.items():
            start = time.time()
            by_protocol = collections.defaultdict(list)
            fold_metrics = []
            for split in splits:
                tr, te = split['train'], split['test']
                if key == 'majority':
                    clf = DummyClassifier(strategy='most_frequent')
                else:
                    clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10, class_weight='balanced',
                                                                           solver='lsqr', tol=.001, max_iter=2000))
                with threadpool_limits(limits=4):
                    clf.fit(z[tr], y[tr])
                    pred = clf.predict(z[te])
                    scores = clf.predict_proba(z[te]) if key == 'majority' else clf.decision_function(z[te])
                assert scores.shape == (len(te), 4) and np.isfinite(scores).all()
                for ix, pr, sc in zip(te, pred, scores):
                    w.writerow([key, split['protocol'], split['fold'], int(ix), int(y[ix]), int(pr),
                                *[format(float(v), '.9g') for v in sc]])
                by_protocol[split['protocol']].append((te, pred))
                fold_metrics.append({'protocol': split['protocol'], 'fold': split['fold'],
                                     'n_train': len(tr), **metrics(y[te], pred)})
            m = {'features': z.shape[1], 'fold_metrics': fold_metrics, 'protocols': {}}
            for protocol, pieces in by_protocol.items():
                ix, pr = np.concatenate([r[0] for r in pieces]), np.concatenate([r[1] for r in pieces])
                assert len(ix) == len(set(ix))
                preds = np.full(n, -1, dtype=int)
                preds[ix] = pr
                oof[(key, protocol)] = preds
                per_dev = {dev: metrics(y[ix][devices[ix] == dev], pr[devices[ix] == dev]) for dev in cfg['devices']}
                m['protocols'][protocol] = {'pooled': metrics(y[ix], pr), 'per_device': per_dev,
                                             'device_mean_macro_f1': float(np.mean([v['macro_f1'] for v in per_dev.values()])),
                                             'unweighted_fold_mean_macro_f1': float(np.mean([v['macro_f1'] for v in fold_metrics if v['protocol'] == protocol]))}
            m['elapsed_seconds'] = time.time() - start
            result['methods'][key] = m
            print('EVALUATED', key, 'LODO', round(m['protocols']['leave_device']['device_mean_macro_f1'], 6),
                  'DAY', round(m['protocols']['within_device_leave_day']['device_mean_macro_f1'], 6),
                  'seconds', round(m['elapsed_seconds'], 1), flush=True)
            (out / 'summary.json').write_text(json.dumps(result, indent=2))
    # Reuse identical day draws across methods. Each observed hardware device
    # gets equal final weight despite its different window/date sample count.
    boot = {}
    rng = np.random.default_rng(20261028)
    repetitions = 2000
    for protocol in ['leave_device', 'within_device_leave_day', 'calibration_zero_day_paired', 'calibration_one_day']:
        draws = {}
        any_preds = oof[(next(iter(features)), protocol)]
        for dev in cfg['devices']:
            active_days = sorted(set(days[(devices == dev) & (any_preds >= 0)]))
            draws[dev] = (active_days, rng.integers(0, len(active_days), size=(repetitions, len(active_days))))
        for key in features:
            pred = oof[(key, protocol)]
            samples = []
            for dev, (active_days, idx) in draws.items():
                cms = np.stack([confusion_matrix(y[(devices == dev) & (days == day) & (pred >= 0)],
                                                 pred[(devices == dev) & (days == day) & (pred >= 0)], labels=[0, 1, 2, 3])
                                for day in active_days])
                samples.append(f1_cm(cms[idx].sum(1)))
            values = np.mean(samples, axis=0)
            boot[(key, protocol)] = values
            result['methods'][key]['protocols'][protocol]['conditional_day_bootstrap_95'] = np.quantile(values, [.025, .975]).tolist()
    comparisons = {}
    for protocol in ['leave_device', 'within_device_leave_day']:
        random_methods = ['mantis_random42', 'mantis_random43', 'mantis_random44']
        delta = boot[('mantis_pretrained', protocol)] - np.mean([boot[(r, protocol)] for r in random_methods], axis=0)
        point = result['methods']['mantis_pretrained']['protocols'][protocol]['device_mean_macro_f1'] - np.mean([
            result['methods'][r]['protocols'][protocol]['device_mean_macro_f1'] for r in random_methods])
        comparisons['mantis_vs_random_mean/' + protocol] = {'difference': float(point), 'conditional_day_bootstrap_95': np.quantile(delta, [.025, .975]).tolist()}
        for model in ['mantis_pretrained', 'tspulse_pretrained', 'fisher_pretrained', 'mantis_plus_spectral']:
            delta = boot[(model, protocol)] - boot[('spectral_trends', protocol)]
            point = result['methods'][model]['protocols'][protocol]['device_mean_macro_f1'] - result['methods']['spectral_trends']['protocols'][protocol]['device_mean_macro_f1']
            comparisons[model + '_vs_spectral/' + protocol] = {'difference': float(point), 'conditional_day_bootstrap_95': np.quantile(delta, [.025, .975]).tolist()}
    result['paired_comparisons'] = comparisons
    for key in features:
        # Matched date draws are required for calibration differences. The
        # preceding marginal draws differ; build an independent paired draw.
        rng = np.random.default_rng(7314)
        vals = []
        for dev in cfg['devices']:
            p0, p1 = oof[(key, 'calibration_zero_day_paired')], oof[(key, 'calibration_one_day')]
            active = sorted(set(days[(devices == dev) & (p0 >= 0)]))
            idx = rng.integers(0, len(active), size=(repetitions, len(active)))
            pair = []
            for pr in [p0, p1]:
                cms = np.stack([confusion_matrix(y[(devices == dev) & (days == day) & (pr >= 0)],
                                                 pr[(devices == dev) & (days == day) & (pr >= 0)], labels=[0, 1, 2, 3]) for day in active])
                pair.append(f1_cm(cms[idx].sum(1)))
            vals.append(pair[1] - pair[0])
        delta = np.mean(vals, axis=0)
        point = result['methods'][key]['protocols']['calibration_one_day']['device_mean_macro_f1'] - result['methods'][key]['protocols']['calibration_zero_day_paired']['device_mean_macro_f1']
        result['methods'][key]['calibration_gain'] = {'difference': point, 'conditional_day_bootstrap_95': np.quantile(delta, [.025, .975]).tolist()}
    result['elapsed_seconds'] = time.time() - result['started_unix']
    result['prediction_file_sha256'] = sha(out / 'predictions.csv.gz')
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    print('ALL_EVALUATIONS_READY', result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
