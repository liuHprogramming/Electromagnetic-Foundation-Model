"""Train-partition-only nested probes and independent temporal alternatives.

Run on scheduled remote CPU, never on the shared desktop. See protocol.json.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def score(y, p, dev):
    return float(np.mean([f1_score(y[dev == d], p[dev == d], labels=range(4), average='macro', zero_division=0)
                          for d in np.unique(dev)]))


def stats(z):
    t = np.linspace(-1, 1, z.shape[-1])
    return np.concatenate([z.mean(-1), z.std(-1), np.quantile(z, .1, axis=-1),
                           np.quantile(z, .9, axis=-1), (z*t).sum(-1)/(t*t).sum(),
                           z[..., -20:].mean(-1)-z[..., :20].mean(-1),
                           np.abs(np.diff(z, axis=-1)).mean(-1)], axis=1).astype('float32')


def fit_predict(z, y, tr, te, config):
    scaler = StandardScaler()
    a, b = scaler.fit_transform(z[tr]), scaler.transform(z[te])
    if config['type'] == 'ridge':
        model = RidgeClassifier(alpha=config['alpha'], class_weight='balanced', solver='lsqr', tol=.001, max_iter=2000)
    else:
        # Euclidean kernels in training-standardized space; no test fit.
        gamma = 1 / z.shape[1]
        kernel_a, kernel_b = rbf_kernel(a, a, gamma=gamma), rbf_kernel(b, a, gamma=gamma)
        a, b = kernel_a, kernel_b
        model = SVC(kernel='precomputed', C=config['C'], class_weight='balanced', decision_function_shape='ovr', break_ties=True)
    model.fit(a, y[tr])
    return model.predict(b), model.decision_function(b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=['nested', 'temporal'], required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(__file__).with_name('protocol.json').read_text())
    records = json.loads((args.base/'data/manifest.json').read_text())['records']
    y = np.array([r['class_id'] for r in records])
    dev = np.array([r['device'] for r in records])
    day = np.array([r['day'] for r in records])
    assert len(records) == 3177 and len({(r['device'], r['day']) for r in records}) == 44
    assert len({r['input_sha256'] for r in records}) == len(records)
    froot = args.base/'features'
    receipt = json.loads((froot/'model_receipt.json').read_text())
    for name, item in receipt['artifacts'].items():
        assert sha(froot/name) == item['sha256']
    x = np.load(args.base/'data/spectrograms.npy')
    # Center each time frame over frequency: spectral shape independent of
    # an additive broadband level shift. Retain signed temporal derivatives.
    shape = x-x.mean(axis=1, keepdims=True)
    original = stats(x)
    np.testing.assert_allclose(original, np.load(froot/'spectral_trends.npy'), atol=2e-4, rtol=2e-5)
    relative = np.column_stack([stats(shape), stats(np.diff(x, axis=-1))]).astype('float32')
    features = {'physical': original, 'relative_dynamics': relative}
    for name in ['fisher', 'mantis', 'tspulse']:
        features[name] = np.load(froot/(name+'_pretrained.npy'))
    features['mantis_random42'] = np.load(froot/'mantis_random42.npy')
    features['fisher_physical'] = np.column_stack([features['fisher'], original])
    folds = []
    for d in np.unique(dev):
        folds.append(('leave_device', d, np.flatnonzero(dev != d), np.flatnonzero(dev == d)))
    for d in np.unique(dev):
        for dt in sorted(set(day[dev == d])):
            folds.append(('within_device_leave_day', d+'/'+dt,
                          np.flatnonzero((dev == d) & (day != dt)), np.flatnonzero((dev == d) & (day == dt))))
    vendor = Path(__file__).parent/'vendor/minirocket_multivariate.py'
    if args.stage == 'temporal':
        import numba
        numba.set_num_threads(4)
        spec = importlib.util.spec_from_file_location('minirocket_multivariate', vendor)
        mr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mr)
        @numba.njit
        def seed_numba(seed):
            np.random.seed(seed)
        bands = np.load(args.base/'data/band_trajectories.npy').astype('float32')
        normalized = (bands-bands.mean(axis=(1, 2), keepdims=True))/np.maximum(bands.std(axis=(1, 2), keepdims=True), 1e-6)
    started = time.time()
    metadata = {'stage': args.stage, 'base_manifest_sha256': sha(args.base/'data/manifest.json'),
                'protocol_sha256': sha(Path(__file__).with_name('protocol.json')),
                'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__,
                'scipy': scipy.__version__, 'host': platform.node(), 'folds': [], 'started': started}
    for protocol, fold, tr, te in folds:
        target = out/args.stage/(protocol+'_'+fold.replace('/', '_'))
        target.mkdir(parents=True, exist_ok=True)
        if (target/'complete.json').exists():
            continue
        trgroup = {(dev[i], day[i]) for i in tr}
        tegroup = {(dev[i], day[i]) for i in te}
        assert not trgroup & tegroup
        if protocol == 'leave_device':
            assert not set(dev[tr]) & set(dev[te])
            inner = [(tr[dev[tr] != d], tr[dev[tr] == d]) for d in np.unique(dev[tr])]
        else:
            inner = [(tr[a], tr[b]) for a, b in GroupKFold(3).split(tr, y[tr], day[tr])]
        inner_audit = []
        for a, b in inner:
            assert not ({(dev[i], day[i]) for i in a} & {(dev[i], day[i]) for i in b})
            assert not set(b) & set(te) and not set(a) & set(te)
            inner_audit.append({'train': a.tolist(), 'valid': b.tolist()})
        details = {'protocol': protocol, 'fold': fold, 'train': tr.tolist(), 'test': te.tolist(),
                   'inner': inner_audit, 'methods': {}, 'elapsed_start': time.time()}
        preds = {}
        with threadpool_limits(limits=4):
            if args.stage == 'nested':
                for key in cfg['nested_features']:
                    z = features[key]
                    candidates = []
                    for config in cfg['head_grid']:
                        valid, predicted = [], []
                        for a, b in inner:
                            p, _ = fit_predict(z, y, a, b, config)
                            valid.extend(b)
                            predicted.extend(p)
                        candidates.append(score(y[valid], np.array(predicted), dev[valid]))
                    best = int(np.argmax(candidates))
                    pred, decision = fit_predict(z, y, tr, te, cfg['head_grid'][best])
                    preds[key+'_nested'] = (pred, decision)
                    details['methods'][key+'_nested'] = {'inner_scores': candidates,
                                                        'chosen': best, 'config': cfg['head_grid'][best]}
                    print('HEAD', protocol, fold, key, 'inner', round(candidates[best], 4), 'chosen', best, flush=True)
                eligible = [k+'_nested' for k in cfg['nested_features'] if not k.startswith('mantis_random')]
                selected = max(eligible, key=lambda k: max(details['methods'][k]['inner_scores']))
                preds['automatic_nested'] = preds[selected]
                details['methods']['automatic_nested'] = {'selected_family': selected}
            else:
                z = np.column_stack([features['physical'], features['relative_dynamics']])
                et = ExtraTreesClassifier(n_estimators=256, min_samples_leaf=2, max_features='sqrt',
                                          class_weight='balanced', random_state=42, n_jobs=4)
                et.fit(z[tr], y[tr])
                preds['extra_trees'] = (et.predict(z[te]), et.predict_proba(z[te]))
                for key, data in [('minirocket', bands), ('minirocket_normalized', normalized)]:
                    np.random.seed(42)
                    seed_numba(42)
                    params = mr.fit(np.ascontiguousarray(data[tr]), num_features=2016)
                    za = mr.transform(np.ascontiguousarray(data[tr]), params)
                    zb = mr.transform(np.ascontiguousarray(data[te]), params)
                    sc = StandardScaler()
                    za, zb = sc.fit_transform(za), sc.transform(zb)
                    clf = RidgeClassifier(alpha=10, class_weight='balanced', solver='lsqr', tol=.001, max_iter=2000)
                    clf.fit(za, y[tr])
                    preds[key] = (clf.predict(zb), clf.decision_function(zb))
                    details['methods'][key] = {'num_features': za.shape[1], 'seed': 42,
                                               'vendor_sha256': sha(vendor),
                                               'bias_sha256': hashlib.sha256(params[-1].tobytes()).hexdigest()}
        with gzip.open(target/'predictions.csv.gz', 'wt', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['method', 'protocol', 'fold', 'sample_index', 'true', 'prediction', 's0', 's1', 's2', 's3'])
            for key, (pr, decision) in preds.items():
                assert decision.shape == (len(te), 4) and np.isfinite(decision).all()
                assert np.array_equal(pr, np.argmax(decision, axis=1))
                for i, p, s in zip(te, pr, decision):
                    w.writerow([key, protocol, fold, int(i), int(y[i]), int(p), *[format(float(v), '.10g') for v in s]])
        details['elapsed_seconds'] = time.time()-details.pop('elapsed_start')
        details['prediction_sha256'] = sha(target/'predictions.csv.gz')
        (target/'complete.json').write_text(json.dumps(details, indent=2))
        metadata['folds'].append({'protocol': protocol, 'fold': fold, 'seconds': details['elapsed_seconds']})
        (out/(args.stage+'_runtime.json')).write_text(json.dumps(metadata, indent=2))
        print('DONE', args.stage, protocol, fold, round(details['elapsed_seconds'], 1), flush=True)
    metadata['elapsed_seconds'] = time.time()-started
    (out/(args.stage+'_runtime.json')).write_text(json.dumps(metadata, indent=2))
    print('COMPLETE', args.stage, metadata['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
