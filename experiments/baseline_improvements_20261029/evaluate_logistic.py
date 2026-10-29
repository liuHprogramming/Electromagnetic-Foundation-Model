"""Fixed logistic probes motivated by the MantisV2 paper, no outer tuning."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    records = json.loads((a.base/'data/manifest.json').read_text())['records']
    y = np.array([r['class_id'] for r in records])
    dev = np.array([r['device'] for r in records])
    day = np.array([r['day'] for r in records])
    methods = [('mantis_logistic_default', 'mantis_pretrained', None),
               ('mantis_logistic_balanced', 'mantis_pretrained', 'balanced'),
               ('fisher_logistic_balanced', 'fisher_pretrained', 'balanced'),
               ('tspulse_logistic_balanced', 'tspulse_pretrained', 'balanced'),
               ('mantis_random42_logistic_balanced', 'mantis_random42', 'balanced')]
    features = {key: np.load(a.base/'features'/(key+'.npy')) for _, key, _ in methods}
    splits = [('leave_device', d, np.flatnonzero(dev != d), np.flatnonzero(dev == d)) for d in np.unique(dev)]
    splits += [('within_device_leave_day', d+'/'+dt, np.flatnonzero((dev == d) & (day != dt)),
                np.flatnonzero((dev == d) & (day == dt))) for d in np.unique(dev) for dt in sorted(set(day[dev == d]))]
    for protocol, fold, tr, te in splits:
        dest = a.out/'logistic'/(protocol+'_'+fold.replace('/', '_'))
        dest.mkdir(parents=True, exist_ok=True)
        if (dest/'complete.json').exists():
            continue
        assert not ({(dev[i], day[i]) for i in tr} & {(dev[i], day[i]) for i in te})
        start = time.time()
        info = {'protocol': protocol, 'fold': fold, 'train': tr.tolist(), 'test': te.tolist(), 'methods': {}}
        with gzip.open(dest/'predictions.csv.gz', 'wt', newline='') as f:
            w = csv.writer(f)
            w.writerow(['method', 'protocol', 'fold', 'sample_index', 'true', 'prediction', 's0', 's1', 's2', 's3'])
            for method, key, balance in methods:
                z = features[key]
                scaler = StandardScaler()
                with threadpool_limits(limits=4), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', ConvergenceWarning)
                    za, zb = scaler.fit_transform(z[tr]), scaler.transform(z[te])
                    clf = LogisticRegression(C=1, solver='lbfgs', max_iter=500, class_weight=balance)
                    clf.fit(za, y[tr])
                    pr, scores = clf.predict(zb), clf.decision_function(zb)
                assert np.array_equal(pr, scores.argmax(1)) and np.isfinite(scores).all()
                for i, prediction, sc in zip(te, pr, scores):
                    w.writerow([method, protocol, fold, i, y[i], prediction, *[format(float(v), '.10g') for v in sc]])
                info['methods'][method] = {'C': 1, 'class_weight': balance, 'iterations': clf.n_iter_.tolist(),
                                           'convergence_warnings': [str(w.message) for w in caught]}
        info['seconds'] = time.time()-start
        info['prediction_sha256'] = hashlib.sha256((dest/'predictions.csv.gz').read_bytes()).hexdigest()
        (dest/'complete.json').write_text(json.dumps(info, indent=2))
        print('DONE logistic', protocol, fold, round(info['seconds'], 1), flush=True)
    print('COMPLETE logistic', flush=True)


if __name__ == '__main__':
    main()
