"""Independent split/score audit, OOF metrics and paired date intervals."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, p):
    cm = confusion_matrix(y, p, labels=range(4))
    return {'macro_f1': float(f1_score(y, p, labels=range(4), average='macro', zero_division=0)),
            'accuracy': float(accuracy_score(y, p)), 'confusion': cm.tolist(),
            'recall': np.divide(cm.diagonal(), cm.sum(1), out=np.zeros(4), where=cm.sum(1) != 0).tolist(),
            'n': len(y)}


def f1_cm(cm):
    den = cm.sum(-1)+cm.sum(-2)
    return np.divide(2*np.diagonal(cm, axis1=-2, axis2=-1), den,
                     out=np.zeros_like(den, dtype=float), where=den != 0).mean(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    r = json.loads((a.base/'data/manifest.json').read_text())['records']
    y = np.array([row['class_id'] for row in r])
    dev = np.array([row['device'] for row in r])
    day = np.array([row['day'] for row in r])
    n = len(r)
    pp = {}
    audit = []
    counts = collections.Counter()
    stage_methods = {'nested': 8, 'temporal': 3, 'logistic': 5}
    with gzip.open(a.out/'predictions.csv.gz', 'wt', newline='', encoding='utf-8') as out:
        writer = None
        for stage in stage_methods:
            complete = sorted((a.out/stage).glob('*/complete.json'))
            assert len(complete) == 47, (stage, len(complete))
            for path in complete:
                item = json.loads(path.read_text())
                tr, te = np.array(item['train']), np.array(item['test'])
                assert not set(tr) & set(te)
                assert not ({(dev[i], day[i]) for i in tr} & {(dev[i], day[i]) for i in te})
                protocol, fold = item['protocol'], item['fold']
                if protocol == 'leave_device':
                    assert not set(dev[tr]) & set(dev[te])
                    assert set(te) == set(np.flatnonzero(dev == fold))
                    assert set(tr) == set(np.flatnonzero(dev != fold))
                else:
                    d, dt = fold.split('/')
                    assert set(te) == set(np.flatnonzero((dev == d) & (day == dt)))
                    assert set(tr) == set(np.flatnonzero((dev == d) & (day != dt)))
                if stage == 'nested':
                    valid = []
                    for inner in item['inner']:
                        i, j = inner['train'], inner['valid']
                        assert set(i) | set(j) == set(tr) and not set(i) & set(j)
                        assert not ({(dev[k], day[k]) for k in i} & {(dev[k], day[k]) for k in j})
                        assert not (set(i) | set(j)) & set(te)
                        if protocol == 'leave_device':
                            assert not set(dev[i]) & set(dev[j])
                        valid.extend(j)
                    assert sorted(valid) == sorted(tr)
                    for method, entry in item['methods'].items():
                        if method != 'automatic_nested':
                            assert entry['chosen'] == int(np.argmax(entry['inner_scores']))
                    selected = item['methods']['automatic_nested']['selected_family']
                    eligible = [k for k in item['methods'] if k not in ['automatic_nested', 'mantis_random42_nested']]
                    assert selected == max(eligible, key=lambda k: max(item['methods'][k]['inner_scores']))
                file = path.with_name('predictions.csv.gz')
                assert sha(file) == item['prediction_sha256']
                method_counts = collections.Counter()
                with gzip.open(file, 'rt') as f:
                    reader = csv.DictReader(f)
                    if writer is None:
                        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    for row in reader:
                        method = row['method']
                        ix, truth, pr = (int(row[k]) for k in ['sample_index', 'true', 'prediction'])
                        assert row['protocol'] == protocol and row['fold'] == fold
                        assert ix in set(te) and y[ix] == truth and 0 <= pr < 4
                        decision = np.array([float(row['s'+str(i)]) for i in range(4)])
                        assert np.isfinite(decision).all() and decision.argmax() == pr
                        key = (method, protocol)
                        if key not in pp:
                            pp[key] = np.full(n, -1, dtype=int)
                        assert pp[key][ix] == -1
                        pp[key][ix] = pr
                        writer.writerow(row)
                        method_counts[method] += 1
                        counts[stage] += 1
                assert len(method_counts) == stage_methods[stage] and set(method_counts.values()) == {len(te)}
                if stage == 'nested':
                    assert np.array_equal(pp[('automatic_nested', protocol)][te], pp[(selected, protocol)][te])
                audit.append(item)
    assert all(np.all(pred >= 0) for pred in pp.values())
    # Preserve and verify old predictions, rather than rounding table values.
    old = json.loads((a.base/'results/summary.json').read_text())
    retained = ['fisher_pretrained', 'mantis_pretrained', 'mantis_random42', 'tspulse_pretrained', 'spectral_trends']
    oldfile = a.base/'results/predictions.csv.gz'
    assert sha(oldfile) == old['prediction_file_sha256']
    with gzip.open(oldfile, 'rt') as f:
        for row in csv.DictReader(f):
            if row['method'] not in retained or row['protocol'] not in ['leave_device', 'within_device_leave_day']:
                continue
            key = (row['method']+'_fixed', row['protocol'])
            pp.setdefault(key, np.full(n, -1, dtype=int))
            i = int(row['sample_index'])
            assert pp[key][i] == -1 and y[i] == int(row['true'])
            pp[key][i] = int(row['prediction'])
    result = {'n': n, 'device_days': 44, 'methods': {}, 'paired_comparisons': {}, 'bootstrap_repetitions': 2000,
              'bootstrap_scope': 'conditional paired resampling of dates within each of three devices; fixed fitted/selected models; not hardware population or procedure-retraining uncertainty',
              'benchmark_manifest_sha256': sha(a.base/'data/manifest.json'), 'predictions_sha256': sha(a.out/'predictions.csv.gz')}
    rng = np.random.default_rng(20261029)
    draws = {d: (sorted(set(day[dev == d])), rng.integers(0, len(set(day[dev == d])), size=(2000, len(set(day[dev == d]))))) for d in sorted(set(dev))}
    boots = {}
    for (method, protocol), pr in pp.items():
        assert np.all(pr >= 0)
        per_device = {d: metrics(y[dev == d], pr[dev == d]) for d in sorted(set(dev))}
        point = float(np.mean([v['macro_f1'] for v in per_device.values()]))
        if method.endswith('_fixed'):
            np.testing.assert_allclose(point, old['methods'][method[:-6]]['protocols'][protocol]['device_mean_macro_f1'], atol=1e-12)
        device_draws = []
        for d, (days, ix) in draws.items():
            cms = np.stack([confusion_matrix(y[(dev == d) & (day == dt)], pr[(dev == d) & (day == dt)], labels=range(4)) for dt in days])
            device_draws.append(f1_cm(cms[ix].sum(1)))
        distribution = np.mean(device_draws, axis=0)
        boots[(method, protocol)] = distribution
        result['methods'].setdefault(method, {})[protocol] = {'device_mean_macro_f1': point, 'per_device': per_device,
                                                            'conditional_95': np.quantile(distribution, [.025, .975]).tolist()}
    baselines = {'fisher_nested':'fisher_pretrained_fixed', 'mantis_nested':'mantis_pretrained_fixed',
                 'tspulse_nested':'tspulse_pretrained_fixed', 'physical_nested':'spectral_trends_fixed',
                 'relative_dynamics_nested':'spectral_trends_fixed', 'fisher_physical_nested':'fisher_pretrained_fixed',
                 'mantis_random42_nested':'mantis_random42_fixed', 'extra_trees':'spectral_trends_fixed',
                 'minirocket':'spectral_trends_fixed', 'minirocket_normalized':'spectral_trends_fixed',
                 'mantis_logistic_default':'mantis_pretrained_fixed', 'mantis_logistic_balanced':'mantis_pretrained_fixed',
                 'fisher_logistic_balanced':'fisher_pretrained_fixed', 'tspulse_logistic_balanced':'tspulse_pretrained_fixed',
                 'mantis_random42_logistic_balanced':'mantis_random42_fixed'}
    for method in set(k[0] for k in pp) - {r+'_fixed' for r in retained}:
        for protocol in ['leave_device', 'within_device_leave_day']:
            baseline = baselines.get(method, 'mantis_pretrained_fixed' if protocol == 'leave_device' else 'fisher_pretrained_fixed')
            point = result['methods'][method][protocol]['device_mean_macro_f1']-result['methods'][baseline][protocol]['device_mean_macro_f1']
            delta = boots[(method, protocol)]-boots[(baseline, protocol)]
            result['paired_comparisons'][method+'/'+protocol] = {'reference': baseline, 'difference': point,
                                                               'conditional_paired_95': np.quantile(delta, [.025, .975]).tolist()}
    for method, random in [('mantis_nested', 'mantis_random42_nested'), ('mantis_logistic_balanced', 'mantis_random42_logistic_balanced')]:
        for protocol in ['leave_device', 'within_device_leave_day']:
            delta = boots[(method, protocol)]-boots[(random, protocol)]
            result['paired_comparisons'][method+'_vs_matched_random/'+protocol] = {
                'reference': random,
                'difference': result['methods'][method][protocol]['device_mean_macro_f1']-result['methods'][random][protocol]['device_mean_macro_f1'],
                'conditional_paired_95': np.quantile(delta, [.025, .975]).tolist()}
    for protocol in ['leave_device', 'within_device_leave_day']:
        baseline = 'mantis_random42_fixed'
        delta = boots[('automatic_nested', protocol)]-boots[(baseline, protocol)]
        result['paired_comparisons']['automatic_vs_original_random42/'+protocol] = {
            'reference': baseline,
            'difference': result['methods']['automatic_nested'][protocol]['device_mean_macro_f1']-result['methods'][baseline][protocol]['device_mean_macro_f1'],
            'conditional_paired_95': np.quantile(delta, [.025, .975]).tolist()}
        baseline = 'fisher_pretrained_fixed'
        delta = boots[('extra_trees', protocol)]-boots[(baseline, protocol)]
        result['paired_comparisons']['extra_trees_vs_original_fisher/'+protocol] = {
            'reference': baseline,
            'difference': result['methods']['extra_trees'][protocol]['device_mean_macro_f1']-result['methods'][baseline][protocol]['device_mean_macro_f1'],
            'conditional_paired_95': np.quantile(delta, [.025, .975]).tolist()}
    with gzip.open(a.out/'selection_and_split_audit.json.gz', 'wt') as f:
        json.dump(audit, f)
    warnings_seen = [entry for item in audit for method, entry in item.get('methods', {}).items() if entry.get('convergence_warnings')]
    verification = {'passed': True, 'new_prediction_rows': sum(counts.values()), 'rows_by_stage': dict(counts),
                     'new_methods': sum(stage_methods.values()), 'folds_per_stage': 47, 'stage_count': 3,
                     'split_audits': len(audit), 'no_device_day_overlap': True, 'no_target_device_in_strict_training': True,
                     'inner_folds_exclude_outer_test': True, 'selection_matches_inner_argmax': True,
                     'all_score_argmax_and_labels_checked': True, 'all_old_metrics_reproduced': True,
                     'logistic_convergence_warning_fits': len(warnings_seen)}
    (a.out/'summary.json').write_text(json.dumps(result, indent=2))
    (a.out/'verification.json').write_text(json.dumps(verification, indent=2))
    with (a.out/'metrics_table.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'within_device_leave_day_macro_f1', 'leave_device_macro_f1'])
        for method, entries in result['methods'].items():
            w.writerow([method, entries['within_device_leave_day']['device_mean_macro_f1'], entries['leave_device']['device_mean_macro_f1']])
    print(json.dumps(verification))
    print((a.out/'metrics_table.csv').read_text())


if __name__ == '__main__':
    main()
