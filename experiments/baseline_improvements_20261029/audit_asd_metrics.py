"""Recompute official domain AUROCs from unchanged stored model outputs."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def components(y, s, d, official):
    masks = [((d == domain) | (y == 1)) if official else (d == domain) for domain in ['source', 'target']]
    return [roc_auc_score(y[m], s[m]) for m in masks] + [roc_auc_score(y, s, max_fpr=.1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--bootstrap', type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (args.input/'scores.jsonl').read_text().splitlines()]
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row['model'], row['variant'], row['machine'])].append(row)
    legacy = json.loads((args.input/'metrics.json').read_text())
    result = {'domain_auc_definition': 'official_domain_normals_vs_all_anomalies',
              'source_scores_sha256': hashlib.sha256((args.input/'scores.jsonl').read_bytes()).hexdigest(),
              'metric_fix_only': True, 'model_inference_repeated': False, 'per_machine': [], 'aggregate': []}
    for key, rs in sorted(groups.items()):
        assert len(rs) == 200
        y = np.array([r['label'] for r in rs])
        d = np.array([r['domain'] for r in rs])
        s = np.array([r['anomaly_score'] for r in rs])
        old, new = components(y, s, d, False), components(y, s, d, True)
        old_record = next(r for r in legacy['metrics'] if (r['model'], r['variant'], r['machine']) == key)
        np.testing.assert_allclose(old, [old_record[k] for k in ['auc_source', 'auc_target', 'pauc_standardized_fpr_0_1']], atol=1e-12)
        # Independent exact Mann-Whitney probability: all 100 anomalies versus
        # 50 normal samples of the indicated domain, including tie half-credit.
        for i, domain in enumerate(['source', 'target']):
            a, n = s[y == 1], s[(y == 0) & (d == domain)]
            diff = a[:, None]-n[None, :]
            reference = ((diff > 0).sum()+.5*(diff == 0).sum())/diff.size
            assert abs(reference-new[i]) < 1e-12
        result['per_machine'].append({'model': key[0], 'variant': key[1], 'machine': key[2],
                                      'legacy_components': old, 'official_components': new})
    for model, variant in sorted({k[:2] for k in groups}):
        selected = [r for r in result['per_machine'] if (r['model'], r['variant']) == (model, variant)]
        hm = lambda values: float(len(values)/np.sum(1/np.asarray(values)))
        result['aggregate'].append({'model': model, 'variant': variant,
                                    'legacy_score': hm(sum([r['legacy_components'] for r in selected], [])),
                                    'official_score': hm(sum([r['official_components'] for r in selected], []))})
    if args.bootstrap:
        rng = np.random.default_rng(20261029)
        machines = sorted({k[2] for k in groups})
        models = ['BEATs_iter3', 'dcase2026_nabeats_seed0']
        aligned = []
        for machine in machines:
            rs = [sorted(groups[(m, 'beam_rdp4', machine)], key=lambda r: r['path']) for m in models]
            assert [(r['path'], r['label'], r['domain'], r['sha256']) for r in rs[0]] == [(r['path'], r['label'], r['domain'], r['sha256']) for r in rs[1]]
            aligned.append((np.array([r['label'] for r in rs[0]]), np.array([r['domain'] for r in rs[0]]),
                            [np.array([r['anomaly_score'] for r in row]) for row in rs]))
        draws = []
        for repeat in range(args.bootstrap):
            cc = [[], []]
            for y, d, ss in aligned:
                ix = np.concatenate([rng.choice(np.flatnonzero((y == label) & (d == domain)), 50, replace=True)
                                     for domain in ['source', 'target'] for label in [0, 1]])
                for j in range(2):
                    cc[j].extend(components(y[ix], ss[j][ix], d[ix], True))
            v = [len(c)/np.sum(1/np.array(c)) for c in cc]
            draws.append([*v, v[1]-v[0]])
        result['paired_bootstrap'] = {'repetitions': args.bootstrap, 'seed': 20261029,
                                      'scope': 'conditional paired stratified clip bootstrap, not independent device/run uncertainty',
                                      'beats_95': np.quantile(np.array(draws)[:, 0], [.025, .975]).tolist(),
                                      'nabeats_95': np.quantile(np.array(draws)[:, 1], [.025, .975]).tolist(),
                                      'difference_95': np.quantile(np.array(draws)[:, 2], [.025, .975]).tolist()}
    result['verification'] = {'groups': len(groups), 'score_rows': len(rows), 'legacy_reproduced': True,
                               'official_auc_matches_pairwise_probability': True}
    (args.out/'asd_metric_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['aggregate'], indent=2))


if __name__ == '__main__':
    main()
