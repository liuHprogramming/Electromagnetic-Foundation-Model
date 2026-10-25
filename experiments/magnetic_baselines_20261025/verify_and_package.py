from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import tarfile
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

root=Path.cwd()
out=root/'results'
manifest=json.loads((out/'data_v2/manifest.json').read_text())
summary=json.loads((out/'summary.json').read_text())
pred=[json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()]
records=manifest['records']
assert len(records)==len(set(r['input_sha256'] for r in records))
assert hashlib.sha256((out/'data_v2/windows.npy').read_bytes()).hexdigest()==manifest['windows_sha256']
features=np.load(out/'features.npz')
for name in features.files:
    assert len(features[name])==len(records) and np.isfinite(features[name]).all()
    for protocol in ('leave_day_out','leave_device_out'):
        rows=[p for p in pred if p['model']==name and p['protocol']==protocol]
        assert sorted(p['id'] for p in rows)==list(range(len(records)))
        item=summary['results'][f'{name}/{protocol}']
        for fold in item['folds']:
            if 'excluded' in fold: continue
            rr=[p for p in rows if records[p['id']]['device']==fold['device'] and
                (fold['day'] is None or records[p['id']]['day']==fold['day'])]
            y=[p['true'] for p in rr]; pp=[p['pred'] for p in rr]
            assert all(p['true']==records[p['id']]['label'] for p in rr)
            assert abs(f1_score(y,pp,labels=[0,1,2,3],average='macro',zero_division=0)-fold['macro_f1'])<1e-12
            assert abs(accuracy_score(y,pp)-fold['accuracy'])<1e-12
comparison={}
rng=np.random.default_rng(123)
for name,reference in [('mantis','mantis_random'),('tspulse','tspulse_random'),('mantis','physical_psd'),('fisher_adapter','physical_psd')]:
    if f'{name}/leave_day_out' not in summary['results']:continue
    a=summary['results'][f'{name}/leave_day_out']['folds']
    b=summary['results'][f'{reference}/leave_day_out']['folds']
    assert [(x['device'],x['day']) for x in a]==[(x['device'],x['day']) for x in b]
    delta=np.array([x['macro_f1']-y['macro_f1'] for x,y in zip(a,b)])
    boots=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
    comparison[f'{name}_minus_{reference}']={'mean_macro_f1_difference':float(delta.mean()),
        'descriptive_paired_device_day_bootstrap_95ci':np.quantile(boots,[.025,.975]).tolist(),
        'n_device_days':len(delta)}
(out/'comparisons.json').write_text(json.dumps(comparison,indent=2))
quality={
    'preselected_windows':manifest['n_preselected'],'accepted_windows':len(records),
    'accepted_devices':sorted(set(r['device'] for r in records)),
    'accepted_device_days':sorted(set((r['device'],r['day']) for r in records)),
    'rejection_counts':dict(Counter(x['reason'] for x in manifest['exclusions'])),
    'sample_clock_counts':dict(Counter(f"{r['device']}:{r['inferred_sample_clock_hz']}" for r in records)),
    'source_file_count':len(set(r['source'] for r in records)),
    'sample_unit':'log-anchored 2-second window; independent unit is device/day/run',
    'estimated_raw_hz_range':[min(r['estimated_raw_hz'] for r in records),max(r['estimated_raw_hz'] for r in records)],
}
(out/'quality_summary.json').write_text(json.dumps(quality,indent=2))
large={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
              'remote_path':str(p)} for p in (out/'features.npz',out/'data_v2/windows.npy')}
(out/'retained_artifacts.json').write_text(json.dumps(large,indent=2))
verification={'passed':True,'verified_utc':datetime.now(timezone.utc).isoformat(),
    'n_windows':len(records),'n_predictions':len(pred),'feature_keys':features.files,
    'checks':['all processed window SHA values unique','window array SHA verified',
              'all feature matrices finite and aligned','every model/protocol predicts every input once',
              'all fold macro-F1 and accuracy independently recomputed','paired day identities match']}
(out/'verification.json').write_text(json.dumps(verification,indent=2))
files=sorted(p for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl') and 'data_v2' in p.parts or
             p.is_file() and p.parent==out and p.suffix in ('.json','.jsonl'))
# Do not package the preliminary timestamp-averaging attempt, raw waveforms or embeddings.
files=[p for p in files if 'data_v2' in p.parts or p.parent==out]
files+=sorted((root/'logs').glob('*.log'))
items=[{'path':str(p.relative_to(root)), 'bytes':p.stat().st_size,
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
(root/'artifact_manifest.json').write_text(json.dumps(items,indent=2))
with tarfile.open(root/'magnetic_results.tar.gz','w:gz') as tar:
    for p in files:tar.add(p,arcname=p.relative_to(root))
    tar.add(root/'artifact_manifest.json',arcname='artifact_manifest.json')
print(json.dumps({'verification':verification,'quality':quality,'comparisons':comparison,
                  'package_bytes':(root/'magnetic_results.tar.gz').stat().st_size},indent=2))
