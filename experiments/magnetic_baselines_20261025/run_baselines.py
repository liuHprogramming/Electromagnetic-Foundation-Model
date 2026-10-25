"""Frozen magnetic-transfer encoders and fixed linear readouts on grouped splits."""
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import os
import time

import numpy as np
import requests
from scipy.signal import welch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(repo, revision, name, dest, expected=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (not expected or sha(dest) == expected):
        return
    errors = []
    for endpoint in ('https://hf-mirror.com', 'https://huggingface.co'):
        try:
            r = requests.get(f'{endpoint}/{repo}/resolve/{revision}/{name}', timeout=45)
            r.raise_for_status()
            dest.write_bytes(r.content)
            if expected and sha(dest) != expected:
                raise ValueError('checkpoint SHA mismatch')
            return
        except Exception as e:
            errors.append(str(e))
    raise RuntimeError(errors)


def metric(y, pred):
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=[0, 1, 2, 3], average='macro', zero_division=0)),
            'confusion': confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist(),
            'n': int(len(y))}


def main():
    root = Path.cwd()
    out = root / 'results'
    out.mkdir(exist_ok=True)
    start = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(24*1024**3/torch.cuda.get_device_properties(0).total_memory)
    x = np.load(out / 'data_v2/windows.npy')
    manifest = json.loads((out / 'data_v2/manifest.json').read_text())
    records = manifest['records']
    y = np.array([r['label'] for r in records])
    devices = np.array([r['device'] for r in records])
    days = np.array([r['day'] for r in records])
    assert len(x) == len(records) and x.shape[1] == 512
    models = {
        'mantis': ('paris-noah/MantisV2', '8f6ca35cb54ab14b120618943c6fca5ddf5a76a6',
                   '49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1'),
        'tspulse': ('ibm-granite/granite-timeseries-tspulse-r1', '66cfb5464ed5e5a5cd7f5df1339af4e51461d955',
                    '71fddf719ebe8e8c64eaec5769620935af48263dc35e4ce6f3f88f7708e973e1'),
    }
    receipts, features, failures = {}, {}, {}
    for key, (repo, rev, h) in models.items():
        folder = root / 'models' / key
        for name in ('config.json', 'model.safetensors'):
            download(repo, rev, name, folder/name, h if name.endswith('safetensors') else None)
        receipts[key] = {'repo': repo, 'revision': rev, 'files': {p.name: sha(p) for p in folder.iterdir()}}
    _, psd = welch(x, fs=256, nperseg=256, axis=-1)
    amp = np.column_stack([x.mean(1), x.std(1), np.sqrt((x*x).mean(1)),
                           np.quantile(x, .95, axis=1)-np.quantile(x, .05, axis=1)])
    features['physical_psd'] = np.column_stack([np.log(psd[:, :101]+1e-12), amp]).astype('float32')
    from mantis.architecture import MantisV2
    from tsfm_public.models.tspulse import TSPulseForReconstruction
    for family in ('mantis', 'tspulse'):
        for pretrained in (True, False):
            key = family if pretrained else family + '_random'
            torch.manual_seed(42)
            if family == 'mantis':
                model = MantisV2(device='cuda', return_transf_layer=2, output_token='combined')
                if pretrained:
                    model = model.from_pretrained(str(root/'models/mantis'))
                encoder = lambda a: model(a[:, None, :])
            else:
                loaded, loading = TSPulseForReconstruction.from_pretrained(
                    root/'models/tspulse', local_files_only=True, output_loading_info=True)
                if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
                    raise ValueError(f'incomplete TSPulse load: {loading}')
                receipts['tspulse']['loading_info'] = loading
                model = loaded if pretrained else TSPulseForReconstruction(loaded.config)
                del loaded
                encoder = lambda a: model.backbone(a[:, :, None], enable_masking=False, return_dict=True).last_hidden_flatten_state.flatten(1)
            model = model.eval().cuda()
            for p in model.parameters():
                p.requires_grad_(False)
            torch.cuda.reset_peak_memory_stats()
            rows = []
            with torch.inference_mode():
                probe = torch.tensor(x[:2], device='cuda')
                aa, bb = encoder(probe), encoder(probe)
                assert torch.isfinite(aa).all() and torch.allclose(aa, bb)
                if pretrained:
                    silence = encoder(torch.zeros_like(probe))
                    if torch.allclose(aa, silence, atol=1e-5):
                        raise ValueError(f'{key} ignored magnetic input')
                for i in range(0, len(x), 32):
                    rows.append(encoder(torch.tensor(x[i:i+32], device='cuda')).float().cpu().numpy())
            z = np.concatenate(rows)
            if not np.isfinite(z).all():
                raise ValueError(f'nonfinite {key} embeddings')
            features[key] = z
            receipts[key] = {**receipts.get(key, {}), 'parameters': sum(p.numel() for p in model.parameters()),
                             'embedding_dim': int(z.shape[1]), 'dtype': 'float32',
                             'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                             'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            print(f'embedded {key}: {z.shape}; elapsed={time.time()-start:.1f}s', flush=True)
            del model, encoder, rows
            gc.collect()
            torch.cuda.empty_cache()
    # FISHER uses the same 0-100Hz/256Hz input. A physically defined 1-second
    # STFT is an explicit adapter, not the published 25ms audio recipe.
    try:
        from transformers import AutoModel
        import torch.nn.functional as F
        candidates = list((root.parent/'models').glob('**/model.safetensors'))
        expected = '9924cdeec51ea796557af0c1920f0a2148303b3c26d2b41a39dbb8c4ff94fb4a'
        fisher_path = next(p.parent for p in candidates if p.stat().st_size < 500_000_000 and sha(p) == expected)
        model, loading = AutoModel.from_pretrained(fisher_path, trust_remote_code=True,
                            local_files_only=True, output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError(f'incomplete FISHER load: {loading}')
        model = model.eval().cuda()
        rows = []
        with torch.inference_mode():
            for i in range(0, len(x), 16):
                a = torch.tensor(x[i:i+16], device='cuda')
                a = a - a.mean(-1, keepdim=True)
                spec = torch.stft(a, n_fft=256, hop_length=16, window=torch.hann_window(256,device='cuda'),
                                  center=False, return_complex=True).abs().clamp_min(1e-10).log().transpose(-1,-2)
                spec = (spec+3.017344307886898)/(2*2.1531635155379805)
                bw = model.cfg.band_width
                spec = F.pad(spec, (0, (-spec.shape[-1]) % bw))
                rows.append(model.extract_features(spec[:, None]).float().cpu().numpy())
        features['fisher_adapter'] = np.concatenate(rows)
        receipts['fisher_adapter'] = {'checkpoint_sha256': expected, 'loading_info': loading,
                                      'embedding_dim': int(features['fisher_adapter'].shape[1]),
                                      'stft_n_fft':256,'hop':16,'sampling_hz':256,'adapted':True}
        del model, rows
        torch.cuda.empty_cache()
    except Exception as exc:
        failures['fisher_adapter'] = repr(exc)
        print('FISHER adapter unavailable:', repr(exc), flush=True)
    np.savez_compressed(out/'features.npz', **features)
    (out/'model_receipts.json').write_text(json.dumps({'models':receipts,'failures':failures},indent=2))
    results, predictions = {}, []
    for name, z in features.items():
        for protocol in ('leave_day_out', 'leave_device_out'):
            folds = []
            groups = [(d, day) for d in np.unique(devices) for day in np.unique(days[devices==d])] if protocol=='leave_day_out' else [(d, None) for d in np.unique(devices)]
            for d, day in groups:
                test = devices==d
                train = devices!=d
                if day is not None:
                    test = test & (days==day)
                    train = (devices==d) & (days!=day)
                if len(np.unique(y[train])) < 4 or not test.any():
                    folds.append({'device':d,'day':day,'excluded':'training classes missing'})
                    continue
                assert not set(np.where(train)[0]) & set(np.where(test)[0])
                assert not {records[k]['input_sha256'] for k in np.where(train)[0]} & {records[k]['input_sha256'] for k in np.where(test)[0]}
                classifier = make_pipeline(StandardScaler(),RidgeClassifier(alpha=10.0,class_weight='balanced'))
                classifier.fit(z[train], y[train])
                pred = classifier.predict(z[test])
                m = metric(y[test],pred)
                folds.append({'device':d,'day':day,'n_train':int(train.sum()), **m})
                predictions.extend({'model':name,'protocol':protocol,'id':int(k),'true':int(y[k]),'pred':int(p)} for k,p in zip(np.where(test)[0],pred))
            valid = [f for f in folds if 'macro_f1' in f]
            results[f'{name}/{protocol}'] = {'folds':folds,'mean_macro_f1':float(np.mean([f['macro_f1'] for f in valid])),
                    'mean_accuracy':float(np.mean([f['accuracy'] for f in valid])), 'n_folds':len(valid)}
            print(name, protocol, results[f'{name}/{protocol}']['mean_macro_f1'], flush=True)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions))
    # Date is the resampling unit; report that 3 device folds do not support a
    # broad population-level device confidence interval.
    rng = np.random.default_rng(42)
    for key, item in results.items():
        vals = np.array([f['macro_f1'] for f in item['folds'] if 'macro_f1' in f])
        if key.endswith('leave_day_out'):
            means = vals[rng.integers(0,len(vals),size=(10000,len(vals)))].mean(1)
            item['descriptive_day_bootstrap_95ci'] = np.quantile(means,[.025,.975]).tolist()
    summary = {'n_windows':len(x),'n_devices':len(np.unique(devices)),
               'n_device_days':len(set(zip(devices.tolist(),days.tolist()))),
               'class_counts':np.bincount(y,minlength=4).tolist(),
               'per_device':{d:{'windows':int((devices==d).sum()),'days':len(np.unique(days[devices==d])),
                    'classes':np.bincount(y[devices==d],minlength=4).tolist()} for d in np.unique(devices)},
               'results':results,'failures':failures,'elapsed_s':time.time()-start,
               'readout':'train-fold StandardScaler + RidgeClassifier(alpha=10,class_weight=balanced); no tuning',
               'scope':'frozen representation transfer on a curated 0-100Hz 2s magnetic pilot; no backbone fine-tuning',
               'environment':{p:importlib.metadata.version(p) for p in ('torch','transformers','granite-tsfm','mantis-tsfm','scikit-learn','numpy','scipy')}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'complete.json').write_text(json.dumps({'success':True,'n_predictions':len(predictions),'elapsed_s':summary['elapsed_s']}))


if __name__ == '__main__':
    main()
