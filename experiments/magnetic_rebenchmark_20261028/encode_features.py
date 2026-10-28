"""Frozen model inference on spectral envelopes; never treats a mel image as waveform."""
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def state_sha(model):
    h = hashlib.sha256()
    for name, v in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    root = Path.cwd()
    out = root / 'features'
    out.mkdir(exist_ok=True)
    cfg = json.loads((root / 'code/encoding_protocol.json').read_text())
    audit = json.loads((root / 'data/data_audit.json').read_text())
    for name, item in audit['artifacts'].items():
        path = root / 'data' / name
        if path.exists() and sha(path) != item['sha256']:
            raise ValueError('Input transfer checksum mismatch: ' + name)
    x = np.load(root / 'data/spectrograms.npy')
    bands = np.load(root / 'data/band_trajectories.npy')
    assert x.shape == (audit['selected_windows'], 48, 79)
    assert bands.shape == (len(x), 8, 79)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(24 * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    old = Path(cfg['old_models_root'])
    expected = cfg['checkpoint_sha256']
    for family in ['mantis', 'tspulse', 'fisher']:
        folder = old / family if family != 'fisher' else Path(cfg['fisher_root'])
        if sha(folder / 'model.safetensors') != expected[family]:
            raise ValueError(f'{family} checkpoint SHA mismatch')
    from mantis.architecture import MantisV2
    from safetensors.torch import load_file
    from tsfm_public.models.tspulse import TSPulseForReconstruction
    from transformers import AutoModel

    # Train-free features retain absolute dB levels and signed time direction.
    t = np.linspace(-1, 1, x.shape[-1], dtype='float32')
    slope = (x * t[None, None]).sum(-1) / float((t * t).sum())
    early_late = x[:, :, -20:].mean(-1) - x[:, :, :20].mean(-1)
    phys = np.concatenate([x.mean(-1), x.std(-1), np.quantile(x, .1, axis=-1),
                           np.quantile(x, .9, axis=-1), slope, early_late,
                           np.abs(np.diff(x, axis=-1)).mean(-1)], axis=1).astype('float32')
    np.save(out / 'spectral_trends.npy', phys)
    np.save(out / 'flat_mel.npy', x.reshape(len(x), -1))
    receipt = {'input_manifest_sha256': sha(root / 'data/manifest.json'),
               'encoding_protocol_sha256': sha(root / 'code/encoding_protocol.json'),
               'models': {}, 'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                                          'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0)},
               'implementations': {}, 'started_unix': time.time()}
    for pkg in ['mantis-tsfm', 'granite-tsfm', 'transformers', 'safetensors']:
        receipt['implementations'][pkg] = importlib.metadata.version(pkg)
    for family, seeds in [('mantis', [None, 42, 43, 44]), ('tspulse', [None, 42]), ('fisher', [None])]:
        for seed in seeds:
            started = time.time()
            key = family + ('_pretrained' if seed is None else f'_random{seed}')
            torch.manual_seed(42 if seed is None else seed)
            loading = {}
            if family == 'mantis':
                # Same architecture/settings for all pretrained/random runs.
                model = MantisV2(device='cuda', return_transf_layer=2, output_token='combined')
                if seed is None:
                    result = model.load_state_dict(load_file(str(old / family / 'model.safetensors')), strict=True)
                    assert not result.missing_keys and not result.unexpected_keys
                def encode(a):
                    b = a.shape[0]
                    seq = F.interpolate(a, size=512, mode='linear', align_corners=False)
                    v = model(seq.reshape(b * 8, 1, 512))
                    return v.reshape(b, -1)
                input_array = bands
            elif family == 'tspulse':
                loaded, loading = TSPulseForReconstruction.from_pretrained(
                    old / family, local_files_only=True, output_loading_info=True)
                assert not any(loading.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
                model = loaded if seed is None else TSPulseForReconstruction(loaded.config)
                if seed is not None:
                    del loaded
                def encode(a):
                    b = a.shape[0]
                    seq = F.interpolate(a, size=512, mode='linear', align_corners=False).reshape(b * 8, 512, 1)
                    z = model.backbone(seq, enable_masking=False, return_dict=True).last_hidden_state[:, 0]
                    # Fixed ordered time blocks (including published register
                    # tokens) preserve direction without a 26k-wide readout.
                    pool = F.adaptive_avg_pool1d(z.transpose(1, 2), 8).transpose(1, 2)
                    return pool.reshape(b, -1)
                input_array = bands
            else:
                model, loading = AutoModel.from_pretrained(cfg['fisher_root'], trust_remote_code=True,
                                                           local_files_only=True, output_loading_info=True)
                assert not any(loading.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
                def encode(a):
                    # dB power -> natural-log magnitude of mel-band energy.
                    # Frequency axis remains LP100 magnetic mel, not the
                    # published audio linear-frequency STFT. Explicit adapter.
                    logmag = a.transpose(1, 2) * (np.log(10.0) / 20.0)
                    z = (logmag + 3.017344307886898) / (2 * 2.1531635155379805)
                    z = F.pad(z, (0, (-z.shape[-1]) % model.cfg.band_width))
                    return model.extract_features(z[:, None])
                input_array = x
            model = model.eval().cuda()
            for par in model.parameters():
                par.requires_grad_(False)
            checksum = state_sha(model)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                probe = torch.tensor(input_array[:2], device='cuda')
                first, repeat = encode(probe), encode(probe)
                zero = encode(torch.zeros_like(probe))
                reverse = encode(probe.flip(-1))
                assert torch.isfinite(first).all() and torch.allclose(first, repeat, rtol=1e-4, atol=1e-5)
                assert not torch.allclose(first, zero, rtol=1e-4, atol=1e-5), 'Model ignored input'
                assert not torch.allclose(first, reverse, rtol=1e-4, atol=1e-5), 'Temporal direction ignored'
                rows = []
                for i in range(0, len(input_array), 16):
                    rows.append(encode(torch.tensor(input_array[i:i + 16], device='cuda')).float().cpu().numpy())
                z = np.concatenate(rows)
                assert len(z) == len(x) and np.isfinite(z).all()
                assert np.std(z, axis=0).max() > 1e-5
            np.save(out / (key + '.npy'), z)
            receipt['models'][key] = {'shape': list(z.shape), 'seed': seed, 'parameters': sum(p.numel() for p in model.parameters()),
                                       'state_sha256': checksum, 'loading_info': loading,
                                       'input_delta_l2': float(torch.linalg.vector_norm(first - zero).cpu()),
                                       'reverse_delta_l2': float(torch.linalg.vector_norm(first - reverse).cpu()),
                                       'elapsed_seconds': time.time() - started,
                                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                       'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            print('EMBEDDED', key, json.dumps(receipt['models'][key]), flush=True)
            (out / 'model_receipt.json').write_text(json.dumps(receipt, indent=2))
            del encode, model, z, rows, probe, first, repeat, zero, reverse
            if family == 'tspulse' and seed is None:
                del loaded
            gc.collect()
            torch.cuda.empty_cache()
    receipt['artifacts'] = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(out.glob('*.npy'))}
    receipt['elapsed_seconds'] = time.time() - receipt['started_unix']
    (out / 'model_receipt.json').write_text(json.dumps(receipt, indent=2))
    print('ALL_EMBEDDINGS_READY', receipt['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
