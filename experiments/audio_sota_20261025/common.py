import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def wilson(k, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = k/n
    den = 1+z*z/n
    centre = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [centre-half, centre+half]


def setup(device):
    torch.set_num_threads(8)
    torch.manual_seed(20261025)
    np.random.seed(20261025)
    if device == 'cuda':
        # User authorized shared GPU use, at most approximately 24 GiB.
        cap = 24 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(cap, 0)
        torch.cuda.reset_peak_memory_stats()
    return device


def runtime(device):
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'device': device, 'threads': torch.get_num_threads(),
            'peak_cuda_allocated_gib': torch.cuda.max_memory_allocated()/1024**3 if device == 'cuda' else 0,
            'peak_cuda_reserved_gib': torch.cuda.max_memory_reserved()/1024**3 if device == 'cuda' else 0,
            'gpu_cap_gib': 24 if device == 'cuda' else 0}


def git_rev(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
