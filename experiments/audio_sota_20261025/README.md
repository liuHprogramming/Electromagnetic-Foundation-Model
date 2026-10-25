# Reproduce the task-specific audio evaluations

This evaluates published checkpoints; it does not repeat large-scale pretraining.
See `protocol.json` for the pre-specified task definitions and limitations.
The final task tables are in `../../results/audio_sota_20261025/REPORT.md`.

## Environment

Python 3.12 was used. Install the matching CUDA runtime on a CUDA host:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python transformers==4.57.1 accelerate timm torchlibrosa safetensors soundfile scikit-learn pandas pyarrow librosa hydra-core omegaconf einops pydantic tiktoken requests
```

`environment.txt` in the results records the versions actually installed.
No FlashAttention build or server daemon is required.

## Assets and source

Choose a work directory `WORK` with at least 60 GB free. Copy this directory's
Python files into `$WORK/code/` and `assets.json` into `$WORK/code/assets.json`.
Run from `$WORK`, with its `.venv` as above:

```bash
.venv/bin/python code/fetch_assets.py --root "$WORK" --manifest code/assets.json
git clone https://github.com/OpenMOSS/MOSS-Audio.git external/moss
git -C external/moss checkout 66326e6e0db34f036c86a76ba005efa4830c69dd
git clone https://github.com/merlresearch/noise-aware-audio-ssl.git external/nassl
git -C external/nassl checkout 2d792f5ff6cd1a119016fbce35e0c618d1e610b6
uv pip install --python .venv/bin/python --no-deps -e external/nassl
```

`fetch_assets.py` uses public `hf-mirror.com` downloads at pinned revisions and
checks publisher LFS SHA-256 values. Install `aria2c` before running it. Download
the original pilot BEATs/Qwen checkpoints using the main repository's
`download_models.py`; set up the original DCASE2020 pilot with
`prepare_subset.py`. `fetch_asd.py` expects the existing BEATs checkpoint at
`$WORK/../models/beats/BEATs_iter3.pt`; it creates a cache link after verifying
the SHA-256. The Qwen path and pilot-data path are explicit CLI arguments.

```bash
.venv/bin/python code/fetch_asd.py --root "$WORK"
```

The receipts record exact hashes of every downloaded asset. Respect the source
datasets' licenses: ESC-50 and DCASE recordings include noncommercial terms;
NA-SSL code is AGPL-3.0-or-later with separate pretrained-model notices. No
source audio or checkpoint is redistributed in this repository.

ESC-50's recipe is pinned to `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`.
The original master-snapshot fetch is retained in the receipt; its metadata
SHA-256 matches that pinned revision. Per-clip WAV hashes in the prediction
files identify the evaluated content regardless of archive root-directory name.

## Evaluate

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NASSL_CACHE_DIR="$WORK/models/nassl"
.venv/bin/python code/run_esc50.py --root "$WORK" --device cuda
.venv/bin/python code/prepare_qa.py --root "$WORK" --pilot-root /path/to/original/dcase2020
.venv/bin/python code/run_qa.py --root "$WORK" --qwen-model /path/to/qwen2-audio --model moss --batch-size 8
.venv/bin/python code/run_qa.py --root "$WORK" --qwen-model /path/to/qwen2-audio --model qwen
.venv/bin/python code/run_asd2026.py --root "$WORK" --device cuda
.venv/bin/python code/analyze_results.py --results "$WORK/results"
.venv/bin/python code/write_report.py --results "$WORK/results"
```

QA outputs are appended after each case and resume by case ID. Use a fresh
output directory when changing model, prompts, decoding or protocol; existing
responses must not be silently reused across configurations. Intermediate ASD
embeddings may also be reused only with the same manifest and model settings.

The 24 GiB GPU cap is implemented in `common.setup`; scripts use one GPU
process at a time. Full original-model training and the DCASE2026 winning
ensemble are not reproduced by these commands.

QA uses BF16 weights. ASD uses CUDA BF16 autocast, float32 pooling/scoring,
and batches of at most 16 clips with identical lengths; no recording is
temporally cropped or padded for batching. ESC-50 uses float32 inference.
