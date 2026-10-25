#!/usr/bin/env bash
set -eu
BASE=${1:?pass remote experiment directory}
cd "$BASE"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NASSL_CACHE_DIR="$BASE/models/nassl"
export PYTHONUNBUFFERED=1
trap 'echo $? > "$BASE/suite.exit"' EXIT
while [ ! -f download.exit ]; do sleep 10; done
test "$(cat download.exit)" = 0
echo "Starting ESC-50 $(date -Iseconds)"
.venv/bin/python code/run_esc50.py --root "$BASE" --device cuda > logs/esc50.log 2>&1
echo "ESC-50 completed $(date -Iseconds)"
.venv/bin/python code/prepare_qa.py --root "$BASE" --pilot-root "$BASE/../dcase2020" > logs/prepare_qa.log 2>&1
echo "Starting MOSS QA $(date -Iseconds)"
.venv/bin/python code/run_qa.py --root "$BASE" --qwen-model "$BASE/../models/qwen2-audio" --model moss > logs/moss.log 2>&1
echo "Starting Qwen QA $(date -Iseconds)"
.venv/bin/python code/run_qa.py --root "$BASE" --qwen-model "$BASE/../models/qwen2-audio" --model qwen > logs/qwen.log 2>&1
while [ ! -f asd_download.exit ]; do sleep 10; done
test "$(cat asd_download.exit)" = 0
echo "Starting ASD $(date -Iseconds)"
.venv/bin/python code/run_asd2026.py --root "$BASE" --device cuda > logs/asd.log 2>&1
echo "All evaluations completed $(date -Iseconds)"
