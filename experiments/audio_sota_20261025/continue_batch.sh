#!/usr/bin/env bash
set -eu
BASE=${1:?remote root}
cd "$BASE"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 NASSL_CACHE_DIR="$BASE/models/nassl"
export PYTHONUNBUFFERED=1
trap 'echo $? > "$BASE/suite_batch.exit"' EXIT
.venv/bin/python code/run_qa.py --root "$BASE" --qwen-model "$BASE/../models/qwen2-audio" --model moss --limit 8 --batch-size 8 --output-name smoke_batch8.jsonl > logs/moss_batch8_smoke.log 2>&1
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('results/qa/moss')
native={r['case_id']:r for r in map(json.loads,(p/'raw_sequential_smoke.jsonl').read_text().splitlines())}
batch=list(map(json.loads,(p/'smoke_batch8.jsonl').read_text().splitlines()))
assert len(batch)==8
for row in batch:
    for key in ['parsed_answer','pcm_sha256','rendered_input_ids_sha256']:
        assert row[key]==native[row['case_id']][key],(row['case_id'],key)
(p/'batch8_equivalence.json').write_text(json.dumps({'passed':True,'n':8,'checks':['parsed answer','PCM hash','un-padded input IDs hash'],'batch_size':8},indent=2))
print('BATCH EQUIVALENCE PASSED')
PY
.venv/bin/python code/run_qa.py --root "$BASE" --qwen-model "$BASE/../models/qwen2-audio" --model moss --batch-size 8 > logs/moss_batch8.log 2>&1
.venv/bin/python code/run_qa.py --root "$BASE" --qwen-model "$BASE/../models/qwen2-audio" --model qwen > logs/qwen.log 2>&1
test "$(cat asd_download.exit)" = 0
.venv/bin/python code/run_asd2026.py --root "$BASE" --device cuda > logs/asd.log 2>&1
.venv/bin/python code/analyze_results.py --results results > logs/analysis.log 2>&1
echo "All evaluations and metrics checks completed $(date -Iseconds)"
