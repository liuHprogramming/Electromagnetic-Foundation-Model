"""Add verified task results while preserving the earlier pilot record."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    a=p.parse_args();path=a.repo/'README.md'
    s=json.loads((a.repo/'results/audio_sota_20261025/summary.json').read_text())
    pct=lambda v:f'{100*v:.2f}%'
    asd={r['model']:r for r in s['asd2026']['aggregate'] if r['variant']=='beam_rdp4'}
    text='''# Audio foundation models: task-specific reproduction and evaluation

This repository contains the original BEATs/FISHER pilot and the 25 October 2026
evaluation of recent leading audio methods on task-matched benchmarks.
The new work evaluates released checkpoints; it does not repeat large-scale
pretraining or claim a new leaderboard SOTA.

## Task-specific evaluations — 25 October 2026

| Task / benchmark | Method | Evaluation size | Measured result |
| --- | --- | ---: | --- |
'''
    text+=f'| Environmental sound classification / ESC-50 | AudioMosaic, five official fold checkpoints | 2,000 held-out clips | Accuracy **{pct(s["esc50"]["accuracy"])}**, macro-F1 {s["esc50"]["macro_f1"]:.4f} |\n'
    for model,display in [('moss','MOSS-Audio-8B-Thinking'),('qwen','Qwen2-Audio-7B-Instruct')]:
        r=s['qa'][model]['mmau_sound']
        text+=f'| Audio reasoning / MMAU-test-mini sound | {display} | {r["n"]} questions | **{r["correct"]}/{r["n"]} = {pct(r["accuracy_or_original_answer_retention"])}** |\n'
    for model in ('BEATs_iter3','dcase2026_nabeats_seed0'):
        r=asd[model]
        text+=f'| Noise-aware ASD / DCASE2026 dev | {model}, frequency memory banks + RDP4 | 7,000 normal train + 1,400 test | Harmonic source AUC / target AUC / pAUC **{pct(r["harmonic_auc_source_target_pauc"])}** |\n'
    for model,display in [('moss','MOSS-Audio-8B-Thinking'),('qwen','Qwen2-Audio-7B-Instruct')]:
        r=s['qa'][model]['machine_full']
        text+=f'| Machine-condition QA / diagnostic only | {display} | 200 distinct clips | Accuracy **{pct(r["accuracy_or_original_answer_retention"])}**, binary answer coverage {pct(r["binary_answer_coverage"])} |\n'
    text+='''
![Task-specific evaluation summary](results/audio_sota_20261025/evaluation_summary.png)

**Reproduction finding:** the released AudioMosaic HF convenience preprocessing
gave 64.35% ESC-50 accuracy. The author training code uses `(fbank-mean)/(2*std)`;
the convenience function omits the factor of two. Using the training formula
recovers the published five-fold mean of 97.45% (rounded to 97.5% in the paper).
Both complete runs are retained; the correction follows source code, not a test
hyperparameter search.

The corrected Qwen run uses Transformers4.57.1's `audio=` interface and asserts
acoustic features before generation. The initial ignored-`audios=` run is
retained as an invalid-run artifact and excluded from the reported scores.
The original pilot below keeps its documented4.45.2 environment/API.

The QA evaluation additionally includes 60 silent and 60 mismatched MMAU inputs
per model, 200 distinct normal/anomalous machine recordings, and 40 silent plus
40 short machine clips for protocol-defined abstention. Invalid and truncated
answers stay in the denominator. The full report includes uncertainty,
paired tests, per-machine metrics, controls and resource usage.
Repeated MMAU recordings are grouped for uncertainty (333 questions from311
audio hashes); answer choices are imbalanced, with an always-A reference of
49.25%. The report also gives balanced choice accuracy.

The machine diagnostic supplies no machine-specific normal reference. MOSS
abstains on146/200 full clips; Qwen answers NORMAL on all200. MOSS meets the
protocol's abstention expectation on80/80 silent/short controls, versus0/80 for
Qwen. General sound-QA success therefore does not establish fault diagnosis.
For ASD, the observed NA-BEATs gain of1.80 points has a paired95% interval of
[-0.12,+3.66] points, so a stable advantage is not resolved in this run.

- [English evaluation report](results/audio_sota_20261025/REPORT.md)
- [Verified metrics](results/audio_sota_20261025/summary.json)
- [Reproduction commands and pinned assets](experiments/audio_sota_20261025/README.md)
- [Fixed protocol and scope](experiments/audio_sota_20261025/protocol.json)
- [Artifact integrity checks](results/audio_sota_20261025/verification.json)

The DCASE2026 run uses frozen seed0 representations and a documented normal-only
detector. Discriminative fine-tuning, score rescaling and three-seed ensembling
are excluded, so these scores are **not the complete 70.24-point winning system**.
ESC-50 accuracy and MMAU reasoning accuracy measure different abilities from
industrial anomaly detection. All results below from the earlier pilot remain
separate from these larger task-specific evaluations.

## Original 17–23 October pilot

'''
    original=path.read_text(encoding='utf-8')
    if '\n## Original 17–23 October pilot\n' in original:
        original=original.split('\n## Original 17–23 October pilot\n',1)[1].lstrip()
    else:original=original.split('\n',1)[1].lstrip()
    path.write_text(text+original,encoding='utf-8')
    print('Updated',path)


if __name__=='__main__':main()
