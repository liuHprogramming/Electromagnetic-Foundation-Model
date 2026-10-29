"""Render English task results from verified summary JSON."""
import argparse
import json
from pathlib import Path


def pct(x):return 'N/A' if x is None else f'{100*x:.2f}%'


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    a=p.parse_args();s=json.loads((a.results/'summary.json').read_text())
    lines=['# Task-specific audio evaluations — 25 October 2026','',
           'We evaluate released checkpoints of recent leading methods on task-matched benchmarks. '
           'This is inference reproduction and transfer evaluation; no new pretraining or leaderboard submission is claimed.','',
           '## Environmental sound classification: AudioMosaic / ESC-50','',
           f'All **2,000 clips** are evaluated using the five official held-out folds. Accuracy is **{pct(s["esc50"]["accuracy"])}** '
           f'and macro-F1 is **{s["esc50"]["macro_f1"]:.4f}**. Each checkpoint is used only for its advertised held-out fold.','',
           '| Fold | Test clips | Correct | Accuracy |','| --- | ---: | ---: | ---: |']
    for r in s['esc50']['folds']:lines.append(f'| {r["fold"]} | {r["n"]} | {r["correct"]} | {pct(r["accuracy"])} |')
    lines+=['','The first run with the HF convenience preprocessing scored **64.35%**. '
            'The author training dataset divides filterbanks by `2 * std`, while the released convenience function divides by `std`. '
            'Using the documented training normalization yields the result above. Both complete runs are retained. '
            'This correction was derived from source code, not a search over test-set normalization values.','',
            '[Author training normalization](https://github.com/HanxunH/AudioMosaic/blob/67a86a8900ce00fb653f1c22f771a9688ec9de33/dataset/audioset.py#L364) '
            'and [official fold checkpoints](https://github.com/HanxunH/AudioMosaic#esc-50-5-fold). '
            'The public fold percentages average to 97.45%, conventionally rounded to the paper\'s 97.5%. '
            'Upstream checkpoint selection may have used validation folds; this is not an independent audit of training-time model selection.','',
            '## Audio reasoning: MMAU-test-mini sound partition','',
            'We use every **333 sound question** in the pinned official snapshot; music and speech are outside this evaluation. '
            'Both models receive identical questions, choices and audio arrays, with greedy decoding and 512 maximum new tokens. '
            'Invalid or truncated answers remain in the denominator. Accuracy uses a strict parsed A/B/C/D choice.','',
            '| Model | Correct / N | Accuracy | Audio-cluster bootstrap 95% interval | Parse failures | Token-cap hits |',
            '| --- | ---: | ---: | --- | ---: | ---: |']
    for model in ('moss','qwen'):
        r=s['qa'][model]['mmau_sound'];ci=r['audio_cluster_bootstrap95']
        lines.append(f'| {model} | {r["correct"]}/{r["n"]} | {pct(r["accuracy_or_original_answer_retention"])} | {pct(ci[0])}–{pct(ci[1])} | {r["unparseable"]} | {r["token_cap_count"]} |')
    r=s['paired_mmau_sound'];ci=r['paired_audio_cluster_bootstrap95']
    prior=s['qa']['moss']['mmau_sound']
    lines+=['',f'The expected-choice counts are {prior["expected_choice_counts"]}; a fixed always-A rule scores '
            f'{pct(prior["fixed_always_A_accuracy"])}. Uniform random guessing still has 25% expected accuracy, '
            'but the fixed-letter baseline exposes the nonuniform answer distribution. Balanced choice accuracy (mean recall over A/B/C/D) is '
            f'{pct(s["qa"]["moss"]["mmau_sound"]["balanced_choice_accuracy"])} for MOSS and '
            f'{pct(s["qa"]["qwen"]["mmau_sound"]["balanced_choice_accuracy"])} for Qwen.','']
    lines+=['',f'Paired MOSS minus Qwen accuracy difference: **{100*r["accuracy_difference_moss_minus_qwen"]:+.2f} percentage points**, '
            f'paired audio-cluster bootstrap 95% interval [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] points; '
            f'audio-cluster sign-flip Monte Carlo p={r["audio_cluster_sign_flip_p_two_sided"]:.5g} (100,000 permutations). '
            f'The 333 questions cover {r["audio_clusters"]} distinct audio hashes; repeated questions are grouped together. '
            'These are conditional test-mini comparisons, not a claim of universal SOTA or independent unseen-data generalization.','',
            'Question-level Wilson intervals and exact McNemar results remain in the JSON as exploratory summaries; '
            'the primary uncertainty above accounts for repeated recordings.','',
            '[Official benchmark](https://github.com/Sakshi113/MMAU) '
            'and [MOSS source](https://github.com/OpenMOSS/MOSS-Audio/tree/66326e6e0db34f036c86a76ba005efa4830c69dd). '
            'The sound-only result is not directly comparable with the paper\'s speech/music/sound average. '
            'Public-test contamination cannot be excluded from upstream model training.','',
            '### Audio-dependence controls','',
            'Sixty fixed questions are repeated with silence and with another recording. '
            'The table measures retention of the **original** correct answer, not accuracy on a newly labeled task. '
            'Matched questions may contain text clues, and mismatched audio can occasionally support the same choice.','',
            '| Model | Condition | Matched-audio accuracy on same 60 | Original answer retained | Rate | Drop from matched |','| --- | --- | ---: | ---: | ---: | ---: |']
    for model in ('moss','qwen'):
        for suite in ('mmau_silence','mmau_mismatch'):
            r=s['qa'][model][suite]
            lines.append(f'| {model} | {suite} | {pct(r["matched_audio_accuracy"])} | {r["correct"]}/{r["n"]} | {pct(r["accuracy_or_original_answer_retention"])} | {100*r["matched_to_control_accuracy_drop"]:+.2f} pp |')
    lines+=['','## Machine-condition QA and abstention','',
            'The diagnostic set contains **200 distinct DCASE2020 test recordings** (100 normal, 100 anomalous; fan and pump). '
            'Normal/anomaly labels are hidden from the model prompt. Another 40 silent and 40 quarter-second versions test '
            'protocol-defined abstention. These controls share source recordings and must not be treated as 280 independent recordings.','',
            '| Model | Full-clip accuracy | Binary coverage | Accuracy when answered | Normal specificity | Anomaly sensitivity | Silent abstention | Short-clip abstention |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model in ('moss','qwen'):
        r=s['qa'][model]['machine_full'];silent=s['qa'][model]['machine_silence'];short=s['qa'][model]['machine_short']
        lines.append(f'| {model} | {pct(r["accuracy_or_original_answer_retention"])} | {pct(r["binary_answer_coverage"])} | {pct(r["accuracy_on_answered"])} | {pct(r["normal_specificity"])} | {pct(r["anomalous_sensitivity"])} | {silent["correct"]}/40 | {short["correct"]}/40 |')
    lines+=['','A 0.25-second crop is not intrinsically unlabeled: “insufficient evidence” is our assessment rule. '
            'Condition labels do not verify the model\'s proposed fault explanation, and no exact physical diagnosis is established.','',
            'Full-clip abstention and conditional accuracy are reported alongside overall correctness. '
            'No machine-specific normal reference was provided, so a refusal can reflect task ambiguity rather than a wrong audible description. '
            'Conditional accuracy must be interpreted at its reported coverage; it is not directly comparable across unequal answer rates.','',
            '## Noise-aware machine anomaly detection: DCASE2026','',
            f'The complete seven-machine development set has **{s["asd2026"]["n"]:,} recordings**: '
            '7,000 normal training clips and 1,400 test clips. Each type has 990 source and 10 target normal training clips, '
            'and 50 normal/50 anomalous test clips in each domain. Archive MD5 and published checkpoint hashes are checked.','',
            'BEATs uses the near microphone. NA-BEATs seed0 also receives the synchronized far microphone as auxiliary noise input. '
            'Both encoders are frozen. All normal training clips form memory banks; no test labels enter features or scoring. '
            'Global/frequency mean pooling and frequency-wise cosine memory banks with RDP gamma=4 are pre-specified and all reported.','',
            '| Frontend | Backend | Harmonic source AUC / target AUC / standardized pAUC |',
            '| --- | --- | ---: |']
    for r in s['asd2026']['aggregate']:
        lines.append(f'| {r["model"]} | {r["variant"]} | {pct(r["harmonic_auc_source_target_pauc"])} |')
    u=s['asd2026']['uncertainty'];ci=u['difference_nabeats_minus_beats_ci95']
    lines+=['',f'For the pre-specified `beam_rdp4` backend, the paired stratified clip bootstrap 95% interval '
            f'for NA-BEATs minus BEATs is [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] percentage points '
            '(1,000 resamples within machine/domain/condition). Intervals describe this released dataset, '
            'not independent recording-day/device uncertainty.','',
            'This implements the published frontend and pooling/memory-bank ideas; it excludes discriminative fine-tuning, '
            'score rescaling and the three-seed ensemble. **It is not a reproduction of the complete 70.24-point winning system.** '
            'Per-machine source/target AUC and standardized pAUC (`max_fpr=0.1`) are in `asd2026/metrics.json`.','',
            '[Official dataset](https://zenodo.org/records/19336329), '
            '[NA-SSL implementation](https://github.com/merlresearch/noise-aware-audio-ssl/tree/2d792f5ff6cd1a119016fbce35e0c618d1e610b6) '
            'and [method paper](https://arxiv.org/abs/2608.00447).','',
            '## Reproduction and resources','',
            'The code is in `experiments/audio_sota_20261025/`; model/dataset revisions and publisher hashes are in `assets.json`. '
            'Raw audio, checkpoints and large intermediate embeddings stay outside Git. '
            'The result directory retains raw replies, per-clip scores, manifests, failure logs and environment metadata.','',
            'Run the preparation and evaluation scripts using the commands in '
            '`experiments/audio_sota_20261025/README.md`. A800 jobs use one GPU process at a time, '
            'with a 24 GiB PyTorch allocation cap and eight CPU threads. Existing GPU processes were left running. '
            'Runtime measurements on a shared GPU are not controlled speed comparisons.','',
            'MOSS batching (batch sizes four/eight) was checked against native single-example inference on four/eight cases '
            '(answers, audio hashes and unpadded input IDs). Qwen uses its native audio processor; '
            'any input exceeding its processor window is flagged in the raw records. '
            'No local training, prompt sweep, threshold fitting or best-test-variant selection was performed.','',
            '### Qwen processor API correction','',
            'Transformers4.57.1 expects `audio=`, whereas the original4.45.2 pilot uses `audios=`. '
            'The initial current-environment Qwen run ignored the legacy keyword and produced no acoustic features. '
            'That run is retained under `qa/qwen_invalid_ignored_audio/` and excluded from every primary comparison. '
            'The corrected run asserts `input_features` and `feature_attention_mask` before generation. '
            '`qa/audio_input_audit.json` verifies that real/silent inputs have identical text IDs and different '
            'acoustic tensors for both models.','',
            '[Current processor source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_audio/processing_qwen2_audio.py) '
            'and [legacy pinned processor source](https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2_audio/processing_qwen2_audio.py).','',
            '## Interpretation','',
            'The ESC-50 reproduction validates the checkpoint and preprocessing path. Its classification accuracy '
            'does not predict machine-anomaly or process-QA performance. The paired audio-QA and audio-dependence controls '
            'separate answer correctness from reliance on audible evidence. Industrial QA remains a domain-transfer diagnostic, '
            'while DCASE2026 tests noise-aware representations under a controlled normal-only detector. '
            'Follow-up instrument experiments still require synchronized recordings and independent day/device splits.','']
    delta=s['paired_mmau_sound']['accuracy_difference_moss_minus_qwen']
    lines+=[f'On the fixed sound partition, MOSS differs from Qwen by {100*delta:+.2f} percentage points. '
            'The paired interval above should determine whether the observed difference is resolved at this sample size. '
            'Audio-control retention must be compared with matched-audio performance on the same 60 questions, not all 333.', '']
    selected={r['model']:r['harmonic_auc_source_target_pauc'] for r in s['asd2026']['aggregate'] if r['variant']=='beam_rdp4'}
    difference=selected['dcase2026_nabeats_seed0']-selected['BEATs_iter3']
    lines+=[f'With the fixed frequency/RDP4 detector, NA-BEATs differs from BEATs by {100*difference:+.2f} percentage points. '
            'This isolates the released noise-aware frontend under the chosen backend; it does not isolate every part of the winning training recipe.', '']
    for model in ('moss','qwen'):
        r=s['qa'][model]['machine_full']
        lines+=[f'{model}: machine-condition accuracy {pct(r["accuracy_or_original_answer_retention"])}, '
                f'normal specificity {pct(r["normal_specificity"])}. '
                'No machine-specific normal reference, operating-state history or fault supervision was supplied to either language model. '
                'That missing context and the shift from general sound QA limit conclusions about industrial diagnosis.', '']
    resources=[]
    for label,relative,scope in [('ESC-50','esc50/metrics.json','all five folds'),
                                 ('MOSS','qa/moss/run_metadata.json','latest resumed batch8 process'),
                                 ('Qwen','qa/qwen/run_metadata.json','corrected full733-case run'),
                                 ('ASD','asd2026/metrics.json','both frontends, all8400 clips each')]:
        metadata=json.loads((a.results/relative).read_text())
        r=metadata['runtime'];secs=metadata.get('seconds',metadata.get('seconds_this_invocation'))
        resources.append(f'| {label} | {r["peak_cuda_allocated_gib"]:.2f} | {r["peak_cuda_reserved_gib"]:.2f} | {secs:.1f} | {scope} |')
    lines+=['## Measured runtime records','',
            '| Stage | Peak allocated GiB | Peak reserved GiB | Recorded invocation seconds | Scope |',
            '| --- | ---: | ---: | ---: | --- |',*resources,'',
            'CUDA figures come from PyTorch process counters, not total board memory. Resumed MOSS timings do not '
            'include earlier completed invocations. The GPU was shared and these are not controlled speed benchmarks.','']
    if s['asd2026']['uncertainty'].get('domain_auc_definition')!='official_domain_normals_vs_all_anomalies':
        lines[2:2]=['> Historical ASD statistics use within-domain-only AUROC. They are not official DCASE domain AUROCs. '
                    'See `../baseline_improvements_20261029/REPORT.md` for the 29 October correction. '
                    'The original model outputs remain unchanged.','']
    (a.results/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote',a.results/'REPORT.md')


if __name__=='__main__':main()
