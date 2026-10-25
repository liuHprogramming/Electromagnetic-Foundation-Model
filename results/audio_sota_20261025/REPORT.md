# Task-specific audio evaluations — 25 October 2026

We evaluate released checkpoints of recent leading methods on task-matched benchmarks. This is inference reproduction and transfer evaluation; no new pretraining or leaderboard submission is claimed.

## Environmental sound classification: AudioMosaic / ESC-50

All **2,000 clips** are evaluated using the five official held-out folds. Accuracy is **97.45%** and macro-F1 is **0.9743**. Each checkpoint is used only for its advertised held-out fold.

| Fold | Test clips | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| 1 | 400 | 389 | 97.25% |
| 2 | 400 | 395 | 98.75% |
| 3 | 400 | 388 | 97.00% |
| 4 | 400 | 394 | 98.50% |
| 5 | 400 | 383 | 95.75% |

The first run with the HF convenience preprocessing scored **64.35%**. The author training dataset divides filterbanks by `2 * std`, while the released convenience function divides by `std`. Using the documented training normalization yields the result above. Both complete runs are retained. This correction was derived from source code, not a search over test-set normalization values.

[Author training normalization](https://github.com/HanxunH/AudioMosaic/blob/67a86a8900ce00fb653f1c22f771a9688ec9de33/dataset/audioset.py#L364) and [official fold checkpoints](https://github.com/HanxunH/AudioMosaic#esc-50-5-fold). The public fold percentages average to 97.45%, conventionally rounded to the paper's 97.5%. Upstream checkpoint selection may have used validation folds; this is not an independent audit of training-time model selection.

## Audio reasoning: MMAU-test-mini sound partition

We use every **333 sound question** in the pinned official snapshot; music and speech are outside this evaluation. Both models receive identical questions, choices and audio arrays, with greedy decoding and 512 maximum new tokens. Invalid or truncated answers remain in the denominator. Accuracy uses a strict parsed A/B/C/D choice.

| Model | Correct / N | Accuracy | Audio-cluster bootstrap 95% interval | Parse failures | Token-cap hits |
| --- | ---: | ---: | --- | ---: | ---: |
| moss | 274/333 | 82.28% | 78.04%–86.22% | 1 | 1 |
| qwen | 214/333 | 64.26% | 59.04%–69.46% | 0 | 0 |

The expected-choice counts are {'A': 164, 'B': 55, 'C': 81, 'D': 33}; a fixed always-A rule scores 49.25%. Uniform random guessing still has 25% expected accuracy, but the fixed-letter baseline exposes the nonuniform answer distribution. Balanced choice accuracy (mean recall over A/B/C/D) is 80.17% for MOSS and 69.26% for Qwen.


Paired MOSS minus Qwen accuracy difference: **+18.02 percentage points**, paired audio-cluster bootstrap 95% interval [+12.23, +23.89] points; audio-cluster sign-flip Monte Carlo p=9.9999e-06 (100,000 permutations). The 333 questions cover 311 distinct audio hashes; repeated questions are grouped together. These are conditional test-mini comparisons, not a claim of universal SOTA or independent unseen-data generalization.

Question-level Wilson intervals and exact McNemar results remain in the JSON as exploratory summaries; the primary uncertainty above accounts for repeated recordings.

[Official benchmark](https://github.com/Sakshi113/MMAU) and [MOSS source](https://github.com/OpenMOSS/MOSS-Audio/tree/66326e6e0db34f036c86a76ba005efa4830c69dd). The sound-only result is not directly comparable with the paper's speech/music/sound average. Public-test contamination cannot be excluded from upstream model training.

### Audio-dependence controls

Sixty fixed questions are repeated with silence and with another recording. The table measures retention of the **original** correct answer, not accuracy on a newly labeled task. Matched questions may contain text clues, and mismatched audio can occasionally support the same choice.

| Model | Condition | Matched-audio accuracy on same 60 | Original answer retained | Rate | Drop from matched |
| --- | --- | ---: | ---: | ---: | ---: |
| moss | mmau_silence | 83.33% | 38/60 | 63.33% | +20.00 pp |
| moss | mmau_mismatch | 83.33% | 34/60 | 56.67% | +26.67 pp |
| qwen | mmau_silence | 75.00% | 26/60 | 43.33% | +31.67 pp |
| qwen | mmau_mismatch | 75.00% | 26/60 | 43.33% | +31.67 pp |

## Machine-condition QA and abstention

The diagnostic set contains **200 distinct DCASE2020 test recordings** (100 normal, 100 anomalous; fan and pump). Normal/anomaly labels are hidden from the model prompt. Another 40 silent and 40 quarter-second versions test protocol-defined abstention. These controls share source recordings and must not be treated as 280 independent recordings.

| Model | Full-clip accuracy | Binary coverage | Accuracy when answered | Normal specificity | Anomaly sensitivity | Silent abstention | Short-clip abstention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| moss | 13.00% | 25.50% | 50.98% | 25.00% | 1.00% | 40/40 | 40/40 |
| qwen | 50.00% | 100.00% | 50.00% | 100.00% | 0.00% | 0/40 | 0/40 |

A 0.25-second crop is not intrinsically unlabeled: “insufficient evidence” is our assessment rule. Condition labels do not verify the model's proposed fault explanation, and no exact physical diagnosis is established.

Full-clip abstention and conditional accuracy are reported alongside overall correctness. No machine-specific normal reference was provided, so a refusal can reflect task ambiguity rather than a wrong audible description. Conditional accuracy must be interpreted at its reported coverage; it is not directly comparable across unequal answer rates.

## Noise-aware machine anomaly detection: DCASE2026

The complete seven-machine development set has **8,400 recordings**: 7,000 normal training clips and 1,400 test clips. Each type has 990 source and 10 target normal training clips, and 50 normal/50 anomalous test clips in each domain. Archive MD5 and published checkpoint hashes are checked.

BEATs uses the near microphone. NA-BEATs seed0 also receives the synchronized far microphone as auxiliary noise input. Both encoders are frozen. All normal training clips form memory banks; no test labels enter features or scoring. Global/frequency mean pooling and frequency-wise cosine memory banks with RDP gamma=4 are pre-specified and all reported.

| Frontend | Backend | Harmonic source AUC / target AUC / standardized pAUC |
| --- | --- | ---: |
| BEATs_iter3 | global_ap | 58.66% |
| BEATs_iter3 | frequency_ap | 60.21% |
| BEATs_iter3 | beam_ap | 60.28% |
| BEATs_iter3 | beam_rdp4 | 60.80% |
| dcase2026_nabeats_seed0 | global_ap | 60.83% |
| dcase2026_nabeats_seed0 | frequency_ap | 62.13% |
| dcase2026_nabeats_seed0 | beam_ap | 62.01% |
| dcase2026_nabeats_seed0 | beam_rdp4 | 62.60% |

For the pre-specified `beam_rdp4` backend, the paired stratified clip bootstrap 95% interval for NA-BEATs minus BEATs is [-0.12, +3.66] percentage points (1,000 resamples within machine/domain/condition). Intervals describe this released dataset, not independent recording-day/device uncertainty.

This implements the published frontend and pooling/memory-bank ideas; it excludes discriminative fine-tuning, score rescaling and the three-seed ensemble. **It is not a reproduction of the complete 70.24-point winning system.** Per-machine source/target AUC and standardized pAUC (`max_fpr=0.1`) are in `asd2026/metrics.json`.

[Official dataset](https://zenodo.org/records/19336329), [NA-SSL implementation](https://github.com/merlresearch/noise-aware-audio-ssl/tree/2d792f5ff6cd1a119016fbce35e0c618d1e610b6) and [method paper](https://arxiv.org/abs/2608.00447).

## Reproduction and resources

The code is in `experiments/audio_sota_20261025/`; model/dataset revisions and publisher hashes are in `assets.json`. Raw audio, checkpoints and large intermediate embeddings stay outside Git. The result directory retains raw replies, per-clip scores, manifests, failure logs and environment metadata.

Run the preparation and evaluation scripts using the commands in `experiments/audio_sota_20261025/README.md`. A800 jobs use one GPU process at a time, with a 24 GiB PyTorch allocation cap and eight CPU threads. Existing GPU processes were left running. Runtime measurements on a shared GPU are not controlled speed comparisons.

MOSS batching (batch sizes four/eight) was checked against native single-example inference on four/eight cases (answers, audio hashes and unpadded input IDs). Qwen uses its native audio processor; any input exceeding its processor window is flagged in the raw records. No local training, prompt sweep, threshold fitting or best-test-variant selection was performed.

### Qwen processor API correction

Transformers4.57.1 expects `audio=`, whereas the original4.45.2 pilot uses `audios=`. The initial current-environment Qwen run ignored the legacy keyword and produced no acoustic features. That run is retained under `qa/qwen_invalid_ignored_audio/` and excluded from every primary comparison. The corrected run asserts `input_features` and `feature_attention_mask` before generation. `qa/audio_input_audit.json` verifies that real/silent inputs have identical text IDs and different acoustic tensors for both models.

[Current processor source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_audio/processing_qwen2_audio.py) and [legacy pinned processor source](https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2_audio/processing_qwen2_audio.py).

## Interpretation

The ESC-50 reproduction validates the checkpoint and preprocessing path. Its classification accuracy does not predict machine-anomaly or process-QA performance. The paired audio-QA and audio-dependence controls separate answer correctness from reliance on audible evidence. Industrial QA remains a domain-transfer diagnostic, while DCASE2026 tests noise-aware representations under a controlled normal-only detector. Follow-up instrument experiments still require synchronized recordings and independent day/device splits.

On the fixed sound partition, MOSS differs from Qwen by +18.02 percentage points. The paired interval above should determine whether the observed difference is resolved at this sample size. Audio-control retention must be compared with matched-audio performance on the same 60 questions, not all 333.

With the fixed frequency/RDP4 detector, NA-BEATs differs from BEATs by +1.80 percentage points. This isolates the released noise-aware frontend under the chosen backend; it does not isolate every part of the winning training recipe.

moss: machine-condition accuracy 13.00%, normal specificity 25.00%. No machine-specific normal reference, operating-state history or fault supervision was supplied to either language model. That missing context and the shift from general sound QA limit conclusions about industrial diagnosis.

qwen: machine-condition accuracy 50.00%, normal specificity 100.00%. No machine-specific normal reference, operating-state history or fault supervision was supplied to either language model. That missing context and the shift from general sound QA limit conclusions about industrial diagnosis.

## Measured runtime records

| Stage | Peak allocated GiB | Peak reserved GiB | Recorded invocation seconds | Scope |
| --- | ---: | ---: | ---: | --- |
| ESC-50 | 0.39 | 0.46 | 43.9 | all five folds |
| MOSS | 18.73 | 22.98 | 1340.0 | latest resumed batch8 process |
| Qwen | 16.34 | 18.19 | 265.7 | corrected full733-case run |
| ASD | 2.28 | 3.97 | 301.4 | both frontends, all8400 clips each |

CUDA figures come from PyTorch process counters, not total board memory. Resumed MOSS timings do not include earlier completed invocations. The GPU was shared and these are not controlled speed benchmarks.
