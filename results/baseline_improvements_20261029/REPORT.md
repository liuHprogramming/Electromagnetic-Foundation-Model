# Sound metric audit and magnetic improvements — 29 October 2026

The magnetic benchmark, samples and pretrained embeddings are unchanged. This run tests classifiers and temporal alternatives using training-only validation. A separate audit corrects the definition of domain AUROC in the earlier sound report. No new sound inference, GPU job, foundation-model pretraining or hardware clock repair was performed.

## 1. Machine anomalous sound detection

**Correction:** the 25 October code restricted each domain AUC to samples of that domain. The official evaluator instead compares the normal samples of the indicated domain with **all anomalies**, from both domains. The current scorer is fixed; old scores and artifacts are retained as historical outputs, and this report supersedes their official-score interpretation. ESC-50 and audio-QA results are unaffected.

[Official evaluator, pinned commit](https://github.com/nttcslab/dcase2026_task2_evaluator/blob/f6a94a2b5e614a9626c9d1ccff6df0705e6aaa75/dcase2026_task2_evaluator.py#L277) is the source of the mask. All 56 model/backend/machine groups were checked; the recomputation reproduces the old metrics and independently matches exact pairwise AUROC probabilities.

| Frozen frontend + frequency banks / RDP4 | Historical statistic | Correct official-definition dev score |
| --- | ---: | ---: |
| BEATs | 60.80% | **60.61%** |
| NA-BEATs seed0 | 62.60% | **62.62%** |

NA-BEATs minus BEATs is **+2.01 percentage points**; paired 2,000-draw stratified clip-bootstrap 95% interval [+0.12, +3.94] points. This conditions on the seven released machine types and their recordings; it is not an independent device/day or hardware-population interval. The change from the historical numbers is a metric correction, not improved model inference.

### Is BEATs itself SOTA, and what should improve next?

The [official DCASE2026 results](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring-results) rank the complete discriminatively adapted, ensembled NA-BEATs system first at 70.24115. A frozen BEATs encoder alone is not that complete system. This challenge score and our seven-machine development score concern different evaluation scopes and cannot be subtracted as an expected gain.

The [NA-SSL paper, Table 1](https://arxiv.org/html/2608.00447v2#S3.F2) reports the following development-set ensemble totals under its shared backend and rescaling protocol:

| Frontend / task adaptation | Paper development total |
| --- | ---: |
| Original BEATs | 61.32% |
| NA-BEATs | 63.92% |
| Discriminatively adapted NA-BEATs | **64.57%** |
| Discriminatively adapted NA-EAT | 64.20% |
| Discriminatively adapted NA-Dasheng | 63.10% |

These matched comparisons support retaining NA-BEATs. EAT and Dasheng are useful alternatives, but this table does not establish a stronger replacement. Our omitted steps are training-normal-based score rescaling, machine-information / pseudo-label adaptation, and averaging three seeds. Their gains need project experiments; they are not guaranteed improvements. [Official model releases](https://github.com/merlresearch/noise-aware-audio-ssl) include all three NA families.

A practical next experiment should first implement the published k=4 local score rescaling and validate it against the reference, then add LoRA-based discriminative adaptation and fixed seed ensembles. Model and backend choices should use training normals or a separate development selection protocol; test-domain labels must not enter scoring. ROC metrics cannot be repaired by choosing a more favorable binary threshold. The [ASDKit reference](https://github.com/TakuyaFujimura/dcase-asd-toolkit) supplies broader backend/adaptation recipes. This round audits scores and literature; it does not claim those new sound experiments were run.

## 2. Controlled magnetic improvements

All methods use the same **3,177** pure-state LP100 magnetic spectrogram windows, **44 device-days**, and fixed four-state labels as the 28 October benchmark. Outer evaluation remains three held-device folds and 44 held-date folds. Each device receives equal final weight, after pooling its held-date predictions. No target samples, labeled or unlabeled, are fitted in strict held-device evaluation.

Nested models select among balanced Ridge (alpha 0.1/10/1000) and RBF SVM (C 0.1/1/10). For held-device evaluation, inner folds hold out each of the two source devices; for held-date evaluation, three inner folds group the remaining training dates. Every scaler is trained inside its corresponding training partition. The automatic selector also selects the representation from inner scores only, excluding the random control.

Additional fixed alternatives are 256-tree ExtraTrees, the author's multivariate MiniRocket with 2,016 features and training-only bias fitting, and fixed logistic probes. MiniRocket sees eight spectral-envelope channels, not a reconstructed waveform; its sample-normalized counterpart removes each window's global mean and scale. Relative-spectrum features remove broadband level per frame and retain signed temporal differences. Mantis logistic probes follow the classifier motivation in [MantisV2 §5.3](https://arxiv.org/html/2602.17868v1#S5.SS3).

| Method | Within-device held-date OOF macro-F1 | Held-device macro-F1 |
| --- | ---: | ---: |
| FISHER + original fixed Ridge | 0.7332 | 0.3133 |
| MantisV2 + original fixed Ridge | 0.5377 | 0.3396 |
| Mantis random42 + original fixed Ridge | 0.5124 | 0.3460 |
| TSPulse + original fixed Ridge | 0.4170 | 0.2713 |
| Spectral statistics + original fixed Ridge | 0.6952 | 0.2838 |
| FISHER + nested head | 0.7434 | 0.3289 |
| MantisV2 + nested head | 0.6167 | 0.3206 |
| TSPulse + nested head | 0.4981 | 0.2973 |
| Spectral statistics + nested head | 0.7220 | 0.2886 |
| Relative spectrum / signed differences + nested head | 0.7140 | 0.2997 |
| FISHER + physical features + nested head | 0.7580 | 0.2900 |
| Mantis random42 + nested head | 0.5688 | 0.3372 |
| Training-only representation / head selector | 0.7614 | 0.3579 |
| ExtraTrees on physical / relative features | 0.8001 | 0.3095 |
| MiniRocket multivariate, raw band envelopes | 0.5227 | 0.3090 |
| MiniRocket multivariate, sample-normalized envelopes | 0.5301 | 0.3116 |
| MantisV2 + default logistic head | 0.5832 | 0.3248 |
| MantisV2 + balanced logistic head | 0.5848 | 0.3259 |
| FISHER + balanced logistic head | 0.7278 | 0.3230 |
| TSPulse + balanced logistic head | 0.4640 | 0.2788 |
| Mantis random42 + balanced logistic head | 0.5509 | 0.3135 |

### Matched changes and uncertainty

All methods and negative results are retained. The intervals below use the same 2,000 date-resampling draws within each device for method and reference. They condition on these devices and fitted models; they are not adjusted for all reported comparisons, do not retrain the search in every draw or account for population-level uncertainty over unseen instruments.

| Method | Protocol | Reference | Macro-F1 change | Conditional paired 95% interval |
| --- | --- | --- | ---: | --- |
| FISHER + nested head | within_device_leave_day | FISHER + original fixed Ridge | +0.0102 | [-0.0021, +0.0226] |
| FISHER + nested head | leave_device | FISHER + original fixed Ridge | +0.0157 | [+0.0040, +0.0269] |
| MantisV2 + nested head | within_device_leave_day | MantisV2 + original fixed Ridge | +0.0790 | [+0.0519, +0.1060] |
| MantisV2 + nested head | leave_device | MantisV2 + original fixed Ridge | -0.0190 | [-0.0321, -0.0049] |
| TSPulse + nested head | within_device_leave_day | TSPulse + original fixed Ridge | +0.0811 | [+0.0630, +0.0987] |
| TSPulse + nested head | leave_device | TSPulse + original fixed Ridge | +0.0259 | [+0.0123, +0.0402] |
| Training-only representation / head selector | within_device_leave_day | FISHER + original fixed Ridge | +0.0282 | [+0.0056, +0.0502] |
| Training-only representation / head selector | leave_device | MantisV2 + original fixed Ridge | +0.0183 | [+0.0022, +0.0348] |
| ExtraTrees on physical / relative features | within_device_leave_day | Spectral statistics + original fixed Ridge | +0.1050 | [+0.0834, +0.1314] |
| ExtraTrees on physical / relative features | leave_device | Spectral statistics + original fixed Ridge | +0.0257 | [+0.0146, +0.0376] |
| MiniRocket multivariate, sample-normalized envelopes | within_device_leave_day | Spectral statistics + original fixed Ridge | -0.1651 | [-0.1875, -0.1424] |
| MiniRocket multivariate, sample-normalized envelopes | leave_device | Spectral statistics + original fixed Ridge | +0.0278 | [+0.0095, +0.0441] |
| FISHER + balanced logistic head | within_device_leave_day | FISHER + original fixed Ridge | -0.0054 | [-0.0227, +0.0109] |
| FISHER + balanced logistic head | leave_device | FISHER + original fixed Ridge | +0.0097 | [-0.0102, +0.0282] |
| ExtraTrees versus original FISHER | within_device_leave_day | FISHER + original fixed Ridge | +0.0670 | [+0.0470, +0.0892] |
| ExtraTrees versus original FISHER | leave_device | FISHER + original fixed Ridge | -0.0038 | [-0.0233, +0.0154] |
| Automatic selector versus original random42 | within_device_leave_day | Mantis random42 + original fixed Ridge | +0.2490 | [+0.2220, +0.2769] |
| Automatic selector versus original random42 | leave_device | Mantis random42 + original fixed Ridge | +0.0119 | [-0.0060, +0.0302] |

### Pretraining controls

- MantisV2 + nested head, within_device_leave_day, versus matched random seed42: +0.0479, conditional paired 95% [+0.0294, +0.0632].
- MantisV2 + nested head, leave_device, versus matched random seed42: -0.0166, conditional paired 95% [-0.0319, -0.0001].
- MantisV2 + balanced logistic head, within_device_leave_day, versus matched random seed42: +0.0339, conditional paired 95% [+0.0097, +0.0574].
- MantisV2 + balanced logistic head, leave_device, versus matched random seed42: +0.0124, conditional paired 95% [-0.0021, +0.0272].

This round has one matched random seed for the new heads, while the previous fixed-head benchmark retains its three random seeds. A classifier improvement must not be described as newly established magnetic pretraining knowledge.

### Device-specific results

| Method | Test 3302 | Test 3303 | Test 3304 |
| --- | ---: | ---: | ---: |
| FISHER + original fixed Ridge | 0.3588 | 0.2309 | 0.3501 |
| MantisV2 + original fixed Ridge | 0.3220 | 0.3173 | 0.3795 |
| Mantis random42 + original fixed Ridge | 0.3382 | 0.3120 | 0.3878 |
| TSPulse + original fixed Ridge | 0.2581 | 0.2769 | 0.2789 |
| Spectral statistics + original fixed Ridge | 0.3245 | 0.1314 | 0.3955 |
| FISHER + nested head | 0.3588 | 0.2779 | 0.3501 |
| MantisV2 + nested head | 0.3192 | 0.2057 | 0.4369 |
| TSPulse + nested head | 0.2518 | 0.3045 | 0.3355 |
| Spectral statistics + nested head | 0.3208 | 0.1314 | 0.4136 |
| Relative spectrum / signed differences + nested head | 0.3455 | 0.1368 | 0.4169 |
| FISHER + physical features + nested head | 0.3361 | 0.1319 | 0.4021 |
| Mantis random42 + nested head | 0.3393 | 0.2677 | 0.4047 |
| Training-only representation / head selector | 0.3588 | 0.2779 | 0.4369 |
| ExtraTrees on physical / relative features | 0.3366 | 0.1319 | 0.4600 |
| MiniRocket multivariate, raw band envelopes | 0.3330 | 0.2367 | 0.3573 |
| MiniRocket multivariate, sample-normalized envelopes | 0.3068 | 0.2923 | 0.3356 |
| MantisV2 + default logistic head | 0.3387 | 0.2212 | 0.4146 |
| MantisV2 + balanced logistic head | 0.3448 | 0.2215 | 0.4114 |
| FISHER + balanced logistic head | 0.4198 | 0.1385 | 0.4107 |
| TSPulse + balanced logistic head | 0.2558 | 0.2901 | 0.2905 |
| Mantis random42 + balanced logistic head | 0.3207 | 0.2114 | 0.4083 |

### What the experiment can establish

Head choice, nonlinear decision boundaries and temporal-envelope alternatives are now measured on matched outer samples. A model-family row is a reported alternative, not an unbiased claim that picking the highest outer score will generalize. The automatic selector is the explicitly training-selected strategy. Because the previous outer test results were already inspected, the entire round remains exploratory; improvements must be confirmed on newly collected dates and preferably additional hardware.

The benchmark still uses inherited 0–100 Hz, 20-second spectral caches. It excludes transition boundaries and does not certify raw clocks, log alignment, absolute acquisition runs, fault labels or measured RPM. No additional normalization solves those provenance limitations. Target-device labeled calibration is a separate setting, not a substitute for strict held-device generalization.

## 3. Other potentially better methods

- The measured MiniRocket and ExtraTrees results above are strong non-foundation controls. They test whether a more suitable supervised classifier is more valuable than simply enlarging a frozen foundation encoder. MiniRocket is a random-convolution method, not a pretrained model. [Author source](https://github.com/angus924/minirocket/tree/0b1c245d9c9dbc50886f28bc7b32d5d45b5663d6).
- MultiRocket/HYDRA and a small InceptionTime/TCN are appropriate next task-specific comparisons. [Official aeon convolutional classifiers](https://www.aeon-toolkit.org/en/latest/examples/classification/convolution_based.html) provide reference implementations. None was run in this round; no centrifuge superiority is claimed.
- The MantisV2 paper reports complementary time/vision representations and improved token/layer aggregation. Such adapters or a smaller partially fine-tuned student remain candidates; they need training-device-only validation and random-initialization controls.
- [TS2Vec-FM](https://arxiv.org/abs/2610.28602) is a recent compact convolutional candidate with both amplitude and normalized-shape views. Its UCR/UEA evidence is not a magnetometer result; an official downloadable checkpoint was not verified in this round. The older per-dataset [TS2Vec implementation](https://github.com/zhihanyue/ts2vec) can instead be trained on source devices only, with the held device fully excluded.

The next data-side priority remains verified acquisition/run IDs, synchronization, wider frequency information if supported by the hardware, and new operating conditions/dates. These are prospective steps rather than measured gains from this run.

## Verification and artifacts

- Independently verified **101,664** new prediction rows, 141 stage/fold audits, score argmax, inherited labels, strict device/date exclusion, inner/outer exclusion, and every training-score selection.
- All retained historical baseline metrics were independently reproduced from their saved predictions.
- Fixed logistic fits with convergence warnings: 0. Their original warnings and iteration counts remain in the compressed audit.
- [All magnetic metrics and class recalls](summary.json), [table](metrics_table.csv), [new predictions](predictions.csv.gz), [split and selection audit](selection_and_split_audit.json.gz), [verification](verification.json), [sound metric audit](asd_metric_audit.json).
- [Commands and complete protocol](../../experiments/baseline_improvements_20261029/README.md). Large arrays remain in the remote archive; no raw instrument data or model weights are committed.
