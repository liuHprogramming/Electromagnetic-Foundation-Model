# Three-device near-field magnetic baseline rebenchmark — 28 October 2026

**Completed:** a fixed benchmark of **3,177 pure-state 20-second
magnetic-feature windows from 44 device-days**, three held-device
folds, 44 within-device held-date folds, random-encoder controls, and a paired
one-date target calibration experiment. All 134,882 saved
predictions and source pointers passed independent verification.

**Scope:** these are archived LP100 magnetic mel features, not newly certified
raw waveforms. Numerical quality, copied-file integrity and supervised task
labels do not independently establish correct sample clocks or event-log
synchronization. In particular, including 3303 here does **not** mean its raw
timestamp issue was repaired. No physical fault-detection or measured-RPM
regression score is claimed.

## Why this benchmark

The research plan concerns externally sensed, low-frequency near-field magnetic
signals from real instruments. A task on the three actual centrifuges is more
directly relevant than an RF I/Q, wireless-channel or radar leaderboard. We use
four states (off, speed-up, maximum-speed, speed-down) because corresponding
archived state labels exist. This remains a private, exploratory transfer
benchmark rather than an official public magnetometer SOTA protocol.

The official [Mantis implementation](https://github.com/vfeofanov/mantis) supports
frozen feature extraction. [TSPulse](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1)
has a classification-specific variant and a separately trained TSLens module;
our result is frozen backbone extraction, without TSLens. The
[FISHER checkpoint](https://huggingface.co/jiangab/FISHER-small-0723) represents
industrial signals through STFT sub-bands; our magnetic-mel adapter is explicitly
different from its published audio STFT recipe. None is asserted to be a
magnetometer-specific pretrained foundation model.

## Data, sample budget and exclusions

| Device | Archive windows | Numeric-valid | Selected | Dates | off/up/max/down | Date range |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 3302 | 43,775 | 43,763 | 1,490 | 17 | 408/296/403/383 | 20260518–20260629 |
| 3303 | 5,525 | 5,525 | 603 | 9 | 216/84/216/87 | 20260617–20260710 |
| 3304 | 10,477 | 10,468 | 1,084 | 18 | 432/119/412/121 | 20260515–20260622 |

The source has 59,777 windows, of which 59,756 pass the numerical checks.
Retain only windows with a single state across all 1,723 inherited label-grid
positions; this avoids assuming frame-by-frame alignment to 79 magnetic
spectrogram frames. It also deliberately excludes transition boundaries.
Greedy 20-second spacing inside inferred cache blocks reduces overlap; a fixed
cap of 24 windows per device/day/class prevents long steady-state recordings
from overwhelming the sample. Seed: 20261028. Four selected duplicate inputs
were removed; the final benchmark has unique exact input hashes.

Cache-block boundaries use relative-start resets, date changes and the known
running/off append boundary. They are not verified physical run IDs. The
number of windows is not the number of independent runs. Because source
absolute offsets were not cached, a claim of globally verified non-overlap is
not supported. Device/date isolation and exact-duplicate checks protect the
reported splits from the known window/version leakage paths.

The budget is limited by available **independent dates**, especially 3303's
nine dates, rather than raw TB or cache-window count. These class-capped
samples support an affordable comparison; they do not guarantee power to
detect a small model improvement. For a future confirmatory benchmark,
collect at least ten verified dates and thirty complete independent runs per
device, record absolute run IDs and acquisition-clock metadata, then reassess
the budget from between-run variation. If claiming generalization over
instrument hardware, more than these three devices are needed.

## Input adapters and fixed readout

The common input is **[48 frequency bins, 79 temporal frames]** from a 20-second,
0–100 Hz absolute-power-dB magnetic mel representation. MantisV2 and TSPulse
receive eight fixed contiguous frequency-group envelope trajectories. Linear
interpolation from 79 to 512 temporal positions adds no new physical bandwidth;
these inputs are spectral envelopes, never a reconstructed waveform.

Mantis concatenates eight 512-dimensional channel representations (4,096D).
TSPulse concatenates fixed ordered token-block summaries (3,072D). FISHER
converts dB mel power to log magnitude and applies the published fixed input
scale, obtaining 384D under this explicit magnetic adapter. Spectral statistics
and signed temporal trends give 336D; flat ordered mel gives 3,792D. A fixed
Mantis-plus-trend concatenation is a predeclared ablation. Identical Mantis
random encoders use three seeds; TSPulse has one matched random control.

Every non-dummy head uses training-only StandardScaler and balanced Ridge,
alpha=10, LSQR tolerance=.001. No test-label parameter search, fine-tuning,
target normalization or domain alignment is used. IDs, dates, RPM and state
logs are excluded from model features. The majority comparator uses only
training labels.

## Primary and secondary results

Primary: train on two devices, test the third, average the three device macro-F1
values with equal device weight. Secondary: within each device leave one date
out, aggregate all out-of-fold predictions per device, then average the three
device macro-F1 values. Both use the fixed four-class label set.

| Method | Leave-device mean macro-F1 | Conditional 95% day interval | Within-device leave-day OOF mean macro-F1 |
| --- | ---: | --- | ---: |
| Spectral statistics + signed trends | 0.2838 | [0.2683, 0.2989] | 0.6952 |
| Flat ordered magnetic mel | 0.2860 | [0.2692, 0.3026] | 0.5759 |
| MantisV2 pretrained | 0.3396 | [0.3215, 0.3583] | 0.5377 |
| MantisV2 random, seed 42 | 0.3460 | [0.3276, 0.3664] | 0.5124 |
| MantisV2 random, seed 43 | 0.3346 | [0.3168, 0.3537] | 0.5131 |
| MantisV2 random, seed 44 | 0.3317 | [0.3145, 0.3505] | 0.5114 |
| TSPulse pretrained | 0.2713 | [0.2545, 0.2889] | 0.4170 |
| TSPulse random, seed 42 | 0.2847 | [0.2684, 0.2996] | 0.3530 |
| FISHER-small magnetic-mel adapter | 0.3133 | [0.2906, 0.3357] | 0.7332 |
| MantisV2 + spectral trends | 0.3025 | [0.2867, 0.3182] | 0.5893 |
| Training-majority classifier | 0.1273 | [0.1230, 0.1317] | 0.1273 |

| Method | Test 3302 | Test 3303 | Test 3304 |
| --- | ---: | ---: | ---: |
| Spectral statistics + signed trends | 0.3245 | 0.1314 | 0.3955 |
| Flat ordered magnetic mel | 0.3177 | 0.2091 | 0.3311 |
| MantisV2 pretrained | 0.3220 | 0.3173 | 0.3795 |
| MantisV2 random, seed 42 | 0.3382 | 0.3120 | 0.3878 |
| MantisV2 random, seed 43 | 0.3334 | 0.2864 | 0.3841 |
| MantisV2 random, seed 44 | 0.3396 | 0.2858 | 0.3696 |
| TSPulse pretrained | 0.2581 | 0.2769 | 0.2789 |
| TSPulse random, seed 42 | 0.2773 | 0.2833 | 0.2934 |
| FISHER-small magnetic-mel adapter | 0.3588 | 0.2309 | 0.3501 |
| MantisV2 + spectral trends | 0.3240 | 0.2002 | 0.3834 |
| Training-majority classifier | 0.1075 | 0.1319 | 0.1425 |

The highest primary score among the non-random candidate configurations is
**MantisV2 pretrained: 0.3396**. This is a
descriptive ranking on the exploratory benchmark; the candidate with the
highest held-test score has not been validated on a new confirmatory test set.
It does not prove that the best general-purpose pretraining objective has been
identified.

Mantis pretraining versus the three-seed random mean (0.3374) changes
the primary score by **+0.0022**, with conditional 95% interval
[-0.0149, 0.0188]. A reliable positive pretraining benefit is not established in this cached-feature task.

- MantisV2 pretrained minus spectral trends: **+0.0558**, conditional 95% interval [0.0362, 0.0753].
- TSPulse pretrained minus spectral trends: **-0.0125**, conditional 95% interval [-0.0306, 0.0065].
- FISHER-small magnetic-mel adapter minus spectral trends: **+0.0295**, conditional 95% interval [0.0117, 0.0465].
- MantisV2 + spectral trends minus spectral trends: **+0.0188**, conditional 95% interval [0.0056, 0.0328].

Intervals use 2,000 paired bootstrap draws of dates within each of the three
devices, conditional on fixed trained folds. They do not resample/retrain the
entire learning procedure, do not turn neighboring windows into independent
trials, and do not estimate a population interval over unseen hardware.
All raw fold metrics, pooled metrics, class recalls and confusion matrices
remain in [summary.json](summary.json); the CSV table is
[metrics_table.csv](metrics_table.csv).

## Target calibration: one labeled date

Choose the earliest target-device date with all four states; reserve it
entirely for calibration. Train source-only and source-plus-calibration heads,
then evaluate both on the exact same remaining target dates.

| Target | Calibration date | Labeled windows | off/up/max/down |
| --- | --- | ---: | --- |
| 3302 | 20260518 | 85 | 24/14/24/23 |
| 3303 | 20260617 | 60 | 24/6/24/6 |
| 3304 | 20260515 | 78 | 24/21/24/9 |

| Method | Zero target dates, paired subset | One target date | Difference | Conditional paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| Spectral statistics + signed trends | 0.2796 | 0.3699 | +0.0903 | [0.0741, 0.1052] |
| Flat ordered magnetic mel | 0.2835 | 0.3365 | +0.0529 | [0.0375, 0.0675] |
| MantisV2 pretrained | 0.3331 | 0.3709 | +0.0378 | [0.0229, 0.0522] |
| MantisV2 random, seed 42 | 0.3413 | 0.3817 | +0.0404 | [0.0239, 0.0561] |
| MantisV2 random, seed 43 | 0.3311 | 0.3812 | +0.0501 | [0.0383, 0.0618] |
| MantisV2 random, seed 44 | 0.3251 | 0.3717 | +0.0466 | [0.0334, 0.0612] |
| TSPulse pretrained | 0.2682 | 0.2786 | +0.0104 | [-0.0010, 0.0212] |
| TSPulse random, seed 42 | 0.2869 | 0.2943 | +0.0074 | [-0.0022, 0.0167] |
| FISHER-small magnetic-mel adapter | 0.3080 | 0.4637 | +0.1557 | [0.1322, 0.1771] |
| MantisV2 + spectral trends | 0.2967 | 0.3740 | +0.0773 | [0.0575, 0.0942] |
| Training-majority classifier | 0.1274 | 0.1274 | +0.0000 | [0.0000, 0.0000] |

One calibration date contains multiple labeled windows and may contain several
runs. It is not one example, one run, or an unsupervised adaptation result.
Any observed gain is measured on this explicit protocol, not evidence that
collecting a fixed number of seconds will guarantee the same gain elsewhere.

## Error evidence and interpretation

Mantis's held-device recalls and largest off-diagonal counts are:

- Test 3302: recalls off/up/max/down = 0.703/0.074/0.769/0.018; largest errors: speed_down → speed_max: 271, speed_up → speed_max: 223.
- Test 3303: recalls off/up/max/down = 0.792/0.000/0.579/0.057; largest errors: speed_max → off: 74, speed_up → speed_max: 58.
- Test 3304: recalls off/up/max/down = 0.525/0.723/0.391/0.050; largest errors: speed_max → speed_up: 115, speed_max → speed_down: 95.

The gap between within-device dates and unseen devices measures generalization
under this data/label construction. Instrument differences, sensor placement,
operating-point coverage and inherited timing/label quality can all contribute;
this experiment does not independently separate those causes. The calibration
comparison tests whether adding actual target-domain labels helps. The random
controls test how much credit should go to pretraining rather than architecture
and the supervised readout. The spectral-trend and flat-mel controls test
whether a foundation model adds value beyond accessible cached structure.

Keep the strongest simple signal-processing control alongside the pretrained
models. Before claiming a general magnetic foundation model, independently
audit raw timing and log synchronization, restore absolute run provenance,
extend to additional devices, and evaluate a prespecified waveform/full-band
and task-adaptation protocol on newly reserved dates. These remain future
steps, not completed results.

## Reproduction, verification and retained artifacts

- [Exact sample index](samples.csv), [data audit](data_audit.json),
  [source-object manifest](manifest.json), [duplicate exclusions](removed_duplicates.json).
- [Every prediction](predictions.csv.gz), [split audit](split_audit.json),
  [independent verification](verification.json), [model receipt](model_receipt.json).
- [Model/source file hashes](source_file_hashes.json) and code/protocol hashes
  are retained. Immutable data objects and all large embeddings stay remote.
- [Commands and adapter details](../../experiments/magnetic_rebenchmark_20261028/README.md).

CPU preparation/statistics ran on scheduler-selected Zhiyuan J with four
numerical threads. CUDA inference ran serially on A800 under the existing
24 GiB sharing cap. Seven backbone feature passes took
11.76 seconds; peak tensor allocation was
0.283 GiB
(framework tensor accounting, not total driver memory). The GPU process exited.
Readout evaluation/statistics took 542.5 seconds.

The complete private data remain at
`/data_s3200a/Claude/electromagnetic-foundation-model/datasets/centrifuge_magnetic_20261026/`.
The new run uses A800 `magnetic_rebenchmark_20261028/` in the same project root
and Zhiyuan `/cache/electromagnetic-foundation-model_4e6aa0415dec/magnetic_rebenchmark_20261028/`.

This result is not directly comparable to the 25 October two-second raw-waveform
pilot: window duration, input representation, sample coverage, exclusions,
class caps and number of held devices all changed. It must not be presented as
a controlled improvement over that pilot.

![Benchmark, held-device variation and target calibration](benchmark_summary.png)
