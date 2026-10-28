# Three-device magnetic benchmark reproduction

This experiment evaluates transfer from released time-series/industrial models
to **archived near-field magnetic LP100 spectrograms** from centrifuges 3302,
3303 and 3304. It does not claim a pretrained magnetometer SOTA or independently
certified waveform timing. The archived spectra are not raw waveform inputs.

The fixed data protocol is in `protocol.json`; model revisions, hashes and
adapters are in `encoding_protocol.json`. Model configuration is fixed before
the held-device predictions are generated. Results and exclusions must be
reported even when random encoders or handcrafted features perform better.

## Data and sampling

- Canonical versions: 3302/3304 clock-corrected archives and the original 3303
  archive. Alternative/probe versions are excluded to avoid repeated samples.
- Source: 59,777 windows; 59,756 pass numerical quality checks. Retain only
  windows whose 1,723 inherited status labels are all the same class.
- Within an inferred monotone cache block, greedily retain starts at least
  20 seconds apart. Infer boundaries from device/day changes, relative-start
  resets and the known running/off append boundary. These blocks are **not
  verified acquisition runs**, because absolute source offsets were not cached.
- Deterministically cap each device/day/class at 24 windows (seed 20261028).
  Exclude all selected exact-input duplicates shared across dates/devices and
  label-conflicting duplicates; keep the first identical copy within one group.
- Final fixed sample: **3,177 windows, 44 device-days**. Class-balanced caps
  make this a diagnostic benchmark, not an estimate of natural deployment
  prevalence. Transition-boundary windows are outside this pure-state task.

## Models and controls

1. MantisV2: eight fixed, contiguous frequency-group envelope trajectories,
   temporally interpolated from 79 to 512 positions; independently encode each
   trajectory and concatenate. Transformer layer 2, combined tokens. Three
   matched frozen random initializations (42/43/44) test pretraining benefit.
2. TSPulse classification checkpoint: same eight envelope trajectories,
   disabled masking, fixed eight-block pooling of ordered backbone tokens.
   Random42 control. This is frozen extraction, **not TSLens fine-tuning**.
3. FISHER-small: explicit LP100 mel adaptation. Convert dB power to natural-log
   mel-band magnitude, apply the published fixed scale, and extract native
   sub-band features. Magnetic mel bins differ from its published audio STFT.
4. Spectral trends: per-bin mean/std/10th and 90th percentile/signed slope/
   late-minus-early level/mean absolute successive difference (336 features).
5. Flat ordered mel: all 3,792 values with the same linear head.
6. Predeclared Mantis + spectral-trend concatenation, and a training-majority
   classifier. No held-test comparison chooses hyperparameters.

All non-dummy methods use a training-fold `StandardScaler` and
`RidgeClassifier(alpha=10, class_weight='balanced', solver='lsqr', tol=.001)`.
No per-device scalar, timestamp, log RPM or device ID is a model feature.

## Evaluation

- Primary: three leave-one-device-out folds. Train on two devices, test the
  third; average the three four-class macro-F1 values with equal device weight.
- Secondary: 44 within-device leave-one-day-out folds. Aggregate out-of-fold
  predictions per device and average the three macro-F1 values. This differs
  from taking the arithmetic mean of all single-date fold scores, where classes
  absent on a date can dominate the score.
- Calibration: reserve the earliest target-device date with all four classes,
  train with or without that independent target date, and test both settings
  on exactly the same remaining target dates. One date can contain many
  labeled windows; this is **not one-shot** adaptation.
- Save per-device scores, four-class recall/confusion matrices, all predictions
  and split membership hashes. Conditional intervals use 2,000 paired day-block
  bootstrap draws with equal final device weights. They condition on these
  three devices and already fitted models; they do not estimate performance
  over a population of unseen instruments.

## Commands

Run data preparation/statistics on the CPU host chosen by the project scheduler;
run CUDA encoding as one process capped at 24 GiB on A800. The private immutable
data root must already exist; only indexes and compact results enter Git.

```bash
# CPU host, working directory containing code/
python code/prepare_benchmark.py \
  --source /cache/electromagnetic-foundation-model_4e6aa0415dec/magnetic_dataset_20261026 \
  --out data

# Transfer data/{spectrograms.npy,band_trajectories.npy,manifest.json,data_audit.json}
# to the GPU working directory and verify the recorded hashes.
PYTHONPATH=/data/Claude/electromagnetic-foundation-model/magnetic_20261025/vendor \
  /data/Claude/electromagnetic-foundation-model/sota_20261025/.venv/bin/python code/encode_features.py

# Transfer all features/*.npy and model_receipt.json back to the CPU host.
python code/evaluate.py
python code/verify_results.py
```

For the preserved physical dataset layout, data preparation also supports
`--layout dataset --source /path/to/centrifuge_magnetic_20261026`. The independent
verification entry point expects the mirrored immutable objects in the adjacent
`magnetic_dataset_20261026/objects/` directory, matching this actual CPU run.
Keep numerical libraries at four threads and use low process priority.

## Sources and scope

- [Mantis official implementation](https://github.com/vfeofanov/mantis)
- [MantisV2 weights](https://huggingface.co/paris-noah/MantisV2)
- [TSPulse model card and task-specific recipes](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1)
- [FISHER official checkpoint/inference](https://huggingface.co/jiangab/FISHER-small-0723)

The 20-second cached LP100 protocol differs from the earlier two-second,
strict-timestamp waveform pilot. A numerical change between the two runs is
not a controlled improvement estimate. Raw acquisition clocks, the 3303 sample
rate and log synchronization still need an independent audit. Physical fault
labels and trustworthy measured RPM are not available for a fault-detection or
RPM-regression claim in this evaluation.
