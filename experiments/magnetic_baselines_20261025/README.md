# Frozen near-field magnetic baseline transfer

This experiment evaluates published encoders on real centrifuge magnetometer
data, not on radio I/Q or Wi-Fi CSI. It is a pilot transfer evaluation, not a
reproduction of a published magnetometer SOTA score.

## Models

- **MantisV2**: official `paris-noah/MantisV2`, revision
  `8f6ca35cb54ab14b120618943c6fca5ddf5a76a6`. Use intermediate transformer
  layer 2 and combined output, as recommended by the authors for frozen features.
- **TSPulse**: official classification-specialized checkpoint, revision
  `66cfb5464ed5e5a5cd7f5df1339af4e51461d955` of
  `ibm-granite/granite-timeseries-tspulse-r1`. Frozen backbone, masking disabled.
  This particular variant has 467,471 parameters; do not substitute the size of
  the different anomaly-detection variant.
- **FISHER-small**: previously verified checkpoint SHA
  `9924cdeec51ea796557af0c1920f0a2148303b3c26d2b41a39dbb8c4ff94fb4a`.
  Explicit magnetic STFT adapter: 256Hz input, 256-point FFT, 16-point hop,
  Hann window, no centering, log magnitude, official fixed normalization,
  zero padding to a multiple of the checkpoint's frequency-band width.
  The 1-second FFT differs from the published 25ms audio inference recipe.
- **Controls**: physical PSD plus amplitude statistics; random-initialized,
  frozen MantisV2 and TSPulse, seed 42, with the same architecture and readout.

## Execution

Run bulk work on A800. Existing source data remain read only. The recorded run
used Python 3.12, torch 2.9.1+cu128, transformers 4.57.1,
`mantis-tsfm==1.1.0`, and `granite-tsfm==0.3.4`. Extra packages were installed
in a separate `vendor/` directory using `uv pip install --target vendor --no-deps`.
The complete observed numerical package versions are in the results summary.
Some versions exceed granite-tsfm's declared package bounds; strict checkpoint
loading, finite features and deterministic inference were checked, but this is
not the authors' exact software environment.

From an isolated remote job root, place these scripts under `code/`:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$PWD/vendor"
python code/prepare_windows.py --out results/data_v2
python code/run_baselines.py
python code/verify_and_package.py
```

`prepare_windows.py` documents the three historical data roots. Update those
roots only when migrating to a verified source. It inventories bounded file
headers/tails, binary-searches timestamps and reads selected segments; it does
not load all 100+ GB of CSVs. The timestamp-search assumption is strictly
checked within every selected interval. A globally corrupt source may still
be missed by this bounded audit, so a full acquisition audit remains necessary.

The local magnetic encoder input is `[N, 1, 512]`: 2 seconds at 256Hz,
low-pass filtered at 100Hz, retaining magnetic amplitude. Packet-repeated
timestamps must not be averaged as if they were repeated measurements of the
same instant. CSV row order is retained; a 4/5kHz sample clock is inferred and
checked against the local timestamp span. The exact hardware sampling clock
and absolute sensor calibration remain unverified.

## Evaluation

All models use the same accepted windows. A training-fold `StandardScaler`
and `RidgeClassifier(alpha=10, class_weight="balanced")` provide the readout.
There is no test-set search or backbone fine-tuning. Leave-day-out trains on
other dates of the same device. Leave-device-out trains on the other accepted
device. Report macro-F1 over all four predefined states, averaged equally over
folds. Date bootstrap intervals are descriptive, conditional on two devices;
overlapping training folds and limited hardware diversity prevent population
claims.

The raw waveforms and feature arrays are retained on the experiment server;
their hashes are recorded. Published artifacts include selected-window and
exclusion manifests, model hashes, fold metrics, predictions, logs and checks.

See [the English report](../../results/magnetic_baselines_20261025/REPORT.md).
