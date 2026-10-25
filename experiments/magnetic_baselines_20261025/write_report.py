from pathlib import Path
import json

base=Path(__file__).resolve().parents[2]
out=base/'results/magnetic_baselines_20261025'
r=out/'run/results'
s=json.loads((r/'summary.json').read_text())
q=json.loads((r/'quality_summary.json').read_text())
c=json.loads((r/'comparisons.json').read_text())
names={'physical_psd':'Physical PSD + amplitude','mantis':'MantisV2, pretrained',
       'mantis_random':'MantisV2, random frozen','tspulse':'TSPulse classification, pretrained',
       'tspulse_random':'TSPulse, random frozen','fisher_adapter':'FISHER-small + magnetic STFT adapter'}
rows=[]
for key,name in names.items():
    a=s['results'][key+'/leave_day_out'];b=s['results'][key+'/leave_device_out']
    lo,hi=a['descriptive_day_bootstrap_95ci']
    rows.append(f"| {name} | {a['mean_macro_f1']:.4f} | {lo:.4f}–{hi:.4f} | {b['mean_macro_f1']:.4f} |")
text='''# Near-field magnetic foundation-model baseline pilot

## 1. Scope and model choice

The research plan targets external, low-frequency magnetic recordings from
laboratory instruments. We selected MantisV2 and TSPulse as reusable time-series
encoders, and FISHER-small as an industrial-signal transfer baseline with an
explicit magnetic preprocessing adapter. These are published-weight transfer
experiments; none is established here as a pretrained centrifuge-magnetometer
foundation model or a magnetometer SOTA system.

MantisV2 is the preferred baseline to carry forward for state classification:
its 4.19M-parameter encoder supports a frozen readout, and the official checkpoint
and code are available. TSPulse is a compact comparator; the selected
classification-specialized checkpoint has 467,471 parameters. FISHER provides a
bridge to the earlier sound experiments, with a 1-second physical STFT adapter
instead of its published 25ms audio STFT. All three checkpoints loaded without
missing, unexpected or mismatched keys.

Sources: [Mantis official implementation](https://github.com/vfeofanov/mantis),
[MantisV2 weights](https://huggingface.co/paris-noah/MantisV2),
[TSPulse official variants](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1),
[FISHER official implementation](https://github.com/jianganbai/FISHER).

## 2. Benchmark and sample accounting

We used the project's real centrifuge magnetic CSVs and run-log-derived labels.
The sensor field is a **single channel**, `Magnetic_Field_uT`. Its header declares
microtesla, but the original absolute calibration was not independently checked.
The four classes are off, acceleration, full speed and deceleration. They are
operating states, not labeled physical faults.

Up to 12 available dates per device were selected evenly across the available
date range. At most 12 candidates per state/date were selected using a fixed
hash-based seed and log timestamps, with a class-boundary guard. Failed
candidates were not replaced. This is a deliberately stratified pilot, not an
estimate of natural operating-state prevalence.

- Preselected windows: **1,462** from 3302, 3303 and 3304.
- Accepted: **343**, from **2 devices and 14 device-days**.
- Device 3302: 110 windows over 5 dates; device 3304: 233 over 9 dates.
- Four-class counts: **92 / 84 / 95 / 72**.
- Sources: 62 distinct selected CSV slices, including 29 running slices.
  These counts do not certify 29 independent complete physical runs.
- Rejected candidates: **1,119**; additionally, 3 files failed inventory checks.
- All 426 candidates for 3303 failed the selected quality checks, so no
  three-device evaluation is claimed.

The main rejection causes were nonmonotonic timestamps (777 windows), packet
timestamp gaps/truncated context (259), insufficient/nonfinite samples (75),
and a sample-clock/reception-clock span disagreement (8). Retention is only
23.46%; selection bias from quality filtering is a substantial limitation.

The CSVs can repeat a reception timestamp for approximately 150 samples. Those
values are sequential samples, so averaging them would erase the waveform.
We preserve row order, choose the nearest 4/5kHz sample-clock interpretation
from local count/span, and require rate and span agreement. Accepted windows
used the 5kHz interpretation. Some 3303 CSV intervals encode about 5kHz while
the current listener source declares 4kHz; historical firmware/collector
configuration is unresolved. Encoded timestamps alone do not prove hardware
sampling frequency. Raw values were never relabeled as speech-rate audio.

Each accepted window is 2 seconds, with 0.5-second processing context on either
side. A fourth-order 100Hz zero-phase low-pass filter and timestamp-anchored
interpolation produce 512 values at 256Hz. All encoders see the same bandwidth
and amplitude-bearing waveform. Conclusions are limited to **0–100Hz**; this
does not test the full raw sensor band or 20-second temporal context.

## 3. Evaluation and results

Encoders are frozen. Every fold fits its own StandardScaler and the same
balanced RidgeClassifier with fixed alpha=10. No test-set tuning was performed.
Mantis uses intermediate layer 2 with combined output. TSPulse uses the full
flattened backbone embedding with masking disabled. Random-initialized frozen
encoders, seed 42, test whether a useful representation necessarily depends on
pretraining. The non-foundation baseline is a physical Welch spectrum plus
mean, standard deviation, RMS and robust amplitude range.

Leave-day-out has 14 folds, training on the other accepted dates of the same
device. Leave-device-out has 2 folds, training on the other accepted device.
All source slices and neighboring windows within a device-day remain on one
side of each split. Processed input hashes are unique and train/test hash
overlap is checked. Results are **fold-mean macro-F1**, not accuracy.

| Model | Leave-day-out macro-F1 | Descriptive 95% day interval | Leave-device-out macro-F1 |
| --- | ---: | ---: | ---: |
'''+ '\n'.join(rows)+'''

The intervals resample observed device-day scores. They are descriptive and
conditional on these two devices; training folds overlap, and 14 dates do not
represent 14 independent instruments. We do not give a population-level
cross-device confidence interval from only two devices.

## 4. Interpretation

MantisV2 exceeds the physical baseline by 0.0858 macro-F1 on the observed
leave-day-out folds. Its paired descriptive interval is 0.0226–0.1527.
However, pretrained versus random Mantis differs by only **0.0031**, with a
paired interval **−0.0456 to 0.0562**. This pilot does not establish a
pretraining benefit. The random control uses one initialization, so it is not
a complete multi-seed representation study.

TSPulse does not improve on physical features here; its pretrained and random
controls are also close. FISHER's adapted features underperform physical
features on leave-day-out and improve on them in this two-device transfer
comparison. No published FISHER magnetic score or complete author training
recipe is reproduced by this adapter experiment.

Plausible causes to test next include pretraining-domain mismatch, an input
band/context that excludes useful motor harmonics, loss of useful amplitude
through model-native normalization, the fixed linear readout, timestamp quality
and sparse state labels. These are hypotheses, not experimentally separated
causes. Changing a transform after observing these test scores would require
a fresh validation/test split.

**Recommendation:** retain MantisV2 as the main pretrained candidate, retain
physical PSD and random-encoder controls as mandatory comparators, and keep
TSPulse and FISHER as secondary baselines. Improve the acquisition audit before
claiming that any backbone supplies transferable magnetic knowledge.

## 5. Proposed formal benchmark and sample budget — not yet executed

The primary benchmark should remain the same instrument modality and task:

1. Recover and audit all three devices; verify actual sampling clocks, packet
   order, calibration units, clipping and log alignment by recording revision.
2. Aim for **at least 30 complete independent runs per device across at least
   10 dates**, totaling at least 90 runs over the three current devices.
3. Take up to 5 nonoverlapping 2-second windows per state/run, including a
   clearly associated off segment. This budgets up to **1,800 windows** when
   all four states are available. Record missing/short states instead of
   fabricating windows. The run/date/device counts determine independence;
   increasing overlapping windows does not increase the independent sample size.
4. Freeze an untouched final-date test partition and run leave-one-device-out
   separately. Fit preprocessing and choose any readout/adapter settings using
   training dates and validation dates only. Compare 2/5/10/20-second context
   and 0–100Hz/full verified bandwidth using validation data.
5. Primary metrics: four-state macro-F1, balanced accuracy and class recall.
   Add RPM MAE only after checking that logs contain measured speed, and event
   timing error only after independent alignment validation. Add AUROC/pAUC
   only with trustworthy normal/anomalous labels; low speed alone is not a fault.
6. Compare 1/5/10 independent labeled runs per class or a feasible run-level
   label budget, repeating predefined seeds. Include a small supervised
   CNN/TCN, magnetic-only self-supervision, ordinary audio-to-magnetic
   distillation, and the proposed physical alignment method. None of those
   training experiments was executed in this frozen pilot.

This is a collection budget, not a statistical-power guarantee. Use the
expanded pilot's run-level variability to plan detection of a predefined
meaningful effect. Three devices still provide weak evidence for general
hardware transfer; more held-out instruments would strengthen that claim.

For an optional public **magnetometer transfer sanity check**, use only the
magnetometer channels of [UCI PAMAP2](https://archive.ics.uci.edu/dataset/231):
100Hz, three body-worn IMUs and nine participants in the source dataset.
Audit protocol completeness, then fix eligible subjects/classes before
leave-subject-out evaluation. Report the actual eligible subject and window
counts. This checks transfer to real magnetic measurements, but its human
activity task and bandwidth do not validate centrifuge state or RPM estimation.
PAMAP2 inference was **not run** in this session.

## 6. Artifacts and reproduction limits

- [Code and command guide](../../experiments/magnetic_baselines_20261025/README.md)
- [Protocol](../../experiments/magnetic_baselines_20261025/protocol.json)
- [Fold metrics and software versions](run/results/summary.json)
- [All 4,116 predictions](run/results/predictions.jsonl)
- [Model revisions, hashes and loading checks](run/results/model_receipts.json)
- [Selected windows and all exclusions](run/results/data_v2/manifest.json)
- [Quality accounting](run/results/quality_summary.json)
- [Paired comparisons](run/results/comparisons.json)
- [Verification](run/results/verification.json)
- [Package SHA manifest](run/artifact_manifest.json)
- [Remote-only waveform/feature hashes](run/results/retained_artifacts.json)

All bulk work ran on the shared A800 with one evaluation process, a 24GiB
CUDA allocation limit, four numerical threads and low priority. Frozen feature
extraction plus readouts took about 68 seconds after preparation; this is not
a controlled throughput benchmark. Mantis peaked at approximately 122MiB
reserved CUDA memory in its recorded stage; the configured cap is an upper
limit, not its memory requirement. Source CSVs were read only. Raw waveforms,
model weights and feature arrays are retained remotely and are not published
in Git. All 15 packaged files were retrieved and SHA-verified locally.

The package contains a failed preliminary timestamp-averaging preparation log
for auditability. It produced no valid evaluation dataset and is excluded
from every metric. Package dependencies were isolated under vendor/; the
recorded stack exceeds some granite-tsfm declared dependency bounds. Successful
strict weight loading and inference do not establish full library compatibility.
'''
(out/'REPORT.md').write_text(text,encoding='utf-8',newline='\n')
print(out/'REPORT.md')
