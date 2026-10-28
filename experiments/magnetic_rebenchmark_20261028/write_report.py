"""Lightweight report and exportable figures from independently verified metrics."""
import argparse
import collections
import csv
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    'spectral_trends': 'Spectral statistics + signed trends',
    'flat_mel': 'Flat ordered magnetic mel',
    'mantis_pretrained': 'MantisV2 pretrained',
    'mantis_random42': 'MantisV2 random, seed 42',
    'mantis_random43': 'MantisV2 random, seed 43',
    'mantis_random44': 'MantisV2 random, seed 44',
    'tspulse_pretrained': 'TSPulse pretrained',
    'tspulse_random42': 'TSPulse random, seed 42',
    'fisher_pretrained': 'FISHER-small magnetic-mel adapter',
    'mantis_plus_spectral': 'MantisV2 + spectral trends',
    'majority': 'Training-majority classifier',
}
ORDER = list(LABELS)


def score(info, protocol):
    return info['protocols'][protocol]['device_mean_macro_f1']


def interval(item):
    lo, hi = item['conditional_day_bootstrap_95']
    return f'[{lo:.4f}, {hi:.4f}]'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out
    s = json.loads((out / 'summary.json').read_text())
    d = json.loads((out / 'data_audit.json').read_text())
    receipt = json.loads((out / 'model_receipt.json').read_text())
    ver = json.loads((out / 'verification.json').read_text())
    assert ver['state'] == 'passed'
    data = list(csv.DictReader((out / 'samples.csv').open(encoding='utf-8')))
    rows = []
    for dev, info in d['accounting'].items():
        cls = info['classes']
        ds = sorted({r['day'] for r in data if r['device'] == dev})
        rows.append(f'| {dev} | {info["archive_windows"]:,} | {info["numeric_valid"]:,} | {info["selected"]:,} | {info["selected_days"]} | '
                    f'{cls.get("off",0)}/{cls.get("speed_up",0)}/{cls.get("speed_max",0)}/{cls.get("speed_down",0)} | {ds[0]}–{ds[-1]} |')
    main_rows, per_dev_rows, calib_rows, full_rows = [], [], [], []
    for key in ORDER:
        m = s['methods'][key]
        leave, days = m['protocols']['leave_device'], m['protocols']['within_device_leave_day']
        main_rows.append(f'| {LABELS[key]} | {score(m,"leave_device"):.4f} | {interval(leave)} | {score(m,"within_device_leave_day"):.4f} |')
        per_dev_rows.append('| ' + LABELS[key] + ' | ' + ' | '.join(f'{leave["per_device"][dev]["macro_f1"]:.4f}' for dev in ['3302','3303','3304']) + ' |')
        gain = m['calibration_gain']
        calib_rows.append(f'| {LABELS[key]} | {score(m,"calibration_zero_day_paired"):.4f} | {score(m,"calibration_one_day"):.4f} | {gain["difference"]:+.4f} | {interval(gain)} |')
        for proto, v in m['protocols'].items():
            full_rows.append({'method': key, 'protocol': proto, 'device_mean_macro_f1': v['device_mean_macro_f1'],
                              'pooled_macro_f1': v['pooled']['macro_f1'], 'pooled_accuracy': v['pooled']['accuracy'],
                              'conditional_ci_lower': v['conditional_day_bootstrap_95'][0], 'conditional_ci_upper': v['conditional_day_bootstrap_95'][1]})
    with (out / 'metrics_table.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(full_rows[0]))
        w.writeheader()
        w.writerows(full_rows)
    cal = []
    for dev, day in s['calibration_days'].items():
        cnt = collections.Counter(r['state'] for r in data if r['device'] == dev and r['day'] == day)
        cal.append(f'| {dev} | {day} | {sum(cnt.values())} | ' + '/'.join(str(cnt.get(k, 0)) for k in ['off','speed_up','speed_max','speed_down']) + ' |')
    delta = s['paired_comparisons']['mantis_vs_random_mean/leave_device']
    random_mean = np.mean([score(s['methods'][f'mantis_random{k}'], 'leave_device') for k in [42,43,44]])
    pretrain_supported = delta['conditional_day_bootstrap_95'][0] > 0
    conclusion = ('The conditional day interval is above zero in this cached-feature task; the result still does not establish transfer to a population of unseen hardware.'
                  if pretrain_supported else 'A reliable positive pretraining benefit is not established in this cached-feature task.')
    trend_comparisons = []
    for key in ['mantis_pretrained','tspulse_pretrained','fisher_pretrained','mantis_plus_spectral']:
        v = s['paired_comparisons'][key + '_vs_spectral/leave_device']
        trend_comparisons.append(f'- {LABELS[key]} minus spectral trends: **{v["difference"]:+.4f}**, conditional 95% interval {interval(v)}.')
    confusions = []
    states = ['off','speed_up','speed_max','speed_down']
    for dev, info in s['methods']['mantis_pretrained']['protocols']['leave_device']['per_device'].items():
        matrix = np.array(info['confusion'])
        diag = matrix.diagonal()
        recalls = np.divide(diag, matrix.sum(1), out=np.zeros(4), where=matrix.sum(1) > 0)
        offdiag = [(int(matrix[i,j]), states[i], states[j]) for i in range(4) for j in range(4) if i != j]
        offdiag.sort(reverse=True)
        confusions.append(f'- Test {dev}: recalls off/up/max/down = ' + '/'.join(f'{v:.3f}' for v in recalls) + '; largest errors: ' + ', '.join(f'{a} → {b}: {n}' for n,a,b in offdiag[:2]) + '.')
    methods = s['methods']
    best = max(['spectral_trends','flat_mel','mantis_pretrained','tspulse_pretrained','fisher_pretrained','mantis_plus_spectral'], key=lambda k: score(methods[k], 'leave_device'))
    report = f'''# Three-device near-field magnetic baseline rebenchmark — 28 October 2026

**Completed:** a fixed benchmark of **{d['selected_windows']:,} pure-state 20-second
magnetic-feature windows from {d['device_days']} device-days**, three held-device
folds, 44 within-device held-date folds, random-encoder controls, and a paired
one-date target calibration experiment. All {ver['prediction_rows']:,} saved
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
{chr(10).join(rows)}

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
{chr(10).join(main_rows)}

| Method | Test 3302 | Test 3303 | Test 3304 |
| --- | ---: | ---: | ---: |
{chr(10).join(per_dev_rows)}

The highest primary score among the non-random candidate configurations is
**{LABELS[best]}: {score(methods[best], 'leave_device'):.4f}**. This is a
descriptive ranking on the exploratory benchmark; the candidate with the
highest held-test score has not been validated on a new confirmatory test set.
It does not prove that the best general-purpose pretraining objective has been
identified.

Mantis pretraining versus the three-seed random mean ({random_mean:.4f}) changes
the primary score by **{delta['difference']:+.4f}**, with conditional 95% interval
{interval(delta)}. {conclusion}

{chr(10).join(trend_comparisons)}

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
{chr(10).join(cal)}

| Method | Zero target dates, paired subset | One target date | Difference | Conditional paired 95% interval |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(calib_rows)}

One calibration date contains multiple labeled windows and may contain several
runs. It is not one example, one run, or an unsupervised adaptation result.
Any observed gain is measured on this explicit protocol, not evidence that
collecting a fixed number of seconds will guarantee the same gain elsewhere.

## Error evidence and interpretation

Mantis's held-device recalls and largest off-diagonal counts are:

{chr(10).join(confusions)}

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
{receipt['elapsed_seconds']:.2f} seconds; peak tensor allocation was
{max(m['peak_allocated_bytes'] for m in receipt['models'].values())/1024**3:.3f} GiB
(framework tensor accounting, not total driver memory). The GPU process exited.
Readout evaluation/statistics took {s['elapsed_seconds']:.1f} seconds.

The complete private data remain at
`/data_s3200a/Claude/electromagnetic-foundation-model/datasets/centrifuge_magnetic_20261026/`.
The new run uses A800 `magnetic_rebenchmark_20261028/` in the same project root
and Zhiyuan `/cache/electromagnetic-foundation-model_4e6aa0415dec/magnetic_rebenchmark_20261028/`.

This result is not directly comparable to the 25 October two-second raw-waveform
pilot: window duration, input representation, sample coverage, exclusions,
class caps and number of held devices all changed. It must not be presented as
a controlled improvement over that pilot.

![Benchmark, held-device variation and target calibration](benchmark_summary.png)
'''
    (out / 'REPORT.md').write_text(textwrap.dedent(report), encoding='utf-8')
    plot_keys = ['spectral_trends','mantis_pretrained','tspulse_pretrained','fisher_pretrained','mantis_plus_spectral']
    plot_names = ['Spectral trends','MantisV2','TSPulse','FISHER adapter','Mantis + trends']
    colors = ['#426A90','#287A71','#BD7144','#8D609E','#B49B3A']
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), gridspec_kw={'width_ratios': [1.25,1,1.25]})
    yy = np.arange(len(plot_keys))
    for i, key in enumerate(plot_keys):
        m = methods[key]
        v = score(m, 'leave_device')
        lo, hi = m['protocols']['leave_device']['conditional_day_bootstrap_95']
        axes[0].errorbar(v, i-.09, xerr=[[max(0,v-lo)], [max(0,hi-v)]], fmt='o', color=colors[i], capsize=3)
        axes[0].plot(score(m,'within_device_leave_day'),i+.12,marker='D',mfc='none',color=colors[i],ms=5)
    axes[0].set_yticks(yy, plot_names)
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0,1),xlabel='Device-balanced macro-F1',title='Held device (solid) / held date (open)')
    axes[0].grid(axis='x', alpha=.2)
    mat = np.array([[methods[k]['protocols']['leave_device']['per_device'][v]['macro_f1'] for v in ['3302','3303','3304']] for k in plot_keys])
    axes[1].imshow(mat, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    axes[1].set_xticks(range(3), ['3302','3303','3304'])
    axes[1].set_yticks(yy, plot_names)
    axes[1].set(xlabel='Held-out device',title='Per-device macro-F1')
    for i in range(5):
        for j in range(3):
            axes[1].text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',color='white' if mat[i,j]>.55 else '#182630')
    for i,k in enumerate(plot_keys):
        v0,v1=[score(methods[k],p) for p in ['calibration_zero_day_paired','calibration_one_day']]
        axes[2].plot([v0,v1],[i,i],color=colors[i],lw=2)
        axes[2].plot(v0,i,'o',mfc='white',color=colors[i])
        axes[2].plot(v1,i,'o',color=colors[i])
    axes[2].set_yticks(yy, plot_names)
    axes[2].invert_yaxis()
    axes[2].set(xlim=(0,1),xlabel='Device-balanced macro-F1',title='0 target dates (open) / 1 date (solid)')
    axes[2].grid(axis='x',alpha=.2)
    fig.suptitle('Three centrifuges · 3,177 cached LP100 windows · 44 device-days',y=1.02,fontsize=12)
    fig.text(.5,-.03,'Pure-state archived-feature task; timing/synchronization inherited. Date bootstrap conditions on these three devices.',ha='center',fontsize=9)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(out / 'benchmark_summary.png', dpi=180, bbox_inches='tight')
    fig.savefig(out / 'benchmark_summary.pdf', bbox_inches='tight')
    plt.close(fig)
    print('REPORT_READY', out / 'REPORT.md')


if __name__ == '__main__':
    main()
