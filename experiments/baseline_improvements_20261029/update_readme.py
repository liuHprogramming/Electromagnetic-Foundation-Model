"""Add verified new results without rewriting historical artifact bundles."""
import argparse
import json
from pathlib import Path
from write_report import NAMES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    a = p.parse_args()
    root = a.repo/'results/baseline_improvements_20261029'
    s = json.loads((root/'summary.json').read_text())
    aud = json.loads((root/'asd_metric_audit.json').read_text())
    sound = {r['model']:r['official_score'] for r in aud['aggregate'] if r['variant'] == 'beam_rdp4'}
    ci = aud['paired_bootstrap']['difference_95']
    readme = a.repo/'README.md'
    text = readme.read_text(encoding='utf-8')
    text = text.replace('28 October 2026 three-centrifuge rebenchmark.', '29 October 2026 nested-readout and temporal-method comparison.')
    marker = '## Sound audit and magnetic improvements — 29 October 2026'
    if marker in text:
        start = text.index(marker)
        end = text.index('## Three-device magnetic rebenchmark', start)
        text = text[:start]+text[end:]
    lines = [marker, '',
             'On the **same 3,177 windows / 44 device-days**, we tested training-only nested '
             'Ridge/RBF-SVM selection, fixed logistic probes, ExtraTrees and the official '
             'multivariate MiniRocket reference. Strict held-device inference fits no target '
             'samples. The prior test results were already known: these remain exploratory '
             'comparisons and need confirmation on new acquisition dates.', '',
             '| Method | Within-device held-date OOF macro-F1 | Held-device macro-F1 |',
             '| --- | ---: | ---: |']
    for key in ['fisher_pretrained_fixed', 'fisher_nested', 'fisher_logistic_balanced',
                'mantis_pretrained_fixed', 'mantis_nested', 'tspulse_nested',
                'extra_trees', 'minirocket_normalized', 'automatic_nested']:
        item = s['methods'][key]
        lines.append(f'| {NAMES[key]} | {item["within_device_leave_day"]["device_mean_macro_f1"]:.4f} | {item["leave_device"]["device_mean_macro_f1"]:.4f} |')
    lines += ['',
              'The automatic selector chooses both representation and head using only inner '
              'validation. All 16 new configurations, fixed baselines, matched random controls, '
              'per-device errors and paired date intervals are retained in the report. '
              'MiniRocket/ExtraTrees are supervised task baselines, not pretrained foundation models. '
              'The inherited LP100-clock/log/run limitations remain.', '',
              '**Sound metric correction:** official DCASE domain AUROCs compare normals of the '
              'indicated domain with all anomalies. Recomputing unchanged scores gives BEATs '
              f'**{100*sound["BEATs_iter3"]:.2f}%** and NA-BEATs **{100*sound["dcase2026_nabeats_seed0"]:.2f}%**. '
              'The old 60.80%/62.60% values use within-domain-only AUC and are superseded; '
              'the original outputs are kept for audit. ESC-50 and QA metrics are unaffected. '
              'The complete NA-BEATs competition system remains distinct from our frozen frontend.', '',
              '- [Complete report, SOTA evidence and matched changes](results/baseline_improvements_20261029/REPORT.md)',
              '- [New predictions and verified metrics](results/baseline_improvements_20261029/summary.json)',
              '- [Official-domain sound metric audit](results/baseline_improvements_20261029/asd_metric_audit.json)',
              '- [Reproduction protocol and commands](experiments/baseline_improvements_20261029/README.md)', '', '']
    at = text.index('## Three-device magnetic rebenchmark')
    text = text[:at]+'\n'.join(lines)+text[at:]
    text = text.replace('pAUC **60.80%**', f'pAUC **{100*sound["BEATs_iter3"]:.2f}%** (corrected 29 October)')
    text = text.replace('pAUC **62.60%**', f'pAUC **{100*sound["dcase2026_nabeats_seed0"]:.2f}%** (corrected 29 October)')
    old = 'For ASD, the observed NA-BEATs gain of1.80 points has a paired95% interval of\n[-0.12,+3.66] points, so a stable advantage is not resolved in this run.'
    delta = sound['dcase2026_nabeats_seed0']-sound['BEATs_iter3']
    new = (f'For ASD, the corrected official-definition NA-BEATs difference is {100*delta:+.2f} points, '
           f'with conditional paired clip-bootstrap 95% interval [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] points. '
           'This supersedes the historical within-domain-only statistic; see the 29 October audit. '
           'It does not establish device-population generalization.')
    assert old in text or new in text
    text = text.replace(old, new)
    text = text.replace('![Task-specific evaluation summary]', 'Archived figure (ASD numbers use the superseded statistic):\n\n![Task-specific evaluation summary]') if 'Archived figure (ASD' not in text else text
    text = text.replace('[English evaluation report](results/audio_sota_20261025/REPORT.md)', '[Archived 25 October report (ASD superseded)](results/audio_sota_20261025/REPORT.md)')
    text = text.replace('[Verified metrics](results/audio_sota_20261025/summary.json)', '[Archived metrics (legacy ASD domain AUROC)](results/audio_sota_20261025/summary.json)')
    readme.write_bytes(text.replace('\r\n','\n').encode('utf-8'))
    erratum = a.repo/'results/audio_sota_20261025/ERRATUM_20261029.md'
    erratum.write_text('# ASD domain-AUROC correction — 29 October 2026\n\n'
                       'The archived REPORT.md, summary.json, metrics.json, figures and associated hash '
                       'manifest are retained byte-for-byte. Their ASD domain AUROCs use only examples '
                       'within each domain, whereas the official definition includes all anomalies. '
                       'The authoritative recomputation is in '
                       '[the new report](../baseline_improvements_20261029/REPORT.md) and '
                       '[metric audit](../baseline_improvements_20261029/asd_metric_audit.json). '
                       'Model scores, ESC-50 and QA results are unchanged. The current scorer uses '
                       'the correct masks; the analyzer explicitly distinguishes historical and new outputs.\n', encoding='utf-8')
    print('README_UPDATED')


if __name__ == '__main__':
    main()
