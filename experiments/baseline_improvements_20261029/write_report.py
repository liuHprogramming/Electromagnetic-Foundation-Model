"""Render only independently verified outputs; keep historical runs intact."""
import argparse
import collections
import json
from pathlib import Path


NAMES = {
    'fisher_pretrained_fixed':'FISHER + original fixed Ridge',
    'mantis_pretrained_fixed':'MantisV2 + original fixed Ridge',
    'mantis_random42_fixed':'Mantis random42 + original fixed Ridge',
    'tspulse_pretrained_fixed':'TSPulse + original fixed Ridge',
    'spectral_trends_fixed':'Spectral statistics + original fixed Ridge',
    'fisher_nested':'FISHER + nested head', 'mantis_nested':'MantisV2 + nested head',
    'tspulse_nested':'TSPulse + nested head', 'physical_nested':'Spectral statistics + nested head',
    'relative_dynamics_nested':'Relative spectrum / signed differences + nested head',
    'fisher_physical_nested':'FISHER + physical features + nested head',
    'mantis_random42_nested':'Mantis random42 + nested head',
    'automatic_nested':'Training-only representation / head selector',
    'extra_trees':'ExtraTrees on physical / relative features',
    'minirocket':'MiniRocket multivariate, raw band envelopes',
    'minirocket_normalized':'MiniRocket multivariate, sample-normalized envelopes',
    'mantis_logistic_default':'MantisV2 + default logistic head',
    'mantis_logistic_balanced':'MantisV2 + balanced logistic head',
    'fisher_logistic_balanced':'FISHER + balanced logistic head',
    'tspulse_logistic_balanced':'TSPulse + balanced logistic head',
    'mantis_random42_logistic_balanced':'Mantis random42 + balanced logistic head'
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    a = p.parse_args()
    root = a.results
    s = json.loads((root/'summary.json').read_text())
    audit = json.loads((root/'asd_metric_audit.json').read_text())
    verify = json.loads((root/'verification.json').read_text())
    assert verify['passed'] and audit['verification']['official_auc_matches_pairwise_probability']
    sound = {r['model']:r for r in audit['aggregate'] if r['variant'] == 'beam_rdp4'}
    b, n = [sound[k]['official_score'] for k in ['BEATs_iter3', 'dcase2026_nabeats_seed0']]
    ci = audit['paired_bootstrap']['difference_95']
    lines = ['# Sound metric audit and magnetic improvements — 29 October 2026', '',
             'The magnetic benchmark, samples and pretrained embeddings are unchanged. '
             'This run tests classifiers and temporal alternatives using training-only validation. '
             'A separate audit corrects the definition of domain AUROC in the earlier sound report. '
             'No new sound inference, GPU job, foundation-model pretraining or hardware clock repair was performed.', '',
             '## 1. Machine anomalous sound detection', '',
             '**Correction:** the 25 October code restricted each domain AUC to samples of that domain. '
             'The official evaluator instead compares the normal samples of the indicated domain with '
             '**all anomalies**, from both domains. The current scorer is fixed; old scores and artifacts '
             'are retained as historical outputs, and this report supersedes their official-score interpretation. '
             'ESC-50 and audio-QA results are unaffected.', '',
             '[Official evaluator, pinned commit](https://github.com/nttcslab/dcase2026_task2_evaluator/blob/f6a94a2b5e614a9626c9d1ccff6df0705e6aaa75/dcase2026_task2_evaluator.py#L277) '
             'is the source of the mask. All 56 model/backend/machine groups were checked; '
             'the recomputation reproduces the old metrics and independently matches exact pairwise AUROC probabilities.', '',
             '| Frozen frontend + frequency banks / RDP4 | Historical statistic | Correct official-definition dev score |',
             '| --- | ---: | ---: |',
             f'| BEATs | {100*sound["BEATs_iter3"]["legacy_score"]:.2f}% | **{100*b:.2f}%** |',
             f'| NA-BEATs seed0 | {100*sound["dcase2026_nabeats_seed0"]["legacy_score"]:.2f}% | **{100*n:.2f}%** |', '',
             f'NA-BEATs minus BEATs is **{100*(n-b):+.2f} percentage points**; '
             f'paired 2,000-draw stratified clip-bootstrap 95% interval '
             f'[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] points. This conditions on the seven released machine types '
             'and their recordings; it is not an independent device/day or hardware-population interval. '
             'The change from the historical numbers is a metric correction, not improved model inference.', '',
             '### Is BEATs itself SOTA, and what should improve next?', '',
             'The [official DCASE2026 results](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring-results) '
             'rank the complete discriminatively adapted, ensembled NA-BEATs system first at 70.24115. '
             'A frozen BEATs encoder alone is not that complete system. This challenge score and our seven-machine '
             'development score concern different evaluation scopes and cannot be subtracted as an expected gain.', '',
             'The [NA-SSL paper, Table 1](https://arxiv.org/html/2608.00447v2#S3.F2) reports the following '
             'development-set ensemble totals under its shared backend and rescaling protocol:', '',
             '| Frontend / task adaptation | Paper development total |',
             '| --- | ---: |',
             '| Original BEATs | 61.32% |',
             '| NA-BEATs | 63.92% |',
             '| Discriminatively adapted NA-BEATs | **64.57%** |',
             '| Discriminatively adapted NA-EAT | 64.20% |',
             '| Discriminatively adapted NA-Dasheng | 63.10% |', '',
             'These matched comparisons support retaining NA-BEATs. EAT and Dasheng are useful alternatives, '
             'but this table does not establish a stronger replacement. Our omitted steps are training-normal-based '
             'score rescaling, machine-information / pseudo-label adaptation, and averaging three seeds. '
             'Their gains need project experiments; they are not guaranteed improvements. '
             '[Official model releases](https://github.com/merlresearch/noise-aware-audio-ssl) include all three NA families.', '',
             'A practical next experiment should first implement the published k=4 local score rescaling '
             'and validate it against the reference, then add LoRA-based discriminative adaptation and fixed seed ensembles. '
             'Model and backend choices should use training normals or a separate development selection protocol; '
             'test-domain labels must not enter scoring. ROC metrics cannot be repaired by choosing a more favorable binary threshold. '
             'The [ASDKit reference](https://github.com/TakuyaFujimura/dcase-asd-toolkit) supplies broader backend/adaptation recipes. '
             'This round audits scores and literature; it does not claim those new sound experiments were run.', '',
             '## 2. Controlled magnetic improvements', '',
             'All methods use the same **3,177** pure-state LP100 magnetic spectrogram windows, '
             '**44 device-days**, and fixed four-state labels as the 28 October benchmark. '
             'Outer evaluation remains three held-device folds and 44 held-date folds. '
             'Each device receives equal final weight, after pooling its held-date predictions. '
             'No target samples, labeled or unlabeled, are fitted in strict held-device evaluation.', '',
             'Nested models select among balanced Ridge (alpha 0.1/10/1000) and RBF SVM (C 0.1/1/10). '
             'For held-device evaluation, inner folds hold out each of the two source devices; '
             'for held-date evaluation, three inner folds group the remaining training dates. '
             'Every scaler is trained inside its corresponding training partition. '
             'The automatic selector also selects the representation from inner scores only, excluding the random control.', '',
             'Additional fixed alternatives are 256-tree ExtraTrees, the author\'s multivariate MiniRocket '
             'with 2,016 features and training-only bias fitting, and fixed logistic probes. '
             'MiniRocket sees eight spectral-envelope channels, not a reconstructed waveform; '
             'its sample-normalized counterpart removes each window\'s global mean and scale. '
             'Relative-spectrum features remove broadband level per frame and retain signed temporal differences. '
             'Mantis logistic probes follow the classifier motivation in '
             '[MantisV2 §5.3](https://arxiv.org/html/2602.17868v1#S5.SS3).', '',
             '| Method | Within-device held-date OOF macro-F1 | Held-device macro-F1 |',
             '| --- | ---: | ---: |']
    for key in NAMES:
        m = s['methods'][key]
        lines.append(f'| {NAMES[key]} | {m["within_device_leave_day"]["device_mean_macro_f1"]:.4f} | {m["leave_device"]["device_mean_macro_f1"]:.4f} |')
    lines += ['', '### Matched changes and uncertainty', '',
              'All methods and negative results are retained. The intervals below use the same 2,000 '
              'date-resampling draws within each device for method and reference. They condition on '
              'these devices and fitted models; they are not adjusted for all reported comparisons, '
              'do not retrain the search in every draw or account '
              'for population-level uncertainty over unseen instruments.', '',
              '| Method | Protocol | Reference | Macro-F1 change | Conditional paired 95% interval |',
              '| --- | --- | --- | ---: | --- |']
    for key in ['fisher_nested', 'mantis_nested', 'tspulse_nested', 'automatic_nested', 'extra_trees', 'minirocket_normalized', 'fisher_logistic_balanced']:
        for protocol in ['within_device_leave_day', 'leave_device']:
            row = s['paired_comparisons'][key+'/'+protocol]
            lo, hi = row['conditional_paired_95']
            lines.append(f'| {NAMES[key]} | {protocol} | {NAMES[row["reference"]]} | {row["difference"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    for key, label in [('extra_trees_vs_original_fisher', 'ExtraTrees versus original FISHER'),
                       ('automatic_vs_original_random42', 'Automatic selector versus original random42')]:
        for protocol in ['within_device_leave_day', 'leave_device']:
            row = s['paired_comparisons'][key+'/'+protocol]
            lo, hi = row['conditional_paired_95']
            lines.append(f'| {label} | {protocol} | {NAMES[row["reference"]]} | {row["difference"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    lines += ['', '### Pretraining controls', '']
    for key in ['mantis_nested', 'mantis_logistic_balanced']:
        for protocol in ['within_device_leave_day', 'leave_device']:
            row = s['paired_comparisons'][key+'_vs_matched_random/'+protocol]
            lo, hi = row['conditional_paired_95']
            lines.append(f'- {NAMES[key]}, {protocol}, versus matched random seed42: {row["difference"]:+.4f}, conditional paired 95% [{lo:+.4f}, {hi:+.4f}].')
    lines += ['', 'This round has one matched random seed for the new heads, while the previous fixed-head '
              'benchmark retains its three random seeds. A classifier improvement must not be described '
              'as newly established magnetic pretraining knowledge.', '',
              '### Device-specific results', '',
              '| Method | Test 3302 | Test 3303 | Test 3304 |', '| --- | ---: | ---: | ---: |']
    for key in NAMES:
        m = s['methods'][key]['leave_device']['per_device']
        lines.append('| '+NAMES[key]+' | '+' | '.join(f'{m[d]["macro_f1"]:.4f}' for d in ['3302', '3303', '3304'])+' |')
    lines += ['', '### What the experiment can establish', '',
              'Head choice, nonlinear decision boundaries and temporal-envelope alternatives are now '
              'measured on matched outer samples. A model-family row is a reported alternative, not '
              'an unbiased claim that picking the highest outer score will generalize. The automatic '
              'selector is the explicitly training-selected strategy. Because the previous outer '
              'test results were already inspected, the entire round remains exploratory; improvements '
              'must be confirmed on newly collected dates and preferably additional hardware.', '',
              'The benchmark still uses inherited 0–100 Hz, 20-second spectral caches. It excludes '
              'transition boundaries and does not certify raw clocks, log alignment, absolute acquisition '
              'runs, fault labels or measured RPM. No additional normalization solves those provenance '
              'limitations. Target-device labeled calibration is a separate setting, not a substitute '
              'for strict held-device generalization.', '',
              '## 3. Other potentially better methods', '',
              '- The measured MiniRocket and ExtraTrees results above are strong non-foundation controls. '
              'They test whether a more suitable supervised classifier is more valuable than simply enlarging '
              'a frozen foundation encoder. MiniRocket is a random-convolution method, not a pretrained model. '
              '[Author source](https://github.com/angus924/minirocket/tree/0b1c245d9c9dbc50886f28bc7b32d5d45b5663d6).',
              '- MultiRocket/HYDRA and a small InceptionTime/TCN are appropriate next task-specific comparisons. '
              '[Official aeon convolutional classifiers](https://www.aeon-toolkit.org/en/latest/examples/classification/convolution_based.html) '
              'provide reference implementations. None was run in this round; no centrifuge superiority is claimed.',
              '- The MantisV2 paper reports complementary time/vision representations and improved token/layer '
              'aggregation. Such adapters or a smaller partially fine-tuned student remain candidates; '
              'they need training-device-only validation and random-initialization controls.',
              '- [TS2Vec-FM](https://arxiv.org/abs/2610.28602) is a recent compact convolutional candidate '
              'with both amplitude and normalized-shape views. Its UCR/UEA evidence is not a magnetometer '
              'result; an official downloadable checkpoint was not verified in this round. The older '
              'per-dataset [TS2Vec implementation](https://github.com/zhihanyue/ts2vec) can instead be '
              'trained on source devices only, with the held device fully excluded.', '',
              'The next data-side priority remains verified acquisition/run IDs, synchronization, wider '
              'frequency information if supported by the hardware, and new operating conditions/dates. '
              'These are prospective steps rather than measured gains from this run.', '',
              '## Verification and artifacts', '',
              f'- Independently verified **{verify["new_prediction_rows"]:,}** new prediction rows, '
              f'{verify["split_audits"]} stage/fold audits, score argmax, inherited labels, strict device/date '
              'exclusion, inner/outer exclusion, and every training-score selection.',
              '- All retained historical baseline metrics were independently reproduced from their saved predictions.',
              f'- Fixed logistic fits with convergence warnings: {verify["logistic_convergence_warning_fits"]}. '
              'Their original warnings and iteration counts remain in the compressed audit.',
              '- [All magnetic metrics and class recalls](summary.json), [table](metrics_table.csv), '
              '[new predictions](predictions.csv.gz), [split and selection audit](selection_and_split_audit.json.gz), '
              '[verification](verification.json), [sound metric audit](asd_metric_audit.json).',
              '- [Commands and complete protocol](../../experiments/baseline_improvements_20261029/README.md). '
              'Large arrays remain in the remote archive; no raw instrument data or model weights are committed.', '']
    (root/'REPORT.md').write_bytes(('\n'.join(lines)).encode('utf-8'))
    cross = s['paired_comparisons']['automatic_nested/leave_device']
    strong = s['paired_comparisons']['extra_trees_vs_original_fisher/within_device_leave_day']
    zh = ['# 声音与磁场基线复核和改进（2026-10-29）', '',
          '1．声音侧：BEATs 是强骨干，但“冻结 BEATs＋简单打分”不能直接称为任务 SOTA。'
          '本次核实的 DCASE2026 冠军是完整 NA-BEATs 任务适配与集成系统，官方分数70.24。'
          '这与本项目七类开发集评测范围不同，不能直接相减作为可实现的增益。', '',
          f'发现并更正了历史域 AUROC 口径：官方使用该域正常声与两个域的全部异常声。'
          f'不改变模型输出，重算得到 BEATs {100*b:.2f}%、NA-BEATs {100*n:.2f}%，'
          f'差值{100*(n-b):+.2f}个百分点，条件成对95%区间[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]。'
          '这一更正不影响声音分类或问答成绩。', '',
          '作者统一开发集协议下，原BEATs为61.32，NA-BEATs为63.92，判别适配后的NA-BEATs为64.57；'
          '对应NA-EAT为64.20、NA-Dasheng为63.10。因此目前有证据的优先级是补足正常库分数重缩放、'
          '判别适配和多种子集成，而非直接换更大编码器。上述数字来自论文，并非本轮新增推理结果。'
          '[官方结果](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring-results)、'
          '[统一协议论文](https://arxiv.org/html/2608.00447v2)。', '',
          '2．磁场侧：沿用相同3177窗口和44设备日，完成16种新增配置。所有外层日期与设备隔离；'
          '嵌套选参仅使用外层训练集。以下macro-F1先按设备汇总OOF，再对三台设备等权平均。', '',
          '| 方法 | 同设备留日 | 严格留设备 |', '| --- | ---: | ---: |']
    zh_names = [('fisher_pretrained_fixed','FISHER＋原固定线性头'), ('fisher_nested','FISHER＋训练内选头'),
                ('mantis_pretrained_fixed','MantisV2＋原固定线性头'), ('mantis_nested','MantisV2＋训练内选头'),
                ('tspulse_pretrained_fixed','TSPulse＋原固定线性头'), ('tspulse_nested','TSPulse＋训练内选头'),
                ('extra_trees','物理／相对谱特征＋ExtraTrees'), ('minirocket_normalized','MiniRocket归一化频带时序'),
                ('automatic_nested','训练内自动选择表征和分类头')]
    for key, label in zh_names:
        row = s['methods'][key]
        zh.append(f'| {label} | {row["within_device_leave_day"]["device_mean_macro_f1"]:.4f} | {row["leave_device"]["device_mean_macro_f1"]:.4f} |')
    lo, hi = strong['conditional_paired_95']
    zh += ['', f'ExtraTrees相对原FISHER的留日增益为{strong["difference"]:+.4f}，条件成对95%区间[{lo:+.4f}, {hi:+.4f}]。'
           '说明当前设备内任务上，更合适的监督分类方法有实际价值。ExtraTrees并非预训练基座。', '']
    lo, hi = cross['conditional_paired_95']
    zh += [f'自动选择方案相对原MantisV2的留设备增益为{cross["difference"]:+.4f}，'
           f'条件成对95%区间[{lo:+.4f}, {hi:+.4f}]。'
           + ('区间跨零，提升尚不确定。' if lo <= 0 <= hi else '区间未跨零，但仍是既有三台设备上的探索性证据。'), '',
           '改分类头并非总能改善跨设备：Mantis单模型的嵌套选头结果就没有超过原固定头。'
           '本轮不能据分类头收益宣称磁场预训练知识成立；匹配随机骨干的结果与区间已保留。', '',
           '其他方法方面，本轮实际复现了MiniRocket和ExtraTrees；MultiRocket/HYDRA、源设备自监督TS2Vec、'
           '小型TCN/InceptionTime值得作为后续对照。TiViT和TS2Vec-FM等通用时序候选也应以相同划分验证，'
           '其公开时序榜单成绩不能当作本项目磁场成绩，TS2Vec-FM权重可下载性本轮尚未核实。', '',
           '当前更大瓶颈仍是跨设备工况、安装／幅值和特征分布差异；这些是与现象一致的解释，'
           '本轮不能独立分离其因果作用。原始时钟、日志同步、独立回合溯源和0–100Hz缓存的限制未修复。'
           '建议新增已核验日期、运行回合和工况，再确认方法增益；若允许一天标注校准，应单列评测，'
           '不要与完全未见目标设备混用。', '',
           '本轮为已看过旧测试结果后的探索性重评测。区间以日期为分组、条件于已拟合模型，未校正全部多重比较，'
           '不代表硬件总体置信区间。完整数字、逐类召回、选择审计与复核在[英文报告](REPORT.md)和[指标](summary.json)。', '']
    (root/'SUMMARY_ZH.md').write_bytes('\n'.join(zh).encode('utf-8'))
    print('REPORT_WRITTEN', root/'REPORT.md')


if __name__ == '__main__':
    main()
