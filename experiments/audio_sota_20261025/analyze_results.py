"""Recompute all reported metrics from per-case outputs, with paired QA tests."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score

from common import wilson,write_json


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    a=p.parse_args();root=a.results
    esc=read_jsonl(root/'esc50/predictions.jsonl')
    assert len(esc)==2000 and len({r['filename'] for r in esc})==2000
    assert all(int(r['filename'].split('-')[0])==r['fold'] for r in esc)
    em=json.loads((root/'esc50/metrics.json').read_text())
    correct=sum(r['target']==r['prediction'] for r in esc)
    assert correct==em['correct'] and np.isclose(correct/2000,em['accuracy'])
    qa={}
    audit=json.loads((root/'qa/audio_input_audit.json').read_text())
    assert all(audit[m]['passed'] for m in ('moss','qwen'))
    for model in ('moss','qwen'):
        rows=read_jsonl(root/'qa'/model/'raw.jsonl')
        expected=json.loads((root/'qa/cases.json').read_text())
        assert len(rows)==len(expected) and {r['case_id'] for r in rows}=={r['case_id'] for r in expected}
        if model=='qwen':
            assert all(r.get('encoder_feature_shape') is not None for r in rows),'Qwen run missing acoustic features; rerun after fixing processor API'
        results={}
        for suite in sorted({r['suite'] for r in rows}):
            group=[r for r in rows if r['suite']==suite]
            k=sum(r['expected']==r['parsed_answer'] for r in group)
            values={'n':len(group),'correct':k,'accuracy_or_original_answer_retention':k/len(group),
                    'wilson95_question_descriptive':wilson(k,len(group)),
                    'unparseable':sum(r['parsed_answer']=='UNPARSEABLE' for r in group),
                    'token_cap_count':sum(r['hit_token_cap'] for r in group),
                    'mean_seconds':float(np.mean([r['seconds'] for r in group]))}
            if suite=='machine_full':
                y=[r['expected'] for r in group];pred=[r['parsed_answer'] for r in group]
                answered=[r for r in group if r['parsed_answer'] in ('NORMAL','ANOMALOUS')]
                values.update(macro_f1=float(f1_score(y,pred,labels=['NORMAL','ANOMALOUS'],average='macro',zero_division=0)),
                              binary_answer_coverage=len(answered)/len(group),
                              answered_count=len(answered),
                              accuracy_on_answered=sum(r['parsed_answer']==r['expected'] for r in answered)/len(answered) if answered else None,
                              abstention_count=sum(r['parsed_answer']=='INSUFFICIENT EVIDENCE' for r in group),
                              normal_specificity=sum(r['parsed_answer']=='NORMAL' for r in group if r['expected']=='NORMAL')/sum(r['expected']=='NORMAL' for r in group),
                              anomalous_sensitivity=sum(r['parsed_answer']=='ANOMALOUS' for r in group if r['expected']=='ANOMALOUS')/sum(r['expected']=='ANOMALOUS' for r in group))
            if suite=='mmau_sound':
                values['by_difficulty']={d:{'n':sum(r.get('difficulty')==d for r in group),
                                          'correct':sum(r.get('difficulty')==d and r['expected']==r['parsed_answer'] for r in group)} for d in ('easy','medium','hard')}
                values['unique_audio_sha256']=len({r['source_sha256'] for r in group})
                values['expected_choice_counts']={c:sum(r['expected']==c for r in group) for c in 'ABCD'}
                values['fixed_always_A_accuracy']=sum(r['expected']=='A' for r in group)/len(group)
                values['recall_by_choice']={c:sum(r['expected']==c and r['parsed_answer']==c for r in group)/sum(r['expected']==c for r in group) for c in 'ABCD'}
                values['balanced_choice_accuracy']=float(np.mean(list(values['recall_by_choice'].values())))
                values['processor_truncated_audio_count']=sum(r.get('processor_audio_truncated',False) for r in group)
            results[suite]=values
        by_id={r['case_id']:r for r in rows}
        for suite in ('mmau_silence','mmau_mismatch'):
            group=[r for r in rows if r['suite']==suite]
            matched=sum(by_id[r['paired_case']]['parsed_answer']==r['expected'] for r in group)
            results[suite]['matched_audio_correct_on_same_questions']=matched
            results[suite]['matched_audio_accuracy']=matched/len(group)
            results[suite]['matched_to_control_accuracy_drop']=matched/len(group)-results[suite]['accuracy_or_original_answer_retention']
            results[suite]['answer_changed_count']=sum(by_id[r['paired_case']]['parsed_answer']!=r['parsed_answer'] for r in group)
        qa[model]=results
    all_m={r['case_id']:r for r in read_jsonl(root/'qa/moss/raw.jsonl')}
    all_q={r['case_id']:r for r in read_jsonl(root/'qa/qwen/raw.jsonl')}
    assert set(all_m)==set(all_q)
    for k in all_m:
        assert all_m[k]['pcm_sha256']==all_q[k]['pcm_sha256'],('paired waveform mismatch',k)
        assert all_m[k]['prompt']==all_q[k]['prompt'],('paired prompt mismatch',k)
        assert all_m[k]['expected']==all_q[k]['expected']
    m={k:r for k,r in all_m.items() if r['suite']=='mmau_sound'}
    q={k:r for k,r in all_q.items() if r['suite']=='mmau_sound'}
    assert set(m)==set(q)
    keys=sorted(m); mc=np.array([m[k]['expected']==m[k]['parsed_answer'] for k in keys]);qc=np.array([q[k]['expected']==q[k]['parsed_answer'] for k in keys])
    b=int(np.sum(mc&~qc));c=int(np.sum(~mc&qc));delta=mc.astype(float)-qc.astype(float)
    rng=np.random.default_rng(20261025)
    boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
    paired={'n':len(keys),'moss_only_correct':b,'qwen_only_correct':c,
            'accuracy_difference_moss_minus_qwen':float(delta.mean()),
            'paired_question_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
            'mcnemar_exact_p_two_sided':float(binomtest(b,b+c,.5).pvalue) if b+c else 1.0,
            'scope':'Descriptive paired comparison of the fixed sound test-mini partition; no universal SOTA claim.'}
    # Some questions share audio. Primary uncertainty resamples audio groups.
    hashes=sorted({m[k]['source_sha256'] for k in keys})
    cluster_counts=np.array([sum(m[k]['source_sha256']==h for k in keys) for h in hashes])
    cluster_m=np.array([sum(m[k]['source_sha256']==h and bool(mc[i]) for i,k in enumerate(keys)) for h in hashes])
    cluster_q=np.array([sum(m[k]['source_sha256']==h and bool(qc[i]) for i,k in enumerate(keys)) for h in hashes])
    cluster_rng=np.random.default_rng(20261026)
    draws=cluster_rng.integers(0,len(hashes),(10000,len(hashes)))
    denom=cluster_counts[draws].sum(axis=1)
    mb=cluster_m[draws].sum(axis=1)/denom;qb=cluster_q[draws].sum(axis=1)/denom
    paired['audio_clusters']=len(hashes)
    paired['paired_audio_cluster_bootstrap95']=np.quantile(mb-qb,[.025,.975]).tolist()
    paired['question_mcnemar_caveat']='Exploratory question-level statistic; repeated audio violates question independence. Use audio-cluster uncertainty for primary inference.'
    for model,boot_values in [('moss',mb),('qwen',qb)]:
        qa[model]['mmau_sound']['audio_cluster_bootstrap95']=np.quantile(boot_values,[.025,.975]).tolist()
    cluster_delta=cluster_m-cluster_q;observed=int(cluster_delta.sum());extreme=0
    for _ in range(100):
        signs=cluster_rng.integers(0,2,(1000,len(hashes)),dtype=np.int8)*2-1
        extreme+=int(np.sum(np.abs(signs@cluster_delta)>=abs(observed)))
    paired['audio_cluster_sign_flip_permutations']=100000
    paired['audio_cluster_sign_flip_p_two_sided']=(extreme+1)/100001
    asd=read_jsonl(root/'asd2026/scores.jsonl');am=json.loads((root/'asd2026/metrics.json').read_text())
    for group in am['metrics']:
        rows=[r for r in asd if all(r[k]==group[k] for k in ('model','machine','variant'))]
        assert len(rows)==200
        y=np.array([r['label'] for r in rows]);s=np.array([r['anomaly_score'] for r in rows]);ds=np.array([r['domain'] for r in rows])
        for key,mask,fpr in [('auc_source',ds=='source',None),('auc_target',ds=='target',None),('pauc_standardized_fpr_0_1',np.ones(len(rows),bool),.1),('auc_pooled',np.ones(len(rows),bool),None)]:
            assert np.isclose(roc_auc_score(y[mask],s[mask],max_fpr=fpr),group[key])
    # Conditional paired uncertainty for the pre-specified BEAM/RDP4 backend.
    machine_data=[]
    for machine in am['machines']:
        rr={model:sorted([r for r in asd if r['machine']==machine and r['model']==model and r['variant']=='beam_rdp4'],key=lambda r:r['path']) for model in ('BEATs_iter3','dcase2026_nabeats_seed0')}
        b=rr['BEATs_iter3'];n=rr['dcase2026_nabeats_seed0']
        assert [r['path'] for r in b]==[r['path'] for r in n]
        y=np.array([r['label'] for r in b]);domains=np.array([r['domain'] for r in b])
        machine_data.append((y,domains,np.array([r['anomaly_score'] for r in b]),np.array([r['anomaly_score'] for r in n])))
    bs=[]
    for _ in range(1000):
        components=[[],[]]
        for y,domains,s0,s1 in machine_data:
            idx=np.concatenate([rng.choice(np.flatnonzero((y==label)&(domains==domain)),50,replace=True) for domain in ('source','target') for label in (0,1)])
            yy=y[idx];dd=domains[idx]
            for k,s in enumerate((s0,s1)):
                ss=s[idx]
                components[k].extend([roc_auc_score(yy[dd=='source'],ss[dd=='source']),roc_auc_score(yy[dd=='target'],ss[dd=='target']),roc_auc_score(yy,ss,max_fpr=.1)])
        hm=[len(v)/np.sum(1/np.asarray(v)) for v in components]
        bs.append([*hm,hm[1]-hm[0]])
    asd_uncertainty={'variant':'beam_rdp4','resamples':1000,
                     'method':'paired stratified clip bootstrap within machine/domain/label; conditional descriptive intervals',
                     'beats_ci95':np.quantile(np.asarray(bs)[:,0],[.025,.975]).tolist(),
                     'nabeats_ci95':np.quantile(np.asarray(bs)[:,1],[.025,.975]).tolist(),
                     'difference_nabeats_minus_beats_ci95':np.quantile(np.asarray(bs)[:,2],[.025,.975]).tolist()}
    write_json(root/'summary.json',{'esc50':{'n':2000,'accuracy':em['accuracy'],'macro_f1':em['macro_f1'],'folds':em['folds']},
                                  'qa':qa,'paired_mmau_sound':paired,'asd2026':{'n':am['n'],'machines':am['machines'],'aggregate':am['aggregate'],'uncertainty':asd_uncertainty}})
    write_json(root/'verification.json',{'passed':True,'esc_predictions':len(esc),'qa_cases_per_model':len(expected),
                                       'mmau_unique_audio_clusters':len(hashes),
                                       'asd_score_rows':len(asd),'checks':['ESC fold isolation and accuracy','QA expected IDs and denominator','paired prompt and waveform hashes for all733 cases','Qwen acoustic-feature presence','real/silent frontend audit','audio-group bootstrap for repeated MMAU recordings','all ASD per-machine ROC/pAUC recomputed']})
    print(json.dumps({'ESC50':em['accuracy'],'MMAU paired':paired,'ASD':am['aggregate']},indent=2))


if __name__=='__main__':main()
