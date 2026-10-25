"""Prepare fixed evaluation cases without tuning prompts on predictions."""
import argparse
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from common import sha,write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pilot-root',type=Path,required=True)
    a=p.parse_args()
    out=a.root/'results/qa';out.mkdir(parents=True,exist_ok=True)
    data=a.root/'data/mmau_sound';data.mkdir(parents=True,exist_ok=True)
    table=pd.read_parquet(a.root/'data/MMAU-test-mini/test_mini.parquet')
    normalized=[]
    for raw in table.to_dict('records'):
        attrs=raw.get('other_attributes',{})
        if isinstance(attrs,str):attrs=json.loads(attrs)
        normalized.append({**attrs,**raw,'audio':raw.get('audio',raw.get('context')),
                           'question':raw.get('question',raw.get('instruction'))})
    sound=[r for r in normalized if r.get('task',r.get('subset'))=='sound']
    assert 300<=len(sound)<=350,(len(table),len(sound),table.columns.tolist())
    cases=[]
    for i,r in enumerate(sound):
        source=r['audio'];raw=source['bytes']
        path=data/f'{i:04d}.wav'
        if not path.exists():path.write_bytes(raw)
        info=sf.info(path)
        original_choices=list(r['choices']); original_answer=r['answer']
        assert original_answer in original_choices,(original_answer,original_choices)
        answer_index=original_choices.index(original_answer)
        choices=[re.sub(r'^\([ABCD]\)\s*','',c) for c in original_choices]
        answer=choices[answer_index]
        prompt='Listen to the audio and answer the following multiple-choice question.\n'+r['question']+'\n'+'\n'.join(f'{chr(65+k)}. {c}' for k,c in enumerate(choices))+'\nRespond with only the letter A, B, C, or D for the single best answer. Do not explain.'
        cases.append({'case_id':f'mmau_{i:04d}','suite':'mmau_sound','audio_path':str(path),
                      'source_id':r.get('id',source.get('path',str(i))),
                      'source_sha256':sha(path),'question':r['question'],'choices':choices,
                      'answer':answer,'expected':chr(65+choices.index(answer)),'prompt':prompt,
                      'difficulty':r.get('difficulty'),'sub_category':r.get('sub-category'),
                      'duration_seconds':info.duration})
    n=len(cases);rng=np.random.default_rng(20261025)
    controls=sorted(rng.choice(n,60,replace=False).tolist())
    base=list(cases)
    for i in controls:
        for mode in ('silence','mismatch'):
            case=dict(base[i]);case.update(case_id=f'mmau_{mode}_{i:04d}',suite='mmau_'+mode,paired_case=base[i]['case_id'])
            if mode=='mismatch':
                other=base[(i+137)%n]
                case.update(audio_path=other['audio_path'],source_sha256=other['source_sha256'],mismatch_source_case=other['case_id'])
            cases.append(case)
    pilot=json.loads((a.pilot_root/'manifest.json').read_text())
    full=sorted((r for r in pilot if r['split']=='test'),key=lambda r:r['path'])
    prompt=('Assess this machine recording using audible evidence only. Start with exactly NORMAL, ANOMALOUS, or INSUFFICIENT EVIDENCE, followed by one short audible reason. If the audio is silent or too brief to support a condition judgment, answer INSUFFICIENT EVIDENCE. Do not infer an exact fault, device setting, or operating reference that was not supplied.')
    for i,r in enumerate(full):
        cases.append({'case_id':f'machine_{i:04d}','suite':'machine_full','audio_path':str(a.pilot_root/r['path']),
                      'source_sha256':r['sha256'],'source_id':r['path'],'machine':r['machine'],
                      'expected':'ANOMALOUS' if r['condition']=='anomaly' else 'NORMAL','prompt':prompt})
    selected=sorted(rng.choice(len(full),40,replace=False).tolist())
    for i in selected:
        r=full[i]
        for mode in ('silence','short'):
            cases.append({'case_id':f'machine_{mode}_{i:04d}','suite':'machine_'+mode,
                          'paired_case':f'machine_{i:04d}','audio_path':str(a.pilot_root/r['path']),
                          'source_sha256':r['sha256'],'source_id':r['path'],'machine':r['machine'],
                          'expected':'INSUFFICIENT EVIDENCE','prompt':prompt,
                          'protocol_label':'abstention expectation; not a DCASE ground-truth condition'})
    write_json(out/'cases.json',cases)
    counts={s:sum(c['suite']==s for c in cases) for s in sorted({c['suite'] for c in cases})}
    write_json(out/'protocol.json',{'seed':20261025,'mmau_revision':'ccd9696c0111ea7060827598f310558df0b71b0a',
                                  'mmau_version':'MMAU-v05.15.25 official HF snapshot','mmau_total_rows':len(table),
                                  'counts':counts,'generation':{'do_sample':False,'max_new_tokens':512},
                                  'label_visibility':'Answers and source paths never rendered into model input.',
                                  'controls':'MMAU silence/mismatch retain original answer only to measure paired audio dependence; not independently labeled QA accuracy. Machine short/silence expected abstention is a protocol rule.'})
    print('QA prepared',counts,flush=True)


if __name__=='__main__':main()
