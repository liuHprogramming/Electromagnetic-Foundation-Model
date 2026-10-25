"""Verify real/silent audio reach different frontend tensors with fixed text."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor
from common import write_json
from run_qa import load_audio


def digest(x):
    return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--qwen-model',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4)
    cases=json.loads((a.root/'results/qa/cases.json').read_text())
    case=next(c for c in cases if c['suite']=='machine_full')
    reports={}
    for name in ('qwen','moss'):
        if name=='qwen':
            processor=AutoProcessor.from_pretrained(a.qwen_model)
            rate=processor.feature_extractor.sampling_rate
            conversation=[{'role':'user','content':[{'type':'audio','audio_url':'placeholder.wav'}, {'type':'text','text':case['prompt']}]}]
            text=processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=False)
            wav=load_audio(case,rate)
            inputs=[processor(text=text,audio=[w],sampling_rate=rate,return_tensors='pt',padding=True) for w in (wav,np.zeros_like(wav))]
            key='input_features'
            token_id=processor.tokenizer.convert_tokens_to_ids('<|AUDIO|>')
        else:
            sys.path.insert(0,str(a.root/'external/moss'))
            from src.processing_moss_audio import MossAudioProcessor
            processor=MossAudioProcessor.from_pretrained(a.root/'models/MOSS-Audio-8B-Thinking',enable_time_marker=True)
            rate=processor.config.mel_sr;wav=load_audio(case,rate)
            inputs=[processor(text=case['prompt'],audios=[w],return_tensors='pt') for w in (wav,np.zeros_like(wav))]
            key='audio_data';token_id=processor.audio_token_id
        real,silent=inputs
        assert torch.equal(real['input_ids'],silent['input_ids'])
        assert (real['input_ids']==token_id).sum()>0
        assert real[key].shape==silent[key].shape
        difference=float((real[key].float()-silent[key].float()).abs().max())
        assert difference>1e-3
        reports[name]={'passed':True,'case_id':case['case_id'],'feature_shape':list(real[key].shape),
                       'real_feature_sha256':digest(real[key]),'silent_feature_sha256':digest(silent[key]),
                       'max_absolute_feature_difference':difference,'input_ids_identical':True,
                       'audio_token_count':int((real['input_ids']==token_id).sum()),
                       'source_pcm_rms':float(np.sqrt(np.mean(wav.astype(float)**2)))}
    write_json(a.root/'results/qa/audio_input_audit.json',reports)
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
