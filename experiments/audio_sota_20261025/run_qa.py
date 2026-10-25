"""Evaluate MOSS and Qwen on identical benchmark/diagnostic cases."""
import argparse
import gc
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from common import runtime,setup,write_json,git_rev


def parse(text,case):
    final=text.split('</think>')[-1].strip()
    final=re.sub(r'^[\s`*#]+','',final)
    if case['suite'].startswith('machine'):
        m=re.match(r'(INSUFFICIENT EVIDENCE|ANOMALOUS|NORMAL)\b',final,re.I)
        return m[1].upper() if m else 'UNPARSEABLE'
    m=re.match(r'^(?:[Aa]nswer\s*[:：]?\s*)?[\(\[]?([ABCD])[\)\].:\s]*(?:$|\s)',final)
    if m:return m[1]
    def normalize(s):return re.sub(r'[^\w ]','',s.lower()).strip()
    exact=[chr(65+i) for i,c in enumerate(case['choices']) if normalize(c)==normalize(final)]
    return exact[0] if len(exact)==1 else 'UNPARSEABLE'


def load_audio(case,sample_rate):
    wav,sr=sf.read(case['audio_path'],dtype='float32',always_2d=True)
    wav=torch.from_numpy(wav.mean(axis=1).copy())
    if sr!=sample_rate:wav=torchaudio.functional.resample(wav,sr,sample_rate)
    if case['suite'].endswith('silence'):wav=torch.zeros_like(wav)
    if case['suite']=='machine_short':
        n=round(sample_rate*.25);start=(len(wav)-n)//2;wav=wav[start:start+n]
    return wav.numpy()


def moss_batch(processor,audios,prompts,device,dtype):
    """Collate official single-example outputs; preserve row/audio ordering."""
    single=[processor(text=p,audios=[w],return_tensors='pt') for w,p in zip(audios,prompts)]
    lengths=[int(x['input_ids'].shape[1]) for x in single]
    width=max(lengths);pad=processor.tokenizer.eos_token_id
    ids=torch.full((len(single),width),pad,dtype=torch.long)
    mask=torch.zeros_like(ids)
    mel_width=max(int(x['audio_data'].shape[-1]) for x in single)
    mel=torch.zeros((len(single),single[0]['audio_data'].shape[1],mel_width),dtype=dtype)
    for i,x in enumerate(single):
        ids[i,-lengths[i]:]=x['input_ids'][0]
        mask[i,-lengths[i]:]=1
        mel[i,:,:x['audio_data'].shape[-1]]=x['audio_data'][0]
    inputs={'input_ids':ids.to(device),'attention_mask':mask.to(device),
            'audio_data':mel.to(device),'audio_data_seqlens':torch.cat([x['audio_data_seqlens'] for x in single]).to(device)}
    inputs['audio_input_mask']=inputs['input_ids']==processor.audio_token_id
    return inputs,[x['input_ids'][0].tolist() for x in single]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--qwen-model',type=Path,required=True)
    p.add_argument('--model',choices=['moss','qwen'],required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--output-name',default='raw.jsonl')
    a=p.parse_args();setup(a.device)
    cases=json.loads((a.root/'results/qa/cases.json').read_text())
    if a.limit:cases=cases[:a.limit]
    out=a.root/'results/qa'/a.model;out.mkdir(parents=True,exist_ok=True)
    output=out/a.output_name
    done={json.loads(l)['case_id'] for l in output.read_text().splitlines()} if output.exists() else set()
    if all(c['case_id'] in done for c in cases):return
    start=time.monotonic()
    if a.model=='moss':
        sys.path.insert(0,str(a.root/'external/moss'))
        from src.modeling_moss_audio import MossAudioModel
        from src.processing_moss_audio import MossAudioProcessor
        model,info=MossAudioModel.from_pretrained(a.root/'models/MOSS-Audio-8B-Thinking',
            dtype=torch.bfloat16 if a.device=='cuda' else torch.float32,device_map=a.device,
            output_loading_info=True,attn_implementation='sdpa')
        assert not info.get('missing_keys') and not info.get('unexpected_keys'),info
        processor=MossAudioProcessor.from_pretrained(a.root/'models/MOSS-Audio-8B-Thinking',enable_time_marker=True)
        rate=processor.config.mel_sr
        revision='aacbb8036eb06dd968c105f91d2663af8fbee720'
    else:
        from transformers import AutoProcessor,Qwen2AudioForConditionalGeneration
        model=Qwen2AudioForConditionalGeneration.from_pretrained(a.qwen_model,
            dtype=torch.bfloat16 if a.device=='cuda' else torch.float32,device_map=a.device,
            attn_implementation='sdpa')
        processor=AutoProcessor.from_pretrained(a.qwen_model)
        rate=processor.feature_extractor.sampling_rate
        revision='0a095220c30b7b31434169c3086508ef3ea5bf0a'
    model.eval()
    if a.model=='moss' and a.batch_size>1:
        pending=[c for c in cases if c['case_id'] not in done]
        for offset in range(0,len(pending),a.batch_size):
            batch=pending[offset:offset+a.batch_size];tick=time.monotonic()
            audios=[load_audio(c,rate) for c in batch]
            inputs,ids=moss_batch(processor,audios,[c['prompt'] for c in batch],model.device,model.dtype)
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=512,do_sample=False,use_cache=True,
                                         pad_token_id=processor.tokenizer.eos_token_id)
            for j,(case,audio) in enumerate(zip(batch,audios)):
                tokens=generated[j,inputs['input_ids'].shape[1]:]
                eos=(tokens==processor.tokenizer.eos_token_id).nonzero()
                if len(eos):tokens=tokens[:int(eos[0])+1]
                response=processor.decode(tokens,skip_special_tokens=True)
                row={**{k:v for k,v in case.items() if k!='audio_path'},'model':a.model,'model_revision':revision,
                     'raw_output':response,'parsed_answer':parse(response,case),'generated_tokens':len(tokens),
                     'hit_token_cap':len(tokens)>=512,'sample_rate':rate,'input_duration_seconds':len(audio)/rate,
                     'pcm_sha256':hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest(),
                     'rendered_input_ids_sha256':hashlib.sha256(np.array(ids[j],dtype='<i8').tobytes()).hexdigest(),
                     'seconds':(time.monotonic()-tick)/len(batch),'batch_size_used':len(batch)}
                with output.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(a.model,offset+j+1,'/',len(pending),case['case_id'],row['parsed_answer'],'tokens',len(tokens),flush=True)
        write_json(out/('run_metadata.json' if a.output_name=='raw.jsonl' else a.output_name+'.metadata.json'),{'runtime':runtime(a.device),'generation':{'max_new_tokens':512,'do_sample':False},
                    'official_moss_code_commit':git_rev(a.root/'external/moss'),'n_expected':len(cases),
                    'batch_size':a.batch_size,'seconds_this_invocation':time.monotonic()-start})
        return
    for i,case in enumerate(cases):
        if case['case_id'] in done:continue
        tick=time.monotonic();audio=load_audio(case,rate)
        if a.model=='moss':
            inputs=processor(text=case['prompt'],audios=[audio],return_tensors='pt').to(model.device)
            if inputs.get('audio_data') is not None:inputs['audio_data']=inputs['audio_data'].to(model.dtype)
            inputs['audio_input_mask']=inputs['input_ids']==processor.audio_token_id
            rendered_ids=inputs['input_ids'][0].cpu().tolist()
        else:
            conversation=[{'role':'user','content':[{'type':'audio','audio_url':'placeholder.wav'},
                                                    {'type':'text','text':case['prompt']}]}]
            rendered=processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=False)
            inputs=processor(text=rendered,audio=[audio],sampling_rate=rate,return_tensors='pt',padding=True).to(model.device)
            assert 'input_features' in inputs and 'feature_attention_mask' in inputs, 'Qwen audio was not processed'
            inputs['input_features']=inputs['input_features'].to(model.dtype)
            rendered_ids=inputs['input_ids'][0].cpu().tolist()
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=512,do_sample=False,use_cache=True)
        tokens=generated[0,inputs['input_ids'].shape[1]:]
        response=processor.decode(tokens,skip_special_tokens=True)
        row={**{k:v for k,v in case.items() if k!='audio_path'},'model':a.model,'model_revision':revision,
             'raw_output':response,'parsed_answer':parse(response,case),'generated_tokens':len(tokens),
             'hit_token_cap':len(tokens)>=512,'sample_rate':rate,'input_duration_seconds':len(audio)/rate,
             'pcm_sha256':hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest(),
             'rendered_input_ids_sha256':hashlib.sha256(np.array(rendered_ids,dtype='<i8').tobytes()).hexdigest(),
             'encoder_feature_shape':list(inputs['input_features'].shape) if 'input_features' in inputs else None,
             'processor_audio_truncated':bool(a.model=='qwen' and len(audio)>getattr(processor.feature_extractor,'n_samples',len(audio))),
             'seconds':time.monotonic()-tick}
        with output.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(a.model,i+1,len(cases),case['case_id'],row['parsed_answer'],'tokens',len(tokens),'sec',round(row['seconds'],2),flush=True)
    write_json(out/'run_metadata.json',{'model':a.model,'checkpoint_revision':revision,
                                      'official_moss_code_commit':git_rev(a.root/'external/moss') if a.model=='moss' else None,
                                      'runtime':runtime(a.device),'generation':{'max_new_tokens':512,'do_sample':False},
                                      'seconds_this_invocation':time.monotonic()-start,'n_expected':len(cases)})


if __name__=='__main__':main()
