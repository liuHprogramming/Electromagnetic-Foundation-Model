"""Frozen BEATs/NA-BEATs + frequency memory-bank ASD on DCASE2026.

This reproduces the released frontend and the described pooling/memory-bank
ideas, not the discriminatively fine-tuned three-seed competition ensemble.
"""
import argparse
import hashlib
import json
import os
import re
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import roc_auc_score

from common import runtime, setup, sha, write_json


def l2(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def pool(z):
    b,l,d=z.shape
    assert l%8==0
    z=z.reshape(b,l//8,8,d).float()
    mean=z.mean(1)
    delta=(z-mean[:,None]).norm(dim=-1)
    relative=delta/delta.amax(dim=1,keepdim=True).clamp_min(1e-12)
    weights=(1+relative).pow(4)
    weights=weights/weights.sum(dim=1,keepdim=True)
    rdp=(z*weights[:,:,:,None]).sum(1)
    return {'global_ap':z.mean((1,2)).cpu().numpy(),
            'frequency_ap':mean.cpu().numpy(),'frequency_rdp4':rdp.cpu().numpy()}


def metrics(labels,scores,domains):
    y=np.asarray(labels);s=np.asarray(scores);ds=np.asarray(domains)
    source=(ds=='source')|(y==1)
    target=(ds=='target')|(y==1)
    return {'auc_source':float(roc_auc_score(y[source],s[source])),
            'auc_target':float(roc_auc_score(y[target],s[target])),
            'pauc_standardized_fpr_0_1':float(roc_auc_score(y,s,max_fpr=.1)),
            'auc_pooled':float(roc_auc_score(y,s))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();setup(a.device)
    os.environ['NASSL_CACHE_DIR']=str(a.root/'models/nassl')
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    from nassl.utils.restore import restore_model
    from nassl.model.beats.BEATs import BEATs,BEATsConfig
    out=a.root/'results/asd2026';out.mkdir(parents=True,exist_ok=True)
    data=a.root/'data/dcase2026';data.mkdir(exist_ok=True)
    archives=sorted((a.root/'data').glob('dev_*.zip'))
    from fetch_asd import DATA
    for archive in archives:
        h=hashlib.md5()
        with archive.open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        assert h.hexdigest()==DATA[archive.name],('archive MD5 mismatch',archive)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if member.is_dir() or not member.filename.endswith('.wav'):continue
                target=data/member.filename
                if not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(member) as src,target.open('wb') as dst:
                        import shutil
                        shutil.copyfileobj(src,dst)
    entries=[]
    pattern=re.compile(r'section_(\d+)_((?:source)|(?:target))_(train|test)_(normal|anomaly)_')
    for path in sorted(data.rglob('*.wav')):
        m=pattern.search(path.name)
        if not m:continue
        machine=path.parent.parent.name
        entries.append({'path':str(path.relative_to(data)),'machine':machine,'section':m[1],
                        'domain':m[2],'split':m[3],'label':int(m[4]=='anomaly'),'sha256':sha(path)})
    machines=sorted({e['machine'] for e in entries})
    for machine in machines:
        group=[e for e in entries if e['machine']==machine]
        assert sum(e['split']=='train' for e in group)==1000,(machine,len(group))
        assert sum(e['split']=='test' for e in group)==200
        assert not any(e['label'] for e in group if e['split']=='train')
        assert sum(e['split']=='train' and e['domain']=='source' for e in group)==990
        assert sum(e['split']=='train' and e['domain']=='target' for e in group)==10
        for domain in ('source','target'):
            for label in (0,1):
                assert sum(e['split']=='test' and e['domain']==domain and e['label']==label for e in group)==50
        assert not ({e['sha256'] for e in group if e['split']=='train'} & {e['sha256'] for e in group if e['split']=='test'}),('duplicate train/test WAV',machine)
    assert machines
    write_json(out/'manifest.json',entries)
    if a.prepare_only:
        print('ASD PREPARATION VERIFIED',len(entries),machines,flush=True)
        return
    all_metrics=[];score_rows=[];start=time.monotonic();precision_checks=[]
    checkpoint=a.root/'models/nassl/BEATs_iter3.pt'
    for name in ('BEATs_iter3','dcase2026_nabeats_seed0'):
        embedding_file=out/(name+'.npz')
        if embedding_file.exists():
            arrays=dict(np.load(embedding_file));print('reuse embeddings',name,flush=True)
        else:
            if name=='BEATs_iter3':
                ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
                model=BEATs(BEATsConfig(ckpt['cfg']))
                model.load_state_dict(ckpt['model'],strict=True)
                model=model.eval().to(a.device)
            else:model=restore_model(model_id=name,device=a.device)
            arrays={key:[None]*len(entries) for key in ('global_ap','frequency_ap','frequency_rdp4')}
            from collections import defaultdict
            duration_buckets=defaultdict(list)
            for i,e in enumerate(entries):
                duration_buckets[sf.info(data/e['path']).frames].append(i)
            batches=[]
            for indices in duration_buckets.values():
                batches.extend(indices[i:i+16] for i in range(0,len(indices),16))
            completed=0
            with torch.inference_mode():
                # Exact-length buckets preserve every sample; no cropping/padding.
                for indices in batches:
                    waves=[]
                    for i in indices:
                        e=entries[i];wav,sr=sf.read(data/e['path'],dtype='float32',always_2d=True)
                        assert sr==16000 and wav.shape[1]==2,(e['path'],sr,wav.shape)
                        waves.append(wav)
                    near=torch.from_numpy(np.stack([w[:,0] for w in waves])).to(a.device)
                    far=torch.from_numpy(np.stack([w[:,1] for w in waves])).to(a.device)
                    precision=torch.autocast('cuda',dtype=torch.bfloat16) if a.device=='cuda' else nullcontext()
                    with precision:
                        z=model.extract_features(near)[0] if name=='BEATs_iter3' else model(near,far)
                    assert torch.isfinite(z).all(),('nonfinite embeddings',name,indices)
                    if completed==0 and a.device=='cuda':
                        ref=model.extract_features(near[:1])[0] if name=='BEATs_iter3' else model(near[:1],far[:1])
                        cosine=float(torch.nn.functional.cosine_similarity(z[:1].float().flatten(1),ref.float().flatten(1)).item())
                        precision_checks.append({'model':name,'batch_bf16_vs_single_fp32_cosine':cosine,'threshold':.999})
                        assert cosine>.999,('mixed precision parity failed',precision_checks[-1])
                        del ref
                    for key,value in pool(z).items():
                        for i,v in zip(indices,value):arrays[key][i]=v
                    completed+=len(indices)
                    if completed%100<16:print('ASD embedded',name,completed,len(entries),flush=True)
            arrays={k:np.stack(v) for k,v in arrays.items()}
            np.savez_compressed(embedding_file,**arrays)
            del model
            if a.device=='cuda':torch.cuda.empty_cache()
        for machine in machines:
            train=[i for i,e in enumerate(entries) if e['machine']==machine and e['split']=='train']
            test=[i for i,e in enumerate(entries) if e['machine']==machine and e['split']=='test']
            for variant in ('global_ap','frequency_ap','beam_ap','beam_rdp4'):
                features=arrays['frequency_rdp4' if variant=='beam_rdp4' else 'global_ap' if variant=='global_ap' else 'frequency_ap']
                if variant in ('global_ap','frequency_ap'):
                    features=l2(features.reshape(len(entries),-1))
                    scores=1-(features[test]@features[train].T).max(axis=1)
                else:
                    features=l2(features)
                    scores=np.mean([1-(features[test,f,:]@features[train,f,:].T).max(axis=1) for f in range(8)],axis=0)
                labels=[entries[i]['label'] for i in test];domains=[entries[i]['domain'] for i in test]
                values=metrics(labels,scores,domains)
                all_metrics.append({'model':name,'machine':machine,'variant':variant,
                                    'n_train':len(train),'n_test':len(test),**values})
                for idx,score in zip(test,scores):
                    score_rows.append({**entries[idx],'model':name,'variant':variant,'anomaly_score':float(score)})
    aggregate=[]
    for model in ('BEATs_iter3','dcase2026_nabeats_seed0'):
        for variant in ('global_ap','frequency_ap','beam_ap','beam_rdp4'):
            group=[r for r in all_metrics if r['model']==model and r['variant']==variant]
            components=[r[k] for r in group for k in ('auc_source','auc_target','pauc_standardized_fpr_0_1')]
            aggregate.append({'model':model,'variant':variant,'harmonic_auc_source_target_pauc':len(components)/sum(1/x for x in components),'machines':len(group)})
    write_json(out/'metrics.json',{'dataset':'DCASE2026 development','machines':machines,'n':len(entries),
                                 'domain_auc_definition':'official_domain_normals_vs_all_anomalies',
                                 'metrics':all_metrics,'aggregate':aggregate,'runtime':runtime(a.device),
                                 'seconds':time.monotonic()-start,'far_mic_usage':'auxiliary noise input only for NA-BEATs',
                                 'precision':'CUDA autocast bfloat16; float32 pooling and cosine scoring',
                                 'batching':'at most 16 exact-duration clips, no temporal padding or cropping',
                                 'precision_checks':precision_checks,
                                 'scope':'Released seed0 frontend, normal-only 1NN banks, frequency BEAM-style cosine score, RDP gamma4. No discriminative fine-tuning, score rescaling or three-seed ensemble; not the full winning system.'})
    (out/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in score_rows))
    print('ASD COMPLETE',aggregate,flush=True)


if __name__=='__main__':main()
