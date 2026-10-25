"""Evaluate five official AudioMosaic fold-specific ESC-50 classifiers."""
import argparse
import csv
import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from common import runtime, setup, sha, wilson, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--preprocessing', choices=['training','hf'], default='training')
    a=p.parse_args(); setup(a.device)
    out=a.root/'results/esc50'; out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.root/'data/ESC-50.zip') as z:
        prefix=next(n.split('/')[0] for n in z.namelist() if n.endswith('/meta/esc50.csv'))
        data=a.root/'data'/prefix
        if not data.exists():
            z.extractall(a.root/'data')
    items=list(csv.DictReader((data/'meta/esc50.csv').open()))
    assert len(items)==2000 and len({r['filename'] for r in items})==2000
    rows=[]; stats=[]; start=time.monotonic(); receipts=[]
    for fold in range(1,6):
        repo=a.root/'models'/f'AudioMosaic-vit-b16-finetune-esc-split{fold}'
        cfg=json.loads((repo/'config.json').read_text())
        prep=cfg.pop('audio_preprocessing'); cfg.pop('model_class')
        if a.preprocessing=='training':
            # Author training dataset/audioset.py line 364 divides by std*2.
            # Released modeling.preprocess divides only by std.
            prep['norm_std']*=2
        spec=importlib.util.spec_from_file_location('audiomosaic_fold'+str(fold),repo/'modeling.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        model=module.AudioMosaicClassifier(**cfg)
        model.load_state_dict(load_file(repo/'model.safetensors'), strict=True)
        model=model.eval().to(a.device)
        heldout=[r for r in items if int(r['fold'])==fold]
        assert len(heldout)==400
        # Each checkpoint is evaluated only on its advertised held-out fold.
        for offset in range(0,len(heldout),8):
            batch=heldout[offset:offset+8]; tensors=[]
            for r in batch:
                audio,sr=sf.read(data/'audio'/r['filename'],dtype='float32',always_2d=True)
                tensors.append(module.preprocess((torch.from_numpy(audio.T.copy()),sr),**prep))
            x=torch.cat(tensors).to(a.device)
            with torch.inference_mode():
                logits=model(x).float().cpu()
            for r,logit in zip(batch,logits):
                rows.append({'filename':r['filename'],'fold':fold,'target':int(r['target']),
                             'category':r['category'],'prediction':int(logit.argmax()),
                             'confidence':float(logit.softmax(-1).max()),
                             'audio_sha256':sha(data/'audio'/r['filename'])})
            if offset%80==0: print('ESC',fold,offset+len(batch),'/400',flush=True)
        rfold=[r for r in rows if r['fold']==fold]
        k=sum(r['target']==r['prediction'] for r in rfold)
        stats.append({'fold':fold,'n':400,'correct':k,'accuracy':k/400,'wilson95':wilson(k,400)})
        receipts.append({'fold':fold,'checkpoint_sha256':sha(repo/'model.safetensors'),
                         'source_code_sha256':sha(repo/'modeling.py'),'preprocessing':prep})
        del model
        if a.device=='cuda':torch.cuda.empty_cache()
        print('ESC fold result',stats[-1],flush=True)
    y=[r['target'] for r in rows]; pred=[r['prediction'] for r in rows]
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result={'benchmark':'ESC-50 official five-fold released-checkpoint inference',
            'n':len(rows),'correct':sum(np.asarray(y)==pred).item(),
            'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),
            'wilson95_clip_descriptive':wilson(sum(np.asarray(y)==pred).item(),len(rows)),
            'folds':stats,'confusion_matrix':confusion_matrix(y,pred).tolist(),
            'dataset_archive_sha256':sha(a.root/'data/ESC-50.zip'),
            'metadata_sha256':sha(data/'meta/esc50.csv'),'checkpoints':receipts,
            'seconds':time.monotonic()-start,'runtime':runtime(a.device),
            'preprocessing_protocol':a.preprocessing,
            'training_preprocessing_source':'HanxunH/AudioMosaic@67a86a8900ce00fb653f1c22f771a9688ec9de33 dataset/audioset.py:364',
            'scope':'No retraining; published fold checkpoints may have used test folds for upstream selection. No local test-based tuning.'}
    write_json(out/'metrics.json',result)
    print('ESC COMPLETE',result['accuracy'],result['seconds'],flush=True)


if __name__=='__main__':main()
