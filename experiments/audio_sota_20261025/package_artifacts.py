"""Package small auditable outputs; retain large embeddings outside Git."""
import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from common import sha,write_json,git_rev


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root;out=root/'results'
    assert json.loads((out/'verification.json').read_text())['passed']
    for filename in ('environment.txt','asset_receipt.json','asd_asset_receipt.json'):
        shutil.copy2(root/filename,out/filename)
    for folder in ('moss','nassl'):
        write_json(out/(folder+'_source.json'),{'repo':folder,'commit':git_rev(root/'external'/folder)})
    write_json(out/'hardware.json',{'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True).strip(),
                                   'gpu_process_cap_gib':24,'cpu_threads_per_evaluation':8})
    logs=out/'tool_logs';logs.mkdir(exist_ok=True)
    for path in (root/'logs').glob('*.log'):
        # aria2 signed redirect URLs and data/model download logs are not needed.
        if 'download' in path.name:continue
        shutil.copy2(path,logs/(path.stem+'.txt'))
    write_json(out/'large_artifacts.json',[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p),'retained_on_remote':True} for p in sorted(out.rglob('*.npz'))])
    included=[p for p in sorted(out.rglob('*')) if p.is_file() and p.suffix not in ('.npz','.wav','.zip') and p.name!='artifact_manifest.json']
    write_json(out/'artifact_manifest.json',[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in included])
    with tarfile.open(root/'results_for_git.tar.gz','w:gz') as tar:
        for p in included+[out/'artifact_manifest.json']:
            tar.add(p,arcname=str(p.relative_to(out)),recursive=False)
    print('Packaged',len(included)+1,'files; archive bytes',(root/'results_for_git.tar.gz').stat().st_size,flush=True)


if __name__=='__main__':main()
