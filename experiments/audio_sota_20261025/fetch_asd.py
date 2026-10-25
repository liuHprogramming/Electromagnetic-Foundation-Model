"""Fetch the official DCASE2026 dev set and published NA-BEATs seed0."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path

from common import sha,write_json

DATA={
    'dev_fan.zip':'58b79bac4b05bf14a11e9d53e6cdfae9',
    'dev_bearingEmu.zip':'82c2d04c5f48d09842152c34ef273667',
    'dev_gearboxEmu.zip':'21ef8b6dea66472bc4e7c91a175efdb7',
    'dev_sliderEmu.zip':'1cbc87a09883a1aa05a6b2a4308efb83',
    'dev_ToyCar_r2.zip':'353201db217e25377a7e7eccf9c1c8a7',
    'dev_ToyCarEmu.zip':'187b0b3f84d2452d8944d5c831305f15',
    'dev_valveEmu.zip':'f7f74e59773eefa512f24625d88d58f0',
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();jobs=[]
    for name,digest in DATA.items():
        if name=='dev_fan.zip':continue # downloaded by fetch_assets.py
        jobs.append((a.root/'data'/name,'https://zenodo.org/records/19336329/files/'+name+'?download=1',digest,'md5'))
    jobs.append((a.root/'models/nassl/dcase2026_nabeats_seed0.ckpt',
                 'https://github.com/merlresearch/noise-aware-audio-ssl/releases/download/v1.0.0/dcase2026_nabeats_seed0.ckpt',
                 'e7f5fc1fc1b6505fff8a716a61a5bce4e5f91845d825dd7c633f26cdf837cf89','sha256'))
    def get(job):
        path,url,expected,kind=job;path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix(path.suffix+'.download.log').open('a') as f:
            subprocess.run(['aria2c','-c','-x','4','-s','4','--file-allocation=none','--console-log-level=warn',
                            '--summary-interval=0','--max-tries=5','-d',str(path.parent),'-o',path.name,url],stdout=f,stderr=subprocess.STDOUT,check=True)
        h=hashlib.new(kind)
        with path.open('rb') as f:
            for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
        assert h.hexdigest()==expected,(path,h.hexdigest(),expected)
        print('VERIFIED',path.name,flush=True)
        return {'path':str(path.relative_to(a.root)),'source_url':url,'publisher_hash_type':kind,'publisher_hash':expected,'sha256':sha(path),'bytes':path.stat().st_size}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(get,jobs))
    write_json(a.root/'asd_asset_receipt.json',rows)
    base=a.root/'models/nassl/BEATs_iter3.pt'
    if not base.exists():base.symlink_to(a.root.parent/'models/beats/BEATs_iter3.pt')
    assert sha(base)=='8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34'


if __name__=='__main__':main()
