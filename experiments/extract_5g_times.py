"""Recover epoch timestamps from original Argus files, checking released row IDs.

Uses the official ra client; on Windows an installed Ubuntu WSL distribution
can supply ra. No packets are captured and no attacks are executed.
"""
from pathlib import Path
import argparse,json,shutil,subprocess
import numpy as np
import pandas as pd
from tifs_extension import ROOT,digest


def extract(stations=(1,2)):
    raw=ROOT/'data/real/5gnidd';records=[]
    for station in stations:
        for path in sorted((raw/f'bs{station}').glob('*.csv')):
            if path.name.endswith('.times.csv'):continue
            original=raw/f'argus_bs{station}'/(path.stem+'.argus')
            if not original.exists():
                records.append(dict(station=station,session=path.name,supported=False,reason='original Argus file unavailable; abbreviated CSV clock rejected'))
                continue
            target=path.with_suffix('.times.csv')
            options=['-X','-n','-u','-p','6','-r',str(original),'-s','seq','stime','ltime','saddr:40','daddr:40','dur','-c',',']
            if shutil.which('ra'):command=['ra']+options
            else:
                wsl_path='/mnt/'+original.drive[0].lower()+original.as_posix()[2:]
                options[options.index(str(original))]=wsl_path
                command=['wsl','-d','Ubuntu-24.04','--','ra']+options
            if not target.exists():
                with target.open('wb') as output:
                    result=subprocess.run(command,stdout=output,stderr=subprocess.PIPE)
                if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
            original_csv=pd.read_csv(path,usecols=['Seq','SrcAddr','DstAddr','Dur'],low_memory=False)
            clock=pd.read_csv(target)
            assert len(original_csv)==len(clock),(path.name,len(original_csv),len(clock))
            for key in ['Seq','SrcAddr','DstAddr']:
                assert np.array_equal(original_csv[key].astype(str).str.strip().to_numpy(),clock[key].astype(str).str.strip().to_numpy()),(path.name,key)
            diff=np.abs(pd.to_numeric(original_csv['Dur'])-pd.to_numeric(clock['Dur']))
            assert np.all(diff<=2e-5),(path.name,float(diff.max()))
            assert np.all(clock['LastTime']>=clock['StartTime'])
            records.append(dict(station=station,session=path.name,supported=True,rows=len(clock),
                argus_sha256=digest(original),labelled_csv_sha256=digest(path),epoch_csv_sha256=digest(target),
                begin=float(clock['StartTime'].min()),end=float(clock['LastTime'].max()),
                alignment='Exact Seq, SrcAddr and DstAddr; duration within 20 microseconds',
                command='ra -X -n -u -p 6 -r FILE -s seq stime ltime saddr:40 daddr:40 dur -c ,'))
            print('Verified original epoch alignment',station,path.name,len(clock),flush=True)
    (raw/'epoch_alignment.json').write_text(json.dumps(dict(client='Argus ra 3.0.8.2',records=records),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stations',nargs='+',type=int,default=[1,2]);args=p.parse_args();extract(args.stations)
