"""Fetch publisher-hosted IoT-23 labelled logs, excluding packet payloads.

Include every available individual conn.log.labeled <=200 MB, a resource
criterion fixed before examining labels. This is a reference-availability audit,
not a classifier-performance benchmark. Infeasible captures stay visible.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import urllib.request
import numpy as np
import pandas as pd
from data import windowize

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/real/iot23'
OUT=ROOT/'data/prepared/IoT23'
BASE='https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/'
CAP=200_000_000


def fetch(name):
    url=BASE+name+'/bro/conn.log.labeled'
    try:
        request=urllib.request.Request(url,headers={'Range':'bytes=0-0'})
        with urllib.request.urlopen(request,timeout=40) as r:
            size=int(r.headers['Content-Range'].split('/')[-1]);r.read()
        rec=dict(capture=name,url=url,bytes=size,included=size<=CAP)
        if size>CAP:return rec
        path=RAW/(name+'.log')
        if not path.exists() or path.stat().st_size!=size:
            with urllib.request.urlopen(url,timeout=60) as r,path.open('wb') as out:
                while block:=r.read(4*1024*1024):out.write(block)
        rec['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        print('fetched',name,size,flush=True)
        return rec
    except Exception as e:
        return dict(capture=name,url=url,included=False,error=str(e))


def prepare(path):
    # Logs use tabs, with some original releases separating the final labels
    # by spaces. Split all whitespace; do not treat a label as a predictor.
    cols=['ts','uid','src','sp','dst','dp','proto','service','duration','ob','rb','state',
          'lo','lr','miss','hist','op','oip','rp','rip','tunnel','label','detail']
    df=pd.read_csv(path,sep=r'\s+',comment='#',names=cols,engine='c',low_memory=False)
    numeric=df[['duration','ob','rb','op','rp','oip','rip','miss']].apply(pd.to_numeric,errors='coerce')
    ts=pd.to_numeric(df['ts'],errors='coerce').to_numpy(float)
    labels=df['label'].astype(str).str.casefold()
    valid=np.isfinite(ts)&labels.isin(['benign','malicious']).to_numpy()
    missing=numeric.isna().sum(axis=1).to_numpy(float)
    f=np.column_stack([numeric.fillna(0).to_numpy(float),missing])
    f=np.log1p(np.clip(f,0,None)).astype(np.float32)
    duration=numeric['duration'].fillna(0).to_numpy(float)
    end=np.rint((ts+duration)*1e6).astype(np.int64)
    order=np.flatnonzero(valid)[np.argsort(end[valid],kind='stable')]
    y=(labels.to_numpy()!='benign').astype(np.int8)
    detail=df['detail'].astype(str).to_numpy();detail[y==0]='Benign'
    d=windowize(f[order],y[order],detail[order],end[order],64)
    by_length={};host_audit={}
    host=df["src"].astype("category").cat.codes.to_numpy()
    host_order=np.flatnonzero(valid)[np.lexsort((np.flatnonzero(valid),end[valid],host[valid]))]
    group=host[host_order];ys=y[host_order]
    starts=np.r_[0,np.flatnonzero(group[1:]!=group[:-1])+1];ends=np.r_[starts[1:],len(group)]
    for length in [1,16,64]:
        labels_ordered=y[order][:len(order)//length*length].reshape(-1,length).max(1)
        by_length[str(length)]=dict(windows=len(labels_ordered),benign=int((labels_ordered==0).sum()))
        total=benign=0
        for a,b in zip(starts,ends):
            cut=(b-a)//length*length
            labels_host=ys[a:a+cut].reshape(-1,length).max(1)
            total+=len(labels_host);benign+=int((labels_host==0).sum())
        host_audit[str(length)]=dict(windows=total,benign=benign)
    return d,dict(flows=int(valid.sum()),invalid=int((~valid).sum()),windows=len(d['y']),
                  positives=int(d['y'].sum()),benign=int((d['y']==0).sum()),
                  benign_flows=int(((y==0)&valid).sum()),window_length_audit=by_length,host_window_audit=host_audit,
                  begin=int(d['t_end'].min()) if len(d['y']) else None,
                  end=int(d['t_end'].max()) if len(d['y']) else None)


def main():
    RAW.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    html=urllib.request.urlopen(BASE,timeout=40).read().decode()
    names=sorted(set(re.findall(r'href="(CTU-[^"/]+)/"',html)))
    with ThreadPoolExecutor(max_workers=5) as pool:records=list(pool.map(fetch,names))
    (RAW/'manifest.json').write_text(json.dumps(dict(publisher='https://www.stratosphereips.org/datasets-iot23',
        inclusion='all publisher individual conn.log.labeled files <=200,000,000 bytes',files=records),indent=2))
    metadata=[]
    for rec in records:
        if not rec['included']:continue
        d,m=prepare(RAW/(rec['capture']+'.log'))
        if not len(d['y']):continue
        np.savez_compressed(OUT/(rec['capture']+'.npz'),**d)
        metadata.append(dict(capture=rec['capture'],**m))
        print('prepared',rec['capture'],m,flush=True)
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
