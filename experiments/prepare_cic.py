"""Fetch selected members of the dataset authors' corrected CIC ZIP releases.

HTTP ranges avoid downloading unrelated members of the 10-GB 2018 archive.
Each member is checked against ZIP CRC32 and its declared uncompressed length;
extracted CSV hashes and the archive URL are retained. No code is downloaded.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import io
import json
import re
import struct
import urllib.request
import zlib
import zipfile
import hashlib
import time
import numpy as np
import pandas as pd
from data import windowize

ROOT=Path(__file__).resolve().parents[1]
BASE='https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/'


def get_range(url,start,size):
    for attempt in range(5):
        try:
            q=urllib.request.Request(url,headers={'Range':f'bytes={start}-{start+size-1}'})
            with urllib.request.urlopen(q,timeout=60) as r:
                if r.status!=206 or not r.headers['Content-Range'].startswith(f'bytes {start}-'):
                    raise ValueError('Server did not honor exact byte range')
                b=r.read()
            if len(b)!=size:raise ValueError('Incomplete range')
            return b
        except Exception:
            if attempt==4:raise
            time.sleep(min(attempt+1,3))


class Remote(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.pos=0
        with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-0'}),timeout=40) as r:
            self.size=int(r.headers['Content-Range'].split('/')[-1]);r.read()
    def seekable(self):return True
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
        return self.pos
    def tell(self):return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if n<=0:return b''
        b=get_range(self.url,self.pos,n);self.pos+=len(b);return b


def acquire(corpus):
    archive='CICIDS2017_improved.zip' if corpus=='cic17' else 'CSECICIDS2018_improved.zip'
    url=BASE+archive;raw=ROOT/'data/real'/('corrected_'+corpus);raw.mkdir(parents=True,exist_ok=True)
    reader=Remote(url)
    selected=(None if corpus=='cic17' else {'Wednesday-14-02-2018.csv','Thursday-15-02-2018.csv','Friday-16-02-2018.csv','Friday-02-03-2018.csv'})
    with zipfile.ZipFile(reader) as z:members=[i for i in z.infolist() if selected is None or i.filename in selected]
    records=[]
    for info in members:
        path=raw/Path(info.filename).name
        checksum=path.with_suffix('.verified.json')
        if path.exists() and checksum.exists():records.append(json.loads(checksum.read_text()));continue
        header=get_range(url,info.header_offset,30)
        n,e=struct.unpack('<HH',header[26:30]);start=info.header_offset+30+n+e
        part_dir=raw/(path.stem+'.parts');part_dir.mkdir(exist_ok=True)
        chunk=8*1024*1024
        tasks=list(range(0,info.compress_size,chunk))
        def part(offset):
            count=min(chunk,info.compress_size-offset)
            p=part_dir/str(offset)
            if not p.exists() or p.stat().st_size!=count:p.write_bytes(get_range(url,start+offset,count))
            return p
        print('Fetching corrected',corpus,info.filename,round(info.compress_size/1e6),'MB',flush=True)
        with ThreadPoolExecutor(max_workers=10) as pool:list(pool.map(part,tasks))
        decomp=zlib.decompressobj(-15);crc=0;total=0;digest=hashlib.sha256()
        with path.open('wb') as out:
            for offset in tasks:
                b=decomp.decompress((part_dir/str(offset)).read_bytes())
                crc=zlib.crc32(b,crc);total+=len(b);digest.update(b);out.write(b)
            b=decomp.flush();crc=zlib.crc32(b,crc);total+=len(b);digest.update(b);out.write(b)
        assert total==info.file_size and crc==info.CRC,(info.filename,total,crc)
        rec=dict(archive_url=url,archive_bytes=reader.size,member=info.filename,bytes=total,
                 compressed_bytes=info.compress_size,crc32=f'{crc:08x}',sha256=digest.hexdigest())
        checksum.write_text(json.dumps(rec,indent=2));records.append(rec)
        print('Verified corrected',corpus,info.filename,total,flush=True)
    (raw/'manifest.json').write_text(json.dumps(dict(version='CNS2022 corrected release, updated 2023',files=records),indent=2))
    return raw


def norm(x):return re.sub('[^a-z0-9]','',x.casefold())


ALIASES=[['flowduration'],['totfwdpkts','totalfwdpacket','totalfwdpackets'],
 ['totbwdpkts','totalbwdpacket','totalbwdpackets'],['totlenfwdpkts','totallengthoffwdpacket','totallengthoffwdpackets'],
 ['totlenbwdpkts','totallengthofbwdpacket','totallengthofbwdpackets'],['flowiatmean'],['flowiatstd'],
 ['pktlenmean','packetlengthmean'],['pktlenstd','packetlengthstd'],['synflagcnt','synflagcount'],
 ['ackflagcnt','ackflagcount'],['fwdpktss','fwdpacketss']]


def prepare_file(path):
    header=pd.read_csv(path,nrows=0).columns.tolist();mapping={norm(c):c for c in header}
    features=[]
    for choices in ALIASES:
        found=next((mapping[c] for c in choices if c in mapping),None)
        if found is None:raise ValueError((path.name,choices,header))
        features.append(found)
    timestamp=next(mapping[c] for c in ['timestamp','flowstarttime'] if c in mapping)
    label=mapping['label'];attempt=mapping.get('attemptedcategory')
    wanted=list(dict.fromkeys(features+[timestamp,label]+([attempt] if attempt else [])))
    df=pd.read_csv(path,usecols=wanted,low_memory=False)
    f=df[features].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    ts=pd.to_datetime(df[timestamp],dayfirst=True,format='mixed',utc=True,errors='coerce')
    labels=df[label].astype(str).str.strip()
    attempted=labels.str.contains('Attempted',case=False,regex=False).to_numpy()
    if attempt:attempted|=pd.to_numeric(df[attempt],errors='coerce').fillna(-1).to_numpy()>=0
    y=(labels.str.casefold().ne('benign').to_numpy()&~attempted).astype(np.int8)
    valid=np.isfinite(f).all(1)&(f>=0).all(1)&ts.notna().to_numpy()
    end=ts.astype('int64').to_numpy()+np.rint(f[:,0]*1000).astype(np.int64)
    order=np.flatnonzero(valid)[np.argsort(end[valid],kind='stable')]
    tags=labels.to_numpy(str);tags[y==0]='Benign'
    d=windowize(np.log1p(f[order]).astype(np.float32),y[order],tags[order],end[order],64)
    rec=dict(file=path.name,flows=len(df),valid=int(valid.sum()),attempted_as_benign=int(attempted.sum()),
             windows=len(d['y']),benign=int((d['y']==0).sum()),attacks=int(d['y'].sum()),
             features=features,timestamp_field=timestamp,timestamp_begin=str(ts[valid].min()),timestamp_end=str(ts[valid].max()),
             end_begin=int(d['t_end'].min()),end_end=int(d['t_end'].max()))
    print('Prepared corrected',rec,flush=True)
    return d,rec


def main(corpus):
    raw=acquire(corpus);out=ROOT/'data/prepared'/corpus
    out.mkdir(parents=True,exist_ok=True);prepared={};meta=[]
    for path in sorted(raw.glob('*.csv')):
        if (out/(path.stem+'.npz')).exists():
            with np.load(out/(path.stem+'.npz')) as z:d={k:z[k] for k in z.files}
            rec=json.loads((out/(path.stem+'.json')).read_text())
        else:
            d,rec=prepare_file(path)
            np.savez_compressed(out/(path.stem+'.npz'),**d)
            (out/(path.stem+'.json')).write_text(json.dumps(rec,indent=2))
        prepared[path.stem]=d;meta.append(rec)
    if corpus=='cic17':
        train={k:np.concatenate([prepared['monday'][k],prepared['tuesday'][k]]) for k in prepared['monday']}
        splits={'train':train,'validation':prepared['wednesday'],'dos_test':prepared['thursday'],'bot_test':prepared['friday']}
    else:
        splits={split:prepared[day] for split,day in [('train','Wednesday-14-02-2018'),('validation','Thursday-15-02-2018'),
                ('dos_test','Friday-16-02-2018'),('bot_test','Friday-02-03-2018')]}
    for split,d in splits.items():np.savez_compressed(out/(split+'.npz'),**d)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('corpus',choices=['cic17','cic18']);args=p.parse_args()
    main(args.corpus)
