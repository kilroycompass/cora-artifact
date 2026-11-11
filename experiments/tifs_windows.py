"""Fixed-boundary window/grouping controls and independent-capture validation.

Window labels use any constituent attack. Clock delay, physical source cutoff,
feature fields, and training cap are held fixed across each representation sweep.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from data import windowize,nf_features,nf_splits,NF_COLS
from prepare_cic import ALIASES,norm
from tifs_extension import ROOT,OUT,fit_tabular,digest

PREP=ROOT/'data/prepared/tifs'
CASES=['c17_thu','c17_fri','unsw','bot','5g_bs1','5g_bs2']


def cic_flows(day):
    path=ROOT/'data/real/corrected_cic17'/(day+'.csv')
    mapping={norm(k):k for k in pd.read_csv(path,nrows=0).columns}
    fields=[next(mapping[k] for k in choices if k in mapping) for choices in ALIASES]
    tc=mapping['timestamp'];lc=mapping['label'];ac=mapping.get('attemptedcategory');hc=mapping['srcip']
    cols=list(dict.fromkeys(fields+[tc,lc,hc]+([ac] if ac else [])))
    df=pd.read_csv(path,usecols=cols,low_memory=False);f=df[fields].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    ts=pd.to_datetime(df[tc],dayfirst=True,format='mixed',utc=True,errors='coerce');tag=df[lc].astype(str).str.strip()
    attempt=tag.str.contains('Attempted',case=False,regex=False).to_numpy()
    if ac:attempt|=pd.to_numeric(df[ac],errors='coerce').fillna(-1).to_numpy()>=0
    y=(tag.str.casefold().ne('benign').to_numpy()&~attempt).astype(np.int8)
    t=ts.astype('int64').to_numpy()/1e9+f[:,0]/1e6
    valid=np.isfinite(f).all(1)&(f>=0).all(1)&ts.notna().to_numpy()
    host=df[hc].astype('category').cat.codes.to_numpy(np.int64)
    return dict(x=np.log1p(f[valid]).astype(np.float32),y=y[valid],t=t[valid],host=host[valid]),dict(raw_file=str(path.relative_to(ROOT)),valid_rows=int(valid.sum()),discarded=int((~valid).sum()),fields=fields)


def netflow_flows(name):
    path=ROOT/'data/real/netflow-v3'/name/(name+'.csv')
    df=pd.read_csv(path,usecols=NF_COLS,low_memory=False);f=nf_features(df);t=df['FLOW_END_MILLISECONDS'].to_numpy(np.int64)
    y=df['Label'].to_numpy(np.int8);host=df['IPV4_SRC_ADDR'].astype('category').cat.codes.to_numpy(np.int64)
    valid=np.isfinite(f).all(1)&(t>=0)&np.isin(y,[0,1]);splits=nf_splits(name,t) if name=='NF-UNSW-NB15-v3' else None
    d=dict(x=f[valid],y=y[valid],t=t[valid]/1e3,host=host[valid])
    if splits:
        d['partition']=np.where(splits['train'][valid],0,np.where(splits['validation'][valid],1,2)).astype(np.int8)
    return d,dict(raw_file=str(path.relative_to(ROOT)),valid_rows=int(valid.sum()),discarded=int((~valid).sum()),fields=NF_COLS)


def fiveg_flows(station):
    """Epoch times are re-extracted from original Argus; CSV clock strings are rejected."""
    chunks=[];records=[]
    fields=['Dur','SrcPkts','DstPkts','SrcBytes','DstBytes','SIntPkt','SIntDist','sMeanPktSz','dMeanPktSz','SrcRate','DstRate','TcpRtt']
    for path in sorted((ROOT/'data/real/5gnidd'/f'bs{station}').glob('*.csv')):
        if path.name.endswith('.times.csv'):continue
        times=path.with_suffix('.times.csv')
        if not times.exists():raise FileNotFoundError(f'Original Argus times required: {times}')
        df=pd.read_csv(path,low_memory=False);df.columns=df.columns.str.strip();clock=pd.read_csv(times)
        if len(df)!=len(clock):raise ValueError(f'Argus/labelled CSV row count mismatch: {path.name}')
        # Exact integer sequence alignment prevents attaching labels to another flow.
        if not np.array_equal(pd.to_numeric(df['Seq']).to_numpy(),pd.to_numeric(clock['Seq']).to_numpy()):
            raise ValueError(f'Argus/labelled CSV sequence mismatch: {path.name}')
        f=df[fields].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        t=pd.to_numeric(clock['LastTime'],errors='coerce').to_numpy(float)
        released_labels=df['Label'].astype(str).str.strip().str.casefold()
        y=released_labels.ne('benign').to_numpy(np.int8)
        missing=np.sum(~np.isfinite(f),axis=1)
        f=np.column_stack([np.maximum(np.nan_to_num(f,nan=0.,posinf=0.,neginf=0.),0.),missing])
        valid=np.isfinite(t)&(t>1e9)&released_labels.isin(['benign','malicious']).to_numpy()
        chunks.append(dict(x=np.log1p(f[valid]).astype(np.float32),y=y[valid],t=t[valid],host=df['SrcAddr'].astype(str).to_numpy()[valid]))
        records.append(dict(file=path.name,sha256=digest(path),epoch_file_sha256=digest(times),rows=len(df),valid=int(valid.sum())))
    if not chunks:raise ValueError('No independent 5G capture files')
    d={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]};d['host']=pd.Categorical(d['host']).codes.astype(np.int64)
    return d,dict(base_station=station,fields=fields+['missing_numeric_count'],imputation='nonfinite values to zero; negatives clipped to zero; log1p',sessions=records,timestamps='original Argus epoch, exact Seq alignment',testbed='one shared 5G testbed')


def prepare_flow(case):
    PREP.mkdir(parents=True,exist_ok=True);path=PREP/(case+'_flows.npz');meta=PREP/(case+'_flows.json')
    if path.exists() and meta.exists():return
    if case.startswith('c17'):d,m=cic_flows('thursday' if case.endswith('thu') else 'friday')
    elif case in ('unsw','bot'):d,m=netflow_flows('NF-UNSW-NB15-v3' if case=='unsw' else 'NF-BoT-IoT-v3')
    else:d,m=fiveg_flows(int(case[-1]))
    order=np.argsort(d['t'],kind='stable');d={k:v[order] for k,v in d.items()}
    if 'partition' not in d:
        boundary=float(d['t'][int(.6*len(d['t']))]);d['partition']=(d['t']>=boundary).astype(np.int8)*2;m['source_cutoff_seconds']=boundary
    np.savez_compressed(path,**d);m.update(case=case,flows=len(d['y']),begin=float(d['t'][0]),end=float(d['t'][-1]))
    meta.write_text(json.dumps(m,indent=2));print('flow cache',case,len(d['y']),flush=True)


def make_window(d,ids,length,grouping):
    # A one-flow window never mixes hosts. Use the same tie order for both
    # spellings so grouping cannot silently alter the historical split at L=1.
    if length==1:grouping='global'
    order=ids[np.lexsort((ids,d['t'][ids],d['host'][ids]))] if grouping=='host' else ids[np.argsort(d['t'][ids],kind='stable')]
    labels=d['y'][order];tag=np.where(labels==0,'Benign','Attack')
    w=windowize(d['x'][order],labels,tag,d['t'][order],length,d['host'][order] if grouping=='host' else None)
    ix=np.argsort(w['t_end'][:,-1],kind='stable');return {k:v[ix] for k,v in w.items()}


def window_fit(job):
    case,length,grouping=job;representation=f'{grouping}_L{length}';folder=PREP/(case+'__'+representation);folder.mkdir(exist_ok=True)
    if length==1 and grouping=='host':return  # same physical windows and fit as global L=1
    metadata=folder/'metadata.json'
    if not metadata.exists():
        with np.load(PREP/(case+'_flows.npz')) as z:d={k:z[k] for k in z.files}
        # Split physical flows first: no window crosses the historical/target boundary.
        target=make_window(d,np.flatnonzero(d['partition']==2),length,grouping)
        if case=='unsw':
            tr=make_window(d,np.flatnonzero(d['partition']==0),length,grouping)
            va=make_window(d,np.flatnonzero(d['partition']==1),length,grouping)
        else:
            pool=make_window(d,np.flatnonzero(d['partition']==0),length,grouping)
            if len(np.unique(pool['y']))<2 or np.min(np.bincount(pool['y'],minlength=2))<3:
                metadata.write_text(json.dumps(dict(case=case,representation=representation,supported=False,reason='historical source pool lacks sufficient class support',source_windows=len(pool['y']),source_benign=int(np.sum(pool['y']==0)),target_windows=len(target['y']),target_benign=int(np.sum(target['y']==0))),indent=2));return
            a,b=next(StratifiedShuffleSplit(n_splits=1,train_size=.7,random_state=9912).split(np.zeros(len(pool['y'])),pool['y']))
            a=np.sort(a);b=np.sort(b);tr={k:v[a] for k,v in pool.items()};va={k:v[b] for k,v in pool.items()}
        record=dict(case=case,representation=representation,supported=True,window_length=length,grouping=grouping,
            source_train=len(tr['y']),source_validation=len(va['y']),target_windows=len(target['y']),target_benign=int(np.sum(target['y']==0)),target_attacks=int(np.sum(target['y']==1)),
            discarded_target_tails=int(np.sum(d['partition']==2)-length*len(target['y'])),flow_metadata_sha256=digest(PREP/(case+'_flows.json')))
        for key,v in [('train',tr),('validation',va),('test',target)]:np.savez_compressed(folder/(key+'.npz'),**v)
        metadata.write_text(json.dumps(record,indent=2));del d;gc.collect()
    else:
        record=json.loads(metadata.read_text())
        if not record['supported']:return
        def read(name):
            with np.load(folder/(name+'.npz')) as z:return {k:z[k] for k in z.files}
        tr,va,target=read('train'),read('validation'),read('test')
    models=[('hgb',700)] if case in ('c17_thu','c17_fri','unsw') else [('hgb',700)]+[('xgb',s) for s in (700,701,702)]
    for model,seed in models:
        fit_id=f'window__{case}__{representation}__{model}__{seed}'
        try:fit_tabular(tr,va,[('test',target)],model,seed,fit_id,dict(experiment='representation_control' if case in ('c17_thu','c17_fri','unsw') else 'independent_validation',source=case,representation=representation))
        except ValueError as e:
            (OUT/'training'/(fit_id+'.json')).write_text(json.dumps(dict(fit_id=fit_id,supported=False,reason=str(e),case=case,representation=representation),indent=2))


def collect_jobs():
    jobs=[]
    for case in CASES:
        for length in (1,16,64):
            for grouping in ('global','host'):
                rep=f'{grouping}_L{length}'
                score_rep='global_L1' if length==1 and grouping=='host' else rep
                if score_rep!=rep:
                    src=PREP/(case+'__global_L1')/'metadata.json'
                    if src.exists():
                        record=json.loads(src.read_text());record.update(representation=rep,grouping='host',alias_of='global_L1',fit_reused=True)
                        dest=PREP/(case+'__'+rep);dest.mkdir(exist_ok=True);(dest/'metadata.json').write_text(json.dumps(record,indent=2))
                for model,seed in [('hgb',700)]+([('xgb',s) for s in (700,701,702)] if case not in ('c17_thu','c17_fri','unsw') else []):
                    path=OUT/'scores'/f'window__{case}__{score_rep}__{model}__{seed}.npz'
                    if path.exists():jobs.append(dict(source=case,scenario=case,split='test',regime='window_control' if case in ('c17_thu','c17_fri','unsw') else 'independent',model=model,fit_seed=seed,
                        experiment='representation_control' if case in ('c17_thu','c17_fri','unsw') else 'independent_validation',representation=rep,fit_reused=(score_rep!=rep),score_path=path.relative_to(ROOT).as_posix(),time_divisor=1.,settings=[dict(q=.1,delay=0,delay_seconds=300,r=300,feedback='issued')]))
    (OUT/'window_jobs.json').write_text(json.dumps(jobs,indent=2));print('Window score/target cases',len(jobs),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',default=CASES);args=p.parse_args()
    active=[]
    for c in args.cases:
        exclusion=PREP/(c+'_excluded.json')
        if exclusion.exists() and not json.loads(exclusion.read_text())['supported']:
            print('Excluded input',c,json.loads(exclusion.read_text())['reason'],flush=True)
            continue
        prepare_flow(c);active.append(c)
    tasks=[(case,length,grouping) for case in active for length in (1,16,64) for grouping in ('global','host') if not(length==1 and grouping=='host')]
    with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(window_fit,tasks))
    collect_jobs()


if __name__=='__main__':main()
