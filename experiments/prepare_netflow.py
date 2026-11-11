"""Prepare uncapped complete native host-window streams from public NetFlow v3 CSVs."""
from pathlib import Path
import numpy as np
import pandas as pd
import json
from data import NF_COLS,nf_features,nf_splits,windowize
ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'data/prepared'
def native(name):
    PREP.mkdir(parents=True,exist_ok=True)
    destination=PREP/name
    if (destination/'test.npz').exists():
        print(name,'native files already exist',flush=True);return
    path=ROOT/'data/real/netflow-v3'/name/(name+'.csv')
    print('Reading complete source',name,flush=True)
    df=pd.read_csv(path,usecols=NF_COLS,low_memory=False)
    t=df['FLOW_END_MILLISECONDS'].to_numpy(np.int64)
    features=nf_features(df)
    y=df['Label'].to_numpy(np.int8)
    att=df['Attack'].astype(str).str.strip().to_numpy()
    att[y==0]='Benign'
    host=df['IPV4_SRC_ADDR'].astype('category').cat.codes.to_numpy(np.int64)
    valid=np.isfinite(features).all(1)&(t>=0)&np.isin(y,[0,1])
    masks=nf_splits(name,t)
    destination.mkdir(exist_ok=True,parents=True)
    metadata={}
    for split,m in masks.items():
        ids=np.flatnonzero(m&valid)
        order=ids[np.lexsort((ids,t[ids],host[ids]))]
        d=windowize(features[order],y[order],att[order],t[order],64,host[order])
        order=np.argsort(d['t_end'][:,-1],kind='stable')
        d={k:v[order] for k,v in d.items()}
        np.savez_compressed(destination/f'{split}.npz',**d)
        metadata[split]=dict(flows=int(len(ids)),windows=len(d['y']),positive=int(d['y'].sum()),
                             begin=int(d['t_end'].min()),end=int(d['t_end'].max()))
        print(name,split,metadata[split],flush=True)
        del d
    (destination/'metadata.json').write_text(json.dumps(dict(source=str(path.relative_to(ROOT)),
        invalid_rows=int((~valid).sum()),subsampling=False,splits=metadata),indent=2))



if __name__=='__main__':
 for name in ['NF-UNSW-NB15-v3','NF-ToN-IoT-v3']:native(name)
