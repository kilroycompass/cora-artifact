"""Stronger-source fitting, with convergence traces and full native test streams."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
import time
import copy
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from learning import Scaler, summary, predict, partition_validation, sample_train, seed_all, build_cnn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
import joblib

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'data/prepared'
OUT=ROOT/'experiments/results/cora'
SOURCES={'cic17':PREP/'cic17','cic18':PREP/'cic18',
 'unsw':PREP/'NF-UNSW-NB15-v3','ton':PREP/'NF-ToN-IoT-v3'}
LOCAL={'cic17_thu':('cic17','dos_test'),'cic17_fri':('cic17','bot_test'),
       'cic18_hulk':('cic18','dos_test'),'cic18_bot':('cic18','bot_test')}

def orient(p,y):
 a=float(average_precision_score(y,p));b=float(average_precision_score(y,1-p))
 return (-1,b) if b>a else (1,a)

def datasets(source):
 if source in LOCAL:
  corpus,split=LOCAL[source];d=read(SOURCES[corpus]/(split+'.npz'));cut=int(.6*len(d['y']))
  ids=np.arange(cut);tr,va=next(StratifiedShuffleSplit(n_splits=1,train_size=.7,random_state=9912).split(np.zeros(cut),d['y'][:cut]))
  tr=np.sort(tr);va=np.sort(va)
  return ({k:v[tr] for k,v in d.items()},{k:v[va] for k,v in d.items()},
          [('test',{k:v[cut:] for k,v in d.items()})],set('+'.join(d['tags'][:cut]).split('+'))-{'Benign'})
 tr=read(SOURCES[source]/'train.npz');va=read(SOURCES[source]/'validation.npz')
 items=([('dos',SOURCES[source]/'dos_test.npz'),('bot',SOURCES[source]/'bot_test.npz')] if source.startswith('cic') else
        [('test',SOURCES[source]/'test.npz'),('transfer',SOURCES['ton' if source=='unsw' else 'unsw']/'test.npz')])
 families=set('+'.join(np.r_[tr['tags'],va['tags']]).split('+'))-{'Benign'}
 return tr,va,[(split,read(path)) for split,path in items],families


def read(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def fit_one(job):
    source,name,seed=job
    dest=OUT/'scores'/f'{source}__{name}__s{seed}.npz'
    log=OUT/'training'/f'{source}__{name}__s{seed}.json'
    if dest.exists() and log.exists():return
    torch.set_num_threads(2)
    seed_all(10000+seed)
    t0=time.perf_counter()
    tr,va,targets,families=datasets(source)
    select,cal=partition_validation(va['y'])
    cap=32768 if name=='cnn' else None
    ids=sample_train(tr['y'],cap,seed)
    scaler=Scaler().fit(tr['x'][ids])
    x=scaler.transform(tr['x'][ids]);y=tr['y'][ids]
    xv=scaler.transform(va['x'][select]);yv=va['y'][select]
    trace=[];config={};direction=1
    with threadpool_limits(2):
        if name=='cnn':
            model=build_cnn(x.shape[1])
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
            xx=torch.from_numpy(x);yy=torch.tensor(y,dtype=torch.float32)
            rng=np.random.default_rng(seed);best=-1;state=None;last=0
            for epoch in range(1,61):
                model.train();order=rng.permutation(len(y));total=0.
                for start in range(0,len(y),256):
                    ix=order[start:start+256];opt.zero_grad(set_to_none=True)
                    loss=F.binary_cross_entropy_with_logits(model(xx[ix]),yy[ix])
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    opt.step();total+=float(loss.detach())*len(ix)
                candidate_direction,ap=orient(predict(model,xv),yv)
                trace.append(dict(epoch=epoch,loss=total/len(y),selection_ap=ap,orientation=candidate_direction))
                if ap>best+1e-5:best=ap;state=copy.deepcopy(model.state_dict());last=epoch;direction=candidate_direction
                if epoch-last>=10:break
            model.load_state_dict(state);model.eval()
            config=dict(cap=cap,lr=.001,max_epochs=60,patience=10,selected_epoch=last)
            torch.save(dict(state_dict=state,scaler_mean=scaler.mean,scaler_sd=scaler.sd,orientation=direction),
                       log.with_suffix('.pt'))
        else:
            sx,sv=summary(x),summary(xv)
            grids=([dict(C=c) for c in [.01,.1,1.,10.,100.]] if name=='logistic' else
                   [dict(max_iter=400,max_leaf_nodes=l,learning_rate=r)
                    for l in [15,63] for r in [.05,.1]])
            best=-1;model=None
            for g in grids:
                candidate=(LogisticRegression(max_iter=2000,random_state=seed,**g) if name=='logistic' else
                           HistGradientBoostingClassifier(l2_regularization=1.,early_stopping=False,random_state=seed,**g))
                candidate.fit(sx,y)
                ap=float(average_precision_score(yv,candidate.predict_proba(sv)[:,1]))
                candidate_direction,ap=orient(candidate.predict_proba(sv)[:,1],yv)
                trace.append(dict(configuration=g,selection_ap=ap,orientation=candidate_direction))
                if ap>best:best=ap;model=candidate;config=g;direction=candidate_direction
    raw=predict(model,scaler.transform(va['x'][cal])).astype(np.float32)
    store=dict(ref_p=raw if direction==1 else 1-raw,ref_raw=raw,ref_y=va['y'][cal].astype(np.int8))
    for split,d in targets:
        raw=predict(model,scaler.transform(d['x'])).astype(np.float32)
        store[split+'_p']=raw if direction==1 else 1-raw;store[split+'_raw']=raw
        store[split+'_y']=d['y'].astype(np.int8);store[split+'_t']=d['t_end'][:,-1]
        store[split+'_known']=np.array([not bool(set(str(tag).split('+'))-families-{'Benign'}) for tag in d['tags']],np.uint8)
        store[split+'_tags']=d['tags']
    if name!='cnn':joblib.dump(dict(model=model,scaler=scaler,orientation=direction),log.with_suffix('.joblib'))
    np.savez_compressed(dest,**store)
    log.write_text(json.dumps(dict(source=source,model=name,seed=seed,train_windows=len(ids),
        available_training_windows=len(tr['y']),selection_windows=len(select),configuration=config,
        trace=trace,orientation=direction,source_families=sorted(families),source_calibration_auc=float(roc_auc_score(store['ref_y'],store['ref_p'])),source_raw_auc=float(roc_auc_score(store['ref_y'],store['ref_raw'])),training_seconds=time.perf_counter()-t0),indent=2))
    print('strong fit',source,name,seed,'n',len(ids),'seconds',round(time.perf_counter()-t0),flush=True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--sources',nargs='+',default=list(SOURCES)+list(LOCAL));args=ap.parse_args()
    for d in [OUT/'scores',OUT/'training']:d.mkdir(exist_ok=True,parents=True)
    jobs=[(src,m,s) for src in args.sources for m in ['logistic','hgb','cnn'] for s in range(500,505)]
    with ProcessPoolExecutor(max_workers=5) as pool:list(pool.map(fit_one,jobs))
