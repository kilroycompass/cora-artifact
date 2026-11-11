"""Measured TIFS extension: stronger scorer, independent query seeds and certificates.

Uses the frozen settings in provenance/tifs_extension_protocol.json. Existing
score files are read only. Every target decision, including acquisition, counts.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import argparse,hashlib,json,time
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
from learning import Scaler,summary,partition_validation,sample_train
from train import datasets,orient
from replay import registry,SCENARIOS
from policies import evaluate,cutoff,POLICIES
from acquisition_certificate import certificate

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/tifs'
PLAN=json.loads((ROOT/'provenance/tifs_extension_protocol.json').read_text())
QUERIES=PLAN['query_seeds']


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()


def fit_tabular(tr,va,targets,model_name,seed,fit_id,metadata):
    """All selection, scaling, checkpoint choices and orientation use source only."""
    score=OUT/'scores'/(fit_id+'.npz');log=OUT/'training'/(fit_id+'.json')
    if score.exists() and log.exists():return score
    for name in ('scores','training','models'): (OUT/name).mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    if len(np.unique(tr['y']))<2 or len(np.unique(va['y']))<2:
        log.write_text(json.dumps(dict(fit_id=fit_id,supported=False,reason='historical fitting/validation lacks a class',**metadata),indent=2))
        return None
    select,cal=partition_validation(va['y'])
    ids=sample_train(tr['y'],32768,seed)
    scaler=Scaler().fit(tr['x'][ids])
    x=summary(scaler.transform(tr['x'][ids]));xs=summary(scaler.transform(va['x'][select]));ys=va['y'][select]
    if model_name=='xgb':
        grid=[dict(max_depth=d,learning_rate=l) for d in (3,6) for l in (.05,.1)]
    else:grid=[dict(max_leaf_nodes=l,learning_rate=.05) for l in (15,63)]
    best=-1.;trace=[];fitted=None;direction=1;chosen={}
    with threadpool_limits(2):
        for config in grid:
            if model_name=='xgb':
                model=XGBClassifier(n_estimators=400,tree_method='hist',subsample=.8,colsample_bytree=.8,
                                    reg_lambda=1,n_jobs=2,random_state=seed,eval_metric='logloss',**config)
            else:model=HistGradientBoostingClassifier(max_iter=400,l2_regularization=1.,early_stopping=False,random_state=seed,**config)
            model.fit(x,tr['y'][ids]);candidate_direction,ap=orient(model.predict_proba(xs)[:,1],ys)
            trace.append(dict(configuration=config,selection_ap=float(ap),orientation=candidate_direction))
            if ap>best:best=ap;fitted=model;direction=candidate_direction;chosen=config
    def score_array(a):
        pieces=[]
        for begin in range(0,len(a),8192):
            pieces.append(fitted.predict_proba(summary(scaler.transform(a[begin:begin+8192])))[:,1])
        return np.concatenate(pieces).astype(np.float32)
    raw=score_array(va['x'][cal]);ref=raw if direction==1 else 1-raw
    store=dict(ref_p=ref,ref_raw=raw,ref_y=va['y'][cal].astype(np.int8))
    diagnostics={}
    for split,d in targets:
        raw=score_array(d['x']);p=raw if direction==1 else 1-raw
        store[split+'_p']=p;store[split+'_raw']=raw;store[split+'_y']=d['y'].astype(np.int8)
        store[split+'_t']=d['t_end'][:,-1]
        diagnostics[split]=dict(n=len(p),benign=int(np.sum(d['y']==0)),attacks=int(np.sum(d['y']==1)),
            auc=float(roc_auc_score(d['y'],p)) if len(np.unique(d['y']))==2 else None,
            ap=float(average_precision_score(d['y'],p)) if np.any(d['y']==1) else None)
    np.savez_compressed(score,**store)
    joblib.dump(dict(model=fitted,scaler=scaler,orientation=direction),OUT/'models'/(fit_id+'.joblib'))
    record=dict(fit_id=fit_id,model=model_name,fit_seed=seed,supported=True,train_windows=len(ids),
        available_training_windows=len(tr['y']),selection_windows=len(select),calibration_windows=len(cal),
        source_benign_reference=int(np.sum(va['y'][cal]==0)),orientation=direction,configuration=chosen,
        trace=trace,source_auc=float(roc_auc_score(va['y'][cal],ref)),targets=diagnostics,
        seconds=time.perf_counter()-start,protocol_sha256=digest(ROOT/'provenance/tifs_extension_protocol.json'),**metadata)
    log.write_text(json.dumps(record,indent=2));print('fit',fit_id,round(record['seconds'],1),'s',flush=True)
    return score


def original_xgb_one(job):
    source,seed=job;fit_id=f'original__{source}__xgb__{seed}'
    if (OUT/'scores'/(fit_id+'.npz')).exists() and (OUT/'training'/(fit_id+'.json')).exists():return
    tr,va,targets,_=datasets(source)
    fit_tabular(tr,va,targets,'xgb',seed,fit_id,dict(experiment='stronger_scorer',source=source))


def original_xgb():
    jobs=[(s,k) for s in PLAN['existing_sources'] for k in PLAN['xgboost_fit_seeds']]
    with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(original_xgb_one,jobs))


def original_jobs():
    jobs=[]
    for j in registry():
        if j['model']!='cnn' and j['seed']!=500:continue
        job=dict(j);job['fit_seed']=job.pop('seed');job['experiment']='existing_unique_scores'
        job['score_path']=(ROOT/'experiments/results/cora/scores'/f"{job['source']}__{job['model']}__s{job['fit_seed']}.npz").relative_to(ROOT).as_posix()
        job['time_divisor']=1e9 if job['source'].startswith('cic') else 1e3;jobs.append(job)
    for source,items in SCENARIOS.items():
        for seed in PLAN['xgboost_fit_seeds']:
            path=OUT/'scores'/f'original__{source}__xgb__{seed}.npz'
            if not path.exists():continue
            for scenario,split,regime in items:
                jobs.append(dict(source=source,scenario=scenario,split=split,regime=regime,model='xgb',fit_seed=seed,
                    experiment='stronger_scorer',score_path=path.relative_to(ROOT).as_posix(),time_divisor=1e9 if source.startswith('cic') else 1e3))
    return jobs


def audit_one(job):
    name='__'.join(str(job[k]).replace(' ','_') for k in ('experiment','source','split','model','fit_seed'))
    if job.get('representation'):name+='__'+job['representation']
    dest=OUT/'replays'/(name+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
    score_path=ROOT/job['score_path']
    signature=dict(protocol=digest(ROOT/'provenance/tifs_extension_protocol.json'),extension=digest(Path(__file__)),policies=digest(ROOT/'experiments/policies.py'),
                   score=digest(score_path),certificate=digest(ROOT/'experiments/acquisition_certificate.py'))
    if dest.exists() and json.loads(dest.read_text()).get('signature')==signature:return
    with np.load(score_path) as z:
        split=job['split'];order=np.argsort(z[split+'_t'],kind='stable')
        p=z[split+'_p'][order].astype(float);y=z[split+'_y'][order];t=z[split+'_t'][order]/job['time_divisor']
        source=z['ref_p'][z['ref_y']==0].astype(float)[-1000:]
    if not len(source) or len(np.unique(y))<2:
        dest.write_text(json.dumps(dict(job=job,signature=signature,supported=False,reason='source reference or target class missing'),indent=2));return
    configs=[{'delay':0,**c} for c in job.get('settings',PLAN['primary_settings'])];rows=[];certificates=[]
    initial=float(cutoff(np.sort(source),.01));fallback=p>initial
    for config in configs:
        seconds=config.get('delay_seconds');delay=config.get('delay',0)
        delivery=np.maximum(np.arange(len(p))+1,np.searchsorted(t,t+seconds,side='right')) if seconds is not None else None
        for beta in PLAN['certificate_targets']['tpr']:
            certificates.append(dict(config=config,**certificate(y,fallback,config['r'],int(config['q']*len(y)),delay,.01,beta,delivery)))
        for seed in QUERIES:
            result,pred=evaluate(p,y,source,seed=seed,keep_trace=True,timestamps=t,**config)
            activation=result['activation'];cut=activation if activation>=0 else len(y)
            for k,policy in enumerate(POLICIES):
                m=result['methods'][policy]
                m['prefix_fp']=int(np.sum(pred[:cut,k][y[:cut]==0]));m['prefix_fn']=int(np.sum(1-pred[:cut,k][y[:cut]==1]))
                for cls,metric in ((0,'fpr'),(1,'tpr')):
                    mask=y[cut:]==cls;m['suffix_'+metric]=float(np.mean(pred[cut:,k][mask])) if np.any(mask) else None
                m['false_alarms_per_hour']=m['fp']/((t[-1]-t[0])/3600) if t[-1]>t[0] else None
            result.update(query_seed=seed,warmup_seconds=float(t[activation]-t[0]) if activation>=0 else None,
                suffix_attacks=int(np.sum(y[cut:]==1)),suffix_benign=int(np.sum(y[cut:]==0)),source_reference=len(source))
            rows.append(result)
    dest.write_text(json.dumps(dict(job=job,signature=signature,supported=True,n=len(y),benign=int(np.sum(y==0)),attacks=int(np.sum(y==1)),
        target_auc=float(roc_auc_score(y,p)),target_ap=float(average_precision_score(y,p)),certificates=certificates,results=rows),indent=2))
    print('audit',name,flush=True)


def query_audit():
    jobs=original_jobs();extra=OUT/'window_jobs.json'
    if extra.exists():jobs+=json.loads(extra.read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(audit_one,jobs))
    print('Completed independently seeded query audit:',len(jobs),'score/target cases',flush=True)


def distinct_fit_audit():
    groups={}
    for path in sorted((ROOT/'experiments/results/cora/training').glob('*.json')):
        log=json.loads(path.read_text());key=(log['source'],log['model'])
        score=ROOT/'experiments/results/cora/scores'/f"{key[0]}__{key[1]}__s{log['seed']}.npz"
        h=hashlib.sha256()
        with np.load(score) as z:
            for name in sorted(k for k in z.files if k.endswith('_p')):h.update(name.encode());h.update(z[name].tobytes())
        groups.setdefault(key,[]).append(h.hexdigest())
    rows=[dict(source=k[0],model=k[1],fitting_executions=len(v),distinct_prediction_sets=len(set(v))) for k,v in groups.items()]
    (OUT/'fit_audit.json').write_text(json.dumps(dict(original_executions=sum(x['fitting_executions'] for x in rows),
        original_distinct_prediction_sets=sum(x['distinct_prediction_sets'] for x in rows),rows=rows),indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['fit','query','audit','all'],default='all');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage in ('fit','all'):original_xgb()
    if args.stage in ('audit','all'):distinct_fit_audit()
    if args.stage in ('query','all'):query_audit()


if __name__=='__main__':main()
