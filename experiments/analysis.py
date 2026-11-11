"""Score diagnostics, resource controls, and paired temporal sensitivity."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from scipy.stats import hypergeom,beta
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve
from replay import registry,arrays,identifier,OUT
from policies import evaluate,POLICIES

def acquisition_probability(y,budget,reference,delay=0,horizon=None):
 n=len(y);deadline=n-1 if horizon is None else horizon
 m=int(np.sum((y==0)&(np.arange(n)+delay+1<=deadline)))
 return float(hypergeom.sf(reference-1,n,m,budget))
def threshold(p,alpha=.01):
 rank=int(np.ceil((len(p)+1)*(1-alpha)))
 return np.sort(p)[rank-1] if rank<=len(p) else np.inf
def rates(p,y):
 return dict(fpr=float(p[y==0].mean()) if (y==0).any() else None,tpr=float(p[y==1].mean()) if (y==1).any() else None)
def tuning(job):
 path=OUT/'tuning'/f"{job['source']}__{job['model']}__{job['seed']}.json"
 if path.exists():return json.loads(path.read_text())['selected']
 with np.load(OUT/'scores'/f"{job['source']}__{job['model']}__s{job['seed']}.npz") as z:p=z['ref_p'].astype(float);y=z['ref_y']
 cut=int(.3*len(y));source=p[:cut][y[:cut]==0];trials=[];selected={}
 for field,policy,values in [('gamma','ACI',[.001,.005,.02,.1]),('wealth_scale','UP',[.0001,.01,1.]),('qt_scale','QT',[.001,.01,.1,1.])]:
  choices=[]
  for value in values:
   r,_=evaluate(p[cut:],y[cut:],source,q=1,delay=0,r=300,seed=job['seed'],**{field:value})
   m=r['methods'][policy];f=m['fpr'];t=m['tpr'] or 0.
   key=(0,-t,abs(f-.01)) if f<=.01 else (1,f,-t)
   choices.append((key,value));trials.append(dict(policy=policy,parameter=field,value=value,**m))
  selected[field]=min(choices)[1]
 path.write_text(json.dumps(dict(selected=selected,trials=trials),indent=2));return selected

def extra(job):
 path=OUT/'sensitivity'/(identifier(job)+'.json')
 if path.exists():return
 p,y,t,source,known=arrays(job);params=tuning(job);rows=[]
 for q,d in [(.01,0),(.1,100),(1.,0)]:
  r,_=evaluate(p,y,source,q=q,delay=d,r=300,seed=job['seed'],**params)
  rows.append(dict(experiment='source_tuned',parameters=params,**job,**r))
 for fraction in [.25,.5,.75]:
  cut=int(fraction*len(y))
  for q in [.1,1.]:
   r,_=evaluate(p[cut:],y[cut:],source,q=q,delay=100,r=300,seed=job['seed'])
   rows.append(dict(experiment='origin',fraction=fraction,**job,**r))
 ix=np.flatnonzero(y==0)
 if len(ix)>=2000 and ix[1999]+1<len(y):
  cut=int(ix[1999]+1);pool=p[:cut][y[:cut]==0]
  for size in [100,300,1000,2000]:
   for sampler in ['first','recent','uniform']:
    for rep in range(10 if sampler=='uniform' else 1):
     ref=(pool[:size] if sampler=='first' else pool[-size:] if sampler=='recent' else np.random.default_rng(job['seed']+7919*rep).choice(pool,size,replace=False))
     rows.append(dict(experiment='recency',reference=size,sampler=sampler,rep=rep,pool_queries=cut,**job,**rates(p[cut:]>threshold(ref),y[cut:])))
 else:rows.append(dict(experiment='recency_infeasible',**job))
 path.write_text(json.dumps(rows,indent=2))

def score_diagnostics():
 rows=[];laws=[];seen=set()
 for job in registry():
  p,y,t,source,known=arrays(job)
  file=OUT/'scores'/f"{job['source']}__{job['model']}__s{job['seed']}.npz"
  with np.load(file) as z:
   raw=z[job['split']+'_raw'];rp=z['ref_p'];rr=z['ref_raw'];ry=z['ref_y']
   fpr,tpr,_=roc_curve(y,p,drop_intermediate=False)
   family=sorted(set('+'.join(z[job['split']+'_tags']).split('+'))-{'Benign'})
   rows.append(dict(**job,n=len(y),benign=int((y==0).sum()),attacks=int(y.sum()),known_attacks=int(((y==1)&known.astype(bool)).sum()),unseen_attacks=int(((y==1)&~known.astype(bool)).sum()),families=family,
    source_raw_auc=float(roc_auc_score(ry,rr)),source_auc=float(roc_auc_score(ry,rp)),source_ap=float(average_precision_score(ry,rp)),target_auc=float(roc_auc_score(y,p)),target_raw_auc=float(roc_auc_score(y,raw)),target_ap=float(average_precision_score(y,p)),oracle_fixed_tpr=float(max(tpr[fpr<=.01]))))
  if job['model']=='hgb' and job['seed']==500:
   for q in [.01,.05,.1,.25,1.]:
    for ref in [99,100,299,300,1000]:
     laws.append(dict(scenario=job['scenario'],regime=job['regime'],q=q,reference=ref,n=len(y),benign=int((y==0).sum()),completion_probability=acquisition_probability(y,int(q*len(y)),ref,100)))
 (OUT/'score_diagnostics.json').write_text(json.dumps(rows,indent=2));(OUT/'acquisition_law.json').write_text(json.dumps(laws,indent=2))
 precision=[]
 for r in [99,100,299,300,1000,3000]:
  j=int(np.ceil((r+1)*.99));lo,hi=beta.ppf([.025,.975],r+1-j,j)
  precision.append(dict(reference=r,mean=(r+1-j)/(r+1),lo=float(lo),hi=float(hi)))
 (OUT/'reference_precision.json').write_text(json.dumps(precision,indent=2))

def group_intervals():
    """Shared physical-capture block weights preserve overlapping-run dependence."""
    jobs=registry();rows=[]
    def capture(j):
        src=j['source']
        if src=='cic17':return 'c17_'+j['split']
        if src=='cic18':return 'c18_'+j['split']
        if src=='cic17_thu':return 'c17_dos'
        if src=='cic17_fri':return 'c17_bot'
        if src=='cic18_hulk':return 'c18_dos'
        if src=='cic18_bot':return 'c18_bot'
        return ('unsw' if (src=='unsw' and j['split']=='test') or (src=='ton' and j['split']=='transfer') else 'ton')
    sizes={}
    for j in jobs:
        if j['seed']==500 and j['model']=='hgb':
            _,y,_,_,_=arrays(j);c=capture(j);sizes[c]=max(sizes.get(c,0),len(y))
    for q,d in [(1.,0),(.1,100)]:
      for width in [500,2000,5000]:
        rng=np.random.default_rng(19371+width);weights={}
        for c,n in sizes.items():
            k=(n+width-1)//width
            weights[c]=rng.multinomial(k,np.full(k,1/k),size=999).astype(float)
        rates_by_group={}
        for j in jobs:
            f=OUT/'traces'/f"{identifier(j)}__q{q}__d{d}__r300.npz"
            with np.load(f) as z:pred=z['predictions'].astype(np.int64);y=z['y']
            c=capture(j);w=weights[c];k=w.shape[1]
            bins=(np.arange(len(y))+sizes[c]-len(y))//width
            group=j['regime']
            for cls,metric in [(0,'fpr'),(1,'tpr')]:
                count=np.bincount(bins,weights=(y==cls),minlength=k);denom=w@count
                num=np.column_stack([np.bincount(bins,weights=pred[:,i]*(y==cls),minlength=k) for i in range(len(POLICIES))])
                value=(w@num)/np.where(denom[:,None]>0,denom[:,None],np.nan)
                rates_by_group.setdefault((group,metric),[]).append(value)
        for (regime,metric),values in rates_by_group.items():
            aggregate=np.mean(np.stack(values),axis=0)
            for left,right in [('SC','Frozen'),('ACI','SC'),('DtACI','SC'),('UP','SC')]:
                delta=aggregate[:,POLICIES.index(left)]-aggregate[:,POLICIES.index(right)];valid=np.isfinite(delta)
                lo,hi=np.quantile(delta[valid],[.025,.975]) if valid.any() else (None,None)
                rows.append(dict(regime=regime,query_fraction=q,delay=d,block_windows=width,metric=metric,left=left,right=right,lo=lo,hi=hi,valid_resamples=int(valid.sum()),resamples=999))
    (OUT/'group_intervals.json').write_text(json.dumps(rows,indent=2))

def main():
 for p in ['tuning','sensitivity','uncertainty']:(OUT/p).mkdir(parents=True,exist_ok=True)
 jobs=registry();seen=set()
 for j in jobs:
  tag=(j['source'],j['model'],j['seed'])
  if tag not in seen:tuning(j);seen.add(tag)
 with ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(extra,jobs))
 score_diagnostics();group_intervals()
 print('Score audits, controls, and shared-block uncertainty complete',flush=True)
if __name__=='__main__':main()
