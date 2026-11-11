"""Deployment-start replay with common annotation designs and explicit feedback semantics."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from policies import evaluate,POLICIES
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results/cora'
SCENARIOS={
 'cic17':[('C17 Thursday','dos','cross_family'),('C17 Friday','bot','cross_family')],
 'cic18':[('C18 Hulk','dos','cross_family'),('C18 Bot','bot','cross_family')],
 'unsw':[('UNSW','test','temporal'),('UNSW to ToN','transfer','cross_corpus')],
 'ton':[('ToN','test','temporal'),('ToN to UNSW','transfer','cross_corpus')],
 'cic17_thu':[('C17 local Thu','test','capture_local')],
 'cic17_fri':[('C17 local Fri','test','capture_local')],
 'cic18_hulk':[('C18 local Hulk','test','capture_local')],
 'cic18_bot':[('C18 local Bot','test','capture_local')]}
def registry():
 return [dict(source=src,scenario=scenario,split=split,regime=regime,model=m,seed=s)
   for src,items in SCENARIOS.items() for scenario,split,regime in items
   for m in ['logistic','hgb','cnn'] for s in range(500,505)]
def identifier(job):return '__'.join(str(job[k]).replace(' ','_') for k in ['source','split','model','seed'])
def arrays(job):
 with np.load(OUT/'scores'/f"{job['source']}__{job['model']}__s{job['seed']}.npz") as z:
  split=job['split'];o=np.argsort(z[split+'_t'],kind='stable')
  return z[split+'_p'][o].astype(float),z[split+'_y'][o],z[split+'_t'][o],z['ref_p'][z['ref_y']==0].astype(float),z[split+'_known'][o]
def settings():
 configs=[dict(q=q,delay=d,r=r) for q in [.01,.05,.1,.25,1.] for d in [0,100,1000] for r in [100,300,1000]]
 configs += [dict(q=1.,delay=0,r=r,alpha=.001) for r in [1000,3000]]
 configs += [dict(q=.1,delay=0,r=300,delay_seconds=s) for s in [300,3600]]
 configs += [dict(q=q,delay=d,r=300,feedback='issued') for q in [.1,1.] for d in [100,1000]]
 configs += [dict(q=.1,delay=100,r=300,query_design='bernoulli')]
 return configs

def one(job,force=False):
 dest=OUT/'grid'/(identifier(job)+'.json')
 if dest.exists() and not force:return
 p,y,t,source,known=arrays(job);seconds=t/(1e9 if job['source'].startswith('cic') else 1e3)
 duration=float(seconds[-1]-seconds[0]);rows=[]
 for config in settings():
  keep=config in [dict(q=1.,delay=0,r=300),dict(q=.1,delay=100,r=300),dict(q=1.,delay=0,r=1000)]
  result,trace=evaluate(p,y,source,seed=job['seed'],keep_trace=keep,timestamps=seconds,**config)
  result.update(job);result['duration_seconds']=duration
  result['source_benign_reference']=min(len(source),1000)
  if keep:
   cut=result['activation'];nb=int((y==0).sum());na=int(y.sum())
   result['warmup_attacks']=int(y[:cut].sum()) if cut>=0 else na
   result['warmup_benign']=int((y[:cut]==0).sum()) if cut>=0 else nb
   result['warmup_seconds']=float(seconds[cut]-seconds[0]) if cut>=0 else duration
   for j,name in enumerate(POLICIES):
    m=result['methods'][name];m['false_alarms_per_hour']=m['fp']/(duration/3600)
    m['alerts_per_hour']=(m['fp']+m['tp'])/(duration/3600)
    for group,mask in [('known',(y==1)&known.astype(bool)),('unseen',(y==1)&~known.astype(bool))]:
     m[group+'_attacks']=int(mask.sum());m[group+'_tpr']=float(trace[:,j][mask].mean()) if mask.any() else None
    if cut>=0:
     for cls,key in [(0,'fpr'),(1,'tpr')]:
      mask=y[cut:]==cls;m['suffix_'+key]=float(trace[cut:,j][mask].mean()) if mask.any() else None
     m['prefix_fp']=int(trace[:cut,j][y[:cut]==0].sum());m['prefix_tp']=int(trace[:cut,j][y[:cut]==1].sum())
   np.savez_compressed(OUT/'traces'/f"{identifier(job)}__q{config['q']}__d{config['delay']}__r{config['r']}.npz",predictions=trace,y=y,t=seconds,known=known)
  rows.append(result)
 dest.write_text(json.dumps(rows,indent=2))
 print('replay',identifier(job),flush=True)

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--force',action='store_true');args=ap.parse_args()
 for p in [OUT/'grid',OUT/'traces']:p.mkdir(parents=True,exist_ok=True)
 protocol=dict(primary_reference=300,reference_sensitivity=[100,300,1000],alpha=.01,np_delta=.05,
  query_rates=[.01,.05,.1,.25,1.],delays_windows=[0,100,1000],clock_delays_seconds=[300,3600],
  policies=POLICIES,feedback_default='arrival',feedback_sensitivity='issued for ACI, DtACI expert feedback, QT; UP unchanged',
  horizon_free_sensitivity='Bernoulli querying, q=.1,D=100',decisions='all target windows, source fallback on failed acquisition',
  capture_local='first 60% source pool, stratified 70/30 source train/validation; final 40% chronological deployment; exploratory design')
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 with ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(lambda j:one(j,force=args.force),registry()))
 print('Completed',len(registry()),'replays x',len(settings()),'settings',flush=True)
if __name__=='__main__':main()
