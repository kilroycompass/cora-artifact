"""Generate the current manuscript from measured replay records; no manual metrics."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.stats import hypergeom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from replay import registry,identifier,OUT
from policies import POLICIES
ROOT=Path(__file__).resolve().parents[1];TEX=ROOT/'paper/generated';FIG=ROOT/'paper/figures'
SHORT={'capture_local':'Local','temporal':'Temporal','cross_family':'Family','cross_corpus':'Transfer'}
GROUPS={'capture_local':'Capture-local','temporal':'NetFlow temporal','cross_family':'Cross-family CIC','cross_corpus':'Cross-corpus NetFlow'}
def allrows(folder):return [r for f in sorted((OUT/folder).glob('*.json')) for r in json.loads(f.read_text())]
def select(rows,q=.1,d=100,r=300,feedback='arrival',design='exact',alpha=.01,seconds=None):
 return [x for x in rows if x['query_fraction']==q and x['delay']==d and x['reference']==r and x['feedback']==feedback and x['query_design']==design and x['alpha']==alpha and x['delay_seconds']==seconds]
def avg(rows,p,m):
 vals=[x['methods'][p].get(m) for x in rows];vals=[v for v in vals if v is not None]
 return float(np.mean(vals)) if vals else None
def pct(x,d=2):return '--' if x is None else f'{100*x:.{d}f}'
def table(name,caption,header,rows,spec):
 s='\\begin{table}[!t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\caption{'+caption+'}\n\\label{tab:'+name+'}\n\\begin{tabular}{'+spec+'}\\toprule\n'+header+r'\\\midrule'+'\n'
 s+='\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)
 s+='\n\\bottomrule\\end{tabular}\n\\end{table}\n';(TEX/(name+'.tex')).write_text(s)
def main():
 TEX.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
 rows=allrows('grid');assert len(rows)==180*54,len(rows)
 sparse=select(rows);full=select(rows,q=1.,d=0);score=json.loads((OUT/'score_diagnostics.json').read_text());extra=allrows('sensitivity')
 evidence=dict(fits=120,replays=180,settings=len(rows),policies=len(POLICIES),physical_capture_periods=6,regimes={})
 for g in GROUPS:
  evidence['regimes'][g]={}
  for label,subset in [('full',full),('sparse',sparse)]:
   rr=[x for x in subset if x['regime']==g]
   evidence['regimes'][g][label]={p:{m:avg(rr,p,m) for m in ['fpr','tpr','label_fraction','false_alarms_per_hour']} for p in POLICIES}
 # Source/target diagnostics, all source fits selected without target data.
 data=[];quality=[]
 for j in [j for j in registry() if j['seed']==500 and j['model']=='hgb']:
  ss=[x for x in score if x['scenario']==j['scenario']];a=ss[0]
  data.append([j['scenario'],f"{a['n']:,}",f"{a['benign']:,}",'--' if j['regime']=='cross_corpus' else pct(a['known_attacks']/a['attacks'],1)])
  quality.append([j['scenario'],f"{min(x['source_auc'] for x in ss):.2f}--{max(x['source_auc'] for x in ss):.2f}",f"{min(x['target_auc'] for x in ss):.2f}--{max(x['target_auc'] for x in ss):.2f}",pct(np.mean([x['oracle_fixed_tpr'] for x in ss]),1)])
 table('data','Complete target windows. Seen uses source-observed released attack categories; cross-corpus taxonomies are not harmonized and are marked unavailable. Local targets overlap the final 40\\% of the corresponding full capture.','Scenario & Windows & Benign & Seen (\\%)',data,'lrrr')
 table('quality','Source-oriented score diagnostics across all three scorers and five fits: AUROC ranges, and mean retrospective best fixed-threshold TPR at target FPR $\\le1\\%$. The latter uses target labels only for diagnosis.','Scenario & Source AUC & Target AUC & Best TPR (\\%)',quality,'lccr')
 for label,subset in [('full',full),('sparse',sparse)]:
  body=[]
  for g,gl in GROUPS.items():
   body.append([r'\multicolumn{4}{l}{\emph{'+gl+r'}}'])
   rr=[x for x in subset if x['regime']==g]
   for p in ['Frozen','SC','NP','ACI','DtACI','UP']:
    body.append([p,pct(avg(rr,p,'fpr')),pct(avg(rr,p,'tpr')),pct(avg(rr,p,'label_fraction'),1)])
  table('results_'+label,('Whole-deployment results; '+('$q=100\\%,D=0$' if label=='full' else '$q=10\\%,D=100$')+', $R=300$. Each regime is averaged separately. Labels are realized unique queries as a fraction of all target windows.'),'Policy & FPR (\\%) & TPR (\\%) & Labels (\\%)',body,'lrrr')
 budgets=[]
 for r in [100,300,1000]:
  for q in [.01,.1,1.]:
   rr=select(rows,q=q,r=r)
   budgets.append([r,int(q*100)]+[str(sum(x['activation']>=0 for x in rr if x['regime']==g))+'/'+str(sum(x['regime']==g for x in rr)) for g in GROUPS])
 table('budgets','Activation counts at $D=100$. L: capture-local; T: NetFlow temporal; F: cross-family; X: cross-corpus. Repeated fits share traffic and do not represent independent captures.','$R$ & $q$ (\\%) & L & T & F & X',budgets,'rrcccc')
 prefix=[];decisions=[]
 for q,d,subset in [(1.,0,full),(.1,100,sparse)]:
  for g,gl in GROUPS.items():
   rr=[x for x in subset if x['regime']==g];valid=[x for x in rr if x['activation']>=0 and x['methods']['SC'].get('suffix_fpr') is not None and x['methods']['SC'].get('suffix_tpr') is not None]
   irrev=sum(x['methods']['SC']['prefix_fp']>.01*x['methods']['SC']['benign'] for x in valid)
   false1=sum(x['methods']['SC']['suffix_fpr']<=.01 and x['methods']['SC']['fpr']>.01 for x in valid)
   false2=sum(x['methods']['SC']['suffix_fpr']<=.02 and x['methods']['SC']['suffix_tpr']>=.1 and not(x['methods']['SC']['fpr']<=.02 and x['methods']['SC']['tpr']>=.1) for x in valid)
   prefix.append([SHORT[g],int(q*100),f'{len(valid)}/{len(rr)}',irrev,false1,false2]);decisions.append(dict(regime=g,q=q,activated=sum(x['activation']>=0 for x in rr),suffix_eligible=len(valid),n=len(rr),irrecoverable_1pct=irrev,suffix_only_fpr_pass=false1,suffix_only_joint_pass=false2))
 table('prefix','SC acquisition accounting at $R=300$. Suffix $n$: replays with an acquired reference and both test classes remaining. I: prefix false alarms alone exceed 1\\% of all benign windows. F: suffix FPR $\\le1\\%$ but whole FPR $>1\\%$. J: suffix meets FPR $\\le2\\%$ and TPR $\\ge10\\%$, but the whole deployment fails that joint criterion. These are diagnostic criteria, not operational standards.','Regime & $q$ & Suffix $n$ & I & F & J',prefix,'lrrrrr')
 evidence['prefix']=decisions
 recency=[]
 for g,gl in GROUPS.items():
  for metric in ['fpr','tpr']:
   cells=[];support=0
   for sampler in ['first','recent','uniform']:
    rr=[x for x in extra if x['experiment']=='recency' and x['regime']==g and x['reference']==300 and x['sampler']==sampler and x.get(metric) is not None]
    cells.append(pct(float(np.mean([x[metric] for x in rr]))) if rr else '--')
    if sampler=='first':support=len(rr)
   recency.append([SHORT[g] if metric=='fpr' else '',metric.upper(),support]+cells)
 table('recency','Common 2,000-benign pool and fixed future suffix; retain $R=300$ earliest, latest, or uniform reference scores (ten draws). Rates in percent. $n$ counts runs with a defined rate; a missing suffix class is never assigned zero. All arms pay the same acquisition cost.','Regime & Rate & $n$ & First & Recent & Uniform',recency,'llrrrr')
 semantic=[]
 for g,gl in GROUPS.items():
  a=[x for x in select(rows,d=1000) if x['regime']==g];b=[x for x in select(rows,d=1000,feedback='issued') if x['regime']==g]
  for p in ['ACI','DtACI','QT']:
   semantic.append([SHORT[g] if p=='ACI' else '',p,pct(avg(a,p,'fpr')),pct(avg(b,p,'fpr')),pct(avg(a,p,'tpr')),pct(avg(b,p,'tpr'))])
 table('feedback','Delayed-feedback semantics, $q=10\\%,D=1,000,R=300$. A: re-score at arrival; I: update from issued decisions (and issued expert losses for DtACI). UP retains its arrival-score recurrence in both settings.','Regime & Policy & FPR A & FPR I & TPR A & TPR I',semantic,'llrrrr')
 horizon=[]
 for g,gl in GROUPS.items():
  a=[x for x in sparse if x['regime']==g];b=[x for x in select(rows,design='bernoulli') if x['regime']==g]
  horizon.append([SHORT[g],pct(avg(a,'SC','fpr')),pct(avg(b,'SC','fpr')),pct(avg(a,'SC','tpr')),pct(avg(b,'SC','tpr'))])
 table('horizon','Unknown-horizon sensitivity, SC, $q=10\\%,D=100,R=300$. E: exact fixed-horizon opportunity budget; B: independent Bernoulli opportunities. The latter does not require future stream length.','Regime & FPR E & FPR B & TPR E & TPR B',horizon,'lrrrr')
 iot=json.loads((ROOT/'data/prepared/IoT23/metadata.json').read_text())
 table('iot','Independent IoT-23 availability audit: counts of entirely benign windows. G: global completion ordering; H: source-host grouping. Capture inclusion is limited to ten available labelled logs no larger than 200 MB.','Capture & Benign flows & G16 & H16 & G64 & H64',[[x['capture'].replace('CTU-IoT-Malware-Capture-','M').replace('CTU-Honeypot-Capture-','H'),f"{x['benign_flows']:,}",x['window_length_audit']['16']['benign'],x['host_window_audit']['16']['benign'],x['benign'],x['host_window_audit']['64']['benign']] for x in iot],'lrrrrr')
 # Conditional bands for the actual regime means, sharing block weights.
 intervals=json.loads((OUT/'group_intervals.json').read_text());ci=[]
 for q in [1.,.1]:
  for g,gl in GROUPS.items():
   z=[x for x in intervals if x['regime']==g and x['query_fraction']==q and x['left']=='SC' and x['right']=='Frozen' and x['block_windows']==2000]
   cells=[]
   for m in ['fpr','tpr']:
    x=next(x for x in z if x['metric']==m);cells.append('['+pct(x['lo'])+', '+pct(x['hi'])+']')
   ci.append([SHORT[g],int(q*100)]+cells)
 table('uncertainty','Paired 95\\% temporal block bands for regime-mean SC minus Frozen, percentage points. All three backbones and five seeds; 999 resamples of 2,000-window physical-capture blocks, with shared weights for overlapping replays. These condition on recorded trajectories.','Regime & $q$ & $\\Delta$FPR band & $\\Delta$TPR band',ci,'lrcc')
 tuned=[x for x in extra if x['experiment']=='source_tuned'];controls=[]
 for g,gl in GROUPS.items():
  rr=[x for x in sparse if x['regime']==g];tt=[x for x in select(tuned) if x['regime']==g]
  for p in ['ACI','UP','QT']:
   controls.append([SHORT[g] if p=='ACI' else '',p,pct(avg(rr,p,'fpr')),pct(avg(tt,p,'fpr')),pct(avg(rr,p,'tpr')),pct(avg(tt,p,'tpr'))])
 table('tuning','Source-only controller selection sensitivity at $q=10\\%,D=100,R=300$. D: default; S: source-selected. Target labels are excluded from selection.','Regime & Policy & FPR D & FPR S & TPR D & TPR S',controls,'llrrrr')
 # Quality and exposure facts are machine-readable for manuscript verification.
 training=[json.loads(f.read_text()) for f in (OUT/'training').glob('*.json')]
 evidence['source_auc_min']=min(x['source_calibration_auc'] for x in training);evidence['orientations_reversed']=sum(x['orientation']==-1 for x in training)
 evidence['source_auc_max']=max(x['source_calibration_auc'] for x in training)
 evidence['iot_host64_max']=max(x['host_window_audit']['64']['benign'] for x in iot)
 (OUT/'paper_evidence.json').write_text(json.dumps(evidence,indent=2))
 plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
 fig,axes=plt.subplots(2,2,figsize=(6.8,5.2))
 for ax,(g,gl) in zip(axes.flat,GROUPS.items()):
  for p in ['Frozen','SC','NP','ACI','DtACI','UP']:
   points=[]
   for q in [.01,.1,1.]:
    rr=[x for x in select(rows,q=q) if x['regime']==g];points.append([100*avg(rr,p,'fpr'),100*avg(rr,p,'tpr'),avg(rr,p,'label_fraction')])
   points=np.array(points);line=ax.plot(points[:,0],points[:,1],lw=1,label=p)[0]
   ax.scatter(points[:,0],points[:,1],s=12+80*points[:,2],color=line.get_color())
  ax.set(title=gl,xlabel='Whole-deployment FPR (%)',ylabel='TPR (%)');ax.axvline(1,c='gray',ls=':',lw=1);ax.grid(alpha=.2)
 method_handles,method_labels=axes[0,0].get_legend_handles_labels()
 fig.legend(method_handles,method_labels,loc='lower center',bbox_to_anchor=(.5,.115),ncol=6,frameon=False,fontsize=8)
 costs=[0.,.01,.1,1.]
 cost_handles=[axes[0,0].scatter([],[],s=12+80*c,color='0.45') for c in costs]
 fig.legend(cost_handles,[f'{100*c:g}%' for c in costs],title='Unique labels queried (% of all target windows)',
            loc='lower center',bbox_to_anchor=(.5,.015),ncol=4,frameon=False,fontsize=8,title_fontsize=8,scatterpoints=1)
 fig.tight_layout(rect=(0,.19,1,1));fig.savefig(FIG/'frontiers.pdf');plt.close(fig)
 # Minimum exact opportunity fraction for 95% acquisition probability.
 jobs=[j for j in registry() if j['model']=='hgb' and j['seed']==500];heat=[]
 from replay import arrays
 for j in jobs:
  _,y,_,_,_=arrays(j);n=len(y);m=int(np.sum((y==0)&(np.arange(n)+101<=n-1)));line=[]
  for ref in [100,300,1000]:
   if m<ref:line.append(np.nan);continue
   lo,hi=0,n
   while lo<hi:
    b=(lo+hi)//2
    if hypergeom.sf(ref-1,n,m,b)>=.95:hi=b
    else:lo=b+1
   line.append(100*lo/n)
  heat.append(line)
 fig,ax=plt.subplots(figsize=(6.8,3.4));im=ax.imshow(np.ma.masked_invalid(heat),aspect='auto',cmap='viridis',vmin=0,vmax=100)
 ax.set_yticks(range(len(jobs)),[j['scenario'] for j in jobs],fontsize=8);ax.set_xticks(range(3),['100','300','1,000']);ax.set_xlabel('Required benign reference R')
 for i,line in enumerate(heat):
  for k,v in enumerate(line):ax.text(k,i,'infeasible' if np.isnan(v) else f'{v:.1f}%',ha='center',va='center',fontsize=8,color='black' if np.isnan(v) or v>55 else 'white')
 fig.colorbar(im,ax=ax,label='Query opportunities (% of all target windows)');fig.tight_layout();fig.savefig(FIG/'acquisition.pdf');plt.close(fig)
 # A concrete timeline selected by a stated failure diagnostic, not a best result.
 candidates=[x for x in sparse if x['model']=='hgb' and x['seed']==500 and x['activation']>=0]
 case=max(candidates,key=lambda x:x['methods']['SC']['prefix_fp']/x['methods']['SC']['benign'])
 with np.load(OUT/'traces'/f"{identifier(case)}__q0.1__d100__r300.npz") as z:pred=z['predictions'];y=z['y'];t=z['t']
 hours=(t-t[0])/3600;display_hours=min(float(hours[-1]),max(2.,3*hours[case['activation']]));draw=hours<=display_hours;fig,axes=plt.subplots(3,1,figsize=(6.8,4.3),sharex=True)
 for p in ['Frozen','SC','ACI']:
  j=POLICIES.index(p);axes[0].plot(hours[draw],np.cumsum(pred[:,j]*(y==0))[draw],label=p);axes[1].plot(hours[draw],100*np.cumsum(pred[:,j]*(y==1))[draw]/y.sum(),label=p)
 axes[0].axhline(.01*(y==0).sum(),color='red',ls=':',label='1% whole-horizon FP allowance');axes[0].set_ylabel('Cumulative false alarms');axes[1].set_ylabel('Detected / all attacks (%)')
 query=np.sort(np.random.default_rng(case['seed']+17019).choice(len(y),int(.1*len(y)),replace=False));allq=np.zeros(len(y));allq[query]=1;scq=allq.copy();scq[case['activation']:]=0
 axes[2].plot(hours[draw],np.cumsum(allq)[draw],label='Continuing queries');axes[2].plot(hours[draw],np.cumsum(scq)[draw],label='SC queries');axes[2].set(xlabel='Recorded hours from deployment start',ylabel='Unique queries')
 for ax in axes:ax.axvline(hours[case['activation']],c='black',ls='--',lw=1);ax.grid(alpha=.2);ax.legend(fontsize=7,loc='best')
 fig.suptitle(case['scenario']+'; first '+f'{display_hours:.1f}'+' recorded hours; HGB seed 500',fontsize=9);fig.tight_layout();fig.savefig(FIG/'timeline.pdf');plt.close(fig)
 evidence['case']={k:case[k] for k in ['scenario','model','seed','activation','warmup_seconds']};(OUT/'paper_evidence.json').write_text(json.dumps(evidence,indent=2))
 # Complete CSV for inspection without the Python JSON structure.
 with (OUT/'metrics.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['scenario','regime','model','seed','q','delay','reference','feedback','query_design','alpha','policy','fpr','tpr','labels'])
  for r in rows:
   for p,m in r['methods'].items():writer.writerow([r[k] for k in ['scenario','regime','model','seed','query_fraction','delay','reference','feedback','query_design','alpha']]+[p,m['fpr'],m['tpr'],m['labels']])
 print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
