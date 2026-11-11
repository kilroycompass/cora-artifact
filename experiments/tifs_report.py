"""Factor-balanced tables and figures from the measured TIFS extension."""
from pathlib import Path
from collections import defaultdict
import json,csv
import numpy as np
from scipy.stats import hypergeom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tifs_extension import ROOT,OUT,QUERIES,digest

TEX=ROOT/'paper/generated';FIG=ROOT/'paper/figures'
REGIMES={'capture_local':'Local','temporal':'Temporal','cross_family':'Family','cross_corpus':'Transfer'}
CASES={'c17_thu':'C17 Thu','c17_fri':'C17 Fri','unsw':'UNSW','bot':'BoT','5g_bs1':'5G BS1'}


def balanced(rows,value):
    """Average query draws, then fits, then scorers, then scenarios equally."""
    fits=defaultdict(list)
    for r in rows:
        v=value(r)
        if v is not None and np.isfinite(v):fits[(r['scenario'],r['model'],r['fit_seed'])].append(float(v))
    models=defaultdict(list)
    for (scenario,model,seed),values in fits.items():models[(scenario,model)].append(np.mean(values))
    scenarios=defaultdict(list)
    for (scenario,model),values in models.items():scenarios[scenario].append(np.mean(values))
    return float(np.mean([np.mean(v) for v in scenarios.values()])) if scenarios else None


def mvalue(policy,metric):return lambda r:r['methods'][policy].get(metric)
def pct(v,d=2):return '--' if v is None else f'{100*v:.{d}f}'


def table(name,caption,header,rows,spec):
    s='\\begin{table}[!t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\caption{'+caption+'}\n\\label{tab:'+name+'}\n\\begin{tabular}{'+spec+'}\\toprule\n'+header+r'\\\midrule'+'\n'
    s+='\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)
    s+='\n\\bottomrule\\end{tabular}\n\\end{table}\n';(TEX/(name+'.tex')).write_text(s)


def load():
    blocks=[];rows=[];unsupported=[]
    for p in sorted((OUT/'replays').glob('*.json')):
        block=json.loads(p.read_text())
        if not block['supported']:unsupported.append(dict(job=block['job'],reason=block['reason']));continue
        blocks.append(block);j=block['job']
        for r in block['results']:
            matching=[c for c in block['certificates'] if c['config']['q']==r['query_fraction'] and c['config']['r']==r['reference'] and c['config'].get('delay',0)==r['delay'] and c['config'].get('delay_seconds')==r['delay_seconds']]
            c=next(c for c in matching if c['beta']==.1)
            impossible=(not c['fallback_pass']) and min(c['eligible_benign'],c['opportunities'])<c['reference']
            sc=r['methods']['SC'];eligible=r['activation']>=0 and r['suffix_attacks']>0 and r['suffix_benign']>0
            passed=sc['fpr']<=.01 and sc['tpr']>=.1
            if impossible:assert not passed,(p.name,r['query_seed'])
            rows.append(dict(**j,**r,certificate=c['joint_success_upper'],impossible=impossible,
                certificate50=next(c['joint_success_upper'] for c in matching if c['beta']==.5),
                prefix_exhausted=sc['prefix_fp']>.01*block['benign'],eligible_suffix=eligible,
                joint_pass=passed,joint_pass50=(sc['fpr']<=.01 and sc['tpr']>=.5),false_suffix=(eligible and sc['suffix_fpr']<=.01 and sc['fpr']>.01),
                false_suffix_joint=(eligible and sc['suffix_fpr']<=.01 and sc['suffix_tpr']>=.1 and not passed)))
    return blocks,rows,unsupported


def subset(rows,experiment,regime=None,q=.1):
    return [r for r in rows if r['experiment']==experiment and r['query_fraction']==q and (regime is None or r['regime']==regime)]


def summarize(rows):
    out={}
    for policy in ['Frozen','SC','NP','Rolling','ACI','DtACI','UP','QT']:
        out[policy]={k:balanced(rows,mvalue(policy,k)) for k in ['fpr','tpr','label_fraction']}
    for k in ['prefix_exhausted','eligible_suffix','joint_pass','joint_pass50','false_suffix','false_suffix_joint','certificate','certificate50','impossible']:
        out[k]=balanced(rows,lambda r:r[k])
    query_means={k:[balanced([r for r in rows if r['query_seed']==s],mvalue('SC',k)) for s in QUERIES] for k in ['fpr','tpr']}
    out['query_ranges']={k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in query_means.items() if v and all(x is not None for x in v)}
    return out


def figure(blocks):
    selected={}
    for b in blocks:
        j=b['job']
        if j['experiment'] in ('representation_control','independent_validation') and j['model']=='hgb':selected[(j['source'],j['representation'])]=b
    left=[];right=[];labels=[];impossible_left=[];impossible_right=[]
    for case,title in CASES.items():
        for grouping in ['global','host']:
            a=[];b=[];ia=[];ib=[]
            for length in [1,16,64]:
                block=selected[(case,f'{grouping}_L{length}')];job=block['job'];path=Path(job['score_path'])
                if not path.is_absolute():path=ROOT/path
                with np.load(path) as z:
                    t=z['test_t'].astype(float);y=z['test_y'];o=np.argsort(t,kind='stable');t=t[o];y=y[o]
                n=len(y);delivery=np.maximum(np.arange(n)+1,np.searchsorted(t,t+300,side='right'))
                m=int(np.sum((y==0)&(delivery<=n-1)));budget=int(.1*n)
                a.append(float(hypergeom.sf(299,n,m,budget)));ia.append(min(m,budget)<300)
                c=next(c for c in block['certificates'] if c['beta']==.1)
                b.append(c['joint_success_upper']);ib.append((not c['fallback_pass']) and min(c['eligible_benign'],c['opportunities'])<300)
            left.append(a);right.append(b);labels.append(title+(' G' if grouping=='global' else ' H'));impossible_left.append(ia);impossible_right.append(ib)
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.4),sharey=True,layout='constrained')
    for ax,data,flags,title in zip(axes,[left,right],[impossible_left,impossible_right],['Reference completion','Optimistic joint success']):
        values=np.asarray(data);im=ax.imshow(100*values,vmin=0,vmax=100,cmap='viridis',aspect='auto')
        ax.set_xticks(range(3),['1','16','64']);ax.set_yticks(range(len(labels)),labels);ax.set_xlabel('Flows per window');ax.set_title(title)
        for i in range(len(labels)):
            for j in range(3):
                val=100*values[i,j];label='0' if flags[i][j] else ('<0.1' if val<.1 else f'{val:.1f}')
                ax.text(j,i,label,ha='center',va='center',fontsize=7,color='black' if val>55 else 'white')
    fig.colorbar(im,ax=axes,label='Conditional probability (%)',shrink=.9)
    fig.savefig(FIG/'tifs_feasibility.pdf');plt.close(fig)


def main():
    TEX.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    blocks,rows,unsupported=load();facts=dict(query_seeds=len(QUERIES),supported_score_target_cases=len(blocks),
        query_replays=len(rows),method_results=sum(len(r['methods']) for r in rows),unsupported=unsupported,core={})
    facts['unsupported_method_results']=sum(not m['supported'] for r in rows for m in r['methods'].values())
    facts['valid_method_results']=facts['method_results']-facts['unsupported_method_results']
    fits=[json.loads(p.read_text()) for p in (OUT/'training').glob('*.json')]
    facts['additional_supported_fits']=sum(x['supported'] for x in fits)
    facts['fit_audit']=json.loads((OUT/'fit_audit.json').read_text())
    main_rows=[];full_rows=[];certificate_rows=[];facts['core_full']={}
    for regime,title in REGIMES.items():
        base=subset(rows,'existing_unique_scores',regime);xgb=subset(rows,'stronger_scorer',regime)
        a=summarize(base);b=summarize(xgb);facts['core'][regime]=dict(baseline=a,xgb=b)
        main_rows.append([r'\multicolumn{6}{l}{\emph{'+title+r'}}'])
        for policy in ['Frozen','SC','NP','ACI','DtACI','UP']:
            assert abs(a[policy]['label_fraction']-b[policy]['label_fraction'])<1e-12
            main_rows.append([policy,pct(a[policy]['fpr']),pct(a[policy]['tpr']),pct(b[policy]['fpr']),pct(b[policy]['tpr']),pct(a[policy]['label_fraction'],1)])
        af=summarize(subset(rows,'existing_unique_scores',regime,q=1.));bf=summarize(subset(rows,'stronger_scorer',regime,q=1.));facts['core_full'][regime]=dict(baseline=af,xgb=bf)
        full_rows.append([r'\multicolumn{6}{l}{\emph{'+title+r'}}'])
        for policy in ['Frozen','SC','NP','ACI','DtACI','UP']:
            full_rows.append([policy,pct(af[policy]['fpr']),pct(af[policy]['tpr']),pct(bf[policy]['fpr']),pct(bf[policy]['tpr']),pct(af[policy]['label_fraction'],1)])
        for label,summary in [('B',a),('X',b)]:
            certificate_rows.append([title,label,pct(1-summary['impossible'],1),pct(summary['certificate'],1),pct(summary['joint_pass'],1),pct(summary['false_suffix'],1)])
    table('tifs_core','Core comparison with issued-decision feedback, $q=10\\%,D=100,R=300$. B: three original scorers, balanced across scorer and unique fit; X: XGBoost (three fits). Means include 20 independent query schedules. UP retains its arrival-score adapter. Labels: realized query cost, identical for B/X. All rates are percentages.','Policy & FPR B & TPR B & FPR X & TPR X & Labels',main_rows,'lrrrrr')
    table('tifs_full','Full-feedback sensitivity, $q=100\\%,D=0,R=300$. B/X and aggregation as in Table~\\ref{tab:tifs_core}. Labels: realized queried-window percentage.','Policy & FPR B & TPR B & FPR X & TPR X & Labels',full_rows,'lrrrrr')
    table('tifs_certificate','Joint diagnostic: FPR $\\le1\\%$, TPR $\\ge10\\%$, $q=10\\%,D=100,R=300$. B/X as in Table~\\ref{tab:tifs_core}. Possible: scoring conditions not structurally ruled out by the certificate; bound: mean optimistic success probability; SC: observed joint-pass rate; false suffix: suffix FPR passes but whole-horizon FPR fails. All entries are conditional percentages, not deployment prevalence.','Regime & Set & Possible & Bound & SC & False suffix',certificate_rows,'llrrrr')
    beta_rows=[]
    for regime,title in REGIMES.items():
        for label,key in [('B','baseline'),('X','xgb')]:
            z=facts['core'][regime][key];beta_rows.append([title,label,pct(z['certificate'],1),pct(z['certificate50'],1),pct(z['joint_pass'],1),pct(z['joint_pass50'],1)])
    table('tifs_beta','Detection-target sensitivity at FPR $\\le1\\%$, $q=10\\%,D=100,R=300$. Bound: optimistic joint-success probability; SC: observed joint-pass fraction over independently seeded query schedules. B/X as in Table~\\ref{tab:tifs_core}. All entries are conditional percentages.','Regime & Set & Bound 10 & Bound 50 & SC 10 & SC 50',beta_rows,'llrrrr')
    unit_rows=[];independent_rows=[];facts['windows']={};facts['independent']={}
    for case,title in CASES.items():
        facts['windows'][case]={}
        for grouping in ['global','host']:
            for length in [1,16,64]:
                rep=f'{grouping}_L{length}';rr=[r for r in rows if r.get('representation')==rep and r['source']==case and r['model']=='hgb']
                meta=json.loads((ROOT/'data/prepared/tifs'/(case+'__'+rep)/'metadata.json').read_text());s=summarize(rr)
                unit_rows.append([title+(' G' if grouping=='global' else ' H'),length,f"{meta['target_benign']:,}",pct(balanced(rr,lambda r:r['activation']>=0),1),pct(s['SC']['fpr']),pct(s['SC']['tpr'])])
                facts['windows'][case][rep]=dict(metadata=meta,summary=s)
        if case in ('bot','5g_bs1'):
            facts['independent'][case]={}
            for model in ['hgb','xgb']:
                independent_rows.append([r'\multicolumn{5}{l}{\emph{'+title+' / '+('HGB' if model=='hgb' else 'XGBoost')+r'}}'])
                for grouping in ['global','host']:
                    for length in [1,16,64]:
                        rep=f'{grouping}_L{length}';rr=[r for r in rows if r.get('representation')==rep and r['source']==case and r['model']==model];s=summarize(rr)
                        independent_rows.append([('G' if grouping=='global' else 'H')+str(length),pct(s['Frozen']['fpr'])+'/'+pct(s['Frozen']['tpr']),pct(s['SC']['fpr'])+'/'+pct(s['SC']['tpr']),pct(s['SC']['label_fraction'],1),pct(balanced(rr,lambda r:r['activation']>=0),1)])
                        facts['independent'][case][model+'__'+rep]=s
    table('tifs_windows','SC window/grouping control with HGB, a fixed physical source cutoff, $q=10\\%$, five-minute label delay and $R=300$. G/H: global/host grouping. $N_0$: target benign windows; A: activation rate over 20 query schedules. One-flow G/H conditions reuse the same fit and tie order. FPR/TPR are whole-horizon percentages.','Capture & $L$ & $N_0$ & A (\\%) & FPR & TPR',unit_rows,'lrrrrr')
    table('tifs_independent','Independent validation, $q=10\\%$, five-minute label delay, $R=300$. G/H and window length identify the unit. Each rate cell reports FPR/TPR (\\%); Labels is SC query cost (\\% of target windows); A is activation (\\%). XGBoost averages three fits; both scorers average 20 query schedules.','Unit & Frozen & SC & Labels & A',independent_rows,'lrrrr')
    facts['new_capture_metadata']={c:json.loads((ROOT/'data/prepared/tifs'/(c+'_flows.json')).read_text()) for c in ['bot','5g_bs1']}
    original_xgb_logs=[x for x in fits if x.get('experiment')=='stronger_scorer' and x['supported']]
    facts['xgb_source_auc_range']=[min(x['source_auc'] for x in original_xgb_logs),max(x['source_auc'] for x in original_xgb_logs)]
    # Source-only discrimination is reported separately from chronological target performance.
    facts['independent_source_auc']={}
    for case in ['bot','5g_bs1']:
        facts['independent_source_auc'][case]={x['fit_id']:x['source_auc'] for x in fits if x.get('source')==case and x['supported']}
    data=[]
    original_diagnostics=json.loads((ROOT/'experiments/results/cora/score_diagnostics.json').read_text())
    seen=set()
    for x in original_diagnostics:
        if x['scenario'] in seen:continue
        seen.add(x['scenario']);data.append([x['scenario'],f"{x['n']:,}",f"{x['benign']:,}"])
    for case in ['bot','5g_bs1']:
        meta=facts['windows'][case]['global_L64']['metadata'];data.append([CASES[case]+' global 64',f"{meta['target_windows']:,}",f"{meta['target_benign']:,}"])
    table('tifs_data','Target-window denominators. The twelve original scenarios reuse six target streams; local/full and temporal/transfer conditions overlap. The two additional capture lineages are BoT-IoT and 5G-NIDD BS1. Unit controls appear in Tables~\\ref{tab:tifs_windows} and~\\ref{tab:tifs_independent}.','Scenario & Windows & Benign',data,'lrr')
    facts['exclusions']=[json.loads((ROOT/'data/prepared/tifs/5g_bs2_excluded.json').read_text()),dict(dataset='HIKARI-2021',reason='No timestamps in released feature CSV; no chronological performance claim')]
    figure(blocks)
    (OUT/'summary.json').write_text(json.dumps(facts,indent=2))
    with (OUT/'metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['experiment','source','scenario','representation','model','fit_seed','query_seed','q','delay','delay_seconds','reference','feedback','policy','fpr','tpr','label_fraction','supported','reason'])
        for r in rows:
            for policy,m in r['methods'].items():writer.writerow([r[k] for k in ['experiment','source','scenario']]+[r.get('representation','')]+[r[k] for k in ['model','fit_seed','query_seed','query_fraction','delay','delay_seconds','reference','feedback']]+[policy,m['fpr'],m['tpr'],m['label_fraction'],m['supported'],m.get('reason','')])
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p)) for p in sorted((OUT/'scores').glob('*.npz'))]
    (ROOT/'provenance/tifs_score_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:facts[k] for k in ['supported_score_target_cases','query_replays','method_results','additional_supported_fits']},indent=2))


if __name__=='__main__':main()
