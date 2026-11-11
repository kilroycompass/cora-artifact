"""Predict-then-query, delayed-feedback policies on complete deployment streams.

DtACI follows isgibbs/DtACI at 1288af7 (fixed-share expert recurrence).
UP follows Algorithm / reference implementation AdaMLLab/up-for-ocp 3575f1f.
The reference code's ad-hoc alpha -= .015 adjustment is explicitly omitted:
it is undefined as a nominal 1%-FPR procedure. No target-dependent tuning.
All adaptive controllers consume delivered benign scores in arrival order.
Their original full-feedback guarantees are not claimed for sparse/delayed data.
"""
import math
import os
from pathlib import Path
os.environ.setdefault("NUMBA_CACHE_DIR",str(Path(__file__).resolve().parents[1]/"build/numba"))
import numpy as np
from numba import njit
from scipy.stats import binom

POLICIES=['Frozen','SC','NP','Rolling','ACI','DtACI','UP','QT','Silent']
GAMMAS=np.array([.001,.002,.004,.008,.016,.032,.064,.128])


@njit(cache=True)
def cutoff(s,alpha):
    rank=int(math.ceil((len(s)+1)*(1-alpha)))
    if rank>len(s):return np.inf
    if rank<=0:return -np.inf
    return s[rank-1]


@njit(cache=True)
def _replace(sorted_ref,ring,index,value):
    old=ring[index]
    loc=np.searchsorted(sorted_ref,old,side='left')
    for k in range(loc,len(sorted_ref)-1):sorted_ref[k]=sorted_ref[k+1]
    last=len(sorted_ref)-1
    loc=np.searchsorted(sorted_ref[:last],value,side='right')
    for k in range(last,loc,-1):sorted_ref[k]=sorted_ref[k-1]
    sorted_ref[loc]=value;ring[index]=value


@njit(cache=True,nogil=True)
def run(p,y,source,queries,delivery,uniforms,r=1000,alpha=.01,np_rank=996,
        gamma=.005,wealth_scale=1.,qt_scale=.01,keep_trace=False,issued_feedback=False):
    n=len(p);m=9
    ring=source.copy();s=np.sort(source.copy());ring_index=0
    initial=cutoff(s,alpha)
    cora_tau=initial;np_tau=initial;cora_ref=np.empty(r);got=0;activation=-1
    aci=alpha;expert=np.full(8,alpha);weights=np.full(8,.125);chosen=0
    scale=max(np.quantile(source,.95)-np.quantile(source,.05),.001)
    # Scores have the declared probability scale [0,1]. The UP default initial
    # wealth is one score unit, as in the authors' implementation.
    wealth=wealth_scale;up=0.;up_t=1;covered_count=0
    qt=initial
    fp=np.zeros(m,np.int64);tp=np.zeros(m,np.int64);label_counts=np.zeros(m,np.int64)
    trace=np.zeros((n if keep_trace else 0,m),np.uint8)
    issued_aci=np.zeros(n,np.uint8);issued_qt=np.zeros(n,np.uint8)
    issued_expert=np.zeros((n,8),np.uint8);issued_levels=np.zeros((n,8));issued_beta=np.zeros(n)
    ptr=0;requested=0
    thresholds=np.empty(9)
    for i in range(n):
        while ptr<len(queries) and delivery[ptr]<=i:
            old=queries[ptr]
            if y[old]==0:
                value=p[old]
                if got<r:
                    cora_ref[got]=value;got+=1
                    if got==r:
                        ordered=np.sort(cora_ref)
                        cora_tau=cutoff(ordered,alpha)
                        np_tau=ordered[np_rank-1] if np_rank<=r else np.inf
                        activation=i
                aci_error=bool(issued_aci[old]) if issued_feedback else value>cutoff(s,aci)
                aci+=gamma*(alpha-int(aci_error))
                # Boundary-adjusted attainable alpha, consistent with strict
                # score > order statistic, including ties.
                lower=np.searchsorted(s,value,side='left')
                beta=np.nextafter(1-lower/(len(s)+1),-np.inf)
                for j in range(8):
                    diff=(issued_beta[old]-issued_levels[old,j]) if issued_feedback else beta-expert[j]
                    loss=alpha*diff-min(diff,0.)
                    weights[j]*=math.exp(-2.72*loss)
                    error=bool(issued_expert[old,j]) if issued_feedback else value>cutoff(s,expert[j])
                    expert[j]+=GAMMAS[j]*(alpha-int(error))
                total=weights.sum()
                if total<=1e-300:weights[:]=.125
                else:
                    for j in range(8):weights[j]=.999*weights[j]/total+.001/8
                u=uniforms[old];cum=0.
                for j in range(8):
                    cum+=weights[j]
                    if u<cum:chosen=j;break
                # Universal portfolio recurrence on source-centered scores.
                covered=(value-initial)<=up
                g=alpha if covered else -(1-alpha)
                covered_count+=int(covered)
                wealth=wealth-g*up
                fraction=(up_t-covered_count+.5)/(up_t+1.)
                bet=-1/(1-alpha)+fraction*(1/alpha+1/(1-alpha))
                up=bet*wealth;up_t+=1
                qt_error=bool(issued_qt[old]) if issued_feedback else value>qt
                qt+=qt_scale*scale*(int(qt_error)-alpha)
                _replace(s,ring,ring_index,value)
                ring_index=(ring_index+1)%len(ring)
            ptr+=1
        thresholds[0]=initial;thresholds[1]=cora_tau;thresholds[2]=np_tau
        thresholds[3]=cutoff(s,alpha);thresholds[4]=cutoff(s,aci)
        thresholds[5]=cutoff(s,expert[chosen]);thresholds[6]=initial+up;thresholds[7]=qt;thresholds[8]=np.inf
        issued_aci[i]=int(p[i]>thresholds[4]);issued_qt[i]=int(p[i]>qt)
        if issued_feedback:
            lower=np.searchsorted(s,p[i],side="left")
            issued_beta[i]=np.nextafter(1-lower/(len(s)+1),-np.inf)
            for j in range(8):
                issued_expert[i,j]=int(p[i]>cutoff(s,expert[j]));issued_levels[i,j]=expert[j]
        for j in range(m):
            alert=p[i]>thresholds[j]
            if y[i]==0:fp[j]+=int(alert)
            else:tp[j]+=int(alert)
            if keep_trace:trace[i,j]=int(alert)
        # Query decisions are independent of the current ground-truth label.
        # SC / NP stop requesting new labels once their reference arrives.
        if requested<len(queries) and queries[requested]==i:
            for j in range(3,8):label_counts[j]+=1
            if activation<0:
                label_counts[1]+=1;label_counts[2]+=1
            requested+=1
    return fp,tp,label_counts,activation,got,trace


def evaluate(p,y,source,q=.1,delay=100,seed=500,alpha=.01,r=1000,keep_trace=False,
             timestamps=None,delay_seconds=None,gamma=.005,wealth_scale=1.,qt_scale=.01,
             query_indices=None,random_keys=None,feedback="arrival",query_design="exact"):
    source=np.asarray(source,dtype=float)[-1000:]
    if not len(source):raise ValueError('Source reference is empty')
    rng=np.random.default_rng(seed+17019)
    # A fixed replay horizon and a precommitted exact annotation opportunity
    # budget. Sampling uses no scores or labels. Every policy gets the same IDs.
    if query_indices is not None:query=np.asarray(query_indices,np.int64)
    elif query_design=='bernoulli':query=np.flatnonzero(rng.random(len(p))<q).astype(np.int64)
    else:query=np.sort(rng.choice(len(p),int(q*len(p)),replace=False)).astype(np.int64)
    if delay_seconds is None:delivery=query+int(delay)+1
    else:
        if timestamps is None:raise ValueError('Clock delays require timestamps')
        delivery=np.searchsorted(timestamps,timestamps[query]+delay_seconds,side='right').astype(np.int64)
        delivery=np.maximum(delivery,query+1)
    ranks=np.arange(1,r+1)
    valid=np.flatnonzero(binom.sf(ranks-1,r,1-alpha)<=.05)
    np_rank=int(valid[0]+1) if len(valid) else r+1
    keys=np.arange(len(p)) if random_keys is None else np.asarray(random_keys,np.int64)
    u=np.random.default_rng(seed+51001).random(int(keys.max())+1)[keys]
    fp,tp,cost,activation,got,trace=run(np.asarray(p,float),np.asarray(y,np.int8),source,
         query,delivery,u,r,alpha,np_rank,gamma,wealth_scale,qt_scale,keep_trace,feedback=="issued")
    nb=int((y==0).sum());na=int((y==1).sum())
    result={name:dict(fp=int(fp[j]),tp=int(tp[j]),benign=nb,attacks=na,
        fpr=float(fp[j]/nb) if nb else None,tpr=float(tp[j]/na) if na else None,
        labels=int(cost[j]),label_fraction=float(cost[j]/len(y))) for j,name in enumerate(POLICIES)}
    for name in result:result[name]['supported']=True
    if math.ceil((len(source)+1)*(1-alpha))>len(source):
        # Portfolio score centering requires a finite source cutoff. An
        # unsupported adapter must not be reported as successful zero FPR.
        result['UP'].update(supported=False,reason='nonfinite source score center',fpr=None,tpr=None,fp=0,tp=0,labels=0,label_fraction=0.)
    return dict(methods=result,activation=int(activation),reference_received=int(got),n=len(y),
                query_fraction=q,delay=delay,delay_seconds=delay_seconds,alpha=alpha,reference=r,feedback=feedback,query_design=query_design,opportunities=len(query)),trace
