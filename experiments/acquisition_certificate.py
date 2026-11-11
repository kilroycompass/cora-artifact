"""Optimistic, retrospective joint-risk feasibility certificate for a fixed fallback.

The suffix oracle can perfectly classify every post-activation observation.
This is an upper bound on success probability, not a deployable classifier.
"""
import numpy as np
from scipy.stats import hypergeom


def certificate(y, fallback, reference, budget, delay=0, alpha=.01, beta=.1,
                label_delivery=None):
    y=np.asarray(y,dtype=np.int8); fallback=np.asarray(fallback,dtype=bool)
    if y.shape!=fallback.shape or y.ndim!=1:raise ValueError('Aligned one-dimensional outcomes required')
    n=len(y); nb=int(np.sum(y==0)); na=int(np.sum(y==1))
    if not nb or not na:raise ValueError('Both target classes required')
    if not 0<=budget<=n or reference<1:raise ValueError('Invalid acquisition resources')
    prefix_fp=np.r_[0,np.cumsum(fallback&(y==0))]
    prefix_fn=np.r_[0,np.cumsum((~fallback)&(y==1))]
    feasible=(prefix_fp<=alpha*nb+1e-12)&(prefix_fn<=(1-beta)*na+1e-12)
    deadline=int(np.flatnonzero(feasible)[-1])
    # If the fallback itself passes, non-activation can pass as well.
    if deadline==n:
        upper=1.; eligible=int(np.sum(y==0))
    else:
        delivery=np.arange(n)+int(delay)+1 if label_delivery is None else np.asarray(label_delivery)
        if delivery.shape!=y.shape:raise ValueError('Delivery map must align with traffic')
        eligible=int(np.sum((y==0)&(delivery<=deadline)))
        upper=float(hypergeom.sf(reference-1,n,eligible,budget))
    return dict(deadline=deadline,eligible_benign=eligible,reference=int(reference),
                opportunities=int(budget),joint_success_upper=upper,
                fallback_pass=bool(feasible[-1]),alpha=float(alpha),beta=float(beta))
