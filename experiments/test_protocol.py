"""Temporal information access, budget accounting, and baseline-port checks."""
from pathlib import Path
import ast
import math
import itertools
import unittest
import numpy as np
from policies import evaluate, POLICIES


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.p=np.random.default_rng(17).random(320)
        self.y=(np.arange(320)%3==0).astype(np.int8)
        self.source=np.linspace(.01,.9,100)

    def test_no_future_scores_or_labels_in_decisions(self):
        _,before=evaluate(self.p,self.y,self.source,q=.5,delay=7,r=20,keep_trace=True)
        p=self.p.copy();y=self.y.copy();p[150:]=1-p[150:];y[150:]=1-y[150:]
        _,after=evaluate(p,y,self.source,q=.5,delay=7,r=20,keep_trace=True)
        np.testing.assert_array_equal(before[:150],after[:150])

    def test_label_cannot_affect_decisions_before_delivery(self):
        _,before=evaluate(self.p,self.y,self.source,q=1,delay=20,r=1000,keep_trace=True)
        y=self.y.copy();y[50]=1-y[50]
        _,after=evaluate(self.p,y,self.source,q=1,delay=20,r=1000,keep_trace=True)
        np.testing.assert_array_equal(before[:71],after[:71])

    def test_full_stream_and_exact_shared_budget(self):
        r,tr=evaluate(self.p,self.y,self.source,q=.1,delay=10,r=20,keep_trace=True)
        self.assertEqual(tr.shape,(320,9))
        for name in POLICIES[3:8]:self.assertEqual(r['methods'][name]['labels'],32)
        self.assertLessEqual(r['methods']['SC']['labels'],32)
        self.assertEqual(r['methods']['SC']['benign']+r['methods']['SC']['attacks'],320)

    def test_source_fallback_until_reference_ready(self):
        r,tr=evaluate(self.p,self.y,self.source,q=1,delay=5,r=20,keep_trace=True)
        cut=r['activation'];self.assertGreater(cut,0)
        np.testing.assert_array_equal(tr[:cut,0],tr[:cut,1])
        np.testing.assert_array_equal(tr[:cut,0],tr[:cut,2])

    def test_no_queries_all_policies_are_frozen(self):
        r,tr=evaluate(self.p,self.y,self.source,q=0,keep_trace=True)
        self.assertEqual(r['activation'],-1)
        for j in range(8):np.testing.assert_array_equal(tr[:,0],tr[:,j])

    def test_full_rate_decomposition(self):
        r,tr=evaluate(self.p,self.y,self.source,q=1,r=20,keep_trace=True)
        cut=r['activation']
        for cls,key in [(0,'fpr'),(1,'tpr')]:
            before=self.y[:cut]==cls;after=self.y[cut:]==cls
            w=before.sum()/(before.sum()+after.sum())
            full=r['methods']['SC'][key]-r['methods']['Frozen'][key]
            post=tr[cut:,1][after].mean()-tr[cut:,0][after].mean()
            self.assertAlmostEqual(full,(1-w)*post)

    def test_reference_size_does_not_change_other_policies(self):
        _,a=evaluate(self.p,self.y,self.source,q=.5,r=20,keep_trace=True)
        _,b=evaluate(self.p,self.y,self.source,q=.5,r=100,keep_trace=True)
        np.testing.assert_array_equal(a[:,[0,3,4,5,6,7]],b[:,[0,3,4,5,6,7]])

    def test_universal_portfolio_matches_reviewed_author_code(self):
        # Only reviewed class definitions are executed; imports and all other
        # external code are excluded. Constructor correction is neutralized so
        # that both recurrences have the same actual target alpha=.01.
        path=Path(__file__).resolve().parents[1]/'provenance/baselines/up-for-ocp_core_methods.py'
        tree=ast.parse(path.read_text())
        selected=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['ConformalPredictor','UniversalPortfolio']]
        ns={'np':np};exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
        ref=ns['UniversalPortfolio'](alpha=.025,initial_wealth=1.)
        self.assertAlmostEqual(ref.alpha,.01)
        _,tr=evaluate(self.p,np.zeros(320,dtype=np.int8),self.source,q=1,delay=0,keep_trace=True)
        initial=np.sort(self.source)[99]
        for i,value in enumerate(self.p):
            self.assertEqual(bool(tr[i,6]),bool(value>initial+ref.predict()),i)
            ref.update(value-initial)

    def test_tied_scores_remain_conservative_for_one_shot(self):
        r,tr=evaluate(np.ones(200),np.zeros(200,np.int8),np.ones(100),q=1,r=100,keep_trace=True)
        self.assertEqual(r['methods']['SC']['fp'],0)
        self.assertEqual(r['methods']['NP']['fp'],0)

    def test_acquisition_law_matches_all_query_subsets(self):
        from analysis import acquisition_probability
        y=np.array([0,1,0,0,1,0],np.int8)
        for delay in [0,1,3]:
            subsets=list(itertools.combinations(range(6),3))
            successes=sum(sum(y[i]==0 and i+delay+1<=5 for i in s)>=2 for s in subsets)
            self.assertAlmostEqual(acquisition_probability(y,3,2,delay),successes/len(subsets))

    def test_dtaci_matches_independent_author_recurrence(self):
        # Independent translation of conformalAdaptStable (DtACI.R), with
        # the same boundary convention and recorded uniform expert draws.
        gammas=np.array([.001,.002,.004,.008,.016,.032,.064,.128])
        levels=np.full(8,.01);weights=np.ones(8);chosen=0
        ring=self.source.copy();u=np.random.default_rng(51501).random(len(self.p))
        _,trace=evaluate(self.p,np.zeros(len(self.p),np.int8),self.source,
                         q=1,delay=0,keep_trace=True)
        def quantile(a):
            rank=math.ceil((len(ring)+1)*(1-a))
            return np.inf if rank>len(ring) else -np.inf if rank<=0 else np.sort(ring)[rank-1]
        for i,value in enumerate(self.p):
            self.assertEqual(bool(trace[i,5]),bool(value>quantile(levels[chosen])),i)
            beta=np.nextafter(1-np.sum(ring<value)/(len(ring)+1),-np.inf)
            losses=.01*(beta-levels)-np.minimum(beta-levels,0)
            errors=np.array([value>quantile(a) for a in levels])
            levels=levels+gammas*(.01-errors)
            bar=weights*np.exp(-2.72*losses)
            weights=.999*bar/bar.sum()+.001/8
            chosen=min(np.searchsorted(np.cumsum(weights),u[i]),7)
            ring[i%len(ring)]=value



    def test_issued_and_arrival_agree_without_feedback_delay(self):
        _,a=evaluate(self.p,self.y,self.source,q=.5,delay=0,r=20,keep_trace=True)
        _,b=evaluate(self.p,self.y,self.source,q=.5,delay=0,r=20,keep_trace=True,feedback='issued')
        np.testing.assert_array_equal(a,b)

    def test_issued_feedback_remains_causal_with_delay(self):
        _,a=evaluate(self.p,self.y,self.source,q=1,delay=20,r=20,keep_trace=True,feedback='issued')
        y=self.y.copy();y[50]=1-y[50]
        _,b=evaluate(self.p,y,self.source,q=1,delay=20,r=20,keep_trace=True,feedback='issued')
        np.testing.assert_array_equal(a[:71],b[:71])

    def test_bernoulli_queries_do_not_depend_on_horizon(self):
        _,a=evaluate(self.p,self.y,self.source,q=.3,delay=10,r=20,query_design='bernoulli',keep_trace=True)
        _,b=evaluate(self.p[:200],self.y[:200],self.source,q=.3,delay=10,r=20,query_design='bernoulli',keep_trace=True)
        np.testing.assert_array_equal(a[:200],b)

    def test_silent_policy_has_no_queries_or_alerts(self):
        r,tr=evaluate(self.p,self.y,self.source,q=1,keep_trace=True)
        self.assertEqual(r['methods']['Silent']['labels'],0)
        self.assertEqual(int(tr[:,-1].sum()),0)


    def test_source_pool_does_not_depend_on_target_labels_or_families(self):
        from unittest.mock import patch
        from train import datasets
        d=dict(x=np.arange(1000)[:,None,None],y=(np.arange(1000)%3==0).astype(np.int8),tags=np.where(np.arange(1000)%3==0,'Known','Benign'),t_end=np.arange(1000)[:,None])
        with patch('train.read',return_value=d):a,b,targets,families=datasets('cic18_bot')
        modified={k:v.copy() for k,v in d.items()};modified['y'][600:]=1-modified['y'][600:];modified['tags']=modified['tags'].astype('<U20');modified['tags'][600:]='Future'
        with patch('train.read',return_value=modified):aa,bb,_,ff=datasets('cic18_bot')
        np.testing.assert_array_equal(a['x'],aa['x']);np.testing.assert_array_equal(b['x'],bb['x']);self.assertEqual(families,ff)
        self.assertLess(a['x'].max(),600);self.assertLess(b['x'].max(),600)
        self.assertFalse(set(a['x'].ravel())&set(b['x'].ravel()))
        self.assertEqual(int(targets[0][1]['x'].min()),600)

    def test_np_finite_cutoff_minimum_is_299_at_one_percent(self):
        p=np.linspace(0,1,350);y=np.zeros(350,np.int8)
        for reference,finite in [(298,False),(299,True)]:
            r,tr=evaluate(p,y,np.linspace(0,1,1000),q=1,delay=0,r=reference,keep_trace=True)
            self.assertEqual(bool(tr[r['activation']:,2].any()),finite)


    def test_portfolio_without_finite_source_center_is_unsupported(self):
        r,_=evaluate(self.p,self.y,self.source,alpha=.001)
        self.assertFalse(r['methods']['UP']['supported'])
        self.assertIsNone(r['methods']['UP']['fpr'])
        self.assertIsNone(r['methods']['UP']['tpr'])

if __name__=='__main__':unittest.main()
