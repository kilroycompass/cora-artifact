"""Checks for representation controls and factor-balanced reporting."""
import unittest
import numpy as np
from tifs_windows import make_window
from tifs_report import balanced


class WindowControlTests(unittest.TestCase):
    def test_fit_counts_do_not_reweight_scorers(self):
        rows=[dict(scenario='S',model=m,fit_seed=seed,value=value) for m,n,value in [('A',1,0),('B',1,0),('C',5,1)] for seed in range(n) for q in range(20)]
        self.assertAlmostEqual(balanced(rows,lambda r:r['value']),1/3)

    def test_length_one_grouping_invariant_with_ties(self):
        d=dict(x=np.arange(48,dtype=np.float32).reshape(12,4),y=np.arange(12)%2,
               t=np.repeat(np.arange(4),3),host=np.tile([2,0,1],4))
        a=make_window(d,np.arange(12),1,'global');b=make_window(d,np.arange(12),1,'host')
        for k in a:np.testing.assert_array_equal(a[k],b[k])

    def test_host_windows_do_not_mix_groups(self):
        d=dict(x=np.column_stack([np.arange(10),np.repeat([0,1],5)]).astype(np.float32),
               y=np.zeros(10,dtype=np.int8),t=np.arange(10),host=np.repeat([0,1],5))
        a=make_window(d,np.arange(10),2,'host')
        self.assertEqual(len(a['y']),4)
        self.assertTrue(np.all(a['x'][:,1,0]==a['x'][:,1,1]))

    def test_source_target_windows_cannot_cross_boundary(self):
        d=dict(x=np.arange(24,dtype=np.float32).reshape(12,2),y=np.arange(12)%2,
               t=np.arange(12),host=np.zeros(12,dtype=int))
        a=make_window(d,np.arange(6),4,'global');b=make_window(d,np.arange(6,12),4,'global')
        self.assertLess(a['t_end'].max(),6);self.assertGreaterEqual(b['t_end'].min(),6)


if __name__=='__main__':unittest.main()
