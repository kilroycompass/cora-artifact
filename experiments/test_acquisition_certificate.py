"""Exhaustive query-schedule checks of the retrospective oracle bound."""
import itertools,unittest
import numpy as np
from acquisition_certificate import certificate


class CertificateTests(unittest.TestCase):
    def test_exhaustive_oracle_attainability(self):
        y=np.array([1,0,1,0,0,1,0,1],np.int8)
        for fallback in [np.zeros(8,bool),np.ones(8,bool),np.array([1,1,0,1,0,0,1,0],bool)]:
            for delay in [0,1,3]:
                for reference in [1,2,3]:
                    outcomes=[]
                    for q in itertools.combinations(range(8),4):
                        delivery=sorted(i+delay+1 for i in q if y[i]==0)
                        activation=delivery[reference-1] if len(delivery)>=reference else 8
                        activation=min(8,activation)
                        pred=fallback.copy();pred[activation:]=y[activation:]
                        outcomes.append(pred[y==0].mean()<=.25 and pred[y==1].mean()>=.75)
                    out=certificate(y,fallback,reference,4,delay,alpha=.25,beta=.75)
                    self.assertAlmostEqual(out['joint_success_upper'],np.mean(outcomes),places=12)

    def test_good_fallback_needs_no_activation(self):
        y=np.array([0,1,0,1]);out=certificate(y,y.astype(bool),300,0)
        self.assertEqual(out['joint_success_upper'],1.)
        self.assertTrue(out['fallback_pass'])

    def test_zero_budget_with_failed_fallback(self):
        y=np.array([0,1,0,1]);out=certificate(y,np.ones(4,bool),1,0)
        self.assertEqual(out['joint_success_upper'],0.)

    def test_clock_delivery_map(self):
        y=np.array([0,1,0,1]);fallback=np.ones(4,bool)
        out=certificate(y,fallback,1,4,alpha=.5,beta=.5,label_delivery=np.array([1,2,4,4]))
        self.assertEqual(out['joint_success_upper'],1.)

    def test_missing_class_rejected(self):
        with self.assertRaises(ValueError):certificate(np.zeros(4),np.zeros(4),1,2)


if __name__=='__main__':unittest.main()
