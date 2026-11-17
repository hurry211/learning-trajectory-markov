import itertools
import unittest
import numpy as np
from experiments.risk_core import run_risk, row_tv_bound, paired_summary


class RiskTests(unittest.TestCase):
    def test_exhaustive(self):
        rng = np.random.default_rng(44)
        for horizon in (1, 2, 4, 6):
            p = rng.dirichlet(np.ones(3), size=(1,horizon,3))
            s = rng.dirichlet(np.ones(3), size=1)
            for m in (1,2,3,7):
                total = 0.
                for path in itertools.product(range(3), repeat=horizon+1):
                    probability = s[0,path[0]]
                    for t in range(horizon):
                        probability *= p[0,t,path[t],path[t+1]]
                    if any(all(x==0 for x in path[1+i:1+i+m]) for i in range(horizon-m+1)):
                        total += probability
                self.assertAlmostEqual(run_risk(p,s,m)[0], total, places=12)

    def test_future_counter_reset_and_edges(self):
        p = np.eye(2)[None,None,:,:].repeat(4,axis=1)
        self.assertEqual(run_risk(p,np.array([[1.,0.]]),5)[0],0.)
        self.assertEqual(run_risk(p,np.array([[1.,0.]]),4)[0],1.)
        self.assertEqual(run_risk(p,np.array([[0.,1.]]),1)[0],0.)
        self.assertEqual(run_risk(p[:,:0],np.array([[1.,0.]]),1)[0],0.)

    def test_perturbation_bound(self):
        rng=np.random.default_rng(88)
        p=rng.dirichlet(np.ones(4),size=(200,8,4))
        q=.96*p+.04/4
        s=rng.dirichlet(np.ones(4),size=200)
        difference=abs(run_risk(p,s,3)-run_risk(q,s,3))
        self.assertTrue(np.all(difference <= row_tv_bound(p,q)+1e-12))

    def test_paired_se(self):
        a=np.array([1.,0.,1.,0.])
        self.assertEqual(paired_summary(a,a)["paired_mcse"],0.)

    def test_validation(self):
        with self.assertRaises(ValueError):
            run_risk(np.ones((1,2,2,2)),np.array([[1.,0.]]),2)
        with self.assertRaises(ValueError):
            run_risk(np.eye(2)[None,None],np.array([[1.,0.]]),0)


if __name__ == "__main__":
    unittest.main()
