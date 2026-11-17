import sys
import unittest
from pathlib import Path
import numpy as np
from experiments.run_q1_baselines import fit_order, score_order
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "paper_output/code/modeling"))
from q34_helpers import epsilon_nondominated, epsilon_dominates


class AdditionalTests(unittest.TestCase):
    def test_no_cross_sequence_transitions(self):
        p=fit_order([np.array([0,1]),np.array([2,3])],1)
        self.assertAlmostEqual(p[1,2],.25)
        self.assertAlmostEqual(p[0,1],.5)

    def test_fixed_prefix_alignment(self):
        seq=np.array([0,1,2,3,0,1,3])
        for order in [0,1,2,3]:
            p=np.full((4**order,4),.25)
            self.assertAlmostEqual(score_order(seq,p,order,landmark=4),3*np.log(.25))
        p=np.full((16,4),.25)
        p[1*4+2]=[.1,.2,.3,.4]
        self.assertAlmostEqual(score_order(np.array([0,1,2,3]),p,2,landmark=3),np.log(.4))

    def test_epsilon_compares_all_original_alternatives(self):
        c=np.array([.8,.795,.79,.78])
        r=np.array([.2,.19,.18,.22])
        eps=.006
        expected=np.array([not any(i!=j and epsilon_dominates(c[i],r[i],c[j],r[j],
            epsilon_completion=eps,epsilon_risk=eps) for i in range(len(c))) for j in range(len(c))])
        np.testing.assert_array_equal(expected,epsilon_nondominated(c,r,epsilon_completion=eps,epsilon_risk=eps))


if __name__=="__main__":
    unittest.main()
