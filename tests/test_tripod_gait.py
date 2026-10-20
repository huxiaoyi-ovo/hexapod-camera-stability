import unittest
import numpy as np
from tripod_gait import TripodConfig, foot_targets

class TestTripodGait(unittest.TestCase):
    def setUp(self):
        self.feet = np.array([[-.1,-.1,-.15],[-.1,.1,-.15],[-.2,0,-.15],
                              [.1,-.1,-.15],[.1,.1,-.15],[.2,0,-.15]])
        self.cfg = TripodConfig(frequency_hz=1,duty_factor=.6,
                                swing_height_m=.04,touchdown_softness=.5)

    def test_stance_groups_are_alternating(self):
        out,support=foot_targets(.3,self.feet,0,.1,0,self.cfg)
        self.assertTrue(np.array_equal(support,[True,True,False,False,False,True]))
        self.assertTrue(np.allclose(out[support,2],self.feet[support,2]))

    def test_horizontal_position_and_velocity_c1_at_contact(self):
        eps=1e-5
        i=0
        p0,_=foot_targets(.6-eps,self.feet,0,.1,0,self.cfg)
        p1,_=foot_targets(.6+eps,self.feet,0,.1,0,self.cfg)
        self.assertLess(np.linalg.norm(p0[i]-p1[i]), 1e-4)
        a,_=foot_targets(.6-2*eps,self.feet,0,.1,0,self.cfg)
        b,_=foot_targets(.6+2*eps,self.feet,0,.1,0,self.cfg)
        left=(p0[i]-a[i])/eps
        right=(b[i]-p1[i])/eps
        self.assertAlmostEqual(left[1],-.1,delta=.003)
        self.assertAlmostEqual(right[1],-.1,delta=.003)

    def test_vertical_swing_height(self):
        out,s=foot_targets(.8,self.feet,0,.1,0,self.cfg)
        self.assertAlmostEqual(out[0,2],-.11,places=6)
        self.assertFalse(s[0])

    def test_achieved_velocity_not_assumed(self):
        # Trajectories only: cannot infer real base progress from foot targets.
        p,_=foot_targets(.15,self.feet,0,.1,.2,self.cfg)
        self.assertEqual(p.shape,(6,3))

if __name__=='__main__':unittest.main()
