"""Synthetic geometry interface checks, NOT contact-dynamics validation."""
import unittest
import numpy as np
from stance_feasibility import check_stance_body_correction, rotation_from_rpy


class TestStanceFeasibility(unittest.TestCase):
    def setUp(self):
        self.anchors = np.zeros((6, 3))
        self.anchors[:,:2] = np.array([[-.1,-.1],[-.1,.1],[0.,-.15],[.1,-.1],[.1,.1],[0.,.15]])
        self.xyz = np.array([0.,0.,.2])
        self.rpy = np.zeros(3)
        self.q = self.anchors - self.xyz[None, :]
        self.lo = np.ones((6,3))*(-1.)
        self.hi = np.ones((6,3))
        self.rate = np.ones((6,3))*10.
        self.ik = lambda i, foot_base: np.asarray(foot_base)

    def check(self, delta, **overrides):
        args = dict(stance_legs=(0,1,4),feet_world_m=self.anchors,
                    body_position_world_m=self.xyz,body_rpy_rad=self.rpy,
                    correction_z_roll_pitch=np.asarray(delta),q_current_rad=self.q,
                    q_lower_rad=self.lo,q_upper_rad=self.hi,
                    q_rate_limit_rad_s=self.rate,dt_s=.02,ik_body_frame=self.ik)
        args.update(overrides)
        return check_stance_body_correction(**args)

    def test_zero_correction_is_feasible(self):
        r = self.check([0,0,0]); self.assertTrue(r.feasible)
        self.assertTrue(np.allclose(r.q_target, self.q))

    def test_large_z_change_fails_kinematic_rate(self):
        r = self.check([.5,0,0]); self.assertFalse(r.feasible)
        self.assertIn('rate', r.reason)

    def test_rejected_unreachable_ik(self):
        r = self.check([0,0,0],ik_body_frame=lambda i,p:None)
        self.assertFalse(r.feasible)

    def test_rotation_rpy_order(self):
        self.assertTrue(np.allclose(rotation_from_rpy(np.zeros(3)),np.eye(3)))

if __name__ == '__main__': unittest.main()
