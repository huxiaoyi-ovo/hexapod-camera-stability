"""Mathematical/interface tests ONLY, not experiments supporting the paper."""
import unittest
import numpy as np
from controller import (Gait, Camera, TrackedScene, Candidate, Settings,
                        image_interaction_matrix, select_gait, PhaseConditionedRidge)

class TestController(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(600, 600, 320, 240, 0.01, 1/30)
        self.scene = TrackedScene(np.array([[320.,240.],[370.,240.],[270.,290.]]),
                                  np.array([2.,2.,2.]), np.ones(3))
        self.gait = Gait(1.5, .65, .045, .4)
        self.settings = Settings(.10, .015, .005, np.array([1.,1.,1.,2.,2.,2.]),
                                 regularization=1e-3, smoothing=1e-4, gait_change_penalty=.0001)

    def test_camera_interaction(self):
        L = image_interaction_matrix(self.camera, self.scene)
        # Translation +x in camera frame should shift center scene left.
        self.assertAlmostEqual((L @ np.array([1.,0,0,0,0,0]))[0], -300.)

    def test_choose_lower_residual_candidate(self):
        H = 6
        g2 = Gait(1.7, .65, .045, .4)
        high = np.tile([0, 0, 0, .12, 0, 0], (H,1))
        low = np.tile([0, 0, 0, .035, 0, 0], (H,1))
        budget = np.tile([0.05, 0.05, 0.05, .015, .015, .015], (H,1))
        candidates = [Candidate(self.gait, high, budget, .11, .02, .02, True),
                      Candidate(g2, low, budget, .11, .02, .02, True)]
        result = select_gait(self.camera, self.scene, candidates, self.settings, self.gait)
        self.assertEqual(result.gait, g2)
        self.assertGreater(result.rms_frame_jitter_px, 0)
        self.assertAlmostEqual(result.rms_frame_jitter_px/self.camera.frame_interval_s,
                               result.rms_exposure_motion_px/self.camera.exposure_s,places=8)

    def test_reject_unsafe_candidate(self):
        H=3; zero=np.zeros((H,6)); budget=np.ones((H,6))*0.1
        bad = Candidate(self.gait,zero,budget,.02,.03,.03,True)
        with self.assertRaises(RuntimeError):
            select_gait(self.camera,self.scene,[bad],self.settings,self.gait)

    def test_phase_model_training(self):
        g=self.gait
        X=np.stack([PhaseConditionedRidge.features(g,.02,.1,k*2*np.pi/40) for k in range(80)])
        Y=np.column_stack([.02*X[:,7], .03*X[:,8], np.zeros((80,4))])
        m=PhaseConditionedRidge().fit(X,Y)
        self.assertLess(np.mean((m.predict(X)-Y)**2),1e-6)

if __name__ == '__main__':
    unittest.main()
