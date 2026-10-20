"""Analytical invariant tests. These are NOT robot experimental results."""
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from controller import Camera, TrackedScene
from trace_metrics import read_trace, image_motion_metrics
from camera_pose_recorder import CameraPoseRecorder

class TestTraceMetrics(unittest.TestCase):
    def setUp(self):
        self.cam = Camera(fx=100,fy=100,cx=50,cy=50,exposure_s=0.02,frame_interval_s=0.05)
        self.scene = TrackedScene(uv=np.array([[50.0,50.0]]),
                                  depth_m=np.array([2.0]),weights=np.array([1.0]))
        self.t = np.linspace(0,1,1001)

    def test_stationary_is_zero(self):
        r = image_motion_metrics(self.t, np.zeros((len(self.t),6)), self.cam,self.scene)
        self.assertAlmostEqual(r.jitter_px_rms_per_frame,0,places=10)
        self.assertAlmostEqual(r.exposure_trace_px_rms,0,places=10)

    def test_constant_translation_matches_projection(self):
        d=np.zeros((len(self.t),6)); d[:,0]=.1
        r=image_motion_metrics(self.t,d,self.cam,self.scene)
        self.assertAlmostEqual(r.jitter_px_rms_per_frame, .1*100/2*.05, places=5)
        self.assertAlmostEqual(r.exposure_trace_px_rms, .1*100/2*.02, places=5)

    def test_reversing_motion_has_blur_without_net_displacement(self):
        # A reversible zigzag within exposure: identical initial/final
        # feature position, yet a nonzero blur-path proxy.
        cam=Camera(fx=100,fy=100,cx=50,cy=50,exposure_s=.05,frame_interval_s=.05)
        t=np.arange(0, 1.00001, .001)
        d=np.zeros((len(t),6)); d[:,0]=.1*np.sin(2*np.pi*20*t)
        r=image_motion_metrics(t,d,cam,self.scene)
        self.assertLess(r.exposure_net_displacement_px_rms,1e-9)
        self.assertGreater(r.exposure_trace_px_rms, .1)

    def test_missing_nominal_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.csv'
            p.write_text('t,vx,vy,vz,wx,wy,wz\n0,0,0,0,0,0,0\n',encoding='utf-8')
            with self.assertRaises(ValueError): read_trace(p)

    def test_pose_recorder_constant_velocity(self):
        rec = CameraPoseRecorder([0,0,0], [0,0,0,1])
        for t in np.linspace(0,1,1001):
            rec.append(t,[.1*t,0,0],[0,0,0,1],[.1,0,0,0,0,0])
        with tempfile.TemporaryDirectory() as tmp:
            f=rec.save_twist_csv(Path(tmp)/'trace.csv')
            tt,a,n=read_trace(f)
            self.assertTrue(np.allclose(a[:,0],.1,atol=1e-12))
            self.assertTrue(np.allclose(a[:,3:],0,atol=1e-12))
            self.assertTrue(np.allclose(a,n,atol=1e-12))

if __name__=='__main__': unittest.main()
