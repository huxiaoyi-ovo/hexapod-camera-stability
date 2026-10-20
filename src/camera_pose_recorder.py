"""Isaac Gym / ROS-independent camera pose logger.

The host needs to supply actual (time, base pose) samples and the desired
nominal camera-frame twist. A fixed body-to-camera transform is required.
Use ROS camera optical-frame extrinsics, NOT base_link xyz by assumption.
Quaternions use SciPy xyzw convention.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from trace_metrics import TWIST


class CameraPoseRecorder:
    def __init__(self, base_to_camera_position_m, base_to_camera_quat_xyzw):
        self.p_bc = np.asarray(base_to_camera_position_m, float)
        self.r_bc = Rotation.from_quat(base_to_camera_quat_xyzw)
        if self.p_bc.shape != (3,) or not np.isfinite(self.p_bc).all():
            raise ValueError('base_to_camera_position_m must be 3 finite floats')
        self.samples = []

    def append(self, t_s, base_position_world_m, base_quat_xyzw, nominal_camera_twist):
        """Called AFTER physics step, using simulator state, not setpoints.

        nominal_camera_twist: expected smooth 6D motion in camera optical
        frame, provided by reference trajectory; MUST NOT be unknown/None.
        """
        p = np.asarray(base_position_world_m, float)
        r = Rotation.from_quat(base_quat_xyzw)
        nominal = np.asarray(nominal_camera_twist, float)
        if p.shape != (3,) or nominal.shape != (6,) or not np.isfinite(nominal).all():
            raise ValueError('Bad base position or nominal twist')
        p_wc = p + r.apply(self.p_bc)
        r_wc = r * self.r_bc
        if self.samples and t_s <= self.samples[-1][0]:
            raise ValueError('Sampling times must increase')
        self.samples.append((float(t_s), p_wc, r_wc, nominal))

    def save_twist_csv(self, filename):
        if len(self.samples) < 3:
            raise ValueError('Need at least three camera poses')
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ['t', *TWIST, *('nominal_'+k for k in TWIST)]
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for j, (t, p, r, nominal) in enumerate(self.samples):
                # One-sided interval derivative at first sample; endpoint
                # velocity then reused at all later forward samples.
                i0, i1 = (0, 1) if j == 0 else (j-1, j)
                t0, p0, r0, _ = self.samples[i0]
                t1, p1, r1, _ = self.samples[i1]
                dt = t1-t0
                # Translational component represented in camera optical frame.
                v_cam = r.inv().apply((p1-p0)/dt)
                # Spatial angular velocity in local camera frame at endpoint.
                omega_cam = -(r1.inv() * r0).as_rotvec()/dt
                vec = np.r_[v_cam, omega_cam]
                writer.writerow(dict(zip(fieldnames, [t, *vec, *nominal])))
        return path
