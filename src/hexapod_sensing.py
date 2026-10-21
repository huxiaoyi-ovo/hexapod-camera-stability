"""Proprioceptive contact-load and support-foot odometry prototypes."""
from __future__ import annotations

import math
from typing import Dict, Mapping

import numpy as np

from hexapod_robot import JOINT_TYPES, LEG_ORDER, RobotModel, toe_positions_base
from hexapod_walk import leg_fk_jacobian, quaternion_matrix_xyzw as _quaternion_matrix_xyzw


def quaternion_matrix_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    return _quaternion_matrix_xyzw(quaternion_xyzw)


def damped_tip_force_estimate(jacobian: np.ndarray, torque_nm: np.ndarray,
                              damping: float = 0.005) -> np.ndarray:
    """Estimate external tip force from motor torque via damped virtual work."""
    jacobian = np.asarray(jacobian, dtype=float)
    torque_nm = np.asarray(torque_nm, dtype=float)
    if jacobian.shape != (3, 3) or torque_nm.shape != (3,):
        raise ValueError("tip force estimate expects a 3x3 Jacobian and three torques")
    if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(torque_nm)):
        raise ValueError("tip force estimate inputs must be finite")
    normal = jacobian @ jacobian.T + float(damping) ** 2 * np.eye(3)
    return -np.linalg.solve(normal, jacobian @ torque_nm)


class ProprioceptiveContactOdometry:
    """Uncalibrated torque-derived load latch and confirmed-foot odometry.

    The supplied torque is the previous modeled applied motor torque. Quaternion
    input is an ideal simulator attitude used as an IMU orientation sample.
    """

    def __init__(self, model: RobotModel, initial_body_xyz_m: np.ndarray,
                 initial_quaternion_xyzw: np.ndarray, initial_q_by_name: Mapping[str, float],
                 toe_radius_m: float,
                 on_threshold_n: float = 6.0, off_threshold_n: float = 3.0,
                 confirmation_s: float = 0.04, force_filter_s: float = 0.02,
                 minimum_support: int = 3):
        self.model = model
        self.body_xyz = np.asarray(initial_body_xyz_m, dtype=float).copy()
        self.rotation = quaternion_matrix_xyzw(initial_quaternion_xyzw)
        self._last_quaternion_xyzw = np.asarray(initial_quaternion_xyzw, dtype=float).copy()
        self.on_threshold_n = float(on_threshold_n)
        self.off_threshold_n = float(off_threshold_n)
        self.confirmation_s = float(confirmation_s)
        self.force_filter_s = float(force_filter_s)
        self.minimum_support = int(minimum_support)
        if (not 0 <= self.off_threshold_n < self.on_threshold_n or self.confirmation_s <= 0 or
                self.force_filter_s <= 0 or not 1 <= self.minimum_support <= len(LEG_ORDER)):
            raise ValueError("invalid proprioceptive contact estimator settings")
        self._last_q_by_name = {name: float(value) for name, value in initial_q_by_name.items()}
        toe_positions = toe_positions_base(model, self._last_q_by_name)
        nominal_feet = {leg: toe_positions["l_{}_toe".format(leg)] for leg in LEG_ORDER}
        self.anchors_world = {
            leg: self.body_xyz + self.rotation @ nominal_feet[leg]
            for leg in LEG_ORDER
        }
        for anchor in self.anchors_world.values():
            anchor[2] = float(toe_radius_m)
        self.support_mask = {leg: True for leg in LEG_ORDER}
        self.contact = {leg: False for leg in LEG_ORDER}
        self._on_elapsed = {leg: 0.0 for leg in LEG_ORDER}
        self._off_elapsed = {leg: 0.0 for leg in LEG_ORDER}
        self._filtered_up = {leg: 0.0 for leg in LEG_ORDER}
        self._previous_contact = {leg: False for leg in LEG_ORDER}
        self._last_feet_world = {
            leg: self.body_xyz + self.rotation @ nominal_feet[leg]
            for leg in LEG_ORDER
        }
        self.last_force_body = {leg: np.zeros(3, dtype=float) for leg in LEG_ORDER}
        self.last_force_world = {leg: np.zeros(3, dtype=float) for leg in LEG_ORDER}

    def update(self, q_by_name: Mapping[str, float], quaternion_xyzw: np.ndarray,
               previous_applied_torque_by_name: Mapping[str, float], dt_s: float) -> Dict[str, object]:
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("sensor update dt must be positive and finite")
        rotation = quaternion_matrix_xyzw(quaternion_xyzw)
        forces_body, forces_world = {}, {}
        alpha = 1.0 - math.exp(-dt / self.force_filter_s)
        for leg in LEG_ORDER:
            _, jacobian = leg_fk_jacobian(self.model, leg, q_by_name)
            names = tuple("j_{}_{}".format(leg, kind) for kind in JOINT_TYPES)
            torques = np.array([previous_applied_torque_by_name[name] for name in names], dtype=float)
            force_body = damped_tip_force_estimate(jacobian, torques)
            force_world = rotation @ force_body
            forces_body[leg], forces_world[leg] = force_body, force_world
            filtered = self._filtered_up[leg] + alpha * (float(force_world[2]) - self._filtered_up[leg])
            self._filtered_up[leg] = filtered
            if not self.contact[leg]:
                self._off_elapsed[leg] = 0.0
                self._on_elapsed[leg] = self._on_elapsed[leg] + dt if filtered >= self.on_threshold_n else 0.0
                if self._on_elapsed[leg] + 1e-12 >= self.confirmation_s:
                    self.contact[leg] = True
                    self._on_elapsed[leg] = 0.0
            else:
                self._on_elapsed[leg] = 0.0
                self._off_elapsed[leg] = self._off_elapsed[leg] + dt if filtered <= self.off_threshold_n else 0.0
                if self._off_elapsed[leg] + 1e-12 >= self.confirmation_s:
                    self.contact[leg] = False
                    self._off_elapsed[leg] = 0.0

        self.rotation = rotation
        self._last_quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=float).copy()
        self._last_q_by_name = {name: float(value) for name, value in q_by_name.items()}
        self.last_force_body = forces_body
        self.last_force_world = forces_world
        self._refresh_odometry(q_by_name)
        self._previous_contact = dict(self.contact)
        return self.snapshot()

    def _refresh_odometry(self, q_by_name: Mapping[str, float]) -> None:
        toe_positions = toe_positions_base(self.model, q_by_name)
        local_feet = {leg: toe_positions["l_{}_toe".format(leg)] for leg in LEG_ORDER}
        support_candidates = [self.anchors_world[leg] - self.rotation @ local_feet[leg]
                              for leg in LEG_ORDER if self.support_mask[leg]]
        if len(support_candidates) >= self.minimum_support:
            self.body_xyz = np.median(np.stack(support_candidates), axis=0)
        self._last_feet_world = {
            leg: self.body_xyz + self.rotation @ local_feet[leg] for leg in LEG_ORDER
        }

    def snapshot(self) -> Dict[str, object]:
        return {"body_xyz_m": self.body_xyz.copy(),
                "feet_world_m": {k: v.copy() for k, v in self._last_feet_world.items()},
                "imu_quaternion_xyzw": self._last_quaternion_xyzw.copy(),
                "contact_confirmed": dict(self.contact), "support_mask": dict(self.support_mask),
                "force_body_n": {k: v.copy() for k, v in self.last_force_body.items()},
                "force_world_n": {k: v.copy() for k, v in self.last_force_world.items()},
                "filtered_up_force_n": dict(self._filtered_up),
                "support_count": sum(self.support_mask.values())}

    def set_support_mask(self, support_mask: Mapping[str, bool],
                         estimated_feet_world: Mapping[str, np.ndarray]) -> None:
        for leg in LEG_ORDER:
            active = bool(support_mask[leg])
            if active and not self.support_mask[leg]:
                self.anchors_world[leg] = np.asarray(estimated_feet_world[leg], dtype=float).copy()
            self.support_mask[leg] = active
        self._refresh_odometry(self._last_q_by_name)
