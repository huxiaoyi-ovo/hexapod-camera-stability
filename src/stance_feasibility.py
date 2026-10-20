"""Kinematic feasibility filter for candidate roll/pitch/body-z corrections.

Inputs are real stance-foot WORLD anchors and a host robot IK callback.
This is NOT a simulator, a contact-force optimizer, nor a camera stabilizer.
The module checks stance IK positions and a per-step joint-velocity bound.
Swing-leg clearance, collision, support margin and actuator dynamics must be
checked separately by the host robot and documented in the paper.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np


@dataclass(frozen=True)
class StanceFeasibility:
    feasible: bool
    reason: str
    q_target: np.ndarray | None
    min_position_margin_rad: float
    min_velocity_margin_rad: float


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    """World-from-body R = Rz(yaw)*Ry(pitch)*Rx(roll), angles in radians."""
    r = np.asarray(rpy, dtype=float)
    if r.shape != (3,) or not np.all(np.isfinite(r)):
        raise ValueError('rpy must be a finite shape-(3,) vector')
    return _rz(r[2]) @ _ry(r[1]) @ _rx(r[0])


def check_stance_body_correction(
    *,
    stance_legs: Sequence[int],
    feet_world_m: np.ndarray,
    body_position_world_m: np.ndarray,
    body_rpy_rad: np.ndarray,
    correction_z_roll_pitch: np.ndarray,
    q_current_rad: np.ndarray,
    q_lower_rad: np.ndarray,
    q_upper_rad: np.ndarray,
    q_rate_limit_rad_s: np.ndarray,
    dt_s: float,
    ik_body_frame: Callable[[int, np.ndarray], np.ndarray | None],
) -> StanceFeasibility:
    """Test a proposed body pose while preserving known stance-foot anchors.

    `ik_body_frame(leg_index, foot_point_in_base_link_m) -> (3,) joint angles`.
    It must perform this platform's leg geometry and frame conversion.
    No controller or gaze reference is changed by this pure feasibility check.
    The measured current q must be used for rate checks; this deliberately
    rejects physically impossible instantaneous compensation.
    """
    feet = np.asarray(feet_world_m, float)
    xyz = np.asarray(body_position_world_m, float)
    rpy = np.asarray(body_rpy_rad, float)
    delta = np.asarray(correction_z_roll_pitch, float)
    q = np.asarray(q_current_rad, float)
    lo = np.asarray(q_lower_rad, float)
    hi = np.asarray(q_upper_rad, float)
    rates = np.asarray(q_rate_limit_rad_s, float)
    if (feet.shape != (6,3) or xyz.shape != (3,) or rpy.shape != (3,)
        or delta.shape != (3,) or q.shape != (6,3) or lo.shape != (6,3)
        or hi.shape != (6,3) or rates.shape != (6,3)):
        raise ValueError('Unexpected vector or joint-array shape')
    if not (np.isfinite(feet).all() and np.isfinite(xyz).all()
            and np.isfinite(rpy).all() and np.isfinite(delta).all()
            and np.isfinite(q).all() and np.isfinite(lo).all()
            and np.isfinite(hi).all() and np.isfinite(rates).all()):
        raise ValueError('Non-finite kinematic input')
    if np.any(lo > hi) or np.any(rates < 0) or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError('Invalid joint bounds or time step')
    leg_ids = tuple(int(i) for i in stance_legs)
    if len(leg_ids) < 3 or len(set(leg_ids)) != len(leg_ids) or any(i not in range(6) for i in leg_ids):
        raise ValueError('At least three unique valid stance legs required')
    pos = xyz.copy()
    pos[2] += delta[0]
    pose_rpy = rpy + np.array([delta[1], delta[2], 0.], dtype=float)
    R = rotation_from_rpy(pose_rpy)
    targets = q.copy()
    for leg in leg_ids:
        foot_base = R.T @ (feet[leg] - pos)
        q_leg = ik_body_frame(leg, foot_base)
        if q_leg is None:
            return StanceFeasibility(False, f'leg {leg}: IK unreachable', None, -np.inf, -np.inf)
        q_leg = np.asarray(q_leg, dtype=float)
        if q_leg.shape != (3,) or not np.isfinite(q_leg).all():
            return StanceFeasibility(False, f'leg {leg}: invalid IK output', None, -np.inf, -np.inf)
        targets[leg] = q_leg
    stance = np.array(leg_ids, dtype=int)
    pos_margin = np.minimum(targets[stance] - lo[stance], hi[stance] - targets[stance])
    rate_margin = rates[stance]*dt_s - np.abs(targets[stance]-q[stance])
    pm, vm = float(np.min(pos_margin)), float(np.min(rate_margin))
    if pm < -1e-10:
        return StanceFeasibility(False, 'joint position violation', targets, pm, vm)
    if vm < -1e-10:
        return StanceFeasibility(False, 'joint rate violation', targets, pm, vm)
    return StanceFeasibility(True, 'stance IK/rate feasible; other checks pending', targets, pm, vm)
