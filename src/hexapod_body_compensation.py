"""Bounded proprioceptive roll, pitch, and height body-reference correction."""
from __future__ import annotations

import math

import numpy as np

from hexapod_walk import quaternion_matrix_xyzw


SETTINGS = {
    "roll_angle_gain": 0.8,
    "roll_rate_gain_s": 0.04,
    "pitch_angle_gain": 0.8,
    "pitch_rate_gain_s": 0.04,
    "height_gain_s": 0.5,
    "vertical_velocity_gain_s": 0.04,
    "max_angle_rad": 0.05,
    "max_height_m": 0.01,
    "max_angle_rate_rad_s": 0.3,
    "max_height_rate_m_s": 0.03,
    "command_filter_cutoff_hz": 2.0,
    "command_filter_order": 2,
}


def _rpy_from_rotation(rotation: np.ndarray):
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    pitch = math.asin(max(-1.0, min(1.0, -float(rotation[2, 0]))))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return roll, pitch, yaw


def body_reference_rotation(yaw_rad: float, roll_correction_rad: float,
                            pitch_correction_rad: float) -> np.ndarray:
    """Return Rz(yaw) @ R_rpy(roll, pitch, 0) for the planned body frame."""
    cr, sr = math.cos(roll_correction_rad), math.sin(roll_correction_rad)
    cp, sp = math.cos(pitch_correction_rad), math.sin(pitch_correction_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


class BoundedBodyCompensator:
    """Fixed-gain 50 Hz RPZ correction using only estimated proprioceptive state."""

    def __init__(self, enabled: bool, control_period_s: float = 0.02,
                 rate_feedback: bool = True):
        if type(enabled) is not bool:
            raise ValueError("compensation enabled must be a boolean")
        if type(rate_feedback) is not bool:
            raise ValueError("rate_feedback must be a boolean")
        if type(control_period_s) not in (int, float) or not math.isfinite(float(control_period_s)) or control_period_s <= 0:
            raise ValueError("control period must be positive and finite")
        self.enabled = enabled
        self.rate_feedback = rate_feedback
        self.dt = float(control_period_s)
        self.reset()

    def reset(self) -> None:
        self.roll_rad = 0.0
        self.pitch_rad = 0.0
        self.height_m = 0.0
        self.filtered_roll_target_stage1_rad = 0.0
        self.filtered_roll_target_stage2_rad = 0.0
        self.filtered_pitch_target_stage1_rad = 0.0
        self.filtered_pitch_target_stage2_rad = 0.0
        self.filtered_height_target_stage1_m = 0.0
        self.filtered_height_target_stage2_m = 0.0

    def update(self, imu_quaternion_xyzw, gyro_world_rad_s, estimated_body_xyz_m,
               estimated_body_velocity_m_s, planned_body_z_m: float):
        """Update one bounded correction from the current 50 Hz proprioceptive sample."""
        zero = {"roll_rad": 0.0, "pitch_rad": 0.0, "height_m": 0.0}
        if not self.enabled:
            self.reset()
            return zero
        quat = np.asarray(imu_quaternion_xyzw, dtype=float)
        gyro_world = np.asarray(gyro_world_rad_s, dtype=float)
        xyz = np.asarray(estimated_body_xyz_m, dtype=float)
        velocity = np.asarray(estimated_body_velocity_m_s, dtype=float)
        if (quat.shape != (4,) or gyro_world.shape != (3,) or xyz.shape != (3,) or
                velocity.shape != (3,) or not all(np.all(np.isfinite(x))
                                                  for x in (quat, gyro_world, xyz, velocity)) or
                not math.isfinite(float(planned_body_z_m))):
            raise ValueError("compensation requires finite proprioceptive state samples")
        norm = float(np.linalg.norm(quat))
        if norm < 1e-12:
            raise ValueError("IMU quaternion must be nonzero")
        rotation_world_body = quaternion_matrix_xyzw(quat / norm)
        roll, pitch, _ = _rpy_from_rotation(rotation_world_body)
        omega_body = rotation_world_body.T @ gyro_world
        target_roll = float(np.clip(-SETTINGS["roll_angle_gain"] * roll -
                                    (SETTINGS["roll_rate_gain_s"] * omega_body[0]
                                     if self.rate_feedback else 0.0),
                                    -SETTINGS["max_angle_rad"], SETTINGS["max_angle_rad"]))
        target_pitch = float(np.clip(-SETTINGS["pitch_angle_gain"] * pitch -
                                     (SETTINGS["pitch_rate_gain_s"] * omega_body[1]
                                      if self.rate_feedback else 0.0),
                                     -SETTINGS["max_angle_rad"], SETTINGS["max_angle_rad"]))
        target_height = float(np.clip(SETTINGS["height_gain_s"] *
                                      (float(planned_body_z_m) - xyz[2]) -
                                      SETTINGS["vertical_velocity_gain_s"] * velocity[2],
                                      -SETTINGS["max_height_m"], SETTINGS["max_height_m"]))
        filter_alpha = 1.0 - math.exp(-2.0 * math.pi *
                                      SETTINGS["command_filter_cutoff_hz"] * self.dt)
        self.filtered_roll_target_stage1_rad += filter_alpha * (
            target_roll - self.filtered_roll_target_stage1_rad)
        self.filtered_roll_target_stage2_rad += filter_alpha * (
            self.filtered_roll_target_stage1_rad - self.filtered_roll_target_stage2_rad)
        self.filtered_pitch_target_stage1_rad += filter_alpha * (
            target_pitch - self.filtered_pitch_target_stage1_rad)
        self.filtered_pitch_target_stage2_rad += filter_alpha * (
            self.filtered_pitch_target_stage1_rad - self.filtered_pitch_target_stage2_rad)
        self.filtered_height_target_stage1_m += filter_alpha * (
            target_height - self.filtered_height_target_stage1_m)
        self.filtered_height_target_stage2_m += filter_alpha * (
            self.filtered_height_target_stage1_m - self.filtered_height_target_stage2_m)
        angle_delta = SETTINGS["max_angle_rate_rad_s"] * self.dt
        height_delta = SETTINGS["max_height_rate_m_s"] * self.dt
        self.roll_rad += float(np.clip(self.filtered_roll_target_stage2_rad - self.roll_rad,
                                       -angle_delta, angle_delta))
        self.pitch_rad += float(np.clip(self.filtered_pitch_target_stage2_rad - self.pitch_rad,
                                        -angle_delta, angle_delta))
        self.height_m += float(np.clip(self.filtered_height_target_stage2_m - self.height_m,
                                       -height_delta, height_delta))
        return {"roll_rad": self.roll_rad, "pitch_rad": self.pitch_rad, "height_m": self.height_m}
