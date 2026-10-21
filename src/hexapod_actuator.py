"""Pure-NumPy finite-bandwidth MIT-style joint position actuator model."""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np


class MitPdActuator:
    """400 Hz local PD servo with delayed command packets and torque response lag."""

    def __init__(self, dt_s: float, internal_period_s: float, command_delay_s: float,
                 torque_bandwidth_hz: float, kp: float, kd: float, effort_limits_nm: np.ndarray,
                 initial_position_rad: np.ndarray):
        self.dt = float(dt_s)
        self.internal_period = float(internal_period_s)
        self.delay_s = float(command_delay_s)
        self.bandwidth = float(torque_bandwidth_hz)
        self.kp = float(kp)
        self.kd = float(kd)
        self.effort_limits = np.asarray(effort_limits_nm, dtype=float).copy()
        self.steps_per_update = int(round(self.internal_period / self.dt)) if self.dt > 0 else 0
        delay_ratio = self.delay_s / self.dt if self.dt > 0 else float("nan")
        self.delay_steps = int(round(delay_ratio)) if math.isfinite(delay_ratio) else -1
        if (self.dt <= 0 or self.internal_period <= 0 or self.steps_per_update != 1 or
                abs(self.internal_period / self.dt - self.steps_per_update) > 1e-9):
            raise ValueError("MIT actuator internal update period must match positive physics dt")
        if (not math.isfinite(self.delay_s) or self.delay_s < 0 or self.delay_steps < 0 or
                abs(delay_ratio - self.delay_steps) > 1e-9):
            raise ValueError("MIT actuator command delay must be an integer number of physics steps")
        if not math.isfinite(self.bandwidth) or self.bandwidth <= 0:
            raise ValueError("MIT actuator torque bandwidth must be positive and finite")
        if not math.isfinite(self.kp) or not math.isfinite(self.kd) or self.kp < 0 or self.kd < 0:
            raise ValueError("MIT actuator gains must be finite and nonnegative")
        if self.effort_limits.ndim != 1 or not np.all(np.isfinite(self.effort_limits)) or np.any(self.effort_limits <= 0):
            raise ValueError("MIT actuator effort limits must be a positive finite vector")
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * self.bandwidth * self.dt)
        self.reset(initial_position_rad)

    def reset(self, initial_position_rad: np.ndarray, initial_velocity_rad_s: Optional[np.ndarray] = None,
              initial_feedforward_nm: Optional[np.ndarray] = None) -> None:
        """Explicitly seed delayed packets from measured/configured stand state."""
        position = np.asarray(initial_position_rad, dtype=float)
        velocity = np.zeros_like(position) if initial_velocity_rad_s is None else np.asarray(initial_velocity_rad_s, dtype=float)
        feedforward = np.zeros_like(position) if initial_feedforward_nm is None else np.asarray(initial_feedforward_nm, dtype=float)
        if any(value.shape != self.effort_limits.shape or not np.all(np.isfinite(value))
               for value in (position, velocity, feedforward)):
            raise ValueError("MIT actuator initialization vectors must match effort limits and be finite")
        initial = {"position": position.copy(), "velocity": velocity.copy(), "feedforward": feedforward.copy()}
        self._queue = [dict((key, value.copy()) for key, value in initial.items())
                       for _ in range(self.delay_steps)]
        self._applied_torque = np.zeros_like(position)
        self._last_delayed = initial

    def step(self, measured_position_rad: np.ndarray, measured_velocity_rad_s: np.ndarray,
             command_position_rad: np.ndarray, command_velocity_rad_s: Optional[np.ndarray] = None,
             command_feedforward_nm: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Advance one internal actuator sample using measured q/dq feedback."""
        measured_position = np.asarray(measured_position_rad, dtype=float)
        measured_velocity = np.asarray(measured_velocity_rad_s, dtype=float)
        command_position = np.asarray(command_position_rad, dtype=float)
        command_velocity = np.zeros_like(command_position) if command_velocity_rad_s is None else np.asarray(command_velocity_rad_s, dtype=float)
        command_feedforward = np.zeros_like(command_position) if command_feedforward_nm is None else np.asarray(command_feedforward_nm, dtype=float)
        vectors = (measured_position, measured_velocity, command_position, command_velocity, command_feedforward)
        if any(value.shape != self.effort_limits.shape or not np.all(np.isfinite(value)) for value in vectors):
            raise ValueError("MIT actuator sample vectors must match effort limits and be finite")
        packet = {"position": command_position.copy(), "velocity": command_velocity.copy(),
                  "feedforward": command_feedforward.copy()}
        if self.delay_steps:
            self._queue.append(packet)
            delayed = self._queue.pop(0)
        else:
            delayed = packet
        raw = (self.kp * (delayed["position"] - measured_position) +
               self.kd * (delayed["velocity"] - measured_velocity) + delayed["feedforward"])
        requested = np.clip(raw, -self.effort_limits, self.effort_limits)
        saturated = np.abs(raw) >= self.effort_limits - 1e-7
        self._applied_torque += self.alpha * (requested - self._applied_torque)
        self._applied_torque = np.clip(self._applied_torque, -self.effort_limits, self.effort_limits)
        self._last_delayed = delayed
        return {"desired_position_rad": command_position.copy(),
                "desired_velocity_rad_s": command_velocity.copy(),
                "delayed_position_rad": delayed["position"].copy(),
                "delayed_velocity_rad_s": delayed["velocity"].copy(),
                "delayed_feedforward_nm": delayed["feedforward"].copy(),
                "raw_torque_nm": raw.copy(), "requested_torque_nm": requested.copy(),
                "applied_model_torque_nm": self._applied_torque.copy(),
                "torque_saturated": saturated.copy(), "response_alpha": self.alpha,
                "response_state_nm": self._applied_torque.copy()}
