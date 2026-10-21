"""Project-owned local-URDF tripod gait, leg kinematics, and bounded IK."""
from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import numpy as np

from hexapod_robot import JOINT_TYPES, LEG_ORDER, RobotModel

TRIPOD_A = ("lb", "lf", "rm")
TRIPOD_B = ("lm", "rb", "rf")


class ControlCadence:
    """Map the physics sample index to exact zero-order-hold control ticks."""

    def __init__(self, physics_dt_s: float, control_period_s: float):
        ratio = float(control_period_s) / float(physics_dt_s)
        rounded = int(round(ratio))
        if physics_dt_s <= 0 or control_period_s <= 0 or rounded < 1 or abs(ratio - rounded) > 1e-9:
            raise ValueError("control period must be a positive integer multiple of physics dt")
        self.steps_per_update = rounded

    def is_tick(self, physics_step: int) -> bool:
        return physics_step % self.steps_per_update == 0

    def tick_index(self, physics_step: int) -> int:
        return physics_step // self.steps_per_update


class StandStartGate:
    """Require one continuous measured stand interval before enabling gait."""

    def __init__(self, window_s: float, timeout_s: float):
        if window_s <= 0 or timeout_s < window_s:
            raise ValueError("stand start window and timeout are invalid")
        self.window_s = float(window_s)
        self.timeout_s = float(timeout_s)
        self.passing_since = None
        self.ready = False

    def update(self, elapsed_s: float, passing: bool) -> str:
        if self.ready:
            return "ready"
        if passing:
            if self.passing_since is None:
                self.passing_since = float(elapsed_s)
            if elapsed_s - self.passing_since + 1e-9 >= self.window_s:
                self.ready = True
                return "ready"
        else:
            self.passing_since = None
        if elapsed_s >= self.timeout_s:
            return "failed_start"
        return "waiting"


def walk_progress_failure(cycle_displacements, net_displacement_m: float,
                         minimum_cycles: int, minimum_net_m: float,
                         minimum_cycle_m: float):
    """Return a failure label unless progress is sustained beyond startup."""
    cycles = np.asarray(cycle_displacements, dtype=float)
    if (cycles.ndim != 1 or not np.all(np.isfinite(cycles)) or
            not math.isfinite(float(net_displacement_m))):
        return "invalid_progress_measurement"
    if len(cycles) < int(minimum_cycles):
        return "minimum_cycles_not_reached"
    if float(net_displacement_m) < float(minimum_net_m):
        return "minimum_net_forward_displacement_not_reached"
    if len(cycles) > 1 and np.any(cycles[1:] < float(minimum_cycle_m)):
        return "sustained_cycle_progress_not_reached"
    return None


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = map(float, rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]], dtype=float)


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("URDF revolute joint has a zero axis")
    x, y, z = axis / norm
    c, s, one = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array([[c + x*x*one, x*y*one-z*s, x*z*one+y*s],
                     [y*x*one+z*s, c+y*y*one, y*z*one-x*s],
                     [z*x*one-y*s, z*y*one+x*s, c+z*z*one]], dtype=float)


def _leg_path(model: RobotModel, leg: str):
    if leg not in LEG_ORDER:
        raise ValueError("unknown leg {}".format(leg))
    by_child = {joint.child: joint for joint in model.joints}
    path = []
    link = "l_{}_toe".format(leg)
    while link != model.root_link:
        joint = by_child.get(link)
        if joint is None:
            raise ValueError("toe {} is not connected to the model root".format(link))
        path.append(joint)
        link = joint.parent
    return tuple(reversed(path))


def leg_fk_jacobian(model: RobotModel, leg: str,
                    q_by_name: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return one toe center and its 3x3 translational Jacobian in root axes."""
    rotation = np.eye(3)
    position = np.zeros(3)
    actuated = []
    for joint in _leg_path(model, leg):
        position = position + rotation @ joint.xyz
        rotation = rotation @ _rpy_matrix(joint.rpy)
        if joint.kind in ("revolute", "continuous"):
            if joint.name not in q_by_name:
                raise ValueError("missing joint value {}".format(joint.name))
            joint_axis = rotation @ (joint.axis / np.linalg.norm(joint.axis))
            actuated.append((joint.name, position.copy(), joint_axis))
            rotation = rotation @ _axis_rotation(joint.axis, float(q_by_name[joint.name]))
        elif joint.kind == "prismatic":
            if joint.name not in q_by_name:
                raise ValueError("missing joint value {}".format(joint.name))
            position = position + rotation @ joint.axis * float(q_by_name[joint.name])
    expected = tuple("j_{}_{}".format(leg, kind) for kind in JOINT_TYPES)
    if tuple(item[0] for item in actuated) != expected:
        raise ValueError("{} does not have the canonical three-joint serial chain".format(leg))
    jacobian = np.column_stack([np.cross(axis, position - origin) for _, origin, axis in actuated])
    return position, jacobian


def leg_inverse_kinematics(model: RobotModel, leg: str, target_base_m: np.ndarray,
                           seed_by_name: Mapping[str, float], tolerance_m: float = 2e-5,
                           max_iterations: int = 80) -> Tuple[Dict[str, float], float]:
    """Solve bounded damped least-squares IK; reject unreachable/limit targets."""
    target = np.asarray(target_base_m, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("IK target must be a finite three-vector")
    names = tuple("j_{}_{}".format(leg, kind) for kind in JOINT_TYPES)
    q = np.array([float(seed_by_name[name]) for name in names], dtype=float)
    lower = np.array([model.limits[name].lower_rad for name in names], dtype=float)
    upper = np.array([model.limits[name].upper_rad for name in names], dtype=float)
    if np.any(q < lower) or np.any(q > upper):
        raise ValueError("IK seed violates URDF joint limits")
    damping = 0.015
    for _ in range(max_iterations):
        q_named = dict(seed_by_name)
        q_named.update(zip(names, q))
        position, jacobian = leg_fk_jacobian(model, leg, q_named)
        error = target - position
        residual = float(np.linalg.norm(error))
        if residual <= tolerance_m:
            return dict(zip(names, map(float, q))), residual
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping*damping*np.eye(3), error)
        max_step = float(np.max(np.abs(delta)))
        if max_step > 0.25:
            delta *= 0.25 / max_step
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            candidate = q + scale * delta
            if np.any(candidate < lower) or np.any(candidate > upper):
                continue
            candidate_named = dict(seed_by_name)
            candidate_named.update(zip(names, candidate))
            candidate_position, _ = leg_fk_jacobian(model, leg, candidate_named)
            if np.linalg.norm(target - candidate_position) < residual:
                q = candidate
                accepted = True
                break
        if not accepted:
            break
    q_named = dict(seed_by_name)
    q_named.update(zip(names, q))
    position, _ = leg_fk_jacobian(model, leg, q_named)
    residual = float(np.linalg.norm(target - position))
    if residual > tolerance_m:
        raise ValueError("{} IK target is unreachable within joint limits (residual {:.6f} m)".format(leg, residual))
    return dict(zip(names, map(float, q))), residual


def quaternion_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = map(float, quaternion)
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        raise ValueError("root quaternion is not normalized")
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]], dtype=float)


def feet_world(model: RobotModel, q_by_name: Mapping[str, float], body_xyz_m: np.ndarray,
               body_quaternion_xyzw: np.ndarray) -> Dict[str, np.ndarray]:
    rotation = quaternion_matrix_xyzw(body_quaternion_xyzw)
    origin = np.asarray(body_xyz_m, dtype=float)
    return {leg: origin + rotation @ leg_fk_jacobian(model, leg, q_by_name)[0] for leg in LEG_ORDER}


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, float(value)))
    return value * value * value * (10.0 + value * (-15.0 + 6.0 * value))


def _ramp_displacement(elapsed_s: float, speed_m_s: float, ramp_s: float) -> float:
    time_s = max(0.0, float(elapsed_s))
    if ramp_s <= 0 or time_s >= ramp_s:
        return speed_m_s * max(0.0, time_s - ramp_s) + 0.5 * speed_m_s * ramp_s
    return 0.5 * speed_m_s * time_s * time_s / ramp_s


class TripodGait:
    """Alternating-tripod world-foot scheduler with causal touchdown anchors."""

    def __init__(self, nominal_feet_base: Mapping[str, np.ndarray], initial_feet_world: Mapping[str, np.ndarray],
                 initial_body_xyz_m: np.ndarray, initial_yaw_rad: float, settings: Mapping[str, float]):
        self.nominal_feet_base = {leg: np.asarray(nominal_feet_base[leg], dtype=float).copy() for leg in LEG_ORDER}
        self.anchors = {leg: np.asarray(initial_feet_world[leg], dtype=float).copy() for leg in LEG_ORDER}
        self.initial_body = np.asarray(initial_body_xyz_m, dtype=float).copy()
        self.yaw = float(initial_yaw_rad)
        self.speed = float(settings["speed_m_s"])
        self.frequency = float(settings["frequency_hz"])
        self.duty = float(settings["duty_factor"])
        self.clearance = float(settings["swing_height_m"])
        self.ramp_s = float(settings["speed_ramp_s"])
        self.touchdown_timeout_s = float(settings["touchdown_timeout_s"])
        self.max_slip_m = float(settings["max_stance_slip_m"])
        self.stance = {leg: True for leg in LEG_ORDER}
        self.swing_start: Dict[str, np.ndarray] = {}
        self.swing_end: Dict[str, np.ndarray] = {}
        self.swing_started: Dict[str, float] = {}
        self.pending_touchdown_since: Dict[str, float] = {}
        self.last_phase = 0.0
        self._started = False

    def body_reference(self, elapsed_s: float) -> np.ndarray:
        distance = _ramp_displacement(elapsed_s, self.speed, self.ramp_s)
        forward = np.array([-math.sin(self.yaw), math.cos(self.yaw), 0.0])
        result = self.initial_body.copy()
        result += forward * distance
        return result

    def phase_for_leg(self, leg: str, elapsed_s: float) -> float:
        offset = 0.0 if leg in TRIPOD_A else 0.5
        return (float(elapsed_s) * self.frequency + offset) % 1.0

    def update(self, elapsed_s: float, actual_feet_world: Mapping[str, np.ndarray],
               contacts: Mapping[str, bool]) -> Dict[str, object]:
        if elapsed_s < 0:
            raise ValueError("gait elapsed time must be nonnegative")
        cycle_phase = (float(elapsed_s) * self.frequency) % 1.0
        phases = {leg: self.phase_for_leg(leg, elapsed_s) for leg in LEG_ORDER}
        desired = {}
        planned_support = {}
        for leg in LEG_ORDER:
            phase = phases[leg]
            should_stance = phase < self.duty
            if self.stance[leg] and not should_stance:
                self.stance[leg] = False
                self.swing_start[leg] = self.anchors[leg].copy()
                self.swing_started[leg] = float(elapsed_s)
                swing_duration = (1.0 - self.duty) / self.frequency
                touchdown_time = elapsed_s + max(0.0, swing_duration * (1.0 - (phase - self.duty) / (1.0 - self.duty)))
                self.swing_end[leg] = self.body_reference(touchdown_time) + self._rotate_nominal(self.nominal_feet_base[leg])
            elif not self.stance[leg] and should_stance:
                self.stance[leg] = True
                if contacts.get(leg, False):
                    self.anchors[leg] = np.asarray(actual_feet_world[leg], dtype=float).copy()
                    self.pending_touchdown_since.pop(leg, None)
                else:
                    self.anchors[leg] = self.swing_end[leg].copy()
                    self.pending_touchdown_since[leg] = float(elapsed_s)
            if self.stance[leg] and leg in self.pending_touchdown_since:
                if contacts.get(leg, False):
                    self.anchors[leg] = np.asarray(actual_feet_world[leg], dtype=float).copy()
                    del self.pending_touchdown_since[leg]
                elif elapsed_s - self.pending_touchdown_since[leg] > self.touchdown_timeout_s:
                    raise RuntimeError("{} failed to establish contact at touchdown".format(leg))
            planned_support[leg] = bool(self.stance[leg])
            if self.stance[leg]:
                desired[leg] = self.anchors[leg].copy()
            else:
                swing_duration = (1.0 - self.duty) / self.frequency
                progress = (phase - self.duty) / (1.0 - self.duty)
                smooth = _smoothstep(progress)
                point = self.swing_start[leg] + smooth * (self.swing_end[leg] - self.swing_start[leg])
                point = point.copy()
                point[2] += self.clearance * 16.0 * progress*progress*(1.0-progress)*(1.0-progress)
                desired[leg] = point
        self.last_phase = cycle_phase
        pending = {leg: elapsed_s - started for leg, started in self.pending_touchdown_since.items()}
        return {"desired_feet_world": desired, "anchors_world": {leg: value.copy() for leg, value in self.anchors.items()},
                "planned_support": planned_support, "phase_by_leg": phases, "cycle_phase": cycle_phase,
                "planned_body_xyz_m": self.body_reference(elapsed_s), "pending_touchdown_s": pending}

    def _rotate_nominal(self, vector: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return rotation @ vector


def create_gait_after_stand_gate(stand_gate_verified: bool,
                                 nominal_feet_base: Mapping[str, np.ndarray],
                                 initial_feet_world: Mapping[str, np.ndarray],
                                 initial_body_xyz_m: np.ndarray, initial_yaw_rad: float,
                                 settings: Mapping[str, float], contacts: Mapping[str, bool]):
    """Create and initialize the gait only after a measured stand gate passes."""
    if stand_gate_verified is not True:
        raise ValueError("tripod gait cannot start before a verified stand gate")
    gait = TripodGait(nominal_feet_base, initial_feet_world, initial_body_xyz_m,
                      initial_yaw_rad, settings)
    return gait, gait.update(0.0, initial_feet_world, contacts)
