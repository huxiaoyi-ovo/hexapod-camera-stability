"""Bounded camera-space RPZ correction using a frozen local support model.

This is a simulation prototype, not an identified plant model. It uses a
fixed-posture URDF inertia and rigid confirmed-support approximation.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.linalg import expm
from scipy.optimize import lsq_linear

from controller import Camera, TrackedScene, image_interaction_matrix
from hexapod_robot import JOINT_TYPES, LEG_ORDER, RobotModel, _axis_rotation, _origin, _rpy_matrix
from hexapod_walk import leg_fk_jacobian, quaternion_matrix_xyzw


SETTINGS = {
    "method": "bounded_camera_twist_least_squares",
    "regularization": 1e-3,
    "command_filter_cutoff_hz": 2.0,
    "command_filter_order": 2,
    "max_angle_rad": 0.05,
    "max_height_m": 0.01,
    "max_angle_rate_rad_s": 0.3,
    "max_height_rate_m_s": 0.03,
    "mass_model": "composite URDF link inertias at configured reference posture",
    "joint_response_model": "support-map transpose times configured MIT PD gains",
    "actuator_model": "configured command delay and first-order torque bandwidth; perturbation state starts at zero",
    "prediction_horizon": "command_delay_s + camera.frame_interval_s",
    "local_response_assumption": "frozen posture, rigid confirmed feet, no slip/contact changes",
    "disturbance_prediction": "current residual twist held constant over horizon",
    "objective": "weighted pixel displacement squared + regularization times normalized RPZ increment norm squared",
    "direct_pose_angle_rate_feedback": False,
    "plant_calibrated": False,
}

def _link_inertias(model: RobotModel, q_by_name):
    root = ET.parse(str(model.path)).getroot()
    links = {}
    for node in root.findall("link"):
        inertial = node.find("inertial")
        if inertial is None:
            continue
        mass_node = inertial.find("mass")
        inertia_node = inertial.find("inertia")
        if mass_node is None or inertia_node is None:
            continue
        mass = float(mass_node.get("value"))
        xyz, rpy = _origin(inertial)
        a = inertia_node.attrib
        inertia = np.array([[float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
                            [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
                            [float(a["ixz"]), float(a["iyz"]), float(a["izz"])]], dtype=float)
        links[node.get("name")] = (mass, xyz, _rpy_matrix(rpy), inertia)

    transforms = {model.root_link: (np.eye(3), np.zeros(3))}
    pending = list(model.joints)
    while pending:
        remaining = []
        for joint in pending:
            if joint.parent not in transforms:
                remaining.append(joint)
                continue
            parent_r, parent_p = transforms[joint.parent]
            local_r = _rpy_matrix(joint.rpy)
            local_p = joint.xyz
            if joint.kind in ("revolute", "continuous"):
                local_r = local_r @ _axis_rotation(joint.axis, float(q_by_name[joint.name]))
            elif joint.kind == "prismatic":
                local_p = local_p + joint.axis * float(q_by_name[joint.name])
            transforms[joint.child] = (parent_r @ local_r, parent_p + parent_r @ local_p)
        if len(remaining) == len(pending):
            raise ValueError("URDF link tree is disconnected")
        pending = remaining
    entries = []
    for name, (mass, local_com, local_r, inertia_local) in links.items():
        if name not in transforms:
            continue
        rotation, origin = transforms[name]
        com = origin + rotation @ local_com
        inertia_world = rotation @ local_r @ inertia_local @ local_r.T @ rotation.T
        entries.append((mass, com, inertia_world))
    if not entries:
        raise ValueError("URDF contains no link inertias")
    return entries


def generalized_mass_matrix(model: RobotModel, q_by_name):
    """Frozen-posture mass matrix for root roll, pitch, and vertical motion."""
    entries = _link_inertias(model, q_by_name)
    matrix = np.zeros((3, 3), dtype=float)
    total_mass = 0.0
    for mass, com, inertia in entries:
        x, y, z = com
        matrix[:2, :2] += inertia[:2, :2] + mass * np.array([[y*y + z*z, -x*y],
                                                              [-x*y, x*x + z*z]])
        matrix[0, 2] += mass * y
        matrix[2, 0] += mass * y
        matrix[1, 2] -= mass * x
        matrix[2, 1] -= mass * x
        matrix[2, 2] += mass
        total_mass += mass
    if total_mass <= 0 or np.min(np.linalg.eigvalsh(matrix)) <= 0:
        raise ValueError("URDF reduced mass matrix is not positive definite")
    return matrix


def camera_residual_twist(rotation_wb, rotation_bc, position_bc, velocity_world,
                          omega_world, nominal_velocity_world):
    """Camera-frame twist relative to the shared nominal camera translation."""
    rotation_wc = np.asarray(rotation_wb, dtype=float) @ np.asarray(rotation_bc, dtype=float)
    omega = np.asarray(omega_world, dtype=float)
    velocity_camera_world = np.asarray(velocity_world, dtype=float) + np.cross(
        omega, np.asarray(rotation_wb, dtype=float) @ np.asarray(position_bc, dtype=float))
    return np.r_[rotation_wc.T @ (velocity_camera_world - np.asarray(nominal_velocity_world, dtype=float)),
                 rotation_wc.T @ omega]


def uncontrollable_projection(mapping, residual):
    """Return the component of a pixel residual outside the modeled column space."""
    matrix = np.asarray(mapping, dtype=float)
    vector = np.asarray(residual, dtype=float)
    return vector - matrix @ np.linalg.pinv(matrix, rcond=1e-8) @ vector


def _body_pose_rotation(yaw, correction):
    roll, pitch = map(float, correction[:2])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
    ry = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
    rz = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
    return rz @ ry @ rx


class CameraSpaceCompensator:
    """Choose a bounded RPZ command increment from predicted image residual."""

    def __init__(self, model: RobotModel, reference_q, camera_config, camera_data,
                 scene_data, control_config, actuator_config, control_period_s,
                 physics_dt_s, mass_matrix=None):
        self.model = model
        self.dt = float(control_period_s)
        self.physics_dt = float(physics_dt_s)
        if self.dt <= 0 or self.physics_dt <= 0:
            raise ValueError("controller periods must be positive")
        self.reference_q = dict(reference_q)
        self.camera = Camera(**camera_data)
        self.scene = TrackedScene(np.asarray(scene_data["uv"], dtype=float),
                                  np.asarray(scene_data["depth_m"], dtype=float),
                                  np.asarray(scene_data["weights"], dtype=float))
        self.scene.validate()
        self.p_bc = np.asarray(camera_config["base_to_camera_position_m"], dtype=float)
        self.q_bc = np.asarray(camera_config["base_to_camera_quaternion_xyzw"], dtype=float)
        if self.p_bc.shape != (3,) or self.q_bc.shape != (4,) or not np.all(np.isfinite(self.p_bc)):
            raise ValueError("camera extrinsics must be finite vectors")
        qnorm = float(np.linalg.norm(self.q_bc))
        if qnorm < 1e-12 or not np.isfinite(qnorm):
            raise ValueError("camera extrinsic quaternion must be nonzero")
        self.r_bc = quaternion_matrix_xyzw(self.q_bc / qnorm)
        self.L = image_interaction_matrix(self.camera, self.scene)
        self.weight = np.repeat(self.scene.weights / np.sum(self.scene.weights), 2)
        self.Lw = np.sqrt(self.weight)[:, None] * self.L
        self.kp = float(control_config["kp_nm_rad"])
        self.kd = float(control_config["kd_nm_s_rad"])
        self.effort = float(control_config["effort_limit_nm"])
        self.op_velocity = float(control_config.get("command_velocity_limit_rad_s",
                                                   control_config["joint_velocity_limit_rad_s"]))
        self.delay_s = float(actuator_config["command_delay_s"])
        self.torque_bandwidth_hz = float(actuator_config["torque_bandwidth_hz"])
        if not (self.kp > 0 and self.kd >= 0 and self.effort > 0 and self.op_velocity > 0 and
                self.delay_s >= 0 and self.torque_bandwidth_hz > 0):
            raise ValueError("actuator response settings are invalid")
        self.mass = (generalized_mass_matrix(model, self.reference_q) if mass_matrix is None
                     else np.asarray(mass_matrix, dtype=float))
        if self.mass.shape != (3, 3) or not np.all(np.isfinite(self.mass)):
            raise ValueError("reduced mass matrix must be a finite 3x3 matrix")
        self.filter_alpha = 1.0 - math.exp(-2.0 * math.pi * SETTINGS["command_filter_cutoff_hz"] * self.dt)
        self.filter1 = np.zeros(3)
        self.filter2 = np.zeros(3)
        self.correction = np.zeros(3)
        self.previous_correction = np.zeros(3)
        self.delay_ticks = int(round(self.delay_s / self.dt))
        if abs(self.delay_ticks * self.dt - self.delay_s) > 1e-8:
            raise ValueError("camera-space model requires delay to be an integer control period")
        self.reset()

    def reset(self):
        self.filter1[:] = 0.0
        self.filter2[:] = 0.0
        self.correction[:] = 0.0
        self.previous_correction[:] = 0.0
        self.command_target = np.zeros(3)

    def _support_map(self, q_by_name, planned_body, yaw, anchors, support):
        blocks = []
        all_names = []
        for leg in LEG_ORDER:
            if not support.get(leg, False):
                continue
            anchor = np.asarray(anchors.get(leg), dtype=float)
            if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
                continue
            _, jacobian = leg_fk_jacobian(self.model, leg, q_by_name)
            derivative = np.zeros((3, 3))
            eps = 1e-6
            for axis in range(3):
                plus = np.zeros(3); minus = np.zeros(3)
                plus[axis] = eps; minus[axis] = -eps
                p_plus = _body_pose_rotation(yaw, plus).T @ (anchor - planned_body - np.array([0., 0., plus[2]]))
                p_minus = _body_pose_rotation(yaw, minus).T @ (anchor - planned_body - np.array([0., 0., minus[2]]))
                derivative[:, axis] = (p_plus - p_minus) / (2.0 * eps)
            try:
                q_map = np.linalg.solve(jacobian, derivative)
            except np.linalg.LinAlgError:
                q_map = np.linalg.pinv(jacobian, rcond=1e-5) @ derivative
            blocks.append(q_map)
            all_names.extend("j_{}_{}".format(leg, joint) for joint in JOINT_TYPES)
        if not blocks:
            return np.zeros((0, 3)), all_names
        return np.vstack(blocks), all_names

    def _discrete_response(self, q_map):
        # x=[RPZ, dRPZ, generalized actuator torque]; torque is first-order lagged.
        stiffness = self.kp * (q_map.T @ q_map)
        damping = self.kd * (q_map.T @ q_map)
        state = np.zeros((9, 9), dtype=float)
        state[:3, 3:6] = np.eye(3)
        state[3:6, 6:9] = np.linalg.inv(self.mass)
        bandwidth = 2.0 * math.pi * self.torque_bandwidth_hz
        state[6:9, :3] = -bandwidth * stiffness
        state[6:9, 3:6] = -bandwidth * damping
        state[6:9, 6:9] = -bandwidth * np.eye(3)
        input_matrix = np.zeros((9, 6), dtype=float)
        input_matrix[6:9, :3] = bandwidth * stiffness
        input_matrix[6:9, 3:6] = bandwidth * damping
        augmented = np.zeros((15, 15), dtype=float)
        augmented[:9, :9] = state
        augmented[:9, 9:] = input_matrix
        transition = expm(augmented * self.physics_dt)
        return transition[:9, :9], transition[:9, 9:]

    def _step_response(self, command, transition, input_step, steps):
        state = np.zeros(9)
        f1 = np.zeros(3); f2 = np.zeros(3)
        queue = [np.zeros(3) for _ in range(self.delay_ticks)]
        previous_delayed = np.zeros(3)
        steps_per_tick = max(1, int(round(self.dt / self.physics_dt)))
        delayed = np.zeros(3)
        for k in range(steps):
            if k % steps_per_tick == 0:
                f1 += self.filter_alpha * (command - f1)
                f2 += self.filter_alpha * (f1 - f2)
                if queue:
                    queue.append(f2.copy())
                    delayed = queue.pop(0)
                else:
                    delayed = f2.copy()
                velocity = (delayed - previous_delayed) / self.dt
                previous_delayed = delayed.copy()
            state = transition @ state + input_step @ np.r_[delayed, velocity]
        return state[:3].copy()

    def update(self, *, q_by_name, support, planned_body, yaw, anchors, estimated_xyz,
               estimated_velocity, imu_quaternion, gyro_world, nominal_velocity_world,
               previous_applied_torque_by_name=None):
        support_map, joint_names = self._support_map(q_by_name, planned_body, yaw, anchors, support)
        state_vectors = (np.asarray(planned_body, dtype=float), np.asarray(estimated_xyz, dtype=float),
                         np.asarray(estimated_velocity, dtype=float), np.asarray(gyro_world, dtype=float),
                         np.asarray(nominal_velocity_world, dtype=float))
        if (any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in state_vectors) or
                not math.isfinite(float(yaw))):
            raise ValueError("camera-space controller requires finite body, velocity, gyro, and nominal state")
        count = len(joint_names) // 3
        diagnostics = {"support_count": count, "mapping_rank": 0,
                       "mapping_singular_values": [], "predicted_pixel_before": None,
                       "predicted_pixel_after": None, "uncontrollable_pixel_norm": None,
                       "bound_scale": 0.0, "hold_reason": None,
                       "model_assumption": SETTINGS["local_response_assumption"]}
        if count < 3 or support_map.shape[0] == 0:
            diagnostics["hold_reason"] = "fewer_than_three_confirmed_planned_support_legs"
            return self.correction.copy(), diagnostics
        singular = np.linalg.svd(support_map, compute_uv=False)
        rank = int(np.sum(singular > max(support_map.shape) * np.finfo(float).eps * singular[0]))
        diagnostics.update(mapping_rank=rank, mapping_singular_values=singular.tolist())
        if rank < 3:
            diagnostics["hold_reason"] = "support_mapping_unobservable"
            return self.correction.copy(), diagnostics
        rotation_wb = quaternion_matrix_xyzw(np.asarray(imu_quaternion, dtype=float))
        omega = np.asarray(gyro_world, dtype=float)
        velocity = np.asarray(estimated_velocity, dtype=float)
        rotation_wc = rotation_wb @ self.r_bc
        twist = camera_residual_twist(rotation_wb, self.r_bc, self.p_bc,
                                      velocity, omega, nominal_velocity_world)
        disturbance = self.Lw @ twist
        transition, input_step = self._discrete_response(support_map)
        steps = int(math.ceil((self.delay_s + self.camera.frame_interval_s) / self.physics_dt))
        horizon = steps * self.physics_dt
        response_columns = []
        for index in range(3):
            command = np.zeros(3); command[index] = 1.0
            response_columns.append(self._step_response(command, transition, input_step, steps))
        response = np.column_stack(response_columns)
        body_to_camera = np.zeros((6, 3))
        heading_rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                                     [math.sin(yaw), math.cos(yaw), 0.0],
                                     [0.0, 0.0, 1.0]])
        for index in range(3):
            rate = response[:, index] / max(horizon, 1e-12)
            omega_delta_world = heading_rotation @ np.array([rate[0], rate[1], 0.0])
            v_delta_world = np.cross(omega_delta_world, rotation_wb @ self.p_bc) + np.array([0., 0., rate[2]])
            body_to_camera[:, index] = np.r_[rotation_wc.T @ v_delta_world,
                                             rotation_wc.T @ omega_delta_world]
        frame_s = float(self.camera.frame_interval_s)
        image_map = frame_s * self.Lw @ body_to_camera
        before_vector = frame_s * disturbance
        before = float(np.linalg.norm(before_vector))
        singular_values = np.linalg.svd(image_map, compute_uv=False)
        image_rank = int(np.linalg.matrix_rank(image_map))
        if image_rank == 0:
            diagnostics.update(image_mapping_singular_values=singular_values.tolist(),
                               image_mapping_rank=0,
                               hold_reason="camera_projection_has_no_observable_control_direction")
            return self.correction.copy(), diagnostics
        uncontrollable = uncontrollable_projection(image_map, before_vector)
        step_bounds = np.array([SETTINGS["max_angle_rate_rad_s"] * self.dt,
                                SETTINGS["max_angle_rate_rad_s"] * self.dt,
                                SETTINGS["max_height_rate_m_s"] * self.dt])
        diagnostics.update(predicted_pixel_before=before,
                           uncontrollable_pixel_norm=float(np.linalg.norm(uncontrollable)),
                           image_mapping_singular_values=singular_values.tolist(),
                           image_mapping_rank=image_rank,
                           prediction_horizon_s=horizon,
                           objective_frame_interval_s=frame_s,
                           regularization=SETTINGS["regularization"],
                           regularization_increment_normalization=step_bounds.tolist(),
                           objective_units="weighted pixel displacement squared plus normalized RPZ increment squared")
        regularizer = math.sqrt(float(SETTINGS["regularization"])) * np.diag(1.0 / step_bounds)
        matrix = np.vstack((image_map, regularizer))
        target = np.r_[-before_vector, np.zeros(3)]
        absolute_bounds = np.array([SETTINGS["max_angle_rad"], SETTINGS["max_angle_rad"],
                                    SETTINGS["max_height_m"]])
        lower = np.maximum(-step_bounds, -absolute_bounds - self.command_target)
        upper = np.minimum(step_bounds, absolute_bounds - self.command_target)
        solution = lsq_linear(matrix, target, bounds=(lower, upper), lsmr_tol="auto")
        increment = solution.x
        q_effect = support_map @ increment
        qmax_values = []
        for name in joint_names:
            limit = self.model.limits[name]
            q_now = float(q_by_name[name])
            position_headroom = max(0.0, min(limit.upper_rad - q_now, q_now - limit.lower_rad))
            effort_headroom = (max(0.0, self.effort - abs(float(previous_applied_torque_by_name.get(name, 0.0))))
                               if previous_applied_torque_by_name else self.effort)
            qmax_values.append(min(self.op_velocity * self.dt, position_headroom,
                                   effort_headroom / max(self.kp + self.kd / self.dt, 1e-9)))
        qmax = np.asarray(qmax_values)
        max_ratio = float(np.max(np.abs(q_effect) / np.maximum(qmax, 1e-12))) if len(q_effect) else 0.0
        scale = min(1.0, 1.0 / max_ratio) if max_ratio > 1.0 else 1.0
        increment *= scale
        after_vector = before_vector + image_map @ increment
        prediction_rejected = float(np.linalg.norm(after_vector)) > before + 1e-12
        if prediction_rejected:
            increment[:] = 0.0
            after_vector = before_vector.copy()
        self.previous_correction = self.correction.copy()
        self.command_target = np.clip(self.command_target + increment, -absolute_bounds, absolute_bounds)
        self.filter1 += self.filter_alpha * (self.command_target - self.filter1)
        self.filter2 += self.filter_alpha * (self.filter1 - self.filter2)
        filtered = self.filter2
        delta = np.r_[filtered[:2] - self.previous_correction[:2], filtered[2] - self.previous_correction[2]]
        delta = np.clip(delta, -step_bounds, step_bounds)
        self.correction = self.previous_correction + delta
        diagnostics.update(predicted_pixel_after=float(np.linalg.norm(after_vector)),
                           bound_scale=scale, bound_active=scale < 1.0,
                           prediction_rejected=prediction_rejected,
                           command_increment=increment.tolist(),
                           correction={"roll_rad": float(self.correction[0]),
                                       "pitch_rad": float(self.correction[1]),
                                       "height_m": float(self.correction[2])})
        return self.correction.copy(), diagnostics
