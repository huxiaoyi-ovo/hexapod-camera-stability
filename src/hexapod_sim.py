"""Single-run Isaac Gym flat-stand diagnostic for the local URDF."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
from hexapod_robot import (JOINT_NAMES, LEG_ORDER, load_robot_model, lowest_stand_height,
                          map_named_values, pd_torque, stand_target_by_name)
from hexapod_actuator import MitPdActuator
from hexapod_body_compensation import (BoundedBodyCompensator, SETTINGS as BODY_COMPENSATION_SETTINGS,
                                       body_reference_rotation)
from hexapod_openshc import OpenShcBridge, openshc_source_provenance, resolve_bridge_executable
from hexapod_sensing import ProprioceptiveContactOdometry
from hexapod_terrain import build_terrain
from hexapod_walk import (ControlCadence, StandStartGate, TripodGait, create_gait_after_stand_gate, feet_world,
                          leg_fk_jacobian, leg_inverse_kinematics, quaternion_matrix_xyzw,
                          walk_progress_failure)
from hexapod_pcr_expert import PcrExpertAdapter, pcr_default_joint_angles


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_provenance(project_root: Path, include_camera: bool = False,
                        include_pcr_expert: bool = False) -> Dict[str, Any]:
    """Describe the exact independent Python runtime inputs for this run."""
    runtime_files = ("src/hexapod_sim.py", "src/hexapod_robot.py", "scripts/run_hexapod_pilot.py",
                     "src/hexapod_walk.py", "src/hexapod_actuator.py", "src/hexapod_openshc.py",
                     "src/hexapod_sensing.py", "src/hexapod_terrain.py",
                     "src/hexapod_body_compensation.py")
    if include_camera:
        runtime_files += ("src/hexapod_camera_compensation.py", "src/controller.py")
    if include_pcr_expert:
        runtime_files += ("src/hexapod_pcr_expert.py", "third_party/pcr_expert/expert.py",
                          "third_party/pcr_expert/kinematic.py", "third_party/pcr_expert/provenance.json")
    hashes = {relative: _hash(project_root / relative) for relative in runtime_files}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(project_root), check=True,
            capture_output=True, text=True, timeout=2.0).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=str(project_root),
            check=True, capture_output=True, text=True, timeout=2.0).stdout
    except (OSError, subprocess.SubprocessError):
        git = {"available": False, "revision": None, "dirty": None,
               "status": "unavailable_or_timed_out"}
    else:
        git = {"available": True, "revision": revision or None, "dirty": bool(status.strip()),
               "status": "available"}
    return {"git": git, "runtime_dependency_sha256": hashes}


def _trajectory_velocity_reference(q_target: np.ndarray, previous_q_target: Optional[np.ndarray],
                                   held_velocity: np.ndarray, control_tick: bool,
                                   control_period_s: float, max_velocity: np.ndarray):
    """Update a ZOH joint-velocity reference from upper-rate position targets."""
    target = np.asarray(q_target, dtype=float)
    held = np.asarray(held_velocity, dtype=float)
    limit = np.asarray(max_velocity, dtype=float)
    if not control_tick:
        return held.copy(), previous_q_target
    if previous_q_target is None:
        velocity = np.zeros_like(target)
    else:
        previous = np.asarray(previous_q_target, dtype=float)
        velocity = (target - previous) / float(control_period_s)
        velocity = np.clip(velocity, -limit, limit)
    return velocity, target.copy()


def _limit_joint_position_step(requested_qref: np.ndarray, previous_sent_qref: np.ndarray,
                               hard_max_velocity: np.ndarray, command_limit_rad_s: float,
                               control_period_s: float):
    """Limit a new joint target relative to the reference actually sent last tick."""
    requested = np.asarray(requested_qref, dtype=float)
    previous = np.asarray(previous_sent_qref, dtype=float)
    hard = np.asarray(hard_max_velocity, dtype=float)
    operating = np.minimum(hard, float(command_limit_rad_s))
    max_delta = operating * float(control_period_s)
    delta = requested - previous
    limited = previous + np.clip(delta, -max_delta, max_delta)
    return limited, operating, np.abs(delta) > max_delta


class _StandWindowTracker:
    """Track elapsed time from the first sample in one continuous passing interval."""

    def __init__(self, window_s: float):
        self.window_s = float(window_s)
        self.started_at: Optional[float] = None

    def update(self, time_s: float, passing: bool) -> bool:
        if not passing:
            self.started_at = None
            return False
        if self.started_at is None:
            self.started_at = float(time_s)
        return float(time_s) - self.started_at + 1e-9 >= self.window_s

    def reset(self) -> None:
        self.started_at = None


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _mark_initialization_stage(status_path: Path, status: Dict[str, Any], stage: str) -> None:
    status["initialization_stage"] = stage
    status_path.write_text(json.dumps(_json_safe(status), indent=2, allow_nan=False) + "\n")
    print("initialization_stage={}".format(stage), flush=True)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text())
    config.setdefault("stop_on_violation", True)
    if type(config["stop_on_violation"]) is not bool:
        raise ValueError("stop_on_violation must be a boolean")
    required = ("robot_urdf", "asset_import", "stand_reference_rad", "physics", "control", "gate", "duration_s")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("missing configuration keys: {}".format(missing))
    if tuple(config.get("leg_order", ())) != LEG_ORDER or tuple(config.get("joint_types", ())) != ("thigh", "knee", "ankle"):
        raise ValueError("configuration leg and joint order must match the canonical named mapping")
    if (not isinstance(config["asset_import"], dict) or
            set(config["asset_import"]) != {"flip_visual_attachments"} or
            type(config["asset_import"]["flip_visual_attachments"]) is not bool):
        raise ValueError("asset_import must explicitly define flip_visual_attachments as a boolean")
    p, c, g, initial = config["physics"], config["control"], config["gate"], config.get("initial", {})
    if any(not isinstance(section, dict) for section in (p, c, g, initial, config["stand_reference_rad"])):
        raise ValueError("physics, control, gate, initial, and stand reference sections must be objects")
    p.setdefault("max_depenetration_velocity_m_s", 100.0)
    p.setdefault("bounce_threshold_velocity_m_s", 0.2)
    boolean_fields = ((p, ("gpu_pipeline", "gpu_physics", "self_collision", "fixed_base", "collapse_fixed_joints")),)
    if any(type(values.get(key)) is not bool for values, keys in boolean_fields for key in keys):
        raise ValueError("physics boolean flags must be explicitly configured as booleans")
    if p.get("gpu_pipeline") is not False or p.get("gpu_physics") is not True:
        raise ValueError("this diagnostic requires GPU PhysX with CPU tensor pipeline")
    if p.get("solver_type") != "TGS":
        raise ValueError("the standalone pilot requires the configured TGS solver")
    mode = config.get("mode", "stand")
    if mode not in ("stand", "walk"):
        raise ValueError("mode must be stand or walk")
    expected_control_mode = "effort_pd" if mode == "stand" else "MIT_PD"
    if c.get("mode") != expected_control_mode:
        raise ValueError("{} mode requires control.mode={}".format(mode, expected_control_mode))
    control_period = config.get("control_period_s", p.get("dt_s"))
    numeric_groups = [
        ("duration_s", [config.get("duration_s")]),
        ("physics", [p.get(k) for k in ("dt_s", "contact_offset_m", "rest_offset_m", "plane_static_friction", "plane_dynamic_friction", "plane_restitution", "max_depenetration_velocity_m_s", "bounce_threshold_velocity_m_s")] + list(p.get("gravity_m_s2", []))),
        ("control", [c.get(k) for k in ("kp_nm_rad", "kd_nm_s_rad", "effort_limit_nm", "joint_velocity_limit_rad_s", "position_tolerance_rad", "velocity_tolerance_rad_s")] + [control_period]),
        ("gate", [g.get(k) for k in ("eligible_after_s", "window_s", "max_abs_roll_pitch_rad", "max_linear_speed_m_s", "max_angular_speed_rad_s", "max_abs_vertical_speed_m_s", "max_tilt_rad", "contact_threshold_n")]),
        ("initial", list(initial.get("base_xy_m", [])) + [initial.get("yaw_rad"), initial.get("toe_clearance_m"), initial.get("base_z_min_m")]),
    ]
    for name, values in numeric_groups:
        if not values or any(type(v) not in (int, float) or not math.isfinite(float(v)) for v in values):
            raise ValueError("{} parameters must be finite numbers".format(name))
    integer_fields = ((p, "substeps", 1), (p, "position_iterations", 1), (p, "velocity_iterations", 0), (g, "min_contacts", 1))
    if any(type(values.get(key)) is not int or values[key] < minimum for values, key, minimum in integer_fields):
        raise ValueError("substeps, solver iterations, and min_contacts must be integers in range")
    if g["min_contacts"] > 6:
        raise ValueError("min_contacts must be between one and six")
    thigh_by_leg = config["stand_reference_rad"].get("thigh_by_leg", {})
    if not isinstance(thigh_by_leg, dict):
        raise ValueError("stand reference thigh_by_leg must be an object")
    stand_reference_values = list(thigh_by_leg.values()) + [
        config["stand_reference_rad"].get("knee"), config["stand_reference_rad"].get("ankle")]
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in stand_reference_values):
        raise ValueError("stand reference values must be finite numbers")
    if config["duration_s"] <= 0 or p["dt_s"] <= 0 or c["effort_limit_nm"] <= 0 or c["joint_velocity_limit_rad_s"] <= 0:
        raise ValueError("duration, time step, torque limit, and velocity limit must be positive")
    sensing = config.get("sensing", {"kind": "legacy_simulator"})
    if not isinstance(sensing, dict) or sensing.get("kind", "legacy_simulator") not in ("legacy_simulator", "proprioceptive"):
        raise ValueError("sensing.kind must be legacy_simulator or proprioceptive")
    if sensing.get("kind") == "proprioceptive":
        required_sensing = {"kind", "contact_on_threshold_n", "contact_off_threshold_n",
                            "contact_confirmation_s", "force_filter_s", "minimum_support"}
        if set(sensing) != required_sensing:
            raise ValueError("proprioceptive sensing settings must be explicit and complete")
        sensing_values = [sensing[key] for key in required_sensing - {"kind", "minimum_support"}]
        if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in sensing_values):
            raise ValueError("proprioceptive sensing values must be finite numbers")
        if (type(sensing["minimum_support"]) is not int or not 1 <= sensing["minimum_support"] <= len(LEG_ORDER) or
                not 0 <= sensing["contact_off_threshold_n"] < sensing["contact_on_threshold_n"] or
                sensing["contact_confirmation_s"] <= 0 or sensing["force_filter_s"] <= 0):
            raise ValueError("proprioceptive contact thresholds and support count are invalid")
        if mode != "walk" or config.get("walk_controller", "python_tripod") != "python_tripod":
            raise ValueError("proprioceptive sensing is enabled only for the Python tripod walk profile")
        adaptation = config.get("walk", {}).get("terrain_adaptation")
        required_adaptation = {"probe_speed_m_s", "probe_max_depth_m", "probe_timeout_s",
                               "liftoff_timeout_s", "body_z_rate_m_s"}
        if not isinstance(adaptation, dict) or set(adaptation) != required_adaptation:
            raise ValueError("proprioceptive walk requires explicit bounded terrain adaptation settings")
        if any(type(value) not in (int, float) or not math.isfinite(float(value))
               for value in adaptation.values()):
            raise ValueError("terrain adaptation settings must be finite numbers")
        if any(float(adaptation[key]) <= 0 for key in required_adaptation):
            raise ValueError("terrain adaptation speeds, limits, and timeouts must be positive")
    compensation = config.get("body_compensation")
    if compensation is not None:
        if (not isinstance(compensation, dict) or set(compensation) - {"enabled", "kind"} or
                type(compensation.get("enabled")) is not bool or
                compensation.get("kind", "pose_pd") not in ("pose_pd", "pose_p", "camera_space")):
            raise ValueError("body_compensation requires a boolean enabled flag and supported kind")
        if (mode != "walk" or config.get("walk_controller", "python_tripod") != "python_tripod" or
                sensing.get("kind") != "proprioceptive"):
            raise ValueError("body compensation is available only for the proprioceptive Python tripod profile")
        if compensation.get("kind", "pose_pd") == "camera_space":
            pilot = config.get("compensation_pilot")
            if (not isinstance(pilot, dict) or not isinstance(pilot.get("camera"), dict) or
                    set(pilot["camera"]) != {"base_to_camera_position_m", "base_to_camera_quaternion_xyzw",
                                             "optical_axis_description", "interpretation"}):
                raise ValueError("camera_space compensation requires explicit camera extrinsics and interpretation")
            template = pilot.get("camera_scene_template")
            if not isinstance(template, str) or not template:
                raise ValueError("camera_space compensation requires compensation_pilot.camera_scene_template")
            try:
                camera_spec = json.loads((PROJECT_ROOT / template).resolve().read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("camera-space scene template is unavailable or malformed") from exc
            if set(camera_spec) != {"camera", "scene", "note"}:
                raise ValueError("camera-space scene template must define camera and scene")
            camera_extrinsics = pilot["camera"]
            position = camera_extrinsics.get("base_to_camera_position_m")
            quaternion = camera_extrinsics.get("base_to_camera_quaternion_xyzw")
            if (not isinstance(position, list) or len(position) != 3 or
                    not isinstance(quaternion, list) or len(quaternion) != 4 or
                    any(type(value) not in (int, float) or not math.isfinite(float(value))
                        for value in position + quaternion)):
                raise ValueError("camera-space extrinsics must be finite 3- and 4-vectors")
    terrain_config = config.get("terrain", {"kind": "flat"})
    if not isinstance(terrain_config, dict):
        raise ValueError("terrain configuration must be an object")
    build_terrain(terrain_config)
    if "command_velocity_limit_rad_s" in c:
        command_limit = c["command_velocity_limit_rad_s"]
        if type(command_limit) not in (int, float) or not math.isfinite(float(command_limit)) or command_limit <= 0:
            raise ValueError("control.command_velocity_limit_rad_s must be a positive finite number")
    cadence = ControlCadence(float(p["dt_s"]), float(control_period))
    if mode == "walk":
        walk_controller = config.get("walk_controller", "python_tripod")
        if walk_controller not in ("python_tripod", "openshc", "pcr_expert"):
            raise ValueError("walk_controller must be python_tripod, openshc, or pcr_expert")
        if walk_controller == "pcr_expert":
            pcr_settings = config.get("pcr_expert")
            if (not isinstance(pcr_settings, dict) or
                    set(pcr_settings) != {"link_lengths_m", "body_half_shape_m"} or
                    pcr_settings["link_lengths_m"] != [0.072, 0.13, 0.17] or
                    pcr_settings["body_half_shape_m"] != {"x": 0.1, "y": 0.22}):
                raise ValueError("pcr_expert requires the frozen source geometry settings")
            if sensing.get("kind") != "legacy_simulator" or compensation is not None:
                raise ValueError("pcr_expert uses raw simulator joint state without compensation")
        if not math.isclose(float(p["dt_s"]), 0.0025, abs_tol=1e-12) or not math.isclose(float(control_period), 0.02, abs_tol=1e-12) or cadence.steps_per_update != 8:
            raise ValueError("walk mode requires 2.5 ms physics steps and an exact 20 ms control period")
        walk = config.get("walk")
        if not isinstance(walk, dict):
            raise ValueError("walk mode requires a walk settings object")
        actuator = config.get("actuator")
        actuator_keys = {"kind", "internal_period_s", "command_delay_s", "torque_bandwidth_hz",
                         "torque_feedforward_nm"}
        if (not isinstance(actuator, dict) or not actuator_keys.issubset(actuator) or
                set(actuator) - actuator_keys - {"velocity_reference", "velocity_feedback"}):
            raise ValueError("walk mode requires complete MIT_PD actuator settings")
        if actuator.get("velocity_reference", "zero") not in ("zero", "trajectory"):
            raise ValueError("actuator.velocity_reference must be zero or trajectory")
        if actuator.get("velocity_feedback", "simulator") not in ("simulator", "encoder_difference"):
            raise ValueError("actuator.velocity_feedback must be simulator or encoder_difference")
        if actuator.get("kind") != "MIT_PD":
            raise ValueError("walk mode requires actuator.kind=MIT_PD")
        actuator_values = [actuator.get(key) for key in
                           ("internal_period_s", "command_delay_s", "torque_bandwidth_hz", "torque_feedforward_nm")]
        if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in actuator_values):
            raise ValueError("MIT_PD actuator settings must be finite numbers")
        delay_steps = float(actuator["command_delay_s"]) / float(p["dt_s"])
        if (not math.isclose(float(actuator["internal_period_s"]), float(p["dt_s"]), abs_tol=1e-12) or
                float(actuator["torque_bandwidth_hz"]) <= 0 or float(actuator["command_delay_s"]) < 0 or
                not math.isclose(delay_steps, round(delay_steps), abs_tol=1e-9) or
                float(actuator["torque_feedforward_nm"]) != 0.0):
            raise ValueError("MIT_PD requires internal update at physics dt, integer-step delay, positive bandwidth, and zero baseline feedforward")
        walk_numbers = ("speed_m_s", "frequency_hz", "duty_factor", "swing_height_m", "speed_ramp_s",
                        "start_timeout_s", "touchdown_timeout_s", "max_stance_slip_m", "max_tracking_error_m",
                        "max_saturation_fraction", "minimum_cycles", "support_failure_s", "slip_failure_s",
                        "minimum_net_forward_displacement_m", "minimum_completed_cycle_displacement_m")
        if any(type(walk.get(key)) not in (int, float) or not math.isfinite(float(walk[key])) for key in walk_numbers):
            raise ValueError("walk settings must contain finite numeric values")
        if (type(walk.get("min_support_contacts")) is not int or walk["min_support_contacts"] < 1 or
                type(walk.get("minimum_cycles")) is not int or walk["minimum_cycles"] < 1):
            raise ValueError("walk support and minimum-cycle thresholds must be positive integers")
        duty_valid = (0.5 <= walk["duty_factor"] < 1.0 if walk_controller == "openshc"
                      else 0.5 < walk["duty_factor"] < 1.0)
        if (walk["speed_m_s"] <= 0 or walk["frequency_hz"] <= 0 or not duty_valid or
                walk["swing_height_m"] <= 0 or walk["speed_ramp_s"] <= 0 or walk["start_timeout_s"] <= 0 or
                walk["touchdown_timeout_s"] <= 0 or walk["max_stance_slip_m"] <= 0 or
                walk["max_tracking_error_m"] <= 0 or not 0.0 < walk["max_saturation_fraction"] <= 1.0 or
                walk["minimum_cycles"] < 5 or walk["min_support_contacts"] > 3 or walk["support_failure_s"] <= 0 or
                walk["slip_failure_s"] <= 0 or walk["minimum_net_forward_displacement_m"] <= 0 or
                walk["minimum_completed_cycle_displacement_m"] <= 0):
            raise ValueError("walk settings are outside the supported low-speed tripod ranges")
        if walk_controller == "openshc" and not math.isclose(float(walk["duty_factor"]), 0.5, abs_tol=1e-12):
            raise ValueError("the OpenSHC prototype currently requires a 0.5 duty factor")
    if min(p["contact_offset_m"], p["rest_offset_m"], p["plane_static_friction"], p["plane_dynamic_friction"],
           p["plane_restitution"], c["kp_nm_rad"], c["kd_nm_s_rad"], c["position_tolerance_rad"],
           c["velocity_tolerance_rad_s"], g["eligible_after_s"], g["max_abs_roll_pitch_rad"],
           p["bounce_threshold_velocity_m_s"], g["max_linear_speed_m_s"], g["max_angular_speed_rad_s"], g["max_abs_vertical_speed_m_s"],
           g["max_tilt_rad"], g["contact_threshold_n"], initial["toe_clearance_m"]) < 0:
        raise ValueError("offsets, gains, tolerances, and gate thresholds must be nonnegative")
    if p["max_depenetration_velocity_m_s"] <= 0:
        raise ValueError("max_depenetration_velocity_m_s must be positive")
    if g["window_s"] <= 0 or len(initial["base_xy_m"]) != 2 or len(p["gravity_m_s2"]) != 3:
        raise ValueError("invalid gate window, contact count, base position, or gravity vector")
    if p["plane_restitution"] > 1.0 or initial["base_z_min_m"] <= 0:
        raise ValueError("restitution must be at most one and minimum base height must be positive")
    robot_path = (PROJECT_ROOT / config["robot_urdf"]).resolve()
    model = load_robot_model(robot_path)
    stand_target_by_name(model, config["stand_reference_rad"])
    if "command_velocity_limit_rad_s" in c:
        hard_limits = [min(model.limits[name].velocity_rad_s, float(c["joint_velocity_limit_rad_s"]))
                       for name in JOINT_NAMES]
        if any(float(c["command_velocity_limit_rad_s"]) > limit for limit in hard_limits):
            raise ValueError("command velocity limit cannot exceed any joint hard velocity limit")
    return config


def _orientation_to_rpy(quat: Iterable[float]):
    x, y, z, w = map(float, quat)
    sinr = 2 * (w*x + y*z)
    cosr = 1 - 2 * (x*x + y*y)
    roll = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2 * (w*y - z*x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return roll, pitch, yaw


def _pcr_reference_body_distance(elapsed_s: float, speed_m_s: float, ramp_s: float) -> float:
    elapsed = max(0.0, float(elapsed_s))
    ramp = float(ramp_s)
    if elapsed < ramp:
        return 0.5 * float(speed_m_s) * elapsed * elapsed / ramp
    return float(speed_m_s) * (elapsed - 0.5 * ramp)


def _walk_control_targets(model, gait: TripodGait, elapsed_s: float, q_by_name: Dict[str, float],
                          body_xyz: np.ndarray, body_quat: np.ndarray,
                          contacts: Dict[str, bool], seed_by_name: Dict[str, float],
                          measured_feet_world: Optional[Dict[str, np.ndarray]] = None,
                          body_compensator: Optional[BoundedBodyCompensator] = None,
                          camera_compensator=None,
                          gyro_world_rad_s: Optional[np.ndarray] = None,
                          estimated_body_velocity_m_s: Optional[np.ndarray] = None,
                          previous_applied_torque_by_name=None,
                          control_period_s: float = 0.02):
    actual_feet = measured_feet_world or feet_world(model, q_by_name, body_xyz, body_quat)
    state = gait.update(elapsed_s, actual_feet, contacts)
    planned_body = np.asarray(state["planned_body_xyz_m"], dtype=float)
    c, s = math.cos(gait.yaw), math.sin(gait.yaw)
    nominal_rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    body_position = planned_body
    body_rotation = nominal_rotation
    if body_compensator is not None and body_compensator.enabled:
        corrections = body_compensator.update(
            body_quat, gyro_world_rad_s, body_xyz, estimated_body_velocity_m_s,
            float(planned_body[2]))
        body_position = planned_body + np.array([0.0, 0.0, corrections["height_m"]])
        body_rotation = body_reference_rotation(gait.yaw, corrections["roll_rad"], corrections["pitch_rad"])
        camera_diagnostics = None
    elif camera_compensator is not None:
        ramp = min(1.0, max(0.0, elapsed_s / max(gait.ramp_s, 1e-9)))
        nominal_speed = gait.speed * ramp
        nominal_velocity_world = nominal_speed * np.array([-math.sin(gait.yaw), math.cos(gait.yaw), 0.0])
        support = {leg: bool(state["confirmed_support"].get(leg, False) and
                             state["planned_support"].get(leg, False)) for leg in LEG_ORDER}
        camera_update_started = time.perf_counter()
        correction_vector, camera_diagnostics = camera_compensator.update(
            q_by_name=q_by_name, support=support, planned_body=planned_body, yaw=gait.yaw,
            anchors=state["anchors_world"], estimated_xyz=body_xyz,
            estimated_velocity=estimated_body_velocity_m_s,
            imu_quaternion=body_quat, gyro_world=gyro_world_rad_s,
            nominal_velocity_world=nominal_velocity_world,
            previous_applied_torque_by_name=previous_applied_torque_by_name)
        camera_diagnostics["compute_time_ms"] = (time.perf_counter() - camera_update_started) * 1000.0
        corrections = {"roll_rad": float(correction_vector[0]),
                       "pitch_rad": float(correction_vector[1]),
                       "height_m": float(correction_vector[2])}
        body_position = planned_body + np.array([0.0, 0.0, corrections["height_m"]])
        body_rotation = body_reference_rotation(gait.yaw, corrections["roll_rad"], corrections["pitch_rad"])
    else:
        corrections = {"roll_rad": 0.0, "pitch_rad": 0.0, "height_m": 0.0}
        camera_diagnostics = None
    state["body_compensation"] = corrections
    state["camera_compensation_diagnostics"] = camera_diagnostics
    q_targets = dict(seed_by_name)
    residuals = {}
    for leg in LEG_ORDER:
        target_world = np.asarray(state["desired_feet_world"][leg], dtype=float)
        target_base = body_rotation.T @ (target_world - body_position)
        solved, residuals[leg] = leg_inverse_kinematics(
            model, leg, target_base, q_targets, allow_approximate=not gait.stop_on_violation)
        q_targets.update(solved)
    return q_targets, state, actual_feet, residuals


def _record_violation(status: Dict[str, Any], category: str, time_s: float) -> None:
    record = status.setdefault("walking_violations", {
        "time_base": "simulation_seconds", "counts": {}, "first_time_s": {}, "last_time_s": {}})
    counts = record["counts"]
    counts[category] = int(counts.get(category, 0)) + 1
    record["first_time_s"].setdefault(category, float(time_s))
    record["last_time_s"][category] = float(time_s)
    if counts[category] == 1:
        print("walking_violation={} time_s={:.3f}".format(category, time_s), flush=True)


def _openshc_gait_state(response: Dict[str, Any], bridge: OpenShcBridge,
                        start_xyz: np.ndarray, initial_yaw: float) -> Dict[str, Any]:
    """Express native OpenSHC odometry targets in the initial yaw/world frame."""
    c, s = math.cos(initial_yaw), math.sin(initial_yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    planned_body_odom = np.asarray(response["planned_body_pose_odom"]["xyz_m"], dtype=float)
    planned_body = np.asarray(start_xyz, dtype=float) + rotation @ planned_body_odom
    desired_feet = {}
    planned_support = {}
    anchors = {}
    phases = {}
    for leg in LEG_ORDER:
        native = response["legs"][leg]
        toe_z = float(bridge.legs[leg]["plane_to_body_toe_z_m"])
        target = np.asarray(native["foot_target_odom_m"], dtype=float) + np.array([0.0, 0.0, toe_z])
        desired_feet[leg] = np.asarray(start_xyz, dtype=float) + rotation @ target
        planned_support[leg] = bool(native["planned_support"])
        phases[leg] = {"phase": int(native["phase"]), "state": int(native["state"]),
                       "phase_progress": float(native["phase_progress"]),
                       "joint_velocity_limit_active": native["joint_velocity_limit_active"],
                       "joint_position_limit_active": native["joint_position_limit_active"],
                       "ik_residual_m": float(native["ik_residual_m"]),
                       "ik_quality": float(native["ik_quality"]),
                       "foot_target_odom_m": native["foot_target_odom_m"],
                       "q_target_rad": native["q_target_rad"],
                       "dq_target_rad_s": native["dq_target_rad_s"]}
    return {"desired_feet_world": desired_feet, "anchors_world": anchors,
            "planned_support": planned_support, "planned_body_xyz_m": planned_body,
            "cycle_phase": float(response["legs"]["rf"]["phase_progress"]),
            "completed_cycles": int(response["completed_cycles"]),
            "accepted_velocity_xy_m_s": response["accepted_velocity_xy_m_s"],
            "legs": phases}


def _openshc_velocity_command(speed_m_s: float) -> np.ndarray:
    """Express forward speed in OpenSHC's fixed initial-yaw odometry frame."""
    speed = float(speed_m_s)
    if not math.isfinite(speed):
        raise ValueError("OpenSHC forward speed must be finite")
    return np.array([0.0, speed], dtype=float)


def _prepare_output(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = root / ("standalone_" + stamp)
    path.mkdir(exist_ok=False)
    return path


def run_stand(config_path: Union[str, Path], sim_device: Optional[str] = None, headless: bool = False,
              duration_s: Optional[float] = None, output_root: Optional[Union[str, Path]] = None,
              hold_viewer: bool = False, stand_only: bool = False) -> Path:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    sim_device = sim_device or config.get("sim_device", "cuda:0")
    mode = config.get("mode", "stand")
    walk_backend = config.get("walk_controller", "python_tripod") if mode == "walk" else None
    openshc_executable = (resolve_bridge_executable(PROJECT_ROOT) if walk_backend == "openshc" else None)
    duration = float(duration_s if duration_s is not None else config["duration_s"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    out = _prepare_output(Path(output_root) if output_root else PROJECT_ROOT / "data/raw")
    urdf_path = (PROJECT_ROOT / config["robot_urdf"]).resolve()
    model = load_robot_model(urdf_path)
    qref_named = stand_target_by_name(model, config["stand_reference_rad"])
    initial = config["initial"]
    start_z = lowest_stand_height(model, qref_named, float(initial["toe_clearance_m"]))
    dt = float(config["physics"]["dt_s"])
    if dt <= 0:
        raise ValueError("physics dt must be positive")
    p = config["physics"]
    control = config["control"]
    gate = config["gate"]
    walk_settings = config.get("walk", {})
    stop_on_violation = config.get("stop_on_violation", True)
    if mode == "walk":
        walk_settings = dict(walk_settings, stop_on_violation=stop_on_violation)
    sensing_config = config.get("sensing", {"kind": "legacy_simulator"})
    sensing_mode = sensing_config.get("kind", "legacy_simulator")
    compensation_config = config.get("body_compensation", {"enabled": False})
    compensation_enabled = bool(compensation_config.get("enabled", False))
    compensation_kind = compensation_config.get("kind", "pose_pd")
    camera_scene_path = None
    camera_scene_spec = None
    camera_compensation_settings = None
    camera_mass_matrix = None
    if compensation_enabled and compensation_kind == "camera_space":
        camera_scene_path = (PROJECT_ROOT / config["compensation_pilot"]["camera_scene_template"]).resolve()
        camera_scene_spec = json.loads(camera_scene_path.read_text())
        from hexapod_camera_compensation import (SETTINGS as camera_compensation_settings,
                                                 generalized_mass_matrix)
        camera_mass_matrix = generalized_mass_matrix(model, qref_named)
        effective_compensation_settings = {
            **camera_compensation_settings,
            "enabled": True,
            "kind": "camera_space",
            "configured_joint_pd_gains_nm": {
                "kp_nm_rad": float(control["kp_nm_rad"]),
                "kd_nm_s_rad": float(control["kd_nm_s_rad"])},
            "configured_actuator_response": {
                "command_delay_s": float(config["actuator"]["command_delay_s"]),
                "torque_bandwidth_hz": float(config["actuator"]["torque_bandwidth_hz"])},
            "pose_correction_bounds": {"roll_pitch_rad": float(camera_compensation_settings["max_angle_rad"]),
                                       "height_m": float(camera_compensation_settings["max_height_m"])},
            "pose_correction_rate_limits": {
                "roll_pitch_rad_s": float(camera_compensation_settings["max_angle_rate_rad_s"]),
                "height_m_s": float(camera_compensation_settings["max_height_rate_m_s"])}}
    elif compensation_enabled:
        effective_compensation_settings = dict(BODY_COMPENSATION_SETTINGS)
        if compensation_kind == "pose_p":
            effective_compensation_settings["roll_rate_gain_s"] = 0.0
            effective_compensation_settings["pitch_rate_gain_s"] = 0.0
        effective_compensation_settings.update({"enabled": True, "kind": compensation_kind})
    else:
        effective_compensation_settings = {"enabled": False, "kind": compensation_kind,
                                           "active_feedback": False}
    terrain_vertices, terrain_triangles, terrain_metadata = build_terrain(config.get("terrain", {"kind": "flat"}))
    control_period = float(config.get("control_period_s", dt))
    cadence = ControlCadence(dt, control_period)
    limits = model.limits
    asset_paths = [urdf_path] + sorted((PROJECT_ROOT / "assets/hexapod/two_set_meshes/visual").glob("*.dae")) + sorted((PROJECT_ROOT / "assets/hexapod/two_set_meshes/collision").glob("*.obj"))
    resolved = dict(config)
    hard_velocity_by_name = {name: min(limits[name].velocity_rad_s,
                                       float(control["joint_velocity_limit_rad_s"])) for name in JOINT_NAMES}
    operating_velocity_by_name = {
        name: min(limit, float(control["command_velocity_limit_rad_s"]))
        if "command_velocity_limit_rad_s" in control else limit
        for name, limit in hard_velocity_by_name.items()}
    resolved["resolved"] = {"config_path": str(config_path), "robot_urdf_sha256": _hash(urdf_path),
                            "asset_sha256": {str(x.relative_to(PROJECT_ROOT)): _hash(x) for x in asset_paths},
                            "derived_start_base_z_m": start_z, "sim_device": sim_device, "duration_s": duration,
                            "joint_velocity_limits_rad_s": {
                                name: {"hard": hard_velocity_by_name[name],
                                       "operating": operating_velocity_by_name[name]}
                                for name in JOINT_NAMES},
                            "terrain": terrain_metadata,
                            "sensing": {"kind": sensing_mode,
                                        "contact_force_source": "previous modeled applied motor torque via local URDF Jacobian"
                                        if sensing_mode == "proprioceptive" else "simulator_contact_diagnostic"},
                            "provenance": _runtime_provenance(
                                PROJECT_ROOT, include_camera=compensation_enabled and compensation_kind == "camera_space",
                                include_pcr_expert=walk_backend == "pcr_expert")}
    if sensing_mode == "proprioceptive":
        resolved["resolved"]["sensing"].update({
            "attitude_source": "ideal simulator quaternion treated as an IMU sample; uncalibrated",
            "initialization_assumption": "known flat start; body xy from configuration, body z from local URDF, all six toe anchors initialized at toe radius",
            "odometry_minimum_confirmed_support": int(sensing_config["minimum_support"])})
    resolved["resolved"]["body_compensation"] = {
        "enabled": compensation_enabled,
        "kind": compensation_kind,
        "input_sources": ["support-foot odometry", "ideal IMU quaternion", "ideal IMU gyro sample", "estimated body velocity"],
        "effective_settings": effective_compensation_settings,
        "correction_applied_after_gait_start_only": True,
        "corrected_pose_is_not_gait_tracking_nominal": True}
    if compensation_enabled and compensation_kind == "camera_space":
        resolved["resolved"]["body_compensation"].update({
            "controller_settings": camera_compensation_settings,
            "camera_scene_template": str(camera_scene_path.relative_to(PROJECT_ROOT)),
            "camera_scene_template_sha256": _hash(camera_scene_path),
            "camera_scene_note": camera_scene_spec["note"],
            "extrinsics": config["compensation_pilot"]["camera"],
            "frozen_posture_reduced_mass_matrix": camera_mass_matrix.tolist(),
            "generalized_coordinates": ["roll_rad", "pitch_rad", "z_m"],
            "kinetic_energy_unit_convention": "0.5 * generalized_velocity.T @ mass_matrix @ generalized_velocity; mixed angular/linear generalized units",
            "fixed_regularizer": "see controller_settings.regularization"})
    if openshc_executable is not None:
        resolved["resolved"]["openshc"] = openshc_source_provenance(PROJECT_ROOT, openshc_executable)
    if walk_backend == "pcr_expert":
        resolved["resolved"]["pcr_expert"] = json.loads(
            (PROJECT_ROOT / "third_party/pcr_expert/provenance.json").read_text())
    (out / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")

    gym = sim = viewer = gymapi = None
    csv_file = None
    openshc_bridge = None
    pcr_expert = None
    status: Dict[str, Any] = {"kind": "walk_diagnostic" if mode == "walk" else "stand_diagnostic", "status": "started", "failure_reason": None,
                              "valid_training_data": False, "start_base_z_m": start_z,
                              "configured_initial_base_velocity_m_s": [0.0, 0.0, 0.0],
                              "configured_initial_base_angular_velocity_rad_s": [0.0, 0.0, 0.0],
                              "initial_joint_target_by_name_rad": qref_named,
                              "duration_requested_s": duration, "sample_count": 0,
                              "stable_gate_passed": False, "gate_rejection_counts": {},
                              "mode": mode, "walk_controller": walk_backend,
                              "control_period_s": control_period,
                              "control_steps_per_update": cadence.steps_per_update,
                              "controller_effort_source": "stand explicit effort PD" if config.get("mode", "stand") == "stand" else "50 Hz position target with modeled 400 Hz MIT_PD actuator",
                              "modeled_motor_delay_s": config.get("actuator", {}).get("command_delay_s", 0.0),
                              "motor_torque_bandwidth_hz": config.get("actuator", {}).get("torque_bandwidth_hz"),
                              "motor_torque_bandwidth_calibrated": False,
                              "actuator_torque_is_model_output_not_measured": True,
                              "body_compensation_enabled": compensation_enabled,
                              "body_compensation_kind": compensation_kind,
                              "body_compensation_settings": effective_compensation_settings,
                              "initialization_stage": "pre_native"}
    status["stop_on_violation"] = stop_on_violation
    status["walking_violations"] = {
        "time_base": "simulation_seconds", "counts": {}, "first_time_s": {}, "last_time_s": {}}
    status_path = out / "status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print("stand diagnostic status path: {}".format(status_path), flush=True)
    try:
        # Import ordering is required by the installed Isaac Gym build.
        _mark_initialization_stage(status_path, status, "gym_import")
        from isaacgym import gymapi as imported_gymapi
        gymapi = imported_gymapi
        _mark_initialization_stage(status_path, status, "acquire_gym")
        gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.dt = dt
        sim_params.substeps = int(p["substeps"])
        sim_params.gravity = gymapi.Vec3(*map(float, p["gravity_m_s2"]))
        sim_params.use_gpu_pipeline = bool(p["gpu_pipeline"])
        sim_params.physx.use_gpu = bool(p["gpu_physics"])
        sim_params.up_axis = gymapi.UP_AXIS_Z
        # The installed Isaac Gym 1.x binding uses PhysX solver enum value 1 for TGS.
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = int(p["position_iterations"])
        sim_params.physx.num_velocity_iterations = int(p["velocity_iterations"])
        sim_params.physx.contact_offset = float(p["contact_offset_m"])
        sim_params.physx.rest_offset = float(p["rest_offset_m"])
        sim_params.physx.max_depenetration_velocity = float(p["max_depenetration_velocity_m_s"])
        sim_params.physx.bounce_threshold_velocity = float(p["bounce_threshold_velocity_m_s"])
        device_type, device_id = sim_device.split(":", 1)
        if device_type != "cuda" or int(device_id) < 0:
            raise ValueError("GPU PhysX requires --sim-device cuda:<index>")
        compute_id = int(device_id)
        graphics_id = -1 if headless else compute_id
        _mark_initialization_stage(status_path, status, "create_sim")
        sim = gym.create_sim(compute_id, graphics_id, gymapi.SIM_PHYSX, sim_params)
        if sim is None:
            raise RuntimeError("gym.create_sim returned None")
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane.static_friction = float(p["plane_static_friction"])
        plane.dynamic_friction = float(p["plane_dynamic_friction"])
        plane.restitution = float(p["plane_restitution"])
        _mark_initialization_stage(status_path, status, "add_ground")
        gym.add_ground(sim, plane)
        if terrain_triangles.size:
            terrain_params = gymapi.TriangleMeshParams()
            terrain_params.nb_vertices = int(terrain_vertices.shape[0])
            terrain_params.nb_triangles = int(terrain_triangles.shape[0])
            terrain_params.transform.p.x = 0.0
            terrain_params.transform.p.y = 0.0
            terrain_params.static_friction = float(p["plane_static_friction"])
            terrain_params.dynamic_friction = float(p["plane_dynamic_friction"])
            terrain_params.restitution = float(p["plane_restitution"])
            gym.add_triangle_mesh(sim, terrain_vertices.flatten(),
                                  terrain_triangles.flatten().astype(np.uint32, copy=False), terrain_params)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = bool(p["fixed_base"])
        asset_options.collapse_fixed_joints = bool(p["collapse_fixed_joints"])
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.disable_gravity = False
        asset_options.use_mesh_materials = True
        asset_options.flip_visual_attachments = config["asset_import"]["flip_visual_attachments"]
        _mark_initialization_stage(status_path, status, "load_asset")
        asset = gym.load_asset(sim, str(urdf_path.parent), urdf_path.name, asset_options)
        if asset is None:
            raise RuntimeError("gym.load_asset failed for the local URDF")
        status["asset_import_options"] = {
            "flip_visual_attachments": bool(asset_options.flip_visual_attachments),
            "use_mesh_materials": bool(asset_options.use_mesh_materials),
        }
        _mark_initialization_stage(status_path, status, "create_env")
        env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(float(initial["base_xy_m"][0]), float(initial["base_xy_m"][1]), start_z)
        yaw = float(initial["yaw_rad"])
        pose.r = gymapi.Quat(0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
        _mark_initialization_stage(status_path, status, "create_actor")
        actor = gym.create_actor(env, asset, pose, "hexapod", 0, 0 if p["self_collision"] else 1)
        sim_dof_names = list(gym.get_actor_dof_names(env, actor))
        qref = map_named_values(qref_named, sim_dof_names)
        effort_limits = map_named_values({n: min(limits[n].effort_nm, control["effort_limit_nm"]) for n in sim_dof_names}, sim_dof_names)
        mit_actuator = None
        if mode == "walk":
            actuator_settings = config["actuator"]
            mit_actuator = MitPdActuator(dt, actuator_settings["internal_period_s"],
                                         actuator_settings["command_delay_s"], actuator_settings["torque_bandwidth_hz"],
                                         control["kp_nm_rad"], control["kd_nm_s_rad"], effort_limits, qref)
            actuator_feedforward = np.full_like(qref, float(actuator_settings["torque_feedforward_nm"]))
        lower = map_named_values({n: limits[n].lower_rad for n in sim_dof_names}, sim_dof_names)
        upper = map_named_values({n: limits[n].upper_rad for n in sim_dof_names}, sim_dof_names)
        max_velocity = map_named_values({n: min(limits[n].velocity_rad_s, control["joint_velocity_limit_rad_s"]) for n in sim_dof_names}, sim_dof_names)
        operating_velocity = map_named_values(operating_velocity_by_name, sim_dof_names)
        dof_props = gym.get_actor_dof_properties(env, actor)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        dof_props["stiffness"].fill(0.0)
        dof_props["damping"].fill(0.0)
        if "command_velocity_limit_rad_s" in control:
            dof_props["velocity"][:] = operating_velocity
        gym.set_actor_dof_properties(env, actor, dof_props)
        if "command_velocity_limit_rad_s" in control:
            applied_dof_props = gym.get_actor_dof_properties(env, actor)
            if not np.allclose(applied_dof_props["velocity"], operating_velocity, rtol=0.0, atol=1e-6):
                raise RuntimeError("Gym did not retain the configured operating DOF velocity limits")
        dof_state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        dof_state["pos"][:] = qref
        dof_state["vel"][:] = 0.0
        gym.set_actor_dof_states(env, actor, dof_state, gymapi.STATE_ALL)
        proprioceptive = None
        proprioceptive_state = None
        previous_estimated_body_xyz = None
        estimated_body_velocity = np.zeros(3, dtype=float)
        if sensing_mode == "proprioceptive":
            initial_yaw = float(initial["yaw_rad"])
            initial_quaternion = np.array([0.0, 0.0, math.sin(initial_yaw / 2.0), math.cos(initial_yaw / 2.0)])
            initial_body_xyz = np.array([float(initial["base_xy_m"][0]),
                                         float(initial["base_xy_m"][1]), start_z])
            proprioceptive = ProprioceptiveContactOdometry(
                model, initial_body_xyz, initial_quaternion, qref_named, model.toe_radius_m,
                float(sensing_config["contact_on_threshold_n"]),
                float(sensing_config["contact_off_threshold_n"]),
                float(sensing_config["contact_confirmation_s"]),
                float(sensing_config["force_filter_s"]), int(sensing_config["minimum_support"]))
            proprioceptive_state = proprioceptive.snapshot()
            previous_estimated_body_xyz = proprioceptive_state["body_xyz_m"].copy()
        body_names = list(gym.get_actor_rigid_body_names(env, actor))
        body_dict = gym.get_actor_rigid_body_dict(env, actor)
        toe_body_indices = {leg: int(gym.get_actor_rigid_body_index(env, actor, body_dict["l_{}_toe".format(leg)], gymapi.DOMAIN_SIM)) for leg in LEG_ORDER}
        toe_body_local_indices = {leg: int(body_dict["l_{}_toe".format(leg)]) for leg in LEG_ORDER}
        root_body_index = body_dict[model.root_link]
        root_body_sim_index = int(gym.get_actor_rigid_body_index(env, actor, root_body_index, gymapi.DOMAIN_SIM))
        pre_dof = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        pre_rb = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
        pre_root = pre_rb["pose"][root_body_index]
        pre_vel = pre_rb["vel"][root_body_index]
        pre_xyz = [float(pre_root["p"][k]) for k in ("x", "y", "z")]
        pre_quat = [float(pre_root["r"][k]) for k in ("x", "y", "z", "w")]
        pre_linear = [float(pre_vel["linear"][k]) for k in ("x", "y", "z")]
        pre_angular = [float(pre_vel["angular"][k]) for k in ("x", "y", "z")]
        status["init_state_before_first_physics_step"] = {
            "base_xyz_m_observed": pre_xyz, "base_quaternion_xyzw_observed": pre_quat,
            "root_linear_velocity_m_s_observed": pre_linear, "root_angular_velocity_rad_s_observed": pre_angular,
            "joint_q_rad_observed": {n: float(pre_dof["pos"][i]) for i, n in enumerate(sim_dof_names)},
            "joint_dq_rad_s_observed": {n: float(pre_dof["vel"][i]) for i, n in enumerate(sim_dof_names)},
        }
        _mark_initialization_stage(status_path, status, "prepare_sim")
        gym.prepare_sim(sim)
        from isaacgym import gymtorch
        if walk_backend == "pcr_expert":
            pcr_expert = PcrExpertAdapter(
                device="cpu", default_joint_angles=pcr_default_joint_angles(config),
                link_lengths_m=config["pcr_expert"]["link_lengths_m"],
                body_half_shape_m=config["pcr_expert"]["body_half_shape_m"])
        import torch
        contact_tensor = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
        if not headless:
            _mark_initialization_stage(status_path, status, "create_viewer")
            viewer = gym.create_viewer(sim, gymapi.CameraProperties())
            if viewer is None:
                raise RuntimeError("failed to create viewer")
            gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(0.8, -0.8, 0.6), gymapi.Vec3(0, 0, start_z))
        _mark_initialization_stage(status_path, status, "running_physics")
        names = ["time_s", "root_x_m", "root_y_m", "root_z_m", "roll_rad", "pitch_rad", "yaw_rad",
                 "vx_m_s", "vy_m_s", "vz_m_s", "wx_rad_s", "wy_rad_s", "wz_rad_s", "contact_count",
                 "contact_legs", "max_abs_roll_pitch_rad", "linear_speed_m_s", "angular_speed_rad_s",
                 "limit_violation", "torque_saturated_count", "control_update", "control_tick_index",
                 "actuator_update", "toe_fk_error_max_m"]
        names.extend("body_contact_force_{}_n".format(axis) for axis in "xyz")
        if mode == "walk":
            names.extend(["gait_started", "gait_elapsed_s", "gait_cycle_phase", "commanded_speed_m_s",
                          "openshc_requested_speed_m_s", "openshc_accepted_speed_m_s",
                          "openshc_control_tick", "openshc_completed_cycles",
                          "achieved_forward_speed_m_s", "estimated_forward_speed_m_s",
                          "planned_body_x_m", "planned_body_y_m", "planned_body_z_m",
                          "body_tracking_error_m", "planned_support_contact_count",
                          "estimated_planned_support_contact_count", "confirmed_support_count",
                          "body_compensation_enabled", "body_compensation_roll_rad",
                          "body_compensation_pitch_rad", "body_compensation_height_m",
                          "camera_predicted_pixel_before", "camera_predicted_pixel_after",
                          "camera_uncontrollable_pixel_norm", "camera_mapping_rank",
                          "camera_support_count", "camera_bound_scale", "camera_compute_time_ms",
                          "camera_hold_reason",
                          "stance_slip_count", "max_stance_slip_m", "max_ik_residual_m"])
            for leg in LEG_ORDER:
                names.extend(["contact_force_{}_{}_n".format(leg, axis) for axis in "xyz"])
                names.extend(["actual_foot_{}_{}_m".format(leg, axis) for axis in "xyz"])
                names.extend(["fk_foot_{}_{}_m".format(leg, axis) for axis in "xyz"])
                names.append("toe_fk_error_{}_m".format(leg))
                names.extend(["desired_foot_{}_{}_m".format(leg, axis) for axis in "xyz"])
                names.extend(["anchor_{}_{}_m".format(leg, axis) for axis in "xyz"])
                names.extend(["planned_support_" + leg, "measured_contact_" + leg,
                              "confirmed_contact_" + leg,
                              "stance_slip_{}_m".format(leg), "ik_residual_{}_m".format(leg)])
                names.append("contact_estimated_up_force_{}_n".format(leg))
                if sensing_mode == "proprioceptive":
                    names.extend(["estimated_contact_" + leg,
                                  "estimated_filtered_up_force_{}_n".format(leg)])
                    names.extend(["estimated_force_{}_{}_n".format(leg, axis) for axis in "xyz"])
                    names.extend(["estimated_foot_{}_{}_m".format(leg, axis) for axis in "xyz"])
                names.extend(["openshc_phase_" + leg, "openshc_state_" + leg,
                              "openshc_phase_progress_" + leg])
                names.extend(["openshc_foot_target_{}_{}_m".format(leg, axis) for axis in "xyz"])
                for joint in ("thigh", "knee", "ankle"):
                    names.extend(["openshc_q_target_j_{}_{}_rad".format(leg, joint),
                                  "openshc_dq_target_j_{}_{}_rad_s".format(leg, joint),
                                  "openshc_velocity_limit_j_{}_{}".format(leg, joint),
                                  "openshc_position_limit_j_{}_{}".format(leg, joint)])
                names.extend(["openshc_ik_quality_" + leg])
            for n in sim_dof_names:
                names.extend(["qref_delayed_" + n, "dqref_command_" + n, "dqref_delayed_" + n,
                              "tau_raw_" + n, "tau_requested_" + n, "tau_ff_command_" + n,
                              "tau_ff_delayed_" + n, "tau_applied_model_" + n, "actuator_saturated_" + n])
            if sensing_mode == "proprioceptive":
                names.extend(["estimated_body_x_m", "estimated_body_y_m", "estimated_body_z_m",
                              "estimated_vx_m_s", "estimated_vy_m_s", "estimated_vz_m_s",
                              "estimated_support_count", "ideal_imu_roll_rad", "ideal_imu_pitch_rad"])
        for n in sim_dof_names:
            names.extend(["q_" + n, "dq_" + n, "tau_" + n, "qref_" + n])
            if mode == "walk":
                names.append("dq_feedback_" + n)
        csv_file = (out / "trace.csv").open("w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=names)
        writer.writeheader()
        status.update({"sim_device_requested": sim_device, "backend": "PhysX", "gpu_physics": bool(p["gpu_physics"]),
                       "gpu_tensor_pipeline": bool(p["gpu_pipeline"]), "sim_params_created": {"dt_s": dt, "substeps": p["substeps"],
                       "control_period_s": control_period, "control_steps_per_update": cadence.steps_per_update,
                       "position_iterations": p["position_iterations"], "velocity_iterations": p["velocity_iterations"],
                       "contact_offset_m": p["contact_offset_m"], "rest_offset_m": p["rest_offset_m"],
                       "max_depenetration_velocity_m_s": p["max_depenetration_velocity_m_s"],
                       "bounce_threshold_velocity_m_s": p["bounce_threshold_velocity_m_s"],
                       "solver_type": p["solver_type"], "solver_type_enum": 1, "up_axis": "Z", "self_collision": bool(p["self_collision"]),
                       "fixed_base": bool(p["fixed_base"]), "collapse_fixed_joints": bool(p["collapse_fixed_joints"])},
                       "python_version": platform.python_version(), "numpy_version": np.__version__,
                       "torch_version": torch.__version__, "torch_cuda_available": torch.cuda.is_available(),
                       "sim_dof_names": sim_dof_names, "sim_body_names": body_names,
                       "joint_velocity_limits_rad_s": {
                           name: {"hard": float(max_velocity[index]),
                                  "operating": float(operating_velocity[index])}
                           for index, name in enumerate(sim_dof_names)},
                       "gym_operating_velocity_limit_applied": ("command_velocity_limit_rad_s" in control),
                       "toe_body_indices_sim": toe_body_indices,
                       "actuator_velocity_feedback_source": (config.get("actuator", {}).get("velocity_feedback", "simulator")
                                                              if mode == "walk" else "simulator_raw"),
                       "joint_velocity_protection_source": "raw_gym_dof_velocity",
                       "control_cadence": {"period_s": control_period, "physics_steps_per_update": cadence.steps_per_update,
                                           "upper_target_update_hz": 1.0 / control_period,
                                           "actuator_internal_period_s": config.get("actuator", {}).get("internal_period_s", dt),
                                           "actuator_internal_steps_per_update": 1 if mode == "walk" else None,
                                           "modeled_command_delay_steps": int(round(config.get("actuator", {}).get("command_delay_s", 0.0) / dt)) if mode == "walk" else 0,
                                           "torque_response_alpha_per_physics_step": (1.0 - math.exp(-2.0 * math.pi * config["actuator"]["torque_bandwidth_hz"] * dt)) if mode == "walk" else None,
                                           "inner_pd_enabled": mode == "walk"}})
        start_clock = time.monotonic()
        criteria = {"contacts": 0, "roll_pitch": 0, "linear_speed": 0, "angular_speed": 0, "vertical_speed": 0, "joint_position": 0, "joint_velocity": 0, "torque_saturation": 0}
        stand_window = _StandWindowTracker(float(gate["window_s"]))
        walk_start_gate = StandStartGate(float(gate["window_s"]), float(walk_settings["start_timeout_s"])) if mode == "walk" else None
        gait = None
        gait_state = None
        openshc_bridge = None
        openshc_start_xyz = None
        openshc_start_yaw = None
        openshc_anchor_world = None
        openshc_previous_support = {leg: False for leg in LEG_ORDER}
        openshc_last_response = None
        gait_start_time = None
        walk_start_verified_at = None
        stand_start_verified = False
        walk_started = False
        failed_start = False
        current_ik_residuals = {leg: 0.0 for leg in LEG_ORDER}
        support_mismatch_ticks = 0
        slip_consecutive = {leg: 0 for leg in LEG_ORDER}
        support_contact_samples = 0
        estimated_support_contact_samples = 0
        planned_support_samples = 0
        slip_samples = 0
        slip_events = 0
        walk_samples = 0
        saturation_samples = 0
        q_error_sq_sum = 0.0
        q_error_values = 0
        q_error_max = 0.0
        tracking_error_max = 0.0
        cycle_progress = []
        estimated_cycle_progress = []
        last_cycle_index = 0
        cycle_start_progress = 0.0
        estimated_cycle_start_progress = 0.0
        initial_walk_xyz = None
        initial_walk_truth_xyz = None
        walk_forward = None
        stable = False
        held_tau = np.zeros_like(qref)
        held_saturated = np.zeros_like(qref, dtype=bool)
        velocity_reference_mode = config.get("actuator", {}).get("velocity_reference", "zero")
        velocity_feedback_mode = config.get("actuator", {}).get("velocity_feedback", "simulator")
        body_compensator = None
        camera_compensator = None
        if "body_compensation" in config and compensation_kind != "camera_space":
            body_compensator = BoundedBodyCompensator(
                compensation_enabled, control_period,
                rate_feedback=(compensation_kind != "pose_p"))
        if compensation_enabled and compensation_kind == "camera_space":
            from hexapod_camera_compensation import CameraSpaceCompensator
            camera_compensator = CameraSpaceCompensator(
                model, qref_named, config["compensation_pilot"]["camera"],
                camera_scene_spec["camera"], camera_scene_spec["scene"], control,
                config["actuator"], control_period, dt, mass_matrix=camera_mass_matrix)
        previous_applied_torque_by_name = {name: 0.0 for name in sim_dof_names}
        if proprioceptive_state is not None:
            controller_xyz = proprioceptive_state["body_xyz_m"].copy()
            controller_quat = proprioceptive_state["imu_quaternion_xyzw"].copy()
            controller_feet_world = {leg: value.copy() for leg, value in proprioceptive_state["feet_world_m"].items()}
            controller_contacts = dict(proprioceptive_state["contact_confirmed"])
        else:
            controller_xyz = None
            controller_quat = None
            controller_feet_world = None
            controller_contacts = None
        controller_gyro_world = np.zeros(3, dtype=float)
        held_dqref = np.zeros_like(qref)
        previous_trajectory_qref = None
        previous_sent_qref = qref.copy()
        qref_rate_clamped_samples = 0
        qref_rate_reference_samples = 0
        encoder_previous_q = None
        control_tick_count = 0
        max_steps = int(math.ceil(duration / dt))
        failure = None
        interrupted_by_viewer = False
        last_measured_state = None
        toe_fk_error_max_observed = 0.0
        max_abs_roll_pitch_observed = 0.0
        body_z_min_observed = float("inf")
        body_z_max_observed = float("-inf")
        max_root_contact_force_n = 0.0
        for step in range(max_steps):
            dof = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            q = np.asarray(dof["pos"], dtype=np.float32).copy()
            dq = np.asarray(dof["vel"], dtype=np.float32).copy()
            if mode == "walk":
                measured_q_sample = q.copy()
                if encoder_previous_q is None:
                    dq_encoder = np.zeros_like(measured_q_sample)
                else:
                    dq_encoder = (measured_q_sample - encoder_previous_q) / dt
                encoder_previous_q = measured_q_sample.copy()
                dq_feedback = dq_encoder if velocity_feedback_mode == "encoder_difference" else dq
            else:
                dq_encoder = np.zeros_like(q)
                dq_feedback = dq
            control_update = cadence.is_tick(step)
            control_tick_index = cadence.tick_index(step)
            if control_update:
                if mode == "walk":
                    try:
                        if sensing_mode == "proprioceptive":
                            xyz_control = controller_xyz.copy()
                            quat_control = controller_quat.copy()
                        else:
                            rb_control = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
                            root_control = rb_control["pose"][root_body_index]
                            xyz_control = np.array([root_control["p"][k] for k in ("x", "y", "z")], dtype=float)
                            quat_control = np.array([root_control["r"][k] for k in ("x", "y", "z", "w")], dtype=float)
                        q_control_by_name = {name: float(q[index]) for index, name in enumerate(sim_dof_names)}
                        if walk_backend == "openshc" and openshc_bridge is not None and not stand_only:
                            gait_elapsed = max(0.0, step * dt - gait_start_time)
                            requested_speed = float(walk_settings["speed_m_s"]) * min(
                                1.0, gait_elapsed / float(walk_settings["speed_ramp_s"]))
                            response = openshc_bridge.step(_openshc_velocity_command(requested_speed),
                                                           0.0, q_control_by_name)
                            openshc_last_response = response
                            qref_named = response["q_target_by_name_rad"]
                            qref = map_named_values(qref_named, sim_dof_names)
                            gait_state = _openshc_gait_state(response, openshc_bridge,
                                                             openshc_start_xyz, openshc_start_yaw)
                            gait_state["anchors_world"] = openshc_anchor_world
                            current_ik_residuals = {leg: float(response["legs"][leg]["ik_residual_m"])
                                                    for leg in LEG_ORDER}
                            status["gait_control_ticks"] = status.get("gait_control_ticks", 0) + 1
                            status["openshc_last_request"] = {
                                "requested_speed_m_s": requested_speed,
                                "accepted_velocity_xy_m_s": response["accepted_velocity_xy_m_s"],
                                "accepted_speed_m_s": float(np.linalg.norm(response["accepted_velocity_xy_m_s"])),
                                "upper_control_tick": response["tick"],
                                "completed_cycles": response["completed_cycles"]}
                        elif walk_backend == "python_tripod" and gait is not None and not stand_only:
                            if sensing_mode == "proprioceptive":
                                contact_map = dict(controller_contacts)
                                feet_for_controller = controller_feet_world
                            else:
                                contact_control = contact_tensor.detach().cpu().numpy()
                                contact_map = {leg: float(np.linalg.norm(contact_control[idx])) > float(gate["contact_threshold_n"])
                                               for leg, idx in toe_body_indices.items()}
                                feet_for_controller = {leg: np.array([rb_control["pose"][toe_body_local_indices[leg]]["p"][axis]
                                                                       for axis in ("x", "y", "z")], dtype=float)
                                                       for leg in LEG_ORDER}
                            qref_named, gait_state, _, current_ik_residuals = _walk_control_targets(
                                model, gait, max(0.0, step * dt - gait_start_time), q_control_by_name,
                                xyz_control, quat_control, contact_map, qref_named, feet_for_controller,
                                body_compensator=body_compensator,
                                camera_compensator=camera_compensator,
                                gyro_world_rad_s=controller_gyro_world,
                                estimated_body_velocity_m_s=estimated_body_velocity,
                                previous_applied_torque_by_name=previous_applied_torque_by_name,
                                control_period_s=control_period)
                            if not stop_on_violation:
                                for item in gait_state.get("violations", ()):
                                    _record_violation(status, "gait_" + item, step * dt)
                                for leg, residual in current_ik_residuals.items():
                                    if float(residual) > 2e-5:
                                        _record_violation(status, "ik_approximate_" + leg, step * dt)
                            if proprioceptive is not None:
                                proprioceptive.set_support_mask(gait_state["confirmed_support"],
                                                                controller_feet_world)
                                proprioceptive_state = proprioceptive.snapshot()
                                controller_xyz = proprioceptive_state["body_xyz_m"].copy()
                                controller_feet_world = {leg: value.copy() for leg, value in
                                                         proprioceptive_state["feet_world_m"].items()}
                                controller_contacts = dict(proprioceptive_state["contact_confirmed"])
                            qref = map_named_values(qref_named, sim_dof_names)
                            if camera_compensator is not None:
                                status["camera_space_last_diagnostics"] = gait_state.get(
                                    "camera_compensation_diagnostics")
                            status["gait_control_ticks"] = status.get("gait_control_ticks", 0) + 1
                        elif (walk_backend == "pcr_expert" and pcr_expert is not None and
                              walk_started and not stand_only):
                            gait_elapsed = max(0.0, step * dt - gait_start_time)
                            requested_speed = float(walk_settings["speed_m_s"]) * min(
                                1.0, gait_elapsed / float(walk_settings["speed_ramp_s"]))
                            dq_control_by_name = {name: float(dq[index])
                                                  for index, name in enumerate(sim_dof_names)}
                            qref_named = pcr_expert.step(
                                q_control_by_name, dq_control_by_name, 0.0, requested_speed, 0.0)
                            gait_state = pcr_expert.diagnostics(
                                xyz_control, quaternion_matrix_xyzw(quat_control))
                            gait_state["planned_body_xyz_m"] = (
                                initial_walk_xyz + walk_forward *
                                _pcr_reference_body_distance(gait_elapsed, float(walk_settings["speed_m_s"]),
                                                             float(walk_settings["speed_ramp_s"])))
                            current_ik_residuals = {leg: None for leg in LEG_ORDER}
                            out_of_range = [name for name, value in qref_named.items()
                                            if value < limits[name].lower_rad or value > limits[name].upper_rad]
                            status["pcr_expert_target_joint_limit_exceedance_count"] = (
                                status.get("pcr_expert_target_joint_limit_exceedance_count", 0) + len(out_of_range))
                            status["pcr_expert_target_joint_limit_exceedance_sample_count"] = (
                                status.get("pcr_expert_target_joint_limit_exceedance_sample_count", 0) +
                                int(bool(out_of_range)))
                            status["pcr_expert_last_target_joint_limit_exceedances"] = out_of_range
                            status["pcr_expert_tripod_switch_count"] = pcr_expert.tripod_switch_count
                            status["pcr_expert_reset_sent"] = pcr_expert.reset_sent
                            qref = map_named_values(qref_named, sim_dof_names)
                            status["gait_control_ticks"] = status.get("gait_control_ticks", 0) + 1
                    except (ValueError, RuntimeError, np.linalg.LinAlgError, KeyError) as exc:
                        failure = "walk_control_failure: {}".format(exc)
                        break
                    if (not stand_only and walk_started and
                            "command_velocity_limit_rad_s" in control):
                        qref, _, qref_rate_clamped = _limit_joint_position_step(
                            qref, previous_sent_qref, max_velocity,
                            float(control["command_velocity_limit_rad_s"]), control_period)
                        qref_rate_clamped_samples += int(np.count_nonzero(qref_rate_clamped))
                        qref_rate_reference_samples += int(qref.size)
                        previous_sent_qref = qref.copy()
                else:
                    held_tau, held_saturated = pd_torque(q, dq, qref, float(control["kp_nm_rad"]),
                                                         float(control["kd_nm_s_rad"]), effort_limits)
                control_tick_count += 1
            if mode == "walk":
                if velocity_reference_mode == "trajectory" and walk_started:
                    held_dqref, previous_trajectory_qref = _trajectory_velocity_reference(
                        qref, previous_trajectory_qref, held_dqref, control_update,
                        control_period, max_velocity)
                else:
                    held_dqref = np.zeros_like(qref)
                    previous_trajectory_qref = None
            if mode == "walk":
                actuator_sample = mit_actuator.step(q, dq_feedback, qref, command_velocity_rad_s=held_dqref,
                                                    command_feedforward_nm=actuator_feedforward)
                tau = actuator_sample["applied_model_torque_nm"].astype(np.float32)
                saturated = actuator_sample["torque_saturated"]
                gym.apply_actor_dof_efforts(env, actor, tau)
                previous_applied_torque_by_name = {name: float(tau[index])
                                                   for index, name in enumerate(sim_dof_names)}
            else:
                actuator_sample = None
                tau, saturated = held_tau, held_saturated
                gym.apply_actor_dof_efforts(env, actor, tau)
            q_before_physics = q.copy()
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_net_contact_force_tensor(sim)
            dof = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            q = np.asarray(dof["pos"], dtype=float).copy()
            dq = np.asarray(dof["vel"], dtype=float).copy()
            rb = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
            root_pose = rb["pose"][root_body_index]
            root_vel = rb["vel"][root_body_index]
            xyz = np.array([root_pose["p"][k] for k in ("x", "y", "z")], dtype=float)
            quat = np.array([root_pose["r"][k] for k in ("x", "y", "z", "w")], dtype=float)
            vel = np.array([root_vel["linear"][k] for k in ("x", "y", "z")], dtype=float)
            omega = np.array([root_vel["angular"][k] for k in ("x", "y", "z")], dtype=float)
            measured_feet_world = {leg: np.array([rb["pose"][toe_body_local_indices[leg]]["p"][axis]
                                                   for axis in ("x", "y", "z")], dtype=float)
                                  for leg in LEG_ORDER}
            q_by_name = {name: float(q[index]) for index, name in enumerate(sim_dof_names)}
            fk_feet_world = (feet_world(model, q_by_name, xyz, quat)
                             if np.all(np.isfinite(q)) and np.all(np.isfinite(xyz)) and np.all(np.isfinite(quat))
                             and abs(float(np.linalg.norm(quat)) - 1.0) <= 1e-3 else {})
            toe_fk_error = ({leg: float(np.linalg.norm(measured_feet_world[leg] - fk_feet_world[leg]))
                             for leg in LEG_ORDER} if fk_feet_world else {leg: float("nan") for leg in LEG_ORDER})
            toe_fk_error_max = max(toe_fk_error.values()) if all(math.isfinite(v) for v in toe_fk_error.values()) else float("nan")
            if math.isfinite(toe_fk_error_max):
                toe_fk_error_max_observed = max(toe_fk_error_max_observed, toe_fk_error_max)
                if "toe_fk_error_first_sample_m" not in status:
                    status["toe_fk_error_first_sample_m"] = toe_fk_error
            contact_np = contact_tensor.detach().cpu().numpy()
            quat_norm = float(np.linalg.norm(quat))
            state_finite = all(np.all(np.isfinite(x)) for x in
                               (q, dq, xyz, quat, vel, omega, contact_np, *measured_feet_world.values()))
            quat_valid = math.isfinite(quat_norm) and abs(quat_norm - 1.0) <= 1e-3
            if not state_finite:
                failure = "nonfinite_measured_state"
            elif not quat_valid:
                failure = "invalid_root_quaternion"
            if (proprioceptive is not None and state_finite and quat_valid and
                    cadence.is_tick(step + 1)):
                tau_by_name = {name: float(tau[index]) for index, name in enumerate(sim_dof_names)}
                proprioceptive_state = proprioceptive.update(
                    q_by_name, quat, tau_by_name, control_period)
                controller_xyz = proprioceptive_state["body_xyz_m"].copy()
                controller_quat = proprioceptive_state["imu_quaternion_xyzw"].copy()
                controller_gyro_world = omega.copy()
                controller_feet_world = {leg: value.copy() for leg, value in
                                          proprioceptive_state["feet_world_m"].items()}
                controller_contacts = dict(proprioceptive_state["contact_confirmed"])
                estimated_body_velocity = (controller_xyz - previous_estimated_body_xyz) / control_period
                previous_estimated_body_xyz = controller_xyz.copy()
            elif proprioceptive is None:
                estimated_body_velocity = vel.copy()
            controller_xyz = xyz.copy() if proprioceptive is None else controller_xyz
            controller_quat = quat.copy() if proprioceptive is None else controller_quat
            controller_feet_world = measured_feet_world if proprioceptive is None else controller_feet_world
            if failure:
                roll = pitch = yaw = float("nan")
            else:
                roll, pitch, yaw = _orientation_to_rpy(quat)
            foot_contacts = {leg: (float(np.linalg.norm(contact_np[idx])) > float(gate["contact_threshold_n"]))
                             if state_finite else False for leg, idx in toe_body_indices.items()}
            if proprioceptive is None:
                controller_contacts = foot_contacts
            contact_count = sum(foot_contacts.values())
            linear_speed = float(np.linalg.norm(vel))
            angular_speed = float(np.linalg.norm(omega))
            controller_roll, controller_pitch, controller_yaw = _orientation_to_rpy(controller_quat)
            controller_contact_count = sum(controller_contacts.values())
            controller_linear_speed = float(np.linalg.norm(estimated_body_velocity))
            controller_angular_speed = angular_speed
            pos_bad = bool(state_finite and (np.any(q < lower - float(control["position_tolerance_rad"])) or np.any(q > upper + float(control["position_tolerance_rad"]))))
            vel_bad = bool(state_finite and np.any(np.abs(dq) > max_velocity + float(control["velocity_tolerance_rad_s"])))
            t = (step + 1) * dt
            if math.isfinite(roll) and math.isfinite(pitch):
                max_abs_roll_pitch_observed = max(max_abs_roll_pitch_observed, abs(roll), abs(pitch))
            body_z_min_observed = min(body_z_min_observed, float(xyz[2]))
            body_z_max_observed = max(body_z_max_observed, float(xyz[2]))
            max_root_contact_force_n = max(max_root_contact_force_n,
                                           float(np.linalg.norm(contact_np[root_body_sim_index])))
            gait_row = {"body_compensation_enabled": int(compensation_enabled),
                        "body_compensation_roll_rad": 0.0,
                        "body_compensation_pitch_rad": 0.0,
                        "body_compensation_height_m": 0.0,
                        "camera_predicted_pixel_before": "",
                        "camera_predicted_pixel_after": "",
                        "camera_uncontrollable_pixel_norm": "",
                        "camera_mapping_rank": "",
                        "camera_support_count": "",
                        "camera_bound_scale": "",
                        "camera_compute_time_ms": "",
                        "camera_hold_reason": ""}
            if mode == "walk" and walk_backend == "openshc" and gait_state is not None:
                for leg in LEG_ORDER:
                    support_now = bool(gait_state["planned_support"][leg])
                    if support_now and not openshc_previous_support[leg] and foot_contacts[leg]:
                        openshc_anchor_world[leg] = measured_feet_world[leg].copy()
                    openshc_previous_support[leg] = support_now
                gait_state["anchors_world"] = openshc_anchor_world
            if mode == "walk" and gait_state is not None:
                desired_feet = gait_state["desired_feet_world"]
                anchors = gait_state["anchors_world"]
                planned_support = gait_state["planned_support"]
                planned_body = np.asarray(gait_state["planned_body_xyz_m"], dtype=float)
                estimated_valid_support_count = sum(bool(planned_support[leg]) and controller_contacts[leg]
                                                     for leg in LEG_ORDER)
                truth_valid_support_count = sum(bool(planned_support[leg]) and foot_contacts[leg]
                                                for leg in LEG_ORDER)
                support_count = sum(bool(planned_support[leg]) for leg in LEG_ORDER)
                planned_support_samples += support_count
                support_contact_samples += truth_valid_support_count
                estimated_support_contact_samples += estimated_valid_support_count
                support_guard_count = (sum(bool(value) for value in gait_state["confirmed_support"].values())
                                       if proprioceptive_state is not None else truth_valid_support_count)
                support_mismatch_ticks = support_mismatch_ticks + 1 if support_guard_count < int(walk_settings["min_support_contacts"]) else 0
                slip_by_leg = {}
                for leg in LEG_ORDER:
                    slip = float(np.linalg.norm(controller_feet_world[leg] - anchors[leg])) if planned_support[leg] and controller_contacts[leg] else 0.0
                    slip_by_leg[leg] = slip
                    if slip > float(walk_settings["max_stance_slip_m"]):
                        slip_samples += 1
                        slip_consecutive[leg] += 1
                        if slip_consecutive[leg] == 1:
                            slip_events += 1
                    else:
                        slip_consecutive[leg] = 0
                max_slip = max(slip_by_leg.values(), default=0.0)
                tracking_error = float(np.linalg.norm(controller_xyz - planned_body))
                tracking_error_max = max(tracking_error_max, tracking_error)
                ik_values = [float(value) for value in current_ik_residuals.values() if value is not None]
                max_ik = max(ik_values) if ik_values else float("nan")
                gait_elapsed = max(0.0, t - gait_start_time)
                cycle_index = (openshc_bridge.completed_cycles if walk_backend == "openshc" else
                               int(gait_state["completed_cycles"]))
                progress = float(np.dot(xyz - initial_walk_truth_xyz, walk_forward))
                estimated_progress = float(np.dot(controller_xyz - initial_walk_xyz, walk_forward))
                while cycle_index > last_cycle_index:
                    cycle_progress.append(progress - cycle_start_progress)
                    estimated_cycle_progress.append(estimated_progress - estimated_cycle_start_progress)
                    cycle_start_progress = progress
                    estimated_cycle_start_progress = estimated_progress
                    last_cycle_index += 1
                commanded_speed = float(walk_settings["speed_m_s"]) * min(1.0, gait_elapsed / float(walk_settings["speed_ramp_s"]))
                accepted_speed = (float(np.linalg.norm(openshc_last_response["accepted_velocity_xy_m_s"]))
                                  if walk_backend == "openshc" and openshc_last_response is not None else float("nan"))
                gait_row = {"gait_started": 1, "gait_elapsed_s": gait_elapsed,
                            "gait_cycle_phase": gait_state["cycle_phase"], "commanded_speed_m_s": commanded_speed,
                            "openshc_requested_speed_m_s": commanded_speed if walk_backend == "openshc" else "",
                            "openshc_accepted_speed_m_s": accepted_speed if walk_backend == "openshc" else "",
                            "openshc_control_tick": openshc_last_response["tick"] if walk_backend == "openshc" and openshc_last_response else "",
                            "openshc_completed_cycles": cycle_index if walk_backend == "openshc" else "",
                            "achieved_forward_speed_m_s": float(np.dot(vel, walk_forward)),
                            "estimated_forward_speed_m_s": float(np.dot(estimated_body_velocity, walk_forward)),
                            "planned_body_x_m": planned_body[0], "planned_body_y_m": planned_body[1],
                            "planned_body_z_m": planned_body[2], "body_tracking_error_m": tracking_error,
                            "planned_support_contact_count": truth_valid_support_count,
                            "estimated_planned_support_contact_count": estimated_valid_support_count,
                            "confirmed_support_count": support_guard_count,
                            "stance_slip_count": sum(value > float(walk_settings["max_stance_slip_m"]) for value in slip_by_leg.values()),
                            "max_stance_slip_m": max_slip, "max_ik_residual_m": max_ik,
                            "body_compensation_enabled": int(compensation_enabled),
                            "body_compensation_roll_rad": gait_state.get("body_compensation", {}).get("roll_rad", 0.0),
                            "body_compensation_pitch_rad": gait_state.get("body_compensation", {}).get("pitch_rad", 0.0),
                            "body_compensation_height_m": gait_state.get("body_compensation", {}).get("height_m", 0.0)}
                camera_diag = gait_state.get("camera_compensation_diagnostics")
                if camera_diag is not None:
                    gait_row.update({
                        "camera_predicted_pixel_before": camera_diag.get("predicted_pixel_before", ""),
                        "camera_predicted_pixel_after": camera_diag.get("predicted_pixel_after", ""),
                        "camera_uncontrollable_pixel_norm": camera_diag.get("uncontrollable_pixel_norm", ""),
                        "camera_mapping_rank": camera_diag.get("image_mapping_rank", ""),
                        "camera_support_count": camera_diag.get("support_count", ""),
                        "camera_bound_scale": camera_diag.get("bound_scale", ""),
                        "camera_compute_time_ms": camera_diag.get("compute_time_ms", ""),
                        "camera_hold_reason": camera_diag.get("hold_reason") or ""})
                for leg in LEG_ORDER:
                    for axis, value in zip("xyz", desired_feet[leg]):
                        gait_row["desired_foot_{}_{}_m".format(leg, axis)] = value
                    for axis, value in zip("xyz", anchors[leg]):
                        gait_row["anchor_{}_{}_m".format(leg, axis)] = value
                    gait_row["planned_support_" + leg] = int(bool(planned_support[leg]))
                    gait_row["confirmed_contact_" + leg] = int(controller_contacts[leg])
                    if proprioceptive_state is not None:
                        gait_row["contact_estimated_up_force_{}_n".format(leg)] = proprioceptive_state["filtered_up_force_n"][leg]
                    gait_row["stance_slip_{}_m".format(leg)] = slip_by_leg[leg]
                    gait_row["ik_residual_{}_m".format(leg)] = current_ik_residuals[leg]
                    if walk_backend == "openshc":
                        native_leg = gait_state["legs"][leg]
                        gait_row["openshc_phase_" + leg] = native_leg["phase"]
                        gait_row["openshc_state_" + leg] = native_leg["state"]
                        gait_row["openshc_phase_progress_" + leg] = native_leg["phase_progress"]
                        for axis, value in zip("xyz", native_leg["foot_target_odom_m"]):
                            gait_row["openshc_foot_target_{}_{}_m".format(leg, axis)] = value
                        for index, joint in enumerate(("thigh", "knee", "ankle")):
                            gait_row["openshc_q_target_j_{}_{}_rad".format(leg, joint)] = native_leg["q_target_rad"][index]
                            gait_row["openshc_dq_target_j_{}_{}_rad_s".format(leg, joint)] = native_leg["dq_target_rad_s"][index]
                            gait_row["openshc_velocity_limit_j_{}_{}".format(leg, joint)] = int(native_leg["joint_velocity_limit_active"][index])
                            gait_row["openshc_position_limit_j_{}_{}".format(leg, joint)] = int(native_leg["joint_position_limit_active"][index])
                        gait_row["openshc_ik_quality_" + leg] = native_leg["ik_quality"]
                walk_samples += 1
                saturation_samples += int(np.count_nonzero(saturated))
                q_error = qref.astype(float) - q
                q_error_sq_sum += float(np.dot(q_error, q_error))
                q_error_values += len(q_error)
                q_error_max = max(q_error_max, float(np.max(np.abs(q_error))))
                if stop_on_violation:
                    if tracking_error > float(walk_settings["max_tracking_error_m"]):
                        failure = "planned_body_tracking_error"
                    elif support_mismatch_ticks >= int(math.ceil(float(walk_settings["support_failure_s"]) / dt)):
                        failure = "planned_support_contact_loss"
                    elif max(slip_consecutive.values(), default=0) >= int(math.ceil(float(walk_settings["slip_failure_s"]) / dt)):
                        failure = "persistent_stance_slip"
                else:
                    if tracking_error > float(walk_settings["max_tracking_error_m"]):
                        _record_violation(status, "planned_body_tracking_error", (step + 1) * dt)
                    if support_mismatch_ticks >= int(math.ceil(float(walk_settings["support_failure_s"]) / dt)):
                        _record_violation(status, "planned_support_contact_loss", (step + 1) * dt)
                    if max(slip_consecutive.values(), default=0) >= int(math.ceil(float(walk_settings["slip_failure_s"]) / dt)):
                        _record_violation(status, "persistent_stance_slip", (step + 1) * dt)
            row = {"time_s": (step + 1) * dt, "root_x_m": xyz[0], "root_y_m": xyz[1], "root_z_m": xyz[2], "roll_rad": roll,
                   "pitch_rad": pitch, "yaw_rad": yaw, "vx_m_s": vel[0], "vy_m_s": vel[1], "vz_m_s": vel[2],
                   "wx_rad_s": omega[0], "wy_rad_s": omega[1], "wz_rad_s": omega[2], "contact_count": contact_count,
                   "contact_legs": ",".join(k for k, v in foot_contacts.items() if v), "max_abs_roll_pitch_rad": max(abs(roll), abs(pitch)),
                   "linear_speed_m_s": linear_speed, "angular_speed_rad_s": angular_speed, "limit_violation": int(pos_bad or vel_bad),
                   "torque_saturated_count": int(np.count_nonzero(saturated)),
                   "control_update": int(control_update), "control_tick_index": control_tick_index,
                   "actuator_update": int(mode == "walk"), "toe_fk_error_max_m": toe_fk_error_max}
            if proprioceptive_state is not None:
                row.update({"estimated_body_x_m": controller_xyz[0], "estimated_body_y_m": controller_xyz[1],
                            "estimated_body_z_m": controller_xyz[2],
                            "estimated_vx_m_s": estimated_body_velocity[0],
                            "estimated_vy_m_s": estimated_body_velocity[1],
                            "estimated_vz_m_s": estimated_body_velocity[2],
                            "estimated_support_count": proprioceptive_state["support_count"],
                            "ideal_imu_roll_rad": controller_roll, "ideal_imu_pitch_rad": controller_pitch})
                for leg in LEG_ORDER:
                    row["estimated_contact_" + leg] = int(controller_contacts[leg])
                    row["estimated_filtered_up_force_{}_n".format(leg)] = proprioceptive_state["filtered_up_force_n"][leg]
                    for axis, value in zip("xyz", proprioceptive_state["force_world_n"][leg]):
                        row["estimated_force_{}_{}_n".format(leg, axis)] = value
                    for axis, value in zip("xyz", controller_feet_world[leg]):
                        row["estimated_foot_{}_{}_m".format(leg, axis)] = value
            row.update({"body_contact_force_{}_n".format(axis): contact_np[root_body_sim_index, index]
                        for index, axis in enumerate("xyz")})
            for i, n in enumerate(sim_dof_names):
                row["q_" + n], row["dq_" + n], row["tau_" + n], row["qref_" + n] = q[i], dq[i], tau[i], qref[i]
                if mode == "walk":
                    row["dq_feedback_" + n] = dq_feedback[i]
                    row["qref_delayed_" + n] = actuator_sample["delayed_position_rad"][i]
                    row["dqref_command_" + n] = actuator_sample["desired_velocity_rad_s"][i]
                    row["dqref_delayed_" + n] = actuator_sample["delayed_velocity_rad_s"][i]
                    row["tau_raw_" + n] = actuator_sample["raw_torque_nm"][i]
                    row["tau_requested_" + n] = actuator_sample["requested_torque_nm"][i]
                    row["tau_ff_command_" + n] = actuator_feedforward[i]
                    row["tau_ff_delayed_" + n] = actuator_sample["delayed_feedforward_nm"][i]
                    row["tau_applied_model_" + n] = tau[i]
                    row["actuator_saturated_" + n] = int(actuator_sample["torque_saturated"][i])
            if mode == "walk":
                for leg in LEG_ORDER:
                    for axis, value in zip("xyz", contact_np[toe_body_indices[leg]]):
                        row["contact_force_{}_{}_n".format(leg, axis)] = value
                    for axis, value in zip("xyz", measured_feet_world[leg]):
                        row["actual_foot_{}_{}_m".format(leg, axis)] = value
                    for axis, value in zip("xyz", fk_feet_world.get(leg, (float("nan"),) * 3)):
                        row["fk_foot_{}_{}_m".format(leg, axis)] = value
                    row["toe_fk_error_{}_m".format(leg)] = toe_fk_error[leg]
                    row["measured_contact_" + leg] = int(foot_contacts[leg])
                if gait_state is None:
                    row["gait_started"] = 0
                row.update(gait_row)
            writer.writerow(row)
            # Flush every sample so an early failure leaves a usable partial trace.
            csv_file.flush()
            status["sample_count"] = step + 1
            last_measured_state = {"time_s": t, "root_xyz_m": xyz.tolist(), "root_quaternion_xyzw": quat.tolist(),
                                   "root_linear_velocity_m_s": vel.tolist(), "root_angular_velocity_rad_s": omega.tolist(),
                                   "joint_q_rad": {n: float(q[i]) for i, n in enumerate(sim_dof_names)},
                                   "joint_dq_rad_s": {n: float(dq[i]) for i, n in enumerate(sim_dof_names)},
                                   "toe_contact_force_n": {leg: contact_np[idx].tolist() for leg, idx in toe_body_indices.items()},
                                   "toe_rigid_body_world_position_m": {leg: point.tolist() for leg, point in measured_feet_world.items()},
                                   "toe_fk_estimate_world_position_m": {leg: point.tolist() for leg, point in fk_feet_world.items()},
                                   "toe_fk_error_m": toe_fk_error}
            status["last_measured_state"] = last_measured_state
            if proprioceptive_state is not None:
                status["last_proprioceptive_state"] = {
                    "estimated_body_xyz_m": controller_xyz.tolist(),
                    "ideal_imu_quaternion_xyzw": controller_quat.tolist(),
                    "confirmed_contact": dict(controller_contacts),
                    "confirmed_support": dict(proprioceptive_state["support_mask"]),
                    "filtered_up_force_n": dict(proprioceptive_state["filtered_up_force_n"]),
                    "support_count": proprioceptive_state["support_count"]}
            checks = {"contacts": controller_contact_count >= int(gate["min_contacts"]),
                          "roll_pitch": max(abs(controller_roll), abs(controller_pitch)) <= float(gate["max_abs_roll_pitch_rad"]),
                          "linear_speed": controller_linear_speed <= float(gate["max_linear_speed_m_s"]),
                          "angular_speed": controller_angular_speed <= float(gate["max_angular_speed_rad_s"]),
                          "vertical_speed": abs(estimated_body_velocity[2]) <= float(gate["max_abs_vertical_speed_m_s"]),
                          "joint_position": not pos_bad, "joint_velocity": not vel_bad,
                          "torque_saturation": not np.any(saturated)}
            if t >= float(gate["eligible_after_s"]):
                for key, passed in checks.items():
                    criteria[key] += int(not passed)
            if mode == "walk":
                if not stand_start_verified:
                    gate_result = walk_start_gate.update(
                        t, t >= float(gate["eligible_after_s"])
                        and all(checks.values()) and failure is None)
                    if gate_result == "ready":
                        stand_start_verified = True
                        stable = True
                        walk_start_verified_at = t
                        status["stand_gate_verified_at_s"] = t
                        if not stand_only:
                            yaw0 = controller_yaw
                            if walk_backend == "openshc":
                                openshc_start_xyz = xyz.copy()
                                openshc_start_yaw = yaw0
                                openshc_anchor_world = {leg: measured_feet_world[leg].copy() for leg in LEG_ORDER}
                                q_measured = {name: float(q[index]) for index, name in enumerate(sim_dof_names)}
                                openshc_bridge = OpenShcBridge(
                                    openshc_executable, model, q_measured, measured_feet_world, xyz, quat,
                                    walk_settings, out / "openshc_stderr.log")
                                status["openshc_initialization"] = openshc_bridge.initialization
                                status["openshc_requested_frequency_hz"] = float(walk_settings["frequency_hz"])
                                status["openshc_requested_duty_factor"] = float(walk_settings["duty_factor"])
                                status["openshc_native_frequency_parameter_hz"] = (
                                    float(walk_settings["frequency_hz"]) / float(walk_settings["duty_factor"]))
                                status["openshc_effective_cycle_frequency_hz"] = (
                                    openshc_bridge.initialization["effective_frequency_hz"])
                                status["openshc_workspace_forward_speed_limit_m_s"] = (
                                    openshc_bridge.initialization["forward_speed_limit_m_s"])
                                status["openshc_speed_limit_source"] = "native walk workspace from local URDF joint limits"
                                status["openshc_pose_feedback_enabled"] = False
                                status["openshc_admittance_enabled"] = False
                                status["openshc_autopose_enabled"] = False
                            elif walk_backend == "python_tripod":
                                rotation_yaw = np.array([[math.cos(yaw0), -math.sin(yaw0), 0.0],
                                                        [math.sin(yaw0), math.cos(yaw0), 0.0], [0.0, 0.0, 1.0]])
                                nominal_feet = {leg: rotation_yaw.T @ (controller_feet_world[leg] - controller_xyz)
                                                for leg in LEG_ORDER}
                                gait_settings_for_trial = walk_settings
                                if proprioceptive is not None:
                                    gait_settings_for_trial = dict(
                                        walk_settings,
                                        terrain_adaptation=dict(walk_settings["terrain_adaptation"],
                                                                toe_radius_m=model.toe_radius_m))
                                gait, gait_state = create_gait_after_stand_gate(
                                    stand_start_verified, nominal_feet, controller_feet_world, controller_xyz, yaw0,
                                    gait_settings_for_trial, controller_contacts)
                                if proprioceptive is not None:
                                    proprioceptive.set_support_mask(gait_state["confirmed_support"],
                                                                    controller_feet_world)
                                    proprioceptive_state = proprioceptive.snapshot()
                                    controller_xyz = proprioceptive_state["body_xyz_m"].copy()
                                    controller_feet_world = {leg: value.copy() for leg, value in
                                                             proprioceptive_state["feet_world_m"].items()}
                                    controller_contacts = dict(proprioceptive_state["contact_confirmed"])
                            elif walk_backend == "pcr_expert":
                                status["pcr_expert_initialization"] = {
                                    "state_source": "measured_named_joint_position_and_velocity",
                                    "command_frame": "body axes; forward is +y",
                                    "reset_sent_at": "first post-gate control tick",
                                    "gait_geometry_source": "vendored ExpertGround constants"}
                            gait_start_time = t
                            initial_walk_xyz = controller_xyz.copy()
                            initial_walk_truth_xyz = xyz.copy()
                            walk_forward = np.array([-math.sin(yaw0), math.cos(yaw0), 0.0])
                            walk_started = True
                            status["gait_started"] = True
                            status["gait_start_time_s"] = t
                            status["initial_yaw_rad"] = yaw0
                    elif gate_result == "failed_start":
                        failed_start = True
                        failure = "stand_start_gate_timeout"
                    else:
                        stable = False
                else:
                    stable = True
                if stand_only:
                    stable = stand_window.update(
                        t, t >= float(gate["eligible_after_s"])
                        and all(checks.values()) and failure is None)
            else:
                stable = stand_window.update(t, t >= float(gate["eligible_after_s"])
                                             and all(checks.values()) and failure is None)
            if failure == "nonfinite_measured_state" or failure == "invalid_root_quaternion":
                stand_window.reset()
                if mode != "walk" or not stand_start_verified:
                    stable = False
            threshold_failures = []
            if pos_bad:
                threshold_failures.append("joint_position_limit")
            if vel_bad:
                threshold_failures.append("joint_velocity_limit")
            if max(abs(roll), abs(pitch)) > float(gate["max_tilt_rad"]):
                threshold_failures.append("tilt_limit")
            if controller_xyz[2] < float(initial["base_z_min_m"]):
                threshold_failures.append("base_height_limit")
            if stop_on_violation:
                if threshold_failures:
                    failure = threshold_failures[0]
            else:
                for threshold_failure in threshold_failures:
                    _record_violation(status, threshold_failure, (step + 1) * dt)
            if failure:
                stand_window.reset()
                if mode != "walk" or not stand_start_verified:
                    stable = False
                break
            if viewer is not None:
                if gym.query_viewer_has_closed(viewer):
                    interrupted_by_viewer = True
                    stable = False
                    break
                if cadence.is_tick(step):
                    gym.step_graphics(sim)
                    gym.draw_viewer(viewer, sim, True)
                    gym.sync_frame_time(sim)
        if mode == "walk" and failure == "joint_velocity_limit" and last_measured_state is not None:
            violations = np.flatnonzero(np.abs(dq) > max_velocity + float(control["velocity_tolerance_rad_s"]))
            joints = {joint.name: joint for joint in model.joints}
            velocity_diagnostic = {}
            for index in violations:
                name = sim_dof_names[int(index)]
                joint = joints[name]
                parent_index = int(body_dict[joint.parent])
                child_index = int(body_dict[joint.child])
                child_quat = rb["pose"][child_index]["r"]
                child_quaternion = np.array([child_quat[key] for key in ("x", "y", "z", "w")], dtype=float)
                axis = joint.axis / np.linalg.norm(joint.axis)
                axis_world = quaternion_matrix_xyzw(child_quaternion) @ axis
                parent_omega = np.array([rb["vel"][parent_index]["angular"][key]
                                         for key in ("x", "y", "z")], dtype=float)
                child_omega = np.array([rb["vel"][child_index]["angular"][key]
                                        for key in ("x", "y", "z")], dtype=float)
                velocity_diagnostic[name] = {
                    "raw_gym_dq_rad_s": float(dq[index]),
                    "q_finite_difference_rad_s": float((q[index] - q_before_physics[index]) / dt),
                    "encoder_difference_feedback_input_rad_s": float(dq_encoder[index]),
                    "actuator_feedback_input_rad_s": float(dq_feedback[index]),
                    "parent_child_angular_velocity_projection_rad_s": float(np.dot(child_omega - parent_omega, axis_world)),
                    "q_rad": float(q[index]),
                    "qref_rad": float(qref[index]),
                    "contact_force_child_body_n": [float(contact_np[body_dict[joint.child]][key])
                                                    for key in range(3)],
                }
            status["joint_velocity_failure_diagnostic"] = velocity_diagnostic
        status["gate_rejection_counts"] = criteria
        status["gate_window_s"] = float(gate["window_s"])
        status["stable_gate_passed"] = (stable if stand_only else stand_start_verified) if mode == "walk" else stable
        status["control_tick_count"] = control_tick_count
        status["qref_rate_clamp_count"] = qref_rate_clamped_samples
        status["qref_rate_reference_sample_count"] = qref_rate_reference_samples
        status["qref_rate_clamp_fraction"] = (
            qref_rate_clamped_samples / qref_rate_reference_samples
            if qref_rate_reference_samples else 0.0)
        status["toe_fk_error_max_observed_m"] = toe_fk_error_max_observed
        status["max_abs_roll_pitch_observed_rad"] = max_abs_roll_pitch_observed
        status["simulator_body_z_range_m"] = ([body_z_min_observed, body_z_max_observed]
                                                if math.isfinite(body_z_min_observed) else None)
        status["max_root_body_contact_force_n"] = max_root_contact_force_n
        status["terrain"] = terrain_metadata
        if mode == "walk":
            saturation_fraction = (float(saturation_samples) / (walk_samples * len(sim_dof_names))) if walk_samples else 0.0
            simulator_truth_net_displacement = (float(np.dot(xyz - initial_walk_truth_xyz, walk_forward))
                                                if walk_started and last_measured_state else 0.0)
            estimated_net_displacement = (float(np.dot(controller_xyz - initial_walk_xyz, walk_forward))
                                          if walk_started and last_measured_state else 0.0)
            status.update({"gait_started": walk_started, "completed_gait_cycles": last_cycle_index,
                           "cycle_forward_displacement_m": cycle_progress,
                           "mean_completed_cycle_displacement_m": float(np.mean(cycle_progress)) if cycle_progress else None,
                           "net_forward_displacement_m": simulator_truth_net_displacement,
                           "estimated_net_forward_displacement_m": estimated_net_displacement,
                           "estimated_cycle_forward_displacement_m": estimated_cycle_progress,
                           "planned_support_contact_fraction": (support_contact_samples / planned_support_samples
                                                                if planned_support_samples else None),
                           "estimated_planned_support_contact_fraction": (
                               estimated_support_contact_samples / planned_support_samples
                               if planned_support_samples else None),
                           "stance_slip_threshold_sample_count": slip_samples, "stance_slip_event_count": slip_events,
                           "max_body_tracking_error_m": tracking_error_max,
                           "joint_position_error_rmse_rad": math.sqrt(q_error_sq_sum / q_error_values) if q_error_values else None,
                           "joint_position_error_max_rad": q_error_max if q_error_values else None,
                           "torque_saturation_fraction": saturation_fraction,
                           "walk_sample_count": walk_samples})
            acceptance_reasons = []
            if not stand_only:
                if last_cycle_index < int(walk_settings["minimum_cycles"]):
                    acceptance_reasons.append("minimum_cycles_not_reached")
                progress_reason = walk_progress_failure(
                    cycle_progress, simulator_truth_net_displacement,
                    int(walk_settings["minimum_cycles"]),
                    float(walk_settings["minimum_net_forward_displacement_m"]),
                    float(walk_settings["minimum_completed_cycle_displacement_m"]))
                if progress_reason and progress_reason not in acceptance_reasons:
                    acceptance_reasons.append(progress_reason)
                if saturation_fraction > float(walk_settings["max_saturation_fraction"]):
                    acceptance_reasons.append("torque_saturation_fraction_exceeded")
                if not stop_on_violation:
                    for reason in acceptance_reasons:
                        _record_violation(status, reason, duration)
            status["walking_acceptance"] = {
                "accepted": (not acceptance_reasons and not status["walking_violations"]["counts"] and
                             failure is None and stand_start_verified and
                             status["sample_count"] >= max_steps and not interrupted_by_viewer),
                "reasons": acceptance_reasons + sorted(status["walking_violations"]["counts"]),
                "time_completed": status["sample_count"] >= max_steps and not interrupted_by_viewer}
            if stand_only:
                status["status"] = "stand_verified" if stable and failure is None else "failed_start"
            elif failure and not walk_started:
                status["status"] = "failed_start"
            elif failure:
                status["status"] = "failed_safety_stop"
            elif not stand_start_verified:
                status["status"] = "failed_start"
                failure = "no_contiguous_stand_gate_window"
            elif acceptance_reasons:
                if not stop_on_violation and status["walking_acceptance"]["time_completed"]:
                    status["status"] = "completed_with_violations"
                    status["walking_acceptance"]["accepted"] = False
                else:
                    status["status"] = "walk_incomplete"
                    failure = acceptance_reasons[0]
            elif status["walking_violations"]["counts"]:
                if not stop_on_violation and status["walking_acceptance"]["time_completed"]:
                    status["status"] = "completed_with_violations"
                    status["walking_acceptance"]["accepted"] = False
                else:
                    status["status"] = "walk_incomplete"
            elif status["sample_count"] < max_steps:
                status["status"] = "walk_incomplete"
                failure = "duration_not_completed"
            else:
                status["status"] = "walk_completed"
        else:
            status["status"] = "settled" if stable and failure is None else "failed_stand_gate"
            if failure:
                status["status"] = "failed_safety_stop"
        if failure:
            status["failure_reason"] = failure
        elif interrupted_by_viewer:
            status["status"] = "interrupted_viewer"
            status["failure_reason"] = "viewer_closed_early"
        elif mode == "walk" and stand_only and not stable:
            status["failure_reason"] = "no_contiguous_final_stand_gate_window"
        elif stable or stand_start_verified:
            status["failure_reason"] = None
        else:
            status["failure_reason"] = "no_contiguous_stand_gate_window"
        status["duration_simulated_s"] = status["sample_count"] * dt
        status["wall_elapsed_s"] = time.monotonic() - start_clock
        _mark_initialization_stage(status_path, status, "hold_viewer" if viewer is not None and hold_viewer else "complete")
        print("stand diagnostic written: {} status={}".format(out, status["status"]))
        if viewer is not None and hold_viewer and not gym.query_viewer_has_closed(viewer):
            deadline = time.monotonic() + 1200.0
            while not gym.query_viewer_has_closed(viewer) and time.monotonic() < deadline:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)
                time.sleep(0.02)
        return out
    except Exception as exc:
        status["failure_reason"] = "{}: {}".format(type(exc).__name__, exc)
        status["status"] = "failed_runtime_error"
        raise
    finally:
        if csv_file:
            csv_file.close()
        if openshc_bridge is not None:
            openshc_bridge.close()
        if out.exists():
            status_path.write_text(json.dumps(_json_safe(status), indent=2, allow_nan=False) + "\n")
        if gym is not None and viewer is not None:
            gym.destroy_viewer(viewer)
        if gym is not None and sim is not None:
            gym.destroy_sim(sim)


def run_walk(config_path: Union[str, Path], sim_device: Optional[str] = None, headless: bool = False,
             duration_s: Optional[float] = None, output_root: Optional[Union[str, Path]] = None,
             hold_viewer: bool = False, stand_only: bool = False) -> Path:
    """Run the separate 50 Hz gait target / modeled MIT actuator diagnostic."""
    if load_config(config_path).get("mode", "stand") != "walk":
        raise ValueError("run_walk requires a walk-mode configuration")
    return run_stand(config_path, sim_device, headless, duration_s, output_root, hold_viewer, stand_only)


def run_diagnostic(config_path: Union[str, Path], sim_device: Optional[str] = None, headless: bool = False,
                   duration_s: Optional[float] = None, output_root: Optional[Union[str, Path]] = None,
                   hold_viewer: bool = False, stand_only: bool = False) -> Path:
    """Dispatch the historical stand or independent walk diagnostic by config mode."""
    config = load_config(config_path)
    runner = run_walk if config.get("mode", "stand") == "walk" else run_stand
    return runner(config_path, sim_device, headless, duration_s, output_root, hold_viewer, stand_only)
