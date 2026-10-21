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
from hexapod_walk import (ControlCadence, StandStartGate, TripodGait, create_gait_after_stand_gate, feet_world,
                          leg_fk_jacobian, leg_inverse_kinematics, walk_progress_failure)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_provenance(project_root: Path, include_walk: bool = False) -> Dict[str, Any]:
    """Describe the exact independent Python runtime inputs for this run."""
    runtime_files = ("src/hexapod_sim.py", "src/hexapod_robot.py", "scripts/run_hexapod_pilot.py",
                     "src/hexapod_walk.py", "src/hexapod_actuator.py")
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
        ("physics", [p.get(k) for k in ("dt_s", "contact_offset_m", "rest_offset_m", "plane_static_friction", "plane_dynamic_friction", "plane_restitution")] + list(p.get("gravity_m_s2", []))),
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
    cadence = ControlCadence(float(p["dt_s"]), float(control_period))
    if mode == "walk":
        if not math.isclose(float(p["dt_s"]), 0.0025, abs_tol=1e-12) or not math.isclose(float(control_period), 0.02, abs_tol=1e-12) or cadence.steps_per_update != 8:
            raise ValueError("walk mode requires 2.5 ms physics steps and an exact 20 ms control period")
        walk = config.get("walk")
        if not isinstance(walk, dict):
            raise ValueError("walk mode requires a walk settings object")
        actuator = config.get("actuator")
        if not isinstance(actuator, dict) or set(actuator) != {
                "kind", "internal_period_s", "command_delay_s", "torque_bandwidth_hz", "torque_feedforward_nm"}:
            raise ValueError("walk mode requires complete MIT_PD actuator settings")
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
        if (walk["speed_m_s"] <= 0 or walk["frequency_hz"] <= 0 or not 0.5 < walk["duty_factor"] < 1.0 or
                walk["swing_height_m"] <= 0 or walk["speed_ramp_s"] <= 0 or walk["start_timeout_s"] <= 0 or
                walk["touchdown_timeout_s"] <= 0 or walk["max_stance_slip_m"] <= 0 or
                walk["max_tracking_error_m"] <= 0 or not 0.0 < walk["max_saturation_fraction"] <= 1.0 or
                walk["minimum_cycles"] < 5 or walk["min_support_contacts"] > 3 or walk["support_failure_s"] <= 0 or
                walk["slip_failure_s"] <= 0 or walk["minimum_net_forward_displacement_m"] <= 0 or
                walk["minimum_completed_cycle_displacement_m"] <= 0):
            raise ValueError("walk settings are outside the supported low-speed tripod ranges")
    if min(p["contact_offset_m"], p["rest_offset_m"], p["plane_static_friction"], p["plane_dynamic_friction"],
           p["plane_restitution"], c["kp_nm_rad"], c["kd_nm_s_rad"], c["position_tolerance_rad"],
           c["velocity_tolerance_rad_s"], g["eligible_after_s"], g["max_abs_roll_pitch_rad"],
           g["max_linear_speed_m_s"], g["max_angular_speed_rad_s"], g["max_abs_vertical_speed_m_s"],
           g["max_tilt_rad"], g["contact_threshold_n"], initial["toe_clearance_m"]) < 0:
        raise ValueError("offsets, gains, tolerances, and gate thresholds must be nonnegative")
    if g["window_s"] <= 0 or len(initial["base_xy_m"]) != 2 or len(p["gravity_m_s2"]) != 3:
        raise ValueError("invalid gate window, contact count, base position, or gravity vector")
    if p["plane_restitution"] > 1.0 or initial["base_z_min_m"] <= 0:
        raise ValueError("restitution must be at most one and minimum base height must be positive")
    robot_path = (PROJECT_ROOT / config["robot_urdf"]).resolve()
    model = load_robot_model(robot_path)
    stand_target_by_name(model, config["stand_reference_rad"])
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


def _walk_control_targets(model, gait: TripodGait, elapsed_s: float, q_by_name: Dict[str, float],
                          body_xyz: np.ndarray, body_quat: np.ndarray,
                          contacts: Dict[str, bool], seed_by_name: Dict[str, float],
                          measured_feet_world: Optional[Dict[str, np.ndarray]] = None):
    actual_feet = measured_feet_world or feet_world(model, q_by_name, body_xyz, body_quat)
    state = gait.update(elapsed_s, actual_feet, contacts)
    planned_body = np.asarray(state["planned_body_xyz_m"], dtype=float)
    c, s = math.cos(gait.yaw), math.sin(gait.yaw)
    nominal_rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    q_targets = dict(seed_by_name)
    residuals = {}
    for leg in LEG_ORDER:
        target_world = np.asarray(state["desired_feet_world"][leg], dtype=float)
        target_base = nominal_rotation.T @ (target_world - planned_body)
        solved, residuals[leg] = leg_inverse_kinematics(model, leg, target_base, q_targets)
        q_targets.update(solved)
    return q_targets, state, actual_feet, residuals


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
    mode = config.get("mode", "stand")
    walk_settings = config.get("walk", {})
    control_period = float(config.get("control_period_s", dt))
    cadence = ControlCadence(dt, control_period)
    limits = model.limits
    asset_paths = [urdf_path] + sorted((PROJECT_ROOT / "assets/hexapod/two_set_meshes/visual").glob("*.dae")) + sorted((PROJECT_ROOT / "assets/hexapod/two_set_meshes/collision").glob("*.obj"))
    resolved = dict(config)
    resolved["resolved"] = {"config_path": str(config_path), "robot_urdf_sha256": _hash(urdf_path),
                            "asset_sha256": {str(x.relative_to(PROJECT_ROOT)): _hash(x) for x in asset_paths},
                            "derived_start_base_z_m": start_z, "sim_device": sim_device, "duration_s": duration,
                            "provenance": _runtime_provenance(PROJECT_ROOT, include_walk=(mode == "walk"))}
    (out / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")

    gym = sim = viewer = gymapi = None
    csv_file = None
    status: Dict[str, Any] = {"kind": "walk_diagnostic" if mode == "walk" else "stand_diagnostic", "status": "started", "failure_reason": None,
                              "valid_training_data": False, "start_base_z_m": start_z,
                              "configured_initial_base_velocity_m_s": [0.0, 0.0, 0.0],
                              "configured_initial_base_angular_velocity_rad_s": [0.0, 0.0, 0.0],
                              "initial_joint_target_by_name_rad": qref_named,
                              "duration_requested_s": duration, "sample_count": 0,
                              "stable_gate_passed": False, "gate_rejection_counts": {},
                              "mode": mode, "control_period_s": control_period,
                              "control_steps_per_update": cadence.steps_per_update,
                              "controller_effort_source": "stand explicit effort PD" if config.get("mode", "stand") == "stand" else "50 Hz position target with modeled 400 Hz MIT_PD actuator",
                              "modeled_motor_delay_s": config.get("actuator", {}).get("command_delay_s", 0.0),
                              "motor_torque_bandwidth_hz": config.get("actuator", {}).get("torque_bandwidth_hz"),
                              "motor_torque_bandwidth_calibrated": False,
                              "actuator_torque_is_model_output_not_measured": True,
                              "initialization_stage": "pre_native"}
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
        dof_props = gym.get_actor_dof_properties(env, actor)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        dof_props["stiffness"].fill(0.0)
        dof_props["damping"].fill(0.0)
        gym.set_actor_dof_properties(env, actor, dof_props)
        dof_state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        dof_state["pos"][:] = qref
        dof_state["vel"][:] = 0.0
        gym.set_actor_dof_states(env, actor, dof_state, gymapi.STATE_ALL)
        body_names = list(gym.get_actor_rigid_body_names(env, actor))
        body_dict = gym.get_actor_rigid_body_dict(env, actor)
        toe_body_indices = {leg: int(gym.get_actor_rigid_body_index(env, actor, body_dict["l_{}_toe".format(leg)], gymapi.DOMAIN_SIM)) for leg in LEG_ORDER}
        toe_body_local_indices = {leg: int(body_dict["l_{}_toe".format(leg)]) for leg in LEG_ORDER}
        root_body_index = body_dict[model.root_link]
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
        if mode == "walk":
            names.extend(["gait_started", "gait_elapsed_s", "gait_cycle_phase", "commanded_speed_m_s",
                          "achieved_forward_speed_m_s", "planned_body_x_m", "planned_body_y_m",
                          "planned_body_z_m", "body_tracking_error_m", "planned_support_contact_count",
                          "stance_slip_count", "max_stance_slip_m", "max_ik_residual_m"])
            for leg in LEG_ORDER:
                names.extend(["actual_foot_{}_{}_m".format(leg, axis) for axis in "xyz"])
                names.extend(["fk_foot_{}_{}_m".format(leg, axis) for axis in "xyz"])
                names.append("toe_fk_error_{}_m".format(leg))
                names.extend(["desired_foot_{}_{}_m".format(leg, axis) for axis in "xyz"])
                names.extend(["anchor_{}_{}_m".format(leg, axis) for axis in "xyz"])
                names.extend(["planned_support_" + leg, "measured_contact_" + leg,
                              "stance_slip_{}_m".format(leg), "ik_residual_{}_m".format(leg)])
            for n in sim_dof_names:
                names.extend(["qref_delayed_" + n, "dqref_command_" + n, "dqref_delayed_" + n,
                              "tau_raw_" + n, "tau_requested_" + n, "tau_ff_command_" + n,
                              "tau_ff_delayed_" + n, "tau_applied_model_" + n, "actuator_saturated_" + n])
        for n in sim_dof_names:
            names.extend(["q_" + n, "dq_" + n, "tau_" + n, "qref_" + n])
        csv_file = (out / "trace.csv").open("w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=names)
        writer.writeheader()
        status.update({"sim_device_requested": sim_device, "backend": "PhysX", "gpu_physics": bool(p["gpu_physics"]),
                       "gpu_tensor_pipeline": bool(p["gpu_pipeline"]), "sim_params_created": {"dt_s": dt, "substeps": p["substeps"],
                       "control_period_s": control_period, "control_steps_per_update": cadence.steps_per_update,
                       "position_iterations": p["position_iterations"], "velocity_iterations": p["velocity_iterations"],
                       "contact_offset_m": p["contact_offset_m"], "rest_offset_m": p["rest_offset_m"],
                       "solver_type": p["solver_type"], "solver_type_enum": 1, "up_axis": "Z", "self_collision": bool(p["self_collision"]),
                       "fixed_base": bool(p["fixed_base"]), "collapse_fixed_joints": bool(p["collapse_fixed_joints"])},
                       "python_version": platform.python_version(), "numpy_version": np.__version__,
                       "torch_version": torch.__version__, "torch_cuda_available": torch.cuda.is_available(),
                       "sim_dof_names": sim_dof_names, "sim_body_names": body_names,
                       "toe_body_indices_sim": toe_body_indices,
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
        gait_start_time = None
        walk_start_verified_at = None
        stand_start_verified = False
        walk_started = False
        failed_start = False
        current_ik_residuals = {leg: 0.0 for leg in LEG_ORDER}
        support_mismatch_ticks = 0
        slip_consecutive = {leg: 0 for leg in LEG_ORDER}
        support_contact_samples = 0
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
        last_cycle_index = 0
        cycle_start_progress = 0.0
        initial_walk_xyz = None
        walk_forward = None
        stable = False
        held_tau = np.zeros_like(qref)
        held_saturated = np.zeros_like(qref, dtype=bool)
        control_tick_count = 0
        max_steps = int(math.ceil(duration / dt))
        failure = None
        interrupted_by_viewer = False
        last_measured_state = None
        toe_fk_error_max_observed = 0.0
        for step in range(max_steps):
            dof = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            q = np.asarray(dof["pos"], dtype=np.float32)
            dq = np.asarray(dof["vel"], dtype=np.float32)
            control_update = cadence.is_tick(step)
            control_tick_index = cadence.tick_index(step)
            if control_update:
                if mode == "walk":
                    try:
                        rb_control = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
                        root_control = rb_control["pose"][root_body_index]
                        xyz_control = np.array([root_control["p"][k] for k in ("x", "y", "z")], dtype=float)
                        quat_control = np.array([root_control["r"][k] for k in ("x", "y", "z", "w")], dtype=float)
                        q_control_by_name = {name: float(q[index]) for index, name in enumerate(sim_dof_names)}
                        if gait is not None and not stand_only:
                            contact_control = contact_tensor.detach().cpu().numpy()
                            contact_map = {leg: float(np.linalg.norm(contact_control[idx])) > float(gate["contact_threshold_n"])
                                           for leg, idx in toe_body_indices.items()}
                            feet_control_world = {leg: np.array([rb_control["pose"][toe_body_local_indices[leg]]["p"][axis]
                                                                  for axis in ("x", "y", "z")], dtype=float)
                                                  for leg in LEG_ORDER}
                            qref_named, gait_state, _, current_ik_residuals = _walk_control_targets(
                                model, gait, max(0.0, step * dt - gait_start_time), q_control_by_name,
                                xyz_control, quat_control, contact_map, qref_named, feet_control_world)
                            qref = map_named_values(qref_named, sim_dof_names)
                            status["gait_control_ticks"] = status.get("gait_control_ticks", 0) + 1
                    except (ValueError, RuntimeError, np.linalg.LinAlgError, KeyError) as exc:
                        failure = "walk_control_failure: {}".format(exc)
                        break
                else:
                    held_tau, held_saturated = pd_torque(q, dq, qref, float(control["kp_nm_rad"]),
                                                         float(control["kd_nm_s_rad"]), effort_limits)
                control_tick_count += 1
            if mode == "walk":
                actuator_sample = mit_actuator.step(q, dq, qref, command_feedforward_nm=actuator_feedforward)
                tau = actuator_sample["applied_model_torque_nm"].astype(np.float32)
                saturated = actuator_sample["torque_saturated"]
                gym.apply_actor_dof_efforts(env, actor, tau)
            else:
                actuator_sample = None
                tau, saturated = held_tau, held_saturated
                gym.apply_actor_dof_efforts(env, actor, tau)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_net_contact_force_tensor(sim)
            dof = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            q = np.asarray(dof["pos"], dtype=float)
            dq = np.asarray(dof["vel"], dtype=float)
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
            if failure:
                roll = pitch = yaw = float("nan")
            else:
                roll, pitch, yaw = _orientation_to_rpy(quat)
            foot_contacts = {leg: (float(np.linalg.norm(contact_np[idx])) > float(gate["contact_threshold_n"]))
                             if state_finite else False for leg, idx in toe_body_indices.items()}
            contact_count = sum(foot_contacts.values())
            linear_speed = float(np.linalg.norm(vel))
            angular_speed = float(np.linalg.norm(omega))
            pos_bad = bool(state_finite and (np.any(q < lower - float(control["position_tolerance_rad"])) or np.any(q > upper + float(control["position_tolerance_rad"]))))
            vel_bad = bool(state_finite and np.any(np.abs(dq) > max_velocity + float(control["velocity_tolerance_rad_s"])))
            t = (step + 1) * dt
            gait_row = {}
            if mode == "walk" and gait is not None and gait_state is not None:
                desired_feet = gait_state["desired_feet_world"]
                anchors = gait_state["anchors_world"]
                planned_support = gait_state["planned_support"]
                planned_body = np.asarray(gait_state["planned_body_xyz_m"], dtype=float)
                valid_support_count = sum(bool(planned_support[leg]) and foot_contacts[leg] for leg in LEG_ORDER)
                support_count = sum(bool(planned_support[leg]) for leg in LEG_ORDER)
                planned_support_samples += support_count
                support_contact_samples += valid_support_count
                support_mismatch_ticks = support_mismatch_ticks + 1 if valid_support_count < int(walk_settings["min_support_contacts"]) else 0
                slip_by_leg = {}
                for leg in LEG_ORDER:
                    slip = float(np.linalg.norm(measured_feet_world[leg] - anchors[leg])) if planned_support[leg] and foot_contacts[leg] else 0.0
                    slip_by_leg[leg] = slip
                    if slip > float(walk_settings["max_stance_slip_m"]):
                        slip_samples += 1
                        slip_consecutive[leg] += 1
                        if slip_consecutive[leg] == 1:
                            slip_events += 1
                    else:
                        slip_consecutive[leg] = 0
                max_slip = max(slip_by_leg.values(), default=0.0)
                tracking_error = float(np.linalg.norm(xyz - planned_body))
                tracking_error_max = max(tracking_error_max, tracking_error)
                max_ik = max(current_ik_residuals.values(), default=0.0)
                gait_elapsed = max(0.0, t - gait_start_time)
                cycle_index = int(math.floor(gait_elapsed * float(walk_settings["frequency_hz"]) + 1e-9))
                progress = float(np.dot(xyz - initial_walk_xyz, walk_forward))
                while cycle_index > last_cycle_index:
                    cycle_progress.append(progress - cycle_start_progress)
                    cycle_start_progress = progress
                    last_cycle_index += 1
                commanded_speed = float(walk_settings["speed_m_s"]) * min(1.0, gait_elapsed / float(walk_settings["speed_ramp_s"]))
                gait_row = {"gait_started": 1, "gait_elapsed_s": gait_elapsed,
                            "gait_cycle_phase": gait_state["cycle_phase"], "commanded_speed_m_s": commanded_speed,
                            "achieved_forward_speed_m_s": float(np.dot(vel, walk_forward)),
                            "planned_body_x_m": planned_body[0], "planned_body_y_m": planned_body[1],
                            "planned_body_z_m": planned_body[2], "body_tracking_error_m": tracking_error,
                            "planned_support_contact_count": valid_support_count,
                            "stance_slip_count": sum(value > float(walk_settings["max_stance_slip_m"]) for value in slip_by_leg.values()),
                            "max_stance_slip_m": max_slip, "max_ik_residual_m": max_ik}
                for leg in LEG_ORDER:
                    for axis, value in zip("xyz", desired_feet[leg]):
                        gait_row["desired_foot_{}_{}_m".format(leg, axis)] = value
                    for axis, value in zip("xyz", anchors[leg]):
                        gait_row["anchor_{}_{}_m".format(leg, axis)] = value
                    gait_row["planned_support_" + leg] = int(bool(planned_support[leg]))
                    gait_row["measured_contact_" + leg] = int(foot_contacts[leg])
                    gait_row["stance_slip_{}_m".format(leg)] = slip_by_leg[leg]
                    gait_row["ik_residual_{}_m".format(leg)] = current_ik_residuals[leg]
                walk_samples += 1
                saturation_samples += int(np.count_nonzero(saturated))
                q_error = qref.astype(float) - q
                q_error_sq_sum += float(np.dot(q_error, q_error))
                q_error_values += len(q_error)
                q_error_max = max(q_error_max, float(np.max(np.abs(q_error))))
                if tracking_error > float(walk_settings["max_tracking_error_m"]):
                    failure = "planned_body_tracking_error"
                elif support_mismatch_ticks >= int(math.ceil(float(walk_settings["support_failure_s"]) / dt)):
                    failure = "planned_support_contact_loss"
                elif max(slip_consecutive.values(), default=0) >= int(math.ceil(float(walk_settings["slip_failure_s"]) / dt)):
                    failure = "persistent_stance_slip"
            row = {"time_s": (step + 1) * dt, "root_x_m": xyz[0], "root_y_m": xyz[1], "root_z_m": xyz[2], "roll_rad": roll,
                   "pitch_rad": pitch, "yaw_rad": yaw, "vx_m_s": vel[0], "vy_m_s": vel[1], "vz_m_s": vel[2],
                   "wx_rad_s": omega[0], "wy_rad_s": omega[1], "wz_rad_s": omega[2], "contact_count": contact_count,
                   "contact_legs": ",".join(k for k, v in foot_contacts.items() if v), "max_abs_roll_pitch_rad": max(abs(roll), abs(pitch)),
                   "linear_speed_m_s": linear_speed, "angular_speed_rad_s": angular_speed, "limit_violation": int(pos_bad or vel_bad),
                   "torque_saturated_count": int(np.count_nonzero(saturated)),
                   "control_update": int(control_update), "control_tick_index": control_tick_index,
                   "actuator_update": int(mode == "walk"), "toe_fk_error_max_m": toe_fk_error_max}
            for i, n in enumerate(sim_dof_names):
                row["q_" + n], row["dq_" + n], row["tau_" + n], row["qref_" + n] = q[i], dq[i], tau[i], qref[i]
                if mode == "walk":
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
                    for axis, value in zip("xyz", measured_feet_world[leg]):
                        row["actual_foot_{}_{}_m".format(leg, axis)] = value
                    for axis, value in zip("xyz", fk_feet_world.get(leg, (float("nan"),) * 3)):
                        row["fk_foot_{}_{}_m".format(leg, axis)] = value
                    row["toe_fk_error_{}_m".format(leg)] = toe_fk_error[leg]
                    row["measured_contact_" + leg] = int(foot_contacts[leg])
                if gait is None or gait_state is None:
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
            checks = {"contacts": contact_count >= int(gate["min_contacts"]),
                          "roll_pitch": max(abs(roll), abs(pitch)) <= float(gate["max_abs_roll_pitch_rad"]),
                          "linear_speed": linear_speed <= float(gate["max_linear_speed_m_s"]),
                          "angular_speed": angular_speed <= float(gate["max_angular_speed_rad_s"]),
                          "vertical_speed": abs(vel[2]) <= float(gate["max_abs_vertical_speed_m_s"]),
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
                            yaw0 = _orientation_to_rpy(quat)[2]
                            rotation_yaw = np.array([[math.cos(yaw0), -math.sin(yaw0), 0.0],
                                                    [math.sin(yaw0), math.cos(yaw0), 0.0], [0.0, 0.0, 1.0]])
                            nominal_feet = {leg: rotation_yaw.T @ (measured_feet_world[leg] - xyz)
                                            for leg in LEG_ORDER}
                            gait, gait_state = create_gait_after_stand_gate(
                                stand_start_verified, nominal_feet, measured_feet_world, xyz, yaw0,
                                walk_settings, foot_contacts)
                            gait_start_time = t
                            initial_walk_xyz = xyz.copy()
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
            if pos_bad:
                failure = "joint_position_limit"
            elif vel_bad:
                failure = "joint_velocity_limit"
            elif max(abs(roll), abs(pitch)) > float(gate["max_tilt_rad"]):
                failure = "tilt_limit"
            elif xyz[2] < float(initial["base_z_min_m"]):
                failure = "base_height_limit"
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
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)
        status["gate_rejection_counts"] = criteria
        status["gate_window_s"] = float(gate["window_s"])
        status["stable_gate_passed"] = (stable if stand_only else stand_start_verified) if mode == "walk" else stable
        status["control_tick_count"] = control_tick_count
        status["toe_fk_error_max_observed_m"] = toe_fk_error_max_observed
        if mode == "walk":
            saturation_fraction = (float(saturation_samples) / (walk_samples * len(sim_dof_names))) if walk_samples else 0.0
            status.update({"gait_started": walk_started, "completed_gait_cycles": last_cycle_index,
                           "cycle_forward_displacement_m": cycle_progress,
                           "mean_completed_cycle_displacement_m": float(np.mean(cycle_progress)) if cycle_progress else None,
                           "net_forward_displacement_m": (float(np.dot(xyz - initial_walk_xyz, walk_forward))
                                                           if walk_started and last_measured_state else 0.0),
                           "planned_support_contact_fraction": (support_contact_samples / planned_support_samples
                                                                if planned_support_samples else None),
                           "stance_slip_threshold_sample_count": slip_samples, "stance_slip_event_count": slip_events,
                           "max_body_tracking_error_m": tracking_error_max,
                           "joint_position_error_rmse_rad": math.sqrt(q_error_sq_sum / q_error_values) if q_error_values else None,
                           "joint_position_error_max_rad": q_error_max if q_error_values else None,
                           "torque_saturation_fraction": saturation_fraction,
                           "walk_sample_count": walk_samples})
            if stand_only:
                status["status"] = "stand_verified" if stable and failure is None else "failed_start"
            elif failure and not walk_started:
                status["status"] = "failed_start"
            elif failure:
                status["status"] = "failed_safety_stop"
            elif not stand_start_verified:
                status["status"] = "failed_start"
                failure = "no_contiguous_stand_gate_window"
            elif last_cycle_index < int(walk_settings["minimum_cycles"]):
                status["status"] = "walk_incomplete"
                failure = "minimum_cycles_not_reached"
            elif progress_failure := walk_progress_failure(
                    cycle_progress,
                    float(np.dot(xyz - initial_walk_xyz, walk_forward)) if walk_started and last_measured_state else 0.0,
                    int(walk_settings["minimum_cycles"]),
                    float(walk_settings["minimum_net_forward_displacement_m"]),
                    float(walk_settings["minimum_completed_cycle_displacement_m"])):
                status["status"] = "walk_incomplete"
                failure = progress_failure
            elif saturation_fraction > float(walk_settings["max_saturation_fraction"]):
                status["status"] = "walk_incomplete"
                failure = "torque_saturation_fraction_exceeded"
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
