"""Single-run Isaac Gym flat-stand diagnostic for the local URDF."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
from hexapod_robot import (JOINT_NAMES, LEG_ORDER, load_robot_model, lowest_stand_height,
                          map_named_values, pd_torque, stand_target_by_name)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if set(config["asset_import"]) != {"flip_visual_attachments"} or not isinstance(config["asset_import"]["flip_visual_attachments"], bool):
        raise ValueError("asset_import must explicitly define flip_visual_attachments as a boolean")
    p, c, g, initial = config["physics"], config["control"], config["gate"], config.get("initial", {})
    if p.get("gpu_pipeline") is not False or p.get("gpu_physics") is not True:
        raise ValueError("this diagnostic requires GPU PhysX with CPU tensor pipeline")
    if p.get("solver_type") != "TGS":
        raise ValueError("the standalone pilot requires the configured TGS solver")
    if c.get("mode") != "effort_pd":
        raise ValueError("only explicit effort_pd stand control is supported")
    numeric_groups = [
        ("duration_s", [config.get("duration_s")]),
        ("physics", [p.get(k) for k in ("dt_s", "contact_offset_m", "rest_offset_m", "plane_static_friction", "plane_dynamic_friction", "plane_restitution")] + list(p.get("gravity_m_s2", []))),
        ("control", [c.get(k) for k in ("kp_nm_rad", "kd_nm_s_rad", "effort_limit_nm", "joint_velocity_limit_rad_s", "position_tolerance_rad", "velocity_tolerance_rad_s")]),
        ("gate", [g.get(k) for k in ("eligible_after_s", "window_s", "min_contacts", "max_abs_roll_pitch_rad", "max_linear_speed_m_s", "max_angular_speed_rad_s", "max_abs_vertical_speed_m_s", "max_tilt_rad", "contact_threshold_n")]),
        ("initial", list(initial.get("base_xy_m", [])) + [initial.get("yaw_rad"), initial.get("toe_clearance_m"), initial.get("base_z_min_m")]),
    ]
    for name, values in numeric_groups:
        if not values or any(not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in values):
            raise ValueError("{} parameters must be finite numbers".format(name))
    if config["duration_s"] <= 0 or p["dt_s"] <= 0 or c["effort_limit_nm"] <= 0 or c["joint_velocity_limit_rad_s"] <= 0:
        raise ValueError("duration, time step, torque limit, and velocity limit must be positive")
    if int(p["substeps"]) < 1 or int(p["position_iterations"]) < 1 or int(p["velocity_iterations"]) < 0:
        raise ValueError("substeps and position iterations must be positive; velocity iterations nonnegative")
    if min(p["contact_offset_m"], p["rest_offset_m"], p["plane_static_friction"], p["plane_dynamic_friction"],
           p["plane_restitution"], c["kp_nm_rad"], c["kd_nm_s_rad"], c["position_tolerance_rad"],
           c["velocity_tolerance_rad_s"], g["eligible_after_s"], g["max_abs_roll_pitch_rad"],
           g["max_linear_speed_m_s"], g["max_angular_speed_rad_s"], g["max_abs_vertical_speed_m_s"],
           g["max_tilt_rad"], g["contact_threshold_n"], initial["toe_clearance_m"]) < 0:
        raise ValueError("offsets, gains, tolerances, and gate thresholds must be nonnegative")
    if g["window_s"] <= 0 or int(g["min_contacts"]) < 1 or len(initial["base_xy_m"]) != 2 or len(p["gravity_m_s2"]) != 3:
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


def _prepare_output(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = root / ("standalone_" + stamp)
    path.mkdir(exist_ok=False)
    return path


def run_stand(config_path: Union[str, Path], sim_device: Optional[str] = None, headless: bool = False,
              duration_s: Optional[float] = None, output_root: Optional[Union[str, Path]] = None,
              hold_viewer: bool = False) -> Path:
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
    limits = model.limits
    asset_paths = [urdf_path] + sorted((PROJECT_ROOT / "assets/hexapod/two_set_meshes/visual").glob("*.dae")) + sorted((PROJECT_ROOT / "assets/hexapod/two_set_meshes/collision").glob("*.obj"))
    resolved = dict(config)
    resolved["resolved"] = {"config_path": str(config_path), "robot_urdf_sha256": _hash(urdf_path),
                            "asset_sha256": {str(x.relative_to(PROJECT_ROOT)): _hash(x) for x in asset_paths},
                            "derived_start_base_z_m": start_z, "sim_device": sim_device, "duration_s": duration}
    (out / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")

    gym = sim = viewer = gymapi = None
    csv_file = None
    status: Dict[str, Any] = {"kind": "stand_diagnostic", "status": "started", "failure_reason": None,
                              "valid_training_data": False, "start_base_z_m": start_z,
                              "configured_initial_base_velocity_m_s": [0.0, 0.0, 0.0],
                              "configured_initial_base_angular_velocity_rad_s": [0.0, 0.0, 0.0],
                              "initial_joint_target_by_name_rad": qref_named,
                              "duration_requested_s": duration, "sample_count": 0,
                              "stable_gate_passed": False, "gate_rejection_counts": {},
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
                 "limit_violation", "torque_saturated_count"]
        for n in sim_dof_names:
            names.extend(["q_" + n, "dq_" + n, "tau_" + n, "qref_" + n])
        csv_file = (out / "trace.csv").open("w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=names)
        writer.writeheader()
        status.update({"sim_device_requested": sim_device, "backend": "PhysX", "gpu_physics": bool(p["gpu_physics"]),
                       "gpu_tensor_pipeline": bool(p["gpu_pipeline"]), "sim_params_created": {"dt_s": dt, "substeps": p["substeps"],
                       "position_iterations": p["position_iterations"], "velocity_iterations": p["velocity_iterations"],
                       "contact_offset_m": p["contact_offset_m"], "rest_offset_m": p["rest_offset_m"],
                       "solver_type": p["solver_type"], "solver_type_enum": 1, "up_axis": "Z", "self_collision": bool(p["self_collision"]),
                       "fixed_base": bool(p["fixed_base"]), "collapse_fixed_joints": bool(p["collapse_fixed_joints"])},
                       "python_version": platform.python_version(), "numpy_version": np.__version__,
                       "torch_version": torch.__version__, "torch_cuda_available": torch.cuda.is_available(),
                       "sim_dof_names": sim_dof_names, "sim_body_names": body_names,
                       "toe_body_indices_sim": toe_body_indices})
        start_clock = time.monotonic()
        criteria = {"contacts": 0, "roll_pitch": 0, "linear_speed": 0, "angular_speed": 0, "vertical_speed": 0, "joint_position": 0, "joint_velocity": 0, "torque_saturation": 0}
        candidate = deque()
        stable = False
        max_steps = int(math.ceil(duration / dt))
        failure = None
        interrupted_by_viewer = False
        last_measured_state = None
        for step in range(max_steps):
            dof = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            q = np.asarray(dof["pos"], dtype=np.float32)
            dq = np.asarray(dof["vel"], dtype=np.float32)
            tau, saturated = pd_torque(q, dq, qref, float(control["kp_nm_rad"]), float(control["kd_nm_s_rad"]), effort_limits)
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
            contact_np = contact_tensor.detach().cpu().numpy()
            quat_norm = float(np.linalg.norm(quat))
            state_finite = all(np.all(np.isfinite(x)) for x in (q, dq, xyz, quat, vel, omega, contact_np))
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
            row = {"time_s": (step + 1) * dt, "root_x_m": xyz[0], "root_y_m": xyz[1], "root_z_m": xyz[2], "roll_rad": roll,
                   "pitch_rad": pitch, "yaw_rad": yaw, "vx_m_s": vel[0], "vy_m_s": vel[1], "vz_m_s": vel[2],
                   "wx_rad_s": omega[0], "wy_rad_s": omega[1], "wz_rad_s": omega[2], "contact_count": contact_count,
                   "contact_legs": ",".join(k for k, v in foot_contacts.items() if v), "max_abs_roll_pitch_rad": max(abs(roll), abs(pitch)),
                   "linear_speed_m_s": linear_speed, "angular_speed_rad_s": angular_speed, "limit_violation": int(pos_bad or vel_bad),
                   "torque_saturated_count": int(np.count_nonzero(saturated))}
            for i, n in enumerate(sim_dof_names):
                row["q_" + n], row["dq_" + n], row["tau_" + n], row["qref_" + n] = q[i], dq[i], tau[i], qref[i]
            writer.writerow(row)
            # Flush every sample so an early failure leaves a usable partial trace.
            csv_file.flush()
            status["sample_count"] = step + 1
            t = (step + 1) * dt
            last_measured_state = {"time_s": t, "root_xyz_m": xyz.tolist(), "root_quaternion_xyzw": quat.tolist(),
                                   "root_linear_velocity_m_s": vel.tolist(), "root_angular_velocity_rad_s": omega.tolist(),
                                   "joint_q_rad": {n: float(q[i]) for i, n in enumerate(sim_dof_names)},
                                   "joint_dq_rad_s": {n: float(dq[i]) for i, n in enumerate(sim_dof_names)},
                                   "toe_contact_force_n": {leg: contact_np[idx].tolist() for leg, idx in toe_body_indices.items()}}
            status["last_measured_state"] = last_measured_state
            if t >= float(gate["eligible_after_s"]):
                checks = {"contacts": contact_count >= int(gate["min_contacts"]),
                          "roll_pitch": max(abs(roll), abs(pitch)) <= float(gate["max_abs_roll_pitch_rad"]),
                          "linear_speed": linear_speed <= float(gate["max_linear_speed_m_s"]),
                          "angular_speed": angular_speed <= float(gate["max_angular_speed_rad_s"]),
                          "vertical_speed": abs(vel[2]) <= float(gate["max_abs_vertical_speed_m_s"]),
                          "joint_position": not pos_bad, "joint_velocity": not vel_bad,
                          "torque_saturation": not np.any(saturated)}
                for key, passed in checks.items():
                    criteria[key] += int(not passed)
                if all(checks.values()) and failure is None:
                    candidate.append(t)
                    while candidate and t - candidate[0] > float(gate["window_s"]) + 1e-9:
                        candidate.popleft()
                    stable = t - candidate[0] + 1e-9 >= float(gate["window_s"])
                else:
                    candidate.clear()
                    stable = False
            if failure == "nonfinite_measured_state" or failure == "invalid_root_quaternion":
                candidate.clear()
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
                candidate.clear()
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
        status["stable_gate_passed"] = stable
        status["status"] = "settled" if stable and failure is None else "failed_stand_gate"
        if failure:
            status["status"] = "failed_safety_stop"
            status["failure_reason"] = failure
        elif stable:
            status["failure_reason"] = None
        elif interrupted_by_viewer:
            status["status"] = "interrupted_viewer"
            status["failure_reason"] = "viewer_closed_early"
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
