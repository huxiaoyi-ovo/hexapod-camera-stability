#!/usr/bin/env python3
"""Run the bounded simulation-only compensation feasibility menu."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from camera_pose_recorder import CameraPoseRecorder
from trace_metrics import camera_scene_from_json, image_motion_metrics, read_trace

BASE_CONFIG = ROOT / "configs/compensation_pilot.json"
CAMERA_SCENE = ROOT / "templates/camera_scene.example.json"
CONDITION_ORDER = ("B_OFF", "B_ON", "A_OFF", "A_ON", "C_OFF", "C_ON")
RUNTIME_HASH_FILES = (
    "src/hexapod_sim.py", "src/hexapod_body_compensation.py", "src/hexapod_robot.py",
    "src/hexapod_walk.py", "src/hexapod_sensing.py", "src/hexapod_terrain.py",
    "src/hexapod_actuator.py", "scripts/run_hexapod_pilot.py", "scripts/run_compensation_pilot.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nominal_camera_twist(rotation_world_body: Rotation, nominal_world_velocity_m_s: np.ndarray,
                         rotation_body_camera: Rotation) -> np.ndarray:
    """Convert the declared smooth world velocity into camera optical axes."""
    rotation_world_camera = rotation_world_body * rotation_body_camera
    return np.r_[rotation_world_camera.inv().apply(nominal_world_velocity_m_s), np.zeros(3)]


def _camera_metrics_for_trace(trace_path: Path, output_dir: Path, start_s: float,
                              initial_yaw_rad: float, speed_m_s: float,
                              camera_position_m, camera_quaternion_xyzw):
    camera, scene = camera_scene_from_json(CAMERA_SCENE)
    camera_position = np.asarray(camera_position_m, dtype=float)
    camera_quaternion = np.asarray(camera_quaternion_xyzw, dtype=float)
    recorder = CameraPoseRecorder(camera_position, camera_quaternion)
    rotation_body_camera = Rotation.from_quat(camera_quaternion)
    forward_world = np.array([-math.sin(initial_yaw_rad), math.cos(initial_yaw_rad), 0.0])
    with trace_path.open("r", newline="", encoding="utf-8") as stream:
        rows = []
        for row in csv.DictReader(stream):
            values = [float(row[key]) for key in ("time_s", "root_x_m", "root_y_m", "root_z_m",
                                                   "roll_rad", "pitch_rad", "yaw_rad")]
            if values[0] >= start_s and np.isfinite(values).all():
                rows.append(row)
    for row in rows:
        t = float(row["time_s"])
        position = np.array([float(row["root_{}_m".format(axis)]) for axis in "xyz"])
        roll, pitch, yaw = (float(row[name]) for name in ("roll_rad", "pitch_rad", "yaw_rad"))
        rotation_world_body = Rotation.from_euler("xyz", [roll, pitch, yaw])
        nominal_world_velocity = speed_m_s * forward_world
        nominal = nominal_camera_twist(rotation_world_body, nominal_world_velocity, rotation_body_camera)
        recorder.append(t, position, rotation_world_body.as_quat(), nominal)
    if len(recorder.samples) < 3:
        return {"status": "insufficient_samples", "sample_count": len(recorder.samples)}
    twist_path = recorder.save_twist_csv(output_dir / "optical_twist.csv")
    t, actual, nominal = read_trace(twist_path)
    metrics = image_motion_metrics(t, actual - nominal, camera, scene)
    return {"status": "computed", "sample_count": len(t), "window_start_s": start_s,
            "window_end_s": float(t[-1]), "primary_J_jitter_px_rms_per_frame": metrics.jitter_px_rms_per_frame,
            "exposure_trace_px_rms": metrics.exposure_trace_px_rms,
            "exposure_net_displacement_px_rms": metrics.exposure_net_displacement_px_rms,
            "peak_exposure_trace_px": metrics.peak_exposure_trace_px,
            "median_samples_per_exposure": metrics.median_samples_per_exposure,
            "twist_csv": str(twist_path)}


def _trial_summary(trial_dir: Path, config: dict, condition: str, return_code: int):
    status_paths = sorted(trial_dir.glob("standalone_*/status.json"))
    if not status_paths:
        return {"condition": condition, "return_code": return_code, "status": "missing_status",
                "trial_dir": str(trial_dir)}
    run_dir = status_paths[-1].parent
    status = json.loads(status_paths[-1].read_text())
    trace_path = run_dir / "trace.csv"
    gait_start = status.get("gait_start_time_s")
    ramp_s = float(config["walk"]["speed_ramp_s"])
    camera = {"status": "not_available"}
    actual_speed = None
    evaluation_start = None
    if gait_start is not None and trace_path.is_file():
        # Use one commanded reference heading for both members, not trial truth.
        yaw = float(config["initial"]["yaw_rad"])
        evaluation_start = float(gait_start) + ramp_s
        camera_config = config["compensation_pilot"]["camera"]
        camera = _camera_metrics_for_trace(trace_path, trial_dir, evaluation_start, yaw,
                                           float(config["walk"]["speed_m_s"]),
                                           camera_config["base_to_camera_position_m"],
                                           camera_config["base_to_camera_quaternion_xyzw"])
        with trace_path.open("r", newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if float(row["time_s"]) >= evaluation_start]
        if rows:
            forward = np.array([-math.sin(yaw), math.cos(yaw)])
            actual_speed = float(np.mean([np.dot([float(row["vx_m_s"]), float(row["vy_m_s"])], forward)
                                          for row in rows]))
    target = float(config["walk"]["speed_m_s"])
    speed_match = (actual_speed is not None and abs(actual_speed - target) <= 0.05 * target)
    passed = status.get("status") == "walk_completed" and speed_match
    return {"condition": condition, "return_code": return_code, "status": status.get("status"),
            "failure_reason": status.get("failure_reason"), "trial_dir": str(trial_dir),
            "raw_run_dir": str(run_dir), "resolved_config": str(run_dir / "resolved_config.json"),
            "walk_completed": status.get("status") == "walk_completed",
            "evaluation_start_s": evaluation_start, "actual_mean_forward_speed_m_s": actual_speed,
            "target_speed_m_s": target, "speed_within_target_5_percent": speed_match,
            "valid_pair_member": passed,
            "net_forward_displacement_m": status.get("net_forward_displacement_m"),
            "completed_gait_cycles": status.get("completed_gait_cycles"),
            "cycle_forward_displacement_m": status.get("cycle_forward_displacement_m"),
            "planned_support_contact_fraction": status.get("planned_support_contact_fraction"),
            "max_stance_slip_m": status.get("max_stance_slip_m"),
            "joint_position_error_rmse_rad": status.get("joint_position_error_rmse_rad"),
            "joint_position_error_max_rad": status.get("joint_position_error_max_rad"),
            "torque_saturation_fraction": status.get("torque_saturation_fraction"),
            "max_abs_roll_pitch_observed_rad": status.get("max_abs_roll_pitch_observed_rad"),
            "simulator_body_z_range_m": status.get("simulator_body_z_range_m"),
            "max_root_body_contact_force_n": status.get("max_root_body_contact_force_n"),
            "camera_proxy": camera,
            "runtime_source_sha256": json.loads((run_dir / "resolved_config.json").read_text())
                                      .get("resolved", {}).get("provenance", {}).get("runtime_dependency_sha256", {})}


def _run_condition(experiment_dir: Path, base_config: dict, candidate: str, enabled: bool):
    condition = "{}_{}".format(candidate, "ON" if enabled else "OFF")
    trial_dir = experiment_dir / ("trial_" + condition)
    if trial_dir.exists():
        raise FileExistsError("refusing to overwrite existing trial {}".format(trial_dir))
    trial_dir.mkdir(parents=True)
    config = json.loads(json.dumps(base_config))
    config["walk"].update(base_config["compensation_pilot"]["candidates"][candidate])
    config["body_compensation"]["enabled"] = enabled
    derived = trial_dir / "resolved_input_config.json"
    derived.write_text(json.dumps(config, indent=2) + "\n")
    env = os.environ.copy()
    python_bin = str(Path(sys.executable).resolve().parent)
    env["PATH"] = python_bin + os.pathsep + env.get("PATH", "")
    env.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/physx_pilot_torch_extensions")
    command = [sys.executable, str(ROOT / "scripts/run_hexapod_pilot.py"),
               "--config", str(derived), "--sim-device", "cuda:0", "--headless",
               "--output-root", str(trial_dir)]
    print("starting {}: {}".format(condition, " ".join(command)), flush=True)
    with (trial_dir / "runner.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=str(ROOT), env=env, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    summary = _trial_summary(trial_dir, config, condition, result.returncode)
    summary["config_path"] = str(derived)
    summary["source_sha256"] = {relative: _sha256(ROOT / relative) for relative in RUNTIME_HASH_FILES}
    summary["camera_source_sha256"] = {
        "src/camera_pose_recorder.py": _sha256(ROOT / "src/camera_pose_recorder.py"),
        "src/trace_metrics.py": _sha256(ROOT / "src/trace_metrics.py"),
        "src/controller.py": _sha256(ROOT / "src/controller.py"),
        "templates/camera_scene.example.json": _sha256(CAMERA_SCENE)}
    (trial_dir / "trial_result.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("finished {} status={} speed={}m/s J={}".format(
        condition, summary.get("status"), summary.get("actual_mean_forward_speed_m_s"),
        summary.get("camera_proxy", {}).get("primary_J_jitter_px_rms_per_frame")), flush=True)
    return summary


def _score_manifest(manifest: dict):
    candidates = manifest["candidates"]
    valid = {}
    for name in ("B", "A", "C"):
        pair = candidates.get(name, {})
        off, on = pair.get("OFF"), pair.get("ON")
        if off and on and off.get("valid_pair_member") and on.get("valid_pair_member"):
            valid[name] = {"OFF": off, "ON": on}
    speeds = [entry[condition]["actual_mean_forward_speed_m_s"]
              for entry in valid.values() for condition in ("OFF", "ON")]
    speed_unmatched = bool(speeds and max(speeds) - min(speeds) >
                           0.05 * float(manifest["target_speed_m_s"]))
    if len(valid) < 2 or speed_unmatched:
        manifest["selection_analysis"] = {
            "status": "speed_unmatched" if speed_unmatched else "insufficient_valid_candidates",
            "valid_candidates": list(valid), "candidate_speed_spread_m_s": (max(speeds) - min(speeds) if speeds else None),
            "minimum_valid_candidates": 2,
            "interpretation": "no fair oracle comparison; this is not a predictor or online scheduling result"}
        return
    raw_choice = min(valid, key=lambda name: valid[name]["OFF"]["camera_proxy"]["primary_J_jitter_px_rms_per_frame"])
    residual_choice = min(valid, key=lambda name: valid[name]["ON"]["camera_proxy"]["primary_J_jitter_px_rms_per_frame"])
    j_raw_under_on = valid[raw_choice]["ON"]["camera_proxy"]["primary_J_jitter_px_rms_per_frame"]
    j_best_under_on = valid[residual_choice]["ON"]["camera_proxy"]["primary_J_jitter_px_rms_per_frame"]
    gain = j_raw_under_on - j_best_under_on
    manifest["selection_analysis"] = {
        "status": "computed_oracle_upper_bound",
        "primary_J": "pinhole static-scene jitter_px_rms_per_frame",
        "theta_raw_minimizing_J0": raw_choice,
        "theta_residual_minimizing_Jpi": residual_choice,
        "Jpi_at_theta_raw": j_raw_under_on,
        "Jpi_at_theta_residual": j_best_under_on,
        "oracle_benefit_px": gain,
        "oracle_benefit_fraction": gain / j_raw_under_on if j_raw_under_on != 0.0 else None,
        "valid_candidates": list(valid), "candidate_speed_spread_m_s": max(speeds) - min(speeds),
        "interpretation": "measured-menu oracle upper bound only; no predictor, online method, or positive-gain claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(BASE_CONFIG))
    parser.add_argument("--pair", choices=("B", "A", "C", "all"), default="all",
                        help="run one paired candidate at a time so baseline outcomes can be reviewed")
    parser.add_argument("--experiment-dir", default=None,
                        help="reuse one experiment directory across pair invocations; existing trials are never overwritten")
    args = parser.parse_args()
    base_path = Path(args.config).resolve()
    base = json.loads(base_path.read_text())
    menu = base["compensation_pilot"]["candidates"]
    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        experiment_dir = ROOT / "data/raw" / ("compensation_pilot_" + stamp)
    experiment_dir.mkdir(parents=True, exist_ok=False if not args.experiment_dir else True)
    manifest_path = experiment_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("base_config_sha256") != _sha256(base_path):
            raise ValueError("experiment config changed since the manifest was created")
    else:
        manifest = {"experiment": "bounded_body_compensation_feasibility",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "base_config": str(base_path), "base_config_sha256": _sha256(base_path),
                    "experiment_seed": base["compensation_pilot"]["experiment_seed"],
                    "initialization": "explicit same configured body pose, stand reference, and gait phase zero; no random draws in deterministic ramp profile",
                    "condition_order": list(CONDITION_ORDER),
                    "candidates": {}, "target_speed_m_s": base["walk"]["speed_m_s"],
                    "speed_tolerance_fraction": 0.05,
                    "evaluation_window": "gait_start + speed_ramp_s through trial end",
                    "primary_metric": "jitter_px_rms_per_frame from simulated pinhole pose/static-scene projection",
                    "camera_assumptions": base["compensation_pilot"]["camera"],
                    "camera_scene_template_sha256": _sha256(CAMERA_SCENE),
                    "analysis_source_sha256": {"scripts/run_compensation_pilot.py": _sha256(Path(__file__))},
                    "camera_interpretation": "simulated fixed-depth pinhole geometric proxy; not Gym RGB, optical flow, blur rendering, or hardware calibration"}
    candidate_names = ("B", "A", "C") if args.pair == "all" else (args.pair,)
    for candidate in candidate_names:
        if candidate not in menu:
            raise ValueError("missing candidate {} in configured menu".format(candidate))
        candidate_result = manifest["candidates"].setdefault(candidate, {})
        for enabled in (False, True):
            condition = "ON" if enabled else "OFF"
            if condition in candidate_result:
                raise FileExistsError("{} {} result already exists; refusing retry".format(candidate, condition))
            candidate_result[condition] = _run_condition(experiment_dir, base, candidate, enabled)
            manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        pair = candidate_result
        pair["valid_pair"] = bool(pair["OFF"].get("valid_pair_member") and pair["ON"].get("valid_pair_member"))
        if candidate == "B" and not pair["valid_pair"]:
            manifest["stopped_after_baseline_pair_failure"] = True
            manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            print("B pair invalid; stopping without candidate substitution", flush=True)
            return 2
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        print("pair {} complete valid_pair={}".format(candidate, pair["valid_pair"]), flush=True)
    _score_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print("selection_analysis={}".format(json.dumps(manifest["selection_analysis"], sort_keys=True)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
