"""Deterministic process adapter for the vendored OpenSHC gait core.

The adapter keeps URDF and Gym name mapping in Python.  Its persistent C++
child receives one measured joint sample and one velocity command per 20 ms
upper-control tick, then returns one set of joint targets and gait diagnostics.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import select
import subprocess
import time
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

from hexapod_robot import JOINT_TYPES, LEG_ORDER, RobotModel
from hexapod_walk import _rpy_matrix, leg_fk_jacobian, quaternion_matrix_xyzw

UPSTREAM_LEG_ORDER = ("rf", "rm", "rb", "lf", "lm", "lb")
UPSTREAM_COMMIT = "48ffa8eb25c98a08d3e8b10f00df7fba2d952921"


def resolve_bridge_executable(project_root: Path, environ: Optional[Mapping[str, str]] = None) -> Path:
    """Return the explicitly built bridge or explain how to build it."""
    env = os.environ if environ is None else environ
    configured = env.get("HEXAPOD_OPENSHC_BRIDGE")
    binary = Path(configured).expanduser().resolve() if configured else (
        Path(project_root) / "data/raw/openshc_build/devel/lib/openshc_adapter/openshc_bridge")
    if not binary.is_file() or not os.access(str(binary), os.X_OK):
        raise FileNotFoundError(
            "OpenSHC bridge is not built or executable at {}. Run scripts/build_openshc.sh "
            "with the existing ROS Noetic/Eigen toolchain, or set HEXAPOD_OPENSHC_BRIDGE.".format(binary))
    return binary


def _joint(model: RobotModel, name: str):
    matches = [node for node in model.joints if node.name == name]
    if len(matches) != 1:
        raise ValueError("local URDF must contain exactly one joint {}".format(name))
    return matches[0]


def derive_leg_data(model: RobotModel, leg: str, q_by_name: Mapping[str, float],
                    measured_foot_body_m: Iterable[float]) -> Dict[str, object]:
    """Validate and express one local URDF chain as OpenSHC standard DH rows.

    The first fixed DH row encodes the hip mount.  A constant theta offset on
    the first actuated row cancels its polar bearing, preserving the URDF's
    named thigh angle without changing the body-to-hip transform.
    """
    if leg not in LEG_ORDER:
        raise ValueError("unknown canonical leg {}".format(leg))
    thigh = _joint(model, "j_{}_thigh".format(leg))
    knee = _joint(model, "j_{}_knee".format(leg))
    ankle = _joint(model, "j_{}_ankle".format(leg))
    toe = _joint(model, "j_{}_toe".format(leg))
    if (thigh.parent != model.root_link or thigh.child != "l_{}_thigh".format(leg) or
            knee.parent != thigh.child or knee.child != "l_{}_knee".format(leg) or
            ankle.parent != knee.child or ankle.child != "l_{}_ankle".format(leg) or
            toe.parent != ankle.child or toe.child != "l_{}_toe".format(leg)):
        raise ValueError("{} URDF links do not form the supported body-thigh-knee-ankle-toe chain".format(leg))
    if any(node.kind != kind for node, kind in ((thigh, "revolute"), (knee, "revolute"),
                                                 (ankle, "revolute"), (toe, "fixed"))):
        raise ValueError("{} URDF chain has unsupported joint kinds".format(leg))
    for node in (thigh, knee, ankle):
        axis = np.asarray(node.axis, dtype=float)
        if axis.shape != (3,) or not np.allclose(axis, (0.0, 0.0, 1.0), atol=1e-10, rtol=0.0):
            raise ValueError("{} OpenSHC mapping requires local +Z revolute axes".format(leg))
    if not np.allclose(thigh.rpy[:2], (0.0, 0.0), atol=1e-10, rtol=0.0):
        raise ValueError("{} hip mount roll/pitch cannot be represented by this standard-DH adapter".format(leg))
    if (not np.allclose(knee.rpy, (math.pi / 2.0, 0.0, 0.0), atol=1e-9, rtol=0.0) or
            not np.allclose(ankle.rpy, 0.0, atol=1e-10, rtol=0.0) or
            not np.allclose(toe.rpy, 0.0, atol=1e-10, rtol=0.0)):
        raise ValueError("{} joint origins are outside the verified local-URDF DH pattern".format(leg))
    if any(abs(float(node.xyz[1])) > 1e-10 or abs(float(node.xyz[2])) > 1e-10
           for node in (knee, ankle, toe)):
        raise ValueError("{} downstream link offsets must lie on their local DH x axes".format(leg))

    mount = np.asarray(thigh.xyz, dtype=float)
    bearing = math.atan2(float(mount[1]), float(mount[0]))
    mount_yaw = float(thigh.rpy[2])
    dh = (
        (math.hypot(float(mount[0]), float(mount[1])), bearing, float(mount[2]), 0.0),
        (float(knee.xyz[0]), mount_yaw - bearing, float(knee.xyz[2]), float(knee.rpy[0])),
        (float(ankle.xyz[0]), 0.0, float(ankle.xyz[2]), 0.0),
        (float(toe.xyz[0]), 0.0, float(toe.xyz[2]), 0.0),
    )
    joint_data = []
    for kind in JOINT_TYPES:
        name = "j_{}_{}".format(leg, kind)
        limit = model.limits[name]
        q = float(q_by_name[name])
        if not math.isfinite(q) or q < limit.lower_rad - 1e-6 or q > limit.upper_rad + 1e-6:
            raise ValueError("initial measured {} violates its URDF position limits".format(name))
        joint_data.append((q, limit.lower_rad, limit.upper_rad, limit.effort_nm,
                           limit.velocity_rad_s))
    foot = np.asarray(tuple(measured_foot_body_m), dtype=float)
    if foot.shape != (3,) or not np.all(np.isfinite(foot)):
        raise ValueError("initial measured toe point must be a finite three-vector")
    return {"leg": leg, "mount_xyz_m": mount.tolist(), "mount_rpy_rad": thigh.rpy.tolist(),
            "q_rad": [row[0] for row in joint_data], "measured_foot_body_m": foot.tolist(),
            "plane_to_body_toe_z_m": float(foot[2]), "limits": [list(row[1:]) for row in joint_data],
            "dh_rows": [list(row) for row in dh]}


def derive_initial_legs(model: RobotModel, q_by_name: Mapping[str, float],
                        measured_feet_world_m: Mapping[str, Iterable[float]],
                        body_xyz_m: Iterable[float], body_quaternion_xyzw: Iterable[float]):
    """Build the fixed upstream leg order from measured toe poses and URDF data."""
    if set(measured_feet_world_m) != set(LEG_ORDER):
        raise ValueError("measured toe map must contain all six canonical legs")
    origin = np.asarray(tuple(body_xyz_m), dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("measured body position must be a finite three-vector")
    rotation = quaternion_matrix_xyzw(np.asarray(tuple(body_quaternion_xyzw), dtype=float))
    feet_body = {leg: rotation.T @ (np.asarray(measured_feet_world_m[leg], dtype=float) - origin)
                 for leg in LEG_ORDER}
    return {leg: derive_leg_data(model, leg, q_by_name, feet_body[leg])
            for leg in UPSTREAM_LEG_ORDER}


def _finite_vector(values: Iterable[float], length: int, label: str):
    vector = np.asarray(tuple(values), dtype=float)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError("{} must contain {} finite numbers".format(label, length))
    return vector


class OpenShcBridge:
    """Synchronous one-command/one-response wrapper around the C++ core."""

    def __init__(self, executable: Path, model: RobotModel, q_by_name: Mapping[str, float],
                 measured_feet_world_m: Mapping[str, Iterable[float]], body_xyz_m: Iterable[float],
                 body_quaternion_xyzw: Iterable[float], settings: Mapping[str, float],
                 stderr_path: Path):
        self.executable = Path(executable).resolve()
        self.model = model
        self.settings = dict(settings)
        self.legs = derive_initial_legs(model, q_by_name, measured_feet_world_m,
                                        body_xyz_m, body_quaternion_xyzw)
        self.process = None
        self.stderr_file = Path(stderr_path).open("w")
        self.last_phase = None
        self.completed_cycles = 0
        self.initialization_timeout_s = 60.0
        self.step_timeout_s = 2.0
        self._output_buffer = b""
        self._initialization_row = self._encode_initialization()
        try:
            self.process = subprocess.Popen([str(self.executable)], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=self.stderr_file,
                                            text=True, encoding="utf-8", bufsize=1)
            assert self.process.stdin is not None and self.process.stdout is not None
            self.process.stdin.write(self._initialization_row + "\n")
            self.process.stdin.flush()
            self.initialization = self._read_initialization()
        except Exception:
            self.close()
            raise

    def _encode_initialization(self):
        duty = float(self.settings["duty_factor"])
        if not math.isclose(duty, 0.5, abs_tol=1e-12):
            raise ValueError("the OpenSHC prototype profile requires a 0.5 duty factor")
        full_frequency = float(self.settings["frequency_hz"])
        native_frequency = full_frequency / duty
        global_values = (0.02, native_frequency, 2, 2, 2, float(self.settings["swing_height_m"]),
                         float(self.settings["speed_m_s"]), float(self.settings.get("max_rotation_rad_s", 0.5)),
                         float(self.settings.get("max_acceleration_m_s2", 0.5)))
        fields = ["INIT"] + ["{:.17g}".format(value) for value in global_values]
        for leg in UPSTREAM_LEG_ORDER:
            data = self.legs[leg]
            fields.append(leg)
            fields.extend("{:.17g}".format(value) for value in data["mount_xyz_m"])
            fields.extend("{:.17g}".format(value) for value in data["mount_rpy_rad"])
            fields.extend("{:.17g}".format(value) for value in data["q_rad"])
            fields.extend("{:.17g}".format(value) for value in data["measured_foot_body_m"])
            fields.append("{:.17g}".format(data["plane_to_body_toe_z_m"]))
            for row in data["limits"]:
                fields.extend("{:.17g}".format(value) for value in row)
            for row in data["dh_rows"]:
                fields.extend("{:.17g}".format(value) for value in row)
        return " ".join(fields)

    def _read_line(self, timeout_s=None):
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("OpenSHC bridge is not running")
        timeout = self.step_timeout_s if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + timeout
        while True:
            if b"\n" not in self._output_buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise TimeoutError("OpenSHC bridge response timed out after {:.3g} s".format(timeout))
                chunk = os.read(self.process.stdout.fileno(), 4096)
                if not chunk:
                    code = self.process.poll()
                    raise RuntimeError("OpenSHC bridge closed its response stream (exit={})".format(code))
                self._output_buffer += chunk
                continue
            line, self._output_buffer = self._output_buffer.split(b"\n", 1)
            fields = line.decode("utf-8", errors="replace").strip().split()
            if not fields:
                continue
            if fields[0] in {"[DEBUG]", "[INFO]", "[WARN]", "[ERROR]", "[FATAL]", "[SHC]"}:
                continue
            return fields

    def _read_initialization(self):
        header = self._read_line(self.initialization_timeout_s)
        if len(header) != 5 or header[0] != "READY":
            raise RuntimeError("OpenSHC bridge initialization failed: {}".format(" ".join(header)))
        result = {"effective_period_control_ticks": int(header[1]),
                  "effective_frequency_hz": float(header[2]), "max_fk_error_m": float(header[3]),
                  "forward_speed_limit_m_s": float(header[4]),
                  "fk_error_by_leg_m": {}}
        for _ in UPSTREAM_LEG_ORDER:
            row = self._read_line(self.initialization_timeout_s)
            if len(row) != 3 or row[0] != "FK":
                raise RuntimeError("OpenSHC bridge FK initialization failed: {}".format(" ".join(row)))
            result["fk_error_by_leg_m"][row[1]] = float(row[2])
        if self._read_line(self.initialization_timeout_s) != ["END"]:
            raise RuntimeError("OpenSHC bridge initialization response was malformed")
        if result["max_fk_error_m"] > 2e-5:
            raise RuntimeError("OpenSHC URDF FK mismatch {:.6g} m exceeds tolerance".format(result["max_fk_error_m"]))
        return result

    def step(self, velocity_xy_m_s: Iterable[float], angular_velocity_rad_s: float,
             q_by_name: Mapping[str, float]):
        velocity = _finite_vector(velocity_xy_m_s, 2, "requested planar velocity")
        q = []
        for leg in UPSTREAM_LEG_ORDER:
            for joint in JOINT_TYPES:
                value = float(q_by_name["j_{}_{}".format(leg, joint)])
                if not math.isfinite(value):
                    raise ValueError("measured joint states must be finite")
                q.append(value)
        omega = float(angular_velocity_rad_s)
        if not math.isfinite(omega):
            raise ValueError("requested angular velocity must be finite")
        assert self.process is not None and self.process.stdin is not None
        fields = ["STEP", "{:.17g}".format(velocity[0]), "{:.17g}".format(velocity[1]),
                  "{:.17g}".format(omega)] + ["{:.17g}".format(value) for value in q]
        self.process.stdin.write(" ".join(fields) + "\n")
        self.process.stdin.flush()
        header = self._read_line()
        if len(header) != 16 or header[0] != "OK":
            raise RuntimeError("OpenSHC step failed: {}".format(" ".join(header)))
        values = list(map(float, header[1:]))
        result = {"tick": int(values[0]), "requested_velocity_xy_m_s": values[1:3],
                  "requested_angular_velocity_rad_s": values[3],
                  "accepted_velocity_xy_m_s": values[4:6], "accepted_angular_velocity_rad_s": values[6],
                  "planned_body_pose_odom": {"xyz_m": values[7:10], "quaternion_xyzw": values[10:14]},
                  "effective_period_control_ticks": int(values[14]), "legs": {}}
        for _ in UPSTREAM_LEG_ORDER:
            row = self._read_line()
            if len(row) != 25 or row[0] != "LEG":
                raise RuntimeError("OpenSHC leg response was malformed: {}".format(" ".join(row)))
            leg = row[1]
            item = list(map(float, row[2:]))
            phase = int(item[0])
            if leg == "rf":
                if self.last_phase is not None and phase < self.last_phase:
                    self.completed_cycles += 1
                self.last_phase = phase
            result["legs"][leg] = {"phase": phase, "state": int(item[1]), "phase_progress": item[2],
                                   "stance_progress": item[3], "swing_progress": item[4],
                                   "foot_target_odom_m": item[5:8], "q_target_rad": item[8:11],
                                   "dq_target_rad_s": item[11:14], "joint_velocity_limit_active": [
                                       bool(value) for value in item[14:17]],
                                   "joint_position_limit_active": [bool(value) for value in item[17:20]],
                                   "planned_support": bool(item[20]), "ik_residual_m": item[21],
                                   "ik_quality": item[22]}
        result["q_target_by_name_rad"] = {
            "j_{}_{}".format(leg, joint): float(result["legs"][leg]["q_target_rad"][index])
            for leg in UPSTREAM_LEG_ORDER for index, joint in enumerate(JOINT_TYPES)}
        result["dq_target_by_name_rad_s"] = {
            "j_{}_{}".format(leg, joint): float(result["legs"][leg]["dq_target_rad_s"][index])
            for leg in UPSTREAM_LEG_ORDER for index, joint in enumerate(JOINT_TYPES)}
        if self._read_line() != ["END"]:
            raise RuntimeError("OpenSHC step response terminator was missing")
        result["completed_cycles"] = self.completed_cycles
        return result

    def close(self):
        process = self.process
        self.process = None
        if process is not None:
            try:
                if process.stdin is not None and process.poll() is None:
                    process.stdin.write("QUIT\n")
                    process.stdin.flush()
                    process.stdin.close()
                process.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait()
        if not self.stderr_file.closed:
            self.stderr_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def openshc_source_provenance(project_root: Path, executable: Path):
    """Return pinned vendored-source and executable hashes for resolved logs."""
    root = Path(project_root)
    manifest_path = root / "third_party/openshc/UPSTREAM.json"
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["sha256"].items():
        actual = hashlib.sha256((root / "third_party/openshc" / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("vendored OpenSHC source hash mismatch: {}".format(relative))
    return {"origin": manifest["origin"], "commit": manifest["commit"],
            "license": manifest["license"], "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "files_sha256": manifest["sha256"],
            "adapter_sha256": hashlib.sha256((root / "src/hexapod_openshc.py").read_bytes()).hexdigest(),
            "bridge_source_sha256": hashlib.sha256((root / "integration/openshc/bridge.cpp").read_bytes()).hexdigest(),
            "build_script_sha256": hashlib.sha256((root / "scripts/build_openshc.sh").read_bytes()).hexdigest(),
            "bridge_sha256": hashlib.sha256(Path(executable).read_bytes()).hexdigest()}
