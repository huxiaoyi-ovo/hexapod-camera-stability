"""Offline URDF metadata and deterministic stand-reference helpers."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Mapping, Tuple, Union

import numpy as np

LEG_ORDER = ("lb", "lf", "lm", "rb", "rf", "rm")
JOINT_TYPES = ("thigh", "knee", "ankle")
JOINT_NAMES = tuple("j_{}_{}".format(leg, joint) for leg in LEG_ORDER for joint in JOINT_TYPES)
TOE_LINK_NAMES = tuple("l_{}_toe".format(leg) for leg in LEG_ORDER)
@dataclass(frozen=True)
class JointLimit:
    name: str
    lower_rad: float
    upper_rad: float
    effort_nm: float
    velocity_rad_s: float


@dataclass(frozen=True)
class JointNode:
    name: str
    parent: str
    child: str
    kind: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class RobotModel:
    path: Path
    root_link: str
    limits: Mapping[str, JointLimit]
    joints: Tuple[JointNode, ...]
    toe_link_names: Tuple[str, ...]
    toe_radius_m: float


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


def _origin(node: ET.Element) -> Tuple[np.ndarray, np.ndarray]:
    origin = node.find("origin")
    if origin is None:
        return np.zeros(3), np.zeros(3)
    return (np.fromstring(origin.get("xyz", "0 0 0"), sep=" ", dtype=float),
            np.fromstring(origin.get("rpy", "0 0 0"), sep=" ", dtype=float))


def load_robot_model(path: Union[str, Path]) -> RobotModel:
    """Read named actuated-joint limits and the URDF link tree without Isaac Gym."""
    path = Path(path).resolve()
    root = ET.parse(str(path)).getroot()
    link_names = [link.get("name") for link in root.findall("link")]
    child_links = {j.find("child").get("link") for j in root.findall("joint")}
    roots = [name for name in link_names if name not in child_links]
    if len(roots) != 1:
        raise ValueError("URDF must have one root link, found {}".format(roots))
    limits: Dict[str, JointLimit] = {}
    joints: List[JointNode] = []
    toe_radii = []
    for joint in root.findall("joint"):
        parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
        xyz, rpy = _origin(joint)
        axis_node = joint.find("axis")
        axis = (np.fromstring(axis_node.get("xyz", "1 0 0"), sep=" ", dtype=float)
                if axis_node is not None else np.array([1.0, 0.0, 0.0]))
        kind = joint.get("type", "fixed")
        joints.append(JointNode(joint.get("name"), parent, child, kind, xyz, rpy, axis))
        if kind in ("revolute", "continuous", "prismatic"):
            lim = joint.find("limit")
            if lim is None:
                raise ValueError("actuated joint {} has no limit".format(joint.get("name")))
            limits[joint.get("name")] = JointLimit(
                joint.get("name"), float(lim.get("lower", "-inf")),
                float(lim.get("upper", "inf")), float(lim.get("effort", "inf")),
                float(lim.get("velocity", "inf")))
    for link in root.findall("link"):
        if link.get("name", "").endswith("_toe"):
            for sphere in link.findall("collision/geometry/sphere"):
                toe_radii.append(float(sphere.get("radius")))
    if len(toe_radii) != len(TOE_LINK_NAMES) or not all(np.isclose(r, toe_radii[0]) for r in toe_radii):
        raise ValueError("expected six toes with a shared spherical collision radius")
    if set(limits) != set(JOINT_NAMES):
        raise ValueError("URDF actuated joints differ from canonical set: {}".format(sorted(limits)))
    return RobotModel(path, roots[0], limits, tuple(joints), TOE_LINK_NAMES, toe_radii[0])


def stand_target_by_name(model: RobotModel, reference: Mapping[str, object]) -> Dict[str, float]:
    """Build the configured stand reference and check it against URDF limits."""
    if set(reference) != {"thigh_by_leg", "knee", "ankle"}:
        raise ValueError("stand reference must define thigh_by_leg, knee, and ankle")
    thigh = reference["thigh_by_leg"]
    if not isinstance(thigh, Mapping) or set(thigh) != set(LEG_ORDER):
        raise ValueError("stand reference must define one thigh angle per canonical leg")
    try:
        target: Dict[str, float] = {}
        for leg in LEG_ORDER:
            target["j_{}_thigh".format(leg)] = float(thigh[leg])
            target["j_{}_knee".format(leg)] = float(reference["knee"])
            target["j_{}_ankle".format(leg)] = float(reference["ankle"])
    except (TypeError, ValueError) as exc:
        raise ValueError("stand reference values must be numeric") from exc
    if not all(np.isfinite(value) for value in target.values()):
        raise ValueError("stand reference values must be finite")
    for name, value in target.items():
        lim = model.limits[name]
        if value < lim.lower_rad - 1e-4 or value > lim.upper_rad + 1e-4:
            raise ValueError("stand reference {}={} violates [{}, {}]".format(
                name, value, lim.lower_rad, lim.upper_rad))
    return target


def toe_positions_base(model: RobotModel, q_by_name: Mapping[str, float]) -> Dict[str, np.ndarray]:
    """Compute toe centers in root coordinates using the URDF joint tree."""
    transforms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {model.root_link: (np.eye(3), np.zeros(3))}
    pending = list(model.joints)
    while pending:
        progress = False
        remaining = []
        for joint in pending:
            if joint.parent not in transforms:
                remaining.append(joint)
                continue
            parent_r, parent_p = transforms[joint.parent]
            local_r = _rpy_matrix(joint.rpy)
            local_p = joint.xyz
            if joint.kind in ("revolute", "continuous"):
                if joint.name not in q_by_name:
                    raise ValueError("missing actuated joint value {}".format(joint.name))
                local_r = local_r @ _axis_rotation(joint.axis, float(q_by_name[joint.name]))
            elif joint.kind == "prismatic":
                if joint.name not in q_by_name:
                    raise ValueError("missing actuated joint value {}".format(joint.name))
                local_p = local_p + joint.axis * float(q_by_name[joint.name])
            transforms[joint.child] = (parent_r @ local_r, parent_p + parent_r @ local_p)
            progress = True
        if not progress and remaining:
            raise ValueError("URDF link tree is disconnected or cyclic")
        pending = remaining
    try:
        return {name: transforms[name][1].copy() for name in model.toe_link_names}
    except KeyError as exc:
        raise ValueError("missing toe link in URDF: {}".format(exc)) from exc


def lowest_stand_height(model: RobotModel, q_by_name: Mapping[str, float], clearance_m: float) -> float:
    """Root height that places the lowest spherical toe above the plane."""
    feet = toe_positions_base(model, q_by_name)
    return float(-min(p[2] for p in feet.values()) + model.toe_radius_m + clearance_m)


def map_named_values(values_by_name: Mapping[str, float], simulator_names: List[str]) -> np.ndarray:
    """Map values to the simulator's reported DOF order, rejecting missing/extra names."""
    if len(simulator_names) != len(set(simulator_names)) or set(values_by_name) != set(simulator_names):
        raise ValueError("simulator DOF names do not match supplied named values")
    return np.asarray([values_by_name[name] for name in simulator_names], dtype=np.float32)


def pd_torque(q: np.ndarray, dq: np.ndarray, q_ref: np.ndarray, kp: float, kd: float,
              effort_limit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute transparent effort PD and report pre-clipped torque saturation."""
    q, dq, q_ref, effort_limit = map(lambda a: np.asarray(a, dtype=float), (q, dq, q_ref, effort_limit))
    if not (q.shape == dq.shape == q_ref.shape == effort_limit.shape):
        raise ValueError("PD vectors must have identical shapes")
    if not all(np.all(np.isfinite(a)) for a in (q, dq, q_ref, effort_limit)):
        raise ValueError("PD vectors must be finite")
    if not np.isfinite(kp) or not np.isfinite(kd) or kp < 0 or kd < 0 or np.any(effort_limit <= 0):
        raise ValueError("PD gains must be nonnegative and effort limits positive")
    raw = kp * (q_ref - q) - kd * dq
    saturated = np.abs(raw) >= effort_limit - 1e-7
    return np.clip(raw, -effort_limit, effort_limit).astype(np.float32), saturated
