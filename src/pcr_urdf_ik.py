"""Closed-form toe-position IK tied to PCR-Net hex_ground.urdf, not guessed link lengths.

The currently supported chain per leg is: body --yaw thigh--> (l1,0,0)
--Rx(pi/2),pitch knee--> (l2,0,0) --pitch ankle--> (l3,0,0) toe.
Mount yaw is 0 for right legs and pi for left legs. Coordinate frame is
body +x right, +y forward, +z up. All joints are revolute about local +z.

Fail closed if a URDF revision changes this chain. This module does not claim
force/contact feasibility; PhysX and additional stance guards are still needed.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np

LEG_ORDER = ('lb', 'lf', 'lm', 'rb', 'rf', 'rm')
JOINTS = ('thigh', 'knee', 'ankle')


def _attr3(element, name, default):
    if element is None:
        return np.array(default, dtype=float)
    return np.array([float(x) for x in element.attrib.get(name, ' '.join(map(str, default))).split()], dtype=float)


def _near(a, b, atol=1e-5):
    return np.allclose(a, b, atol=atol, rtol=0)


@dataclass(frozen=True)
class HexUrdfIK:
    hip_xyz: np.ndarray  # (6,3), base frame
    mount_yaw: np.ndarray  # (6,), radians
    l1: float
    l2: float
    l3: float  # URDF ankle->toe, not config nominal length
    joint_lower: np.ndarray  # (6,3)
    joint_upper: np.ndarray  # (6,3)

    @classmethod
    def from_urdf(cls, urdf_path: str | Path) -> 'HexUrdfIK':
        root = ET.parse(urdf_path).getroot()
        by_name = {j.attrib['name']: j for j in root.findall('joint')}
        hip, mount, low, high, lengths = [], [], [], [], []
        for leg in LEG_ORDER:
            chain = []
            for joint in JOINTS + ('toe',):
                key = f'j_{leg}_{joint}'
                if key not in by_name:
                    raise ValueError(f'PCR URDF joint missing: {key}')
                chain.append(by_name[key])
            origins = [j.find('origin') for j in chain]
            positions = [_attr3(node, 'xyz', [0, 0, 0]) for node in origins]
            rotations = [_attr3(node, 'rpy', [0, 0, 0]) for node in origins]
            # Constrain the assumed geometry; no silent use on unrelated robot.
            if not (_near(positions[1], [positions[1][0], 0, 0]) and
                    _near(positions[2], [positions[2][0], 0, 0]) and
                    _near(positions[3], [positions[3][0], 0, 0]) and
                    _near(rotations[0][:2], [0, 0]) and
                    _near(rotations[1], [math.pi / 2, 0, 0]) and
                    _near(rotations[2], [0, 0, 0]) and _near(rotations[3], [0, 0, 0])):
                raise ValueError(f'Unsupported URDF kinematic geometry: {leg}')
            if not all(j.attrib.get('type') == 'revolute' for j in chain[:3]) or chain[3].attrib.get('type') != 'fixed':
                raise ValueError(f'Unsupported PCR joint types: {leg}')
            if not all(_near(_attr3(j.find('axis'), 'xyz', [1, 0, 0]), [0, 0, 1]) for j in chain[:3]):
                raise ValueError(f'Unsupported PCR joint axes: {leg}')
            if not (0 <= rotations[0][2] <= math.pi + 1e-5):
                raise ValueError('Unsupported hip mount rotation')
            hip.append(positions[0]); mount.append(rotations[0][2])
            lengths.append([positions[1][0], positions[2][0], positions[3][0]])
            low.append([float(j.find('limit').attrib['lower']) for j in chain[:3]])
            high.append([float(j.find('limit').attrib['upper']) for j in chain[:3]])
        if not np.allclose(lengths, lengths[0], atol=1e-8):
            raise ValueError('Nonuniform leg geometry requires more general IK')
        if min(lengths[0]) <= 0:
            raise ValueError('Nonpositive leg link length')
        return cls(np.asarray(hip), np.asarray(mount), *lengths[0], np.asarray(low), np.asarray(high))

    @property
    def canonical_dof_names(self):
        return tuple(f'j_{leg}_{j}' for leg in LEG_ORDER for j in JOINTS)

    def to_canonical(self, q_sim, dof_names):
        names = tuple(dof_names)
        if len(names) != 18 or len(set(names)) != 18 or set(names) != set(self.canonical_dof_names):
            raise ValueError('Isaac Gym DOF names mismatch exact PCR URDF 18 joint contract')
        q = np.asarray(q_sim, dtype=float)
        if q.shape[-1] != 18:
            raise ValueError('Need exactly 18 simulator DOFs')
        indices = [names.index(n) for n in self.canonical_dof_names]
        return q[..., indices].reshape(*q.shape[:-1], 6, 3)

    def to_sim_order(self, q_canonical, dof_names):
        names = tuple(dof_names)
        if len(names) != 18 or len(set(names)) != 18 or set(names) != set(self.canonical_dof_names):
            raise ValueError('Isaac Gym DOF names mismatch exact PCR URDF 18 joint contract')
        q = np.asarray(q_canonical, dtype=float)
        if q.shape[-2:] != (6, 3):
            raise ValueError('Need (6,3) joints')
        flat = q.reshape(*q.shape[:-2], 18)
        return flat[..., [self.canonical_dof_names.index(n) for n in names]]

    def fk(self, q):
        """Foot/toe position in body coordinates for (6,3) joint angles."""
        q = np.asarray(q, dtype=float)
        if q.shape != (6, 3):
            raise ValueError('Expected six 3-DOF leg joint triplets')
        a = self.mount_yaw + q[:, 0]
        radial = self.l1 + self.l2 * np.cos(q[:, 1]) + self.l3 * np.cos(q[:, 1] + q[:, 2])
        z = self.l2 * np.sin(q[:, 1]) + self.l3 * np.sin(q[:, 1] + q[:, 2])
        return self.hip_xyz + np.stack([radial * np.cos(a), radial * np.sin(a), z], axis=1)

    def ik(self, feet_body, *, q_reference=None, margin_rad=0.005):
        """Return canonical (6,3) IK, or raise ValueError on unreachable targets.

        Two planar elbow branches exist. Choose the valid branch closest to
        the existing joint state; for default PCR pose the ankle is negative.
        """
        foot = np.asarray(feet_body, dtype=float)
        if foot.shape != (6, 3) or not np.isfinite(foot).all():
            raise ValueError('Expected finite body-frame foot array (6,3)')
        loc = foot - self.hip_xyz
        c, s = np.cos(self.mount_yaw), np.sin(self.mount_yaw)
        xx = c * loc[:, 0] + s * loc[:, 1]
        yy = -s * loc[:, 0] + c * loc[:, 1]
        q_yaw = np.arctan2(yy, xx)
        radius = np.hypot(xx, yy) - self.l1
        z = loc[:, 2]
        cos_a = (radius**2 + z**2 - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3)
        if np.any(cos_a > 1 + 1e-8) or np.any(cos_a < -1 - 1e-8):
            raise ValueError('IK foot target outside radial reach')
        ankle_mag = np.arccos(np.clip(cos_a, -1, 1))
        choices = []
        for sign in (-1, 1):
            ankle = sign * ankle_mag
            knee = np.arctan2(z, radius) - np.arctan2(self.l3 * np.sin(ankle), self.l2 + self.l3 * np.cos(ankle))
            choices.append(np.column_stack([q_yaw, knee, ankle]))
        if q_reference is None:
            reference = np.tile([0.0, 0.67, -2.2], (6, 1))
        else:
            reference = np.asarray(q_reference, dtype=float)
            if reference.shape != (6, 3):
                raise ValueError('Expected (6,3) reference angles')
        valid = [np.all((q >= self.joint_lower + margin_rad) & (q <= self.joint_upper - margin_rad), axis=1) for q in choices]
        distances = [np.sum((q - reference)**2, axis=1) + (~v).astype(float) * 1e10 for q, v in zip(choices, valid)]
        which = np.argmin(np.stack(distances, axis=1), axis=1)
        q = np.stack(choices, axis=1)[np.arange(6), which]
        if not np.all(np.stack(valid, axis=1)[np.arange(6), which]):
            raise ValueError('IK target would violate joint position limits')
        if np.max(np.abs(self.fk(q) - foot)) > 1e-6:
            raise RuntimeError('IK/FK roundtrip consistency check failed')
        return q