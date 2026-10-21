"""PCR-Net / HexGround simulation observations -> auditable trajectory CSV.

Host integration: call `tap.append(env, env_id, t, ..., nominal_cmd_body)`
AFTER each `env.step(actions)`. This module never runs a robot, never changes
its actions or motor model, and does not depend on Isaac Gym for unit tests.

PCR coordinate contract: body +x right, +y forward, +z up; quaternions xyzw.
`nominal_cmd_body` is [vx,vy,wz] in body axes, not world axes.
All future states/foot forces are logged as OUTCOMES only, not online features.
"""
from __future__ import annotations
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import numpy as np


def _array(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def read_hexground_state(env, env_id: int = 0) -> dict:
    """Read actual Isaac Gym state tensors, with no silent zero fallback."""
    required = ('root_states', 'dof_pos', 'dof_vel', 'torques', 'torque_limits',
                'dof_pos_limits', 'dof_vel_limits', 'contact_forces',
                'feet_indices', 'rb_states', 'gym', 'sim')
    absent = [k for k in required if not hasattr(env, k)]
    if absent:
        raise AttributeError(f'HexGround state unavailable: {absent}')
    root = _array(env.root_states[env_id])
    q = _array(env.dof_pos[env_id]).reshape(-1)
    dq = _array(env.dof_vel[env_id]).reshape(-1)
    torque = _array(env.torques[env_id]).reshape(-1)
    torque_limits = _array(env.torque_limits).reshape(-1)
    position_limits = _array(env.dof_pos_limits)
    velocity_limits = _array(env.dof_vel_limits).reshape(-1)
    indices = _array(env.feet_indices).astype(int)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    rigid = _array(env.rb_states[env_id])
    contacts = _array(env.contact_forces[env_id])
    if (root.shape != (13,) or q.shape != (18,) or dq.shape != (18,) or torque.shape != (18,)
            or torque_limits.shape != (18,) or position_limits.shape != (18,2)
            or velocity_limits.shape != (18,)):
        raise ValueError('Expected root (13), joint states (18), and torque (18)')
    if indices.shape != (6,) or contacts.ndim != 2 or contacts.shape[1] != 3:
        raise ValueError('Expected six foot indices and 3D contact-force array')
    if rigid.ndim != 2 or rigid.shape[1] != 13 or np.any(indices < 0) or np.any(indices >= len(contacts)) or np.any(indices >= len(rigid)):
        raise ValueError('Foot rigid-body indices outside state/contact tensor')
    toes = rigid[indices]
    values = np.r_[root, q, dq, torque, torque_limits, position_limits.ravel(),
                   velocity_limits, contacts[indices].ravel(), toes.ravel()]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite simulator state')
    if abs(np.linalg.norm(root[3:7]) - 1.0) > .02:
        raise ValueError('Root quaternion unexpectedly unnormalized')
    margin = np.minimum(q-position_limits[:,0], position_limits[:,1]-q)
    dq_ratio = np.abs(dq)/velocity_limits
    saturation = np.abs(torque) >= torque_limits-1e-4
    return dict(root=root, q=q, dq=dq, torque=torque, foot_forces=contacts[indices],
                toe_positions=toes[:,:3], toe_velocities=toes[:,7:10],
                min_position_margin=float(np.min(margin)),
                max_dq_ratio=float(np.max(dq_ratio)),
                torque_saturation_fraction=float(np.mean(saturation)))


def canonical_toe_permutation(env, env_id: int = 0):
    """Map Isaac Gym's actual feet_indices order to lb,lf,lm,rb,rf,rm.

    Actor rigid-body ordering is NOT guaranteed to match gait-generator order.
    Resolve via the same Gym handle API as PCR's LeggedRobot._create_envs.
    """
    if not all(hasattr(env, x) for x in ('gym', 'envs', 'actor_handles', 'feet_indices')):
        raise AttributeError('Cannot verify PCR toe rigid-body identity')
    from pcr_urdf_ik import LEG_ORDER
    handle_to_slot = {int(x): i for i, x in enumerate(_array(env.feet_indices).astype(int))}
    if len(handle_to_slot) != 6:
        raise ValueError('Foot handles are not six distinct rigid bodies')
    perm = []
    for leg in LEG_ORDER:
        handle = int(env.gym.find_actor_rigid_body_handle(
            env.envs[env_id], env.actor_handles[env_id], f'l_{leg}_toe'))
        if handle not in handle_to_slot:
            raise ValueError(f'PCR toe handle missing or inconsistent: {leg}')
        perm.append(handle_to_slot[handle])
    if len(set(perm)) != 6:
        raise ValueError('Foot permutation contains duplicates')
    return np.array(perm, dtype=int)


@dataclass(frozen=True)
class TrialIdentity:
    trial_id: str
    pair_id: str
    terrain_instance_id: str
    seed: int
    controller_on: bool
    gait_id: str

    def validate(self):
        if not (self.trial_id and self.pair_id and self.terrain_instance_id and self.gait_id):
            raise ValueError('Trial identity must include trial, pair, terrain and gait')
        if self.seed < 0:
            raise ValueError('Seed must be nonnegative')


COLUMNS = (
    'trial_id', 'pair_id', 'terrain_instance_id', 'seed', 'controller_on', 'gait_id',
    'time_s', 'cmd_vx_body_mps', 'cmd_vy_body_mps', 'cmd_wz_body_rad_s',
    *('root_'+i for i in ('px','py','pz','qx','qy','qz','qw','vx','vy','vz','wx','wy','wz')),
    'q_rad_json', 'dq_rad_s_json', 'torque_nm_json', 'foot_forces_n_json',
    'toe_world_positions_m_json', 'toe_world_velocities_mps_json',
    'min_position_margin_rad', 'max_dq_limit_ratio', 'torque_saturation_fraction',
    'phase', 'reference_limited', 'planned_support_json', 'measured_foot_contact_json',
)


class PcrSimTap:
    """One tap per rollout. `controller_on` describes controller executed by host."""
    def __init__(self, output_csv: str | Path, identity: TrialIdentity,
                 measured_contact_threshold_n: float = 1.0,
                 canonical_feet_permutation=None):
        identity.validate()
        self.identity = identity
        self.path = Path(output_csv)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f'Refusing to overwrite existing rollout: {self.path}')
        if measured_contact_threshold_n < 0:
            raise ValueError('Bad contact threshold')
        self.threshold = float(measured_contact_threshold_n)
        if canonical_feet_permutation is None:
            self.feet_perm = np.arange(6, dtype=int)  # legacy caller, unchecked
        else:
            perm = np.asarray(canonical_feet_permutation, dtype=int)
            if perm.shape != (6,) or set(perm.tolist()) != set(range(6)):
                raise ValueError('Invalid canonical foot-contact permutation')
            self.feet_perm = perm
        self.handle = self.path.open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.handle, fieldnames=COLUMNS)
        self.writer.writeheader()
        self.last_time = None
        self.n = 0

    def append(self, env, env_id: int, t_s: float, nominal_cmd_body,
               planned_support=None, phase='walking', reference_limited=False):
        t = float(t_s)
        cmd = np.asarray(nominal_cmd_body, dtype=float)
        if cmd.shape != (3,) or not np.isfinite(cmd).all() or not np.isfinite(t):
            raise ValueError('Command must be finite [vx,vy,wz], time finite')
        if self.last_time is not None and t <= self.last_time:
            raise ValueError('Time must increase within a rollout')
        v = read_hexground_state(env, env_id)
        if planned_support is None:
            raise ValueError('Planned support from gait generator required (6, bool)')
        support = np.asarray(planned_support)
        if support.shape != (6,):
            raise ValueError('Planned support mask must have shape (6,)')
        foot_forces = v['foot_forces'][self.feet_perm]
        toe_positions = v['toe_positions'][self.feet_perm]
        toe_velocities = v['toe_velocities'][self.feet_perm]
        measured_contact = np.linalg.norm(foot_forces, axis=1) > self.threshold
        i = self.identity
        row = dict(trial_id=i.trial_id, pair_id=i.pair_id,
                   terrain_instance_id=i.terrain_instance_id, seed=i.seed,
                   controller_on=int(i.controller_on), gait_id=i.gait_id,
                   time_s=t, cmd_vx_body_mps=cmd[0], cmd_vy_body_mps=cmd[1],
                   cmd_wz_body_rad_s=cmd[2],
                   q_rad_json=json.dumps(v['q'].tolist()),
                   dq_rad_s_json=json.dumps(v['dq'].tolist()),
                   torque_nm_json=json.dumps(v['torque'].tolist()),
                   toe_world_positions_m_json=json.dumps(toe_positions.tolist()),
                   toe_world_velocities_mps_json=json.dumps(toe_velocities.tolist()),
                   min_position_margin_rad=v['min_position_margin'],
                   max_dq_limit_ratio=v['max_dq_ratio'],
                   torque_saturation_fraction=v['torque_saturation_fraction'],
                   phase=str(phase), reference_limited=int(bool(reference_limited)),
                   foot_forces_n_json=json.dumps(foot_forces.tolist()),
                   planned_support_json=json.dumps(support.astype(bool).tolist()),
                   measured_foot_contact_json=json.dumps(measured_contact.tolist()))
        row.update(dict(zip([x for x in COLUMNS if x.startswith('root_')],v['root'].tolist())))
        self.writer.writerow(row)
        self.last_time = t
        self.n += 1

    def close(self):
        if not self.handle.closed:
            self.handle.close()
        if self.n < 3:
            raise ValueError(f'Only {self.n} samples; need >=3 after simulator steps')
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.handle.close()
