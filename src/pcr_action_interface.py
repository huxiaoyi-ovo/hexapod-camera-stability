"""Explicit PCR HexGround action convention: q_target = default_q + action_scale * action.

This checks precomputed IK joint targets and converts them to 18D actions.
It does NOT solve IK, choose swing/stance, or control the physical robot.
No silent saturation: infeasible candidates must be discarded, not clipped.
PCR HexGround._compute_torques applies the actuator model and torque clipping.
"""
from __future__ import annotations
import numpy as np
from pcr_sim_bridge import _array


def targets_to_hexground_actions(env, absolute_q_target, safety_margin_rad=0.01,
                                 check_velocity=True, previous_reference=None):
    target=_array(absolute_q_target)
    if target.ndim==1:
        target=target[None,:]
    if target.ndim!=2 or target.shape[1]!=18 or not np.isfinite(target).all():
        raise ValueError('Absolute joint targets must be finite (N,18)')
    n=len(target)
    default=_array(env.default_dof_pos)
    if default.ndim==1:
        default=default[None,:]
    if default.shape not in ((1,18),(n,18)):
        raise ValueError('PCR default_dof_pos shape unexpected')
    limits=_array(env.dof_pos_limits)
    if limits.shape!=(18,2):
        raise ValueError('Expected 18 joint position limits')
    if safety_margin_rad<0:
        raise ValueError('Bad joint safety margin')
    if np.any(target < limits[:,0]+safety_margin_rad) or np.any(target > limits[:,1]-safety_margin_rad):
        raise ValueError('Proposed IK target violates joint position range')
    if check_velocity:
        reference = env.dof_pos if previous_reference is None else previous_reference
        now=_array(reference)
        if now.ndim==1:
            now=now[None,:]
        maxdq=_array(env.dof_vel_limits).reshape(-1)
        if now.shape != target.shape or maxdq.shape!=(18,):
            raise ValueError('Need current joint state and 18 actuator velocity limits')
        control_dt=float(env.dt)
        if not np.isfinite(control_dt) or control_dt<=0:
            raise ValueError('Invalid simulator control dt')
        if np.any(np.abs(target-now)>maxdq[None,:]*control_dt+1e-8):
            raise ValueError('Reference changes faster than nominal joint velocity limits')
    scale=np.asarray(env.cfg.control.action_scale,dtype=float)
    if scale.ndim==0:
        scale=np.full((1,18),float(scale))
    else:
        scale=scale.reshape(1,18)
    if np.any(scale<=0) or not np.isfinite(scale).all():
        raise ValueError('Nonpositive or nonfinite action_scale')
    actions=(target-default)/scale
    clip=float(env.cfg.normalization.clip_actions)
    if np.any(np.abs(actions)>clip+1e-9):
        raise ValueError('Needed action outside PCR action clip range')
    return actions
