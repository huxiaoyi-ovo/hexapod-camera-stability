"""Single-environment PhysX pilot: gait -> URDF IK -> 18D action.

Not a proven locomotion controller. Stand foot positions are obtained by FK
of the PCR default joint pose; commands and compensation are strictly bounded.
Simulation must validate contact, tracking, falls, and real motor bandwidth.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from pcr_urdf_ik import HexUrdfIK
from tripod_gait import TripodConfig, ContinuousPhaseClock, foot_targets_by_phase
from pcr_action_interface import targets_to_hexground_actions
from pcr_sim_bridge import _array


@dataclass(frozen=True)
class FixedPoseCompensator:
    """A bounded proportional correction, not an ideal six-DoF stabilizer."""
    roll_gain: float = 0.25
    pitch_gain: float = 0.25
    height_gain: float = 0.12
    max_tilt_rad: float = 0.045
    max_height_m: float = 0.012

    def foot_reference(self, reference, body_quat_xyzw, body_z_m, desired_z_m):
        """Transform the foot targets opposite to a small desired body change.

        This is a simple workspace correction. Whether ground reaction forces
        actually correct posture is a PhysX experimental question.
        """
        feet = np.asarray(reference, float)
        if feet.shape != (6, 3):
            raise ValueError('Expected 6 foot references')
        roll, pitch, _ = Rotation.from_quat(body_quat_xyzw).as_euler('xyz')
        dr = np.clip(-self.roll_gain * roll, -self.max_tilt_rad, self.max_tilt_rad)
        dp = np.clip(-self.pitch_gain * pitch, -self.max_tilt_rad, self.max_tilt_rad)
        dz = np.clip(self.height_gain * (desired_z_m - body_z_m), -self.max_height_m, self.max_height_m)
        delta = Rotation.from_euler('xyz', [dr, dp, 0.]).as_matrix()
        out = (delta.T @ (feet - np.array([0, 0, dz])).T).T
        return out


class PcrSingleTrialController:
    """One environment, dry run first; only valid after PCR PhysX rollout."""
    def __init__(self, env, kinematics: HexUrdfIK, gait: TripodConfig,
                 *, compensate: bool, home_z_m: float | None = None):
        if getattr(env, 'num_envs', None) != 1:
            raise ValueError('Single-trial pilot needs num_envs=1')
        self.env = env
        self.kin = kinematics
        self.gait = gait
        self.gait.validate()
        self.compensate = bool(compensate)
        self.clock = ContinuousPhaseClock()
        self.names = tuple(env.dof_names)
        default = _array(env.default_dof_pos).reshape(-1)
        if default.shape != (18,):
            raise ValueError('PCR HexGround default joint pose must be 18 DOF')
        self.home_q = self.kin.to_canonical(default, self.names)
        self.home_feet = self.kin.fk(self.home_q)
        self.compensator = FixedPoseCompensator()
        self.home_z_m = float(home_z_m if home_z_m is not None else _array(env.root_states[0])[2])
        self.last_desired_q = default.copy()
        self.steps = 0

    def action(self, forward_mps: float, lateral_mps: float = 0.0,
               yaw_rad_s: float = 0.0):
        dt = float(self.env.dt)
        if not 0 < dt < 0.2:
            raise ValueError('Unexpected Isaac Gym control period')
        phase_cycles = self.clock.cycles
        foot, stance = foot_targets_by_phase(
            phase_cycles, self.home_feet, lateral_mps, forward_mps,
            yaw_rad_s, self.gait)
        if self.compensate:
            state = _array(self.env.root_states[0])
            foot = self.compensator.foot_reference(foot, state[3:7],
                                                  state[2], self.home_z_m)
        q_now = self.kin.to_canonical(_array(self.env.dof_pos[0]), self.names)
        q_des = self.kin.ik(foot, q_reference=q_now)
        sim_q = self.kin.to_sim_order(q_des, self.names)
        # Only static actuator checks. No hidden clipping on unachievable IK.
        act = targets_to_hexground_actions(self.env, sim_q.reshape(1,18),
                                           check_velocity=True)
        self.last_desired_q = sim_q
        self.clock.advance(dt, self.gait.frequency_hz)
        self.steps += 1
        return act, stance