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
        self.phase = 'stand'
        self.reference_limited = False
        self.steps = 0

    def start_walking(self):
        """Anchor walking to the settled, measured standing pose."""
        if self.phase != 'stand':
            raise RuntimeError('Walking can only start from the stand phase')
        q_sim = _array(self.env.dof_pos[0]).reshape(-1)
        self.home_q = self.kin.to_canonical(q_sim, self.names)
        self.home_feet = self.kin.fk(self.home_q)
        self.home_z_m = float(_array(self.env.root_states[0])[2])
        self.clock = ContinuousPhaseClock()
        self.phase = 'walking'

    def _rate_limit_reference(self, candidate):
        dt = float(self.env.dt)
        max_delta = _array(self.env.dof_vel_limits).reshape(-1) * dt
        delta = candidate - self.last_desired_q
        self.reference_limited = bool(np.any(np.abs(delta) > max_delta + 1e-8))
        return self.last_desired_q + np.clip(delta, -max_delta, max_delta)

    def action(self, forward_mps: float, lateral_mps: float = 0.0,
               yaw_rad_s: float = 0.0):
        dt = float(self.env.dt)
        if not 0 < dt < 0.2:
            raise ValueError('Unexpected Isaac Gym control period')
        if self.phase == 'stand':
            candidate = _array(self.env.default_dof_pos).reshape(-1)
            stance = np.ones(6, dtype=bool)
        else:
            foot, stance = foot_targets_by_phase(
                self.clock.cycles, self.home_feet, lateral_mps, forward_mps,
                yaw_rad_s, self.gait)
            if self.compensate:
                state = _array(self.env.root_states[0])
                foot = self.compensator.foot_reference(foot, state[3:7],
                                                      state[2], self.home_z_m)
            q_now = self.kin.to_canonical(_array(self.env.dof_pos[0]), self.names)
            q_des = self.kin.ik(foot, q_reference=q_now)
            candidate = self.kin.to_sim_order(q_des, self.names)
        # Check the requested IK target before reference slew limiting so a
        # position/action infeasibility cannot be hidden by a bounded ramp.
        targets_to_hexground_actions(self.env, candidate.reshape(1,18),
                                     check_velocity=False)
        sim_q = self._rate_limit_reference(candidate)
        act = targets_to_hexground_actions(
            self.env, sim_q.reshape(1,18), check_velocity=True,
            previous_reference=self.last_desired_q)
        self.last_desired_q = sim_q
        if self.phase == 'walking':
            self.clock.advance(dt, self.gait.frequency_hz)
        self.steps += 1
        return act, stance
