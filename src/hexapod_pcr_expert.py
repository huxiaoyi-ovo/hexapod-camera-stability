"""Lazy standalone adapter for the source-frozen PCR ExpertGround controller."""
from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace


LEG_ORDER = ("lb", "lf", "lm", "rb", "rf", "rm")
JOINT_TYPES = ("thigh", "knee", "ankle")
JOINT_ORDER = tuple("j_{}_{}".format(leg, joint) for leg in LEG_ORDER for joint in JOINT_TYPES)


class PcrExpertAdapter:
    """Map named local simulator state to the original batched Torch controller."""

    def __init__(self, device, default_joint_angles, link_lengths_m, body_half_shape_m,
                 swing_speed_scale=1.0, swing_apex_z_m=0.02):
        # Isaac Gym must initialize before Torch in the installed runtime.
        project_root = str(Path(__file__).resolve().parents[1])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        import torch
        from third_party.pcr_expert.expert import ExpertGround
        from hexapod_body_compensation import SETTINGS as BODY_COMP_SETTINGS

        swing_speed_scale = float(swing_speed_scale)
        swing_apex_z_m = float(swing_apex_z_m)
        if not math.isfinite(swing_speed_scale) or swing_speed_scale <= 0.0:
            raise ValueError("PCR swing_speed_scale must be positive and finite")
        if not math.isfinite(swing_apex_z_m) or swing_apex_z_m > 0.10:
            raise ValueError("PCR swing_apex_z_m must be finite and at most 0.10 m in leg-local coordinates")

        self.torch = torch
        self.device = torch.device(device)
        cfg = SimpleNamespace(
            asset=SimpleNamespace(
                links=SimpleNamespace(l1=float(link_lengths_m[0]), l2=float(link_lengths_m[1]),
                                      l3=float(link_lengths_m[2])),
                body_shape=SimpleNamespace(x=float(body_half_shape_m["x"]),
                                           y=float(body_half_shape_m["y"]))),
            init_state=SimpleNamespace(default_joint_angles=dict(default_joint_angles)))
        class ParameterizedExpertGround(ExpertGround):
            def __init__(self, *args):
                self.swing_speed_scale = swing_speed_scale
                self.swing_apex_z_m = swing_apex_z_m
                super().__init__(*args)
                initial_foot_z = float(self.B_e_init[:, 2].max().item())
                if swing_apex_z_m <= initial_foot_z:
                    raise ValueError("PCR swing apex must be above the initial stand foot")
                self.swing_init_point[:, 2] = swing_apex_z_m
                self.body_correction = None

            def CalJointPoses(self, q_cur_flat, q_dot_cur_flat):
                correction = self.body_correction
                if correction is None or not any(correction):
                    return super().CalJointPoses(q_cur_flat, q_dot_cur_flat)
                saved_feet = self.B_e_des.clone()
                roll, pitch, height = correction
                cr, sr = math.cos(roll), math.sin(roll)
                cp, sp = math.cos(pitch), math.sin(pitch)
                rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
                                  dtype=self.B_e_des.dtype, device=self.B_e_des.device)
                ry = torch.tensor([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
                                  dtype=self.B_e_des.dtype, device=self.B_e_des.device)
                rotation = ry @ rx
                translation = torch.tensor([0.0, 0.0, height], dtype=self.B_e_des.dtype,
                                           device=self.B_e_des.device)
                try:
                    feet_body = self._B2R(self.B_e_des)
                    feet_body[..., :3] = (feet_body[..., :3] - translation) @ rotation
                    self.B_e_des.copy_(self._R2B(feet_body))
                    return super().CalJointPoses(q_cur_flat, q_dot_cur_flat)
                finally:
                    self.B_e_des.copy_(saved_feet)

            def _CalB_e_des(self, command):
                if self.swing_speed_scale == 1.0:
                    return super()._CalB_e_des(command)
                swing_command = command.clone()
                swing_command[:, 1:3] *= self.swing_speed_scale
                swing_command[:, 4] *= self.swing_speed_scale
                return super()._CalB_e_des(swing_command)

        self.swing_speed_scale = swing_speed_scale
        self.swing_apex_z_m = swing_apex_z_m
        self.max_body_angle_rad = float(BODY_COMP_SETTINGS["max_angle_rad"])
        self.max_body_height_m = float(BODY_COMP_SETTINGS["max_height_m"])
        self.expert = ParameterizedExpertGround(cfg, self.device, 1)
        self.reset_sent = False
        self._last_stance_group = 0
        self.tripod_switch_count = 0
        self._activity_tick = 0
        self.tripod_switch_times_s = []
        self.tripod_switch_groups = []
        self.full_cycle_durations_s = []
        self._previous_support = {leg: False for leg in LEG_ORDER}
        self._anchors_world = {}

    def _record_tripod_switch(self, group, time_s):
        group = int(group)
        if group == self._last_stance_group:
            return
        self.tripod_switch_count += 1
        self.tripod_switch_times_s.append(float(time_s))
        self.tripod_switch_groups.append("A" if group == 0 else "B")
        if self.tripod_switch_count % 2 == 0:
            cycle_start_s = (0.0 if self.tripod_switch_count == 2
                             else self.tripod_switch_times_s[-3])
            self.full_cycle_durations_s.append(self.tripod_switch_times_s[-1] - cycle_start_s)
        self._last_stance_group = group

    @property
    def observed_full_cycle_frequency_hz(self):
        total_duration = sum(self.full_cycle_durations_s)
        return (len(self.full_cycle_durations_s) / total_duration
                if total_duration > 0.0 else None)

    def _validated_body_correction(self, body_correction):
        if body_correction is None:
            return None
        expected = {"roll_rad", "pitch_rad", "height_m"}
        if not isinstance(body_correction, dict) or set(body_correction) != expected:
            raise ValueError("body_correction requires roll_rad, pitch_rad, and height_m")
        values = tuple(float(body_correction[key]) for key in ("roll_rad", "pitch_rad", "height_m"))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("body_correction values must be finite")
        if abs(values[0]) > self.max_body_angle_rad or abs(values[1]) > self.max_body_angle_rad:
            raise ValueError("body_correction angles exceed existing body-compensation bounds")
        if abs(values[2]) > self.max_body_height_m:
            raise ValueError("body_correction height exceeds existing body-compensation bound")
        return None if not any(values) else values

    def step(self, q_by_name, dq_by_name, vx_m_s, vy_m_s, yaw_rad_s,
             body_correction=None):
        if set(q_by_name) != set(JOINT_ORDER) or set(dq_by_name) != set(JOINT_ORDER):
            raise ValueError("PCR expert inputs must contain every named canonical joint")
        torch = self.torch
        q = torch.tensor([[q_by_name[name] for name in JOINT_ORDER]], dtype=torch.float32, device=self.device)
        dq = torch.tensor([[dq_by_name[name] for name in JOINT_ORDER]], dtype=torch.float32, device=self.device)
        reset = not self.reset_sent
        command = torch.tensor([[float(reset), float(vx_m_s), float(vy_m_s), 0.0,
                                 float(yaw_rad_s)]], dtype=torch.float32, device=self.device)
        activity_time_s = self._activity_tick * float(self.expert.dt)
        self.expert.body_correction = self._validated_body_correction(body_correction)
        try:
            target = self.expert.ProcessCommand(command, q, dq)
        finally:
            self.expert.body_correction = None
        self.reset_sent = True
        group = int(self.expert.stance_group_index[0].item())
        self._record_tripod_switch(group, activity_time_s)
        self._activity_tick += 1
        values = target.detach().cpu().numpy().reshape(18)
        return {name: float(value) for name, value in zip(JOINT_ORDER, values)}

    def diagnostics(self, body_xyz_m, body_rotation):
        """Build log-only world targets from PCR references and measured pose."""
        import numpy as np

        feet_body = self.expert._B2R(self.expert.B_e_des).detach().cpu().numpy()[0, :, :3]
        supports = self.expert.gaits.detach().cpu().numpy()[0].astype(bool)
        feet_world = {leg: np.asarray(body_xyz_m, dtype=float) + body_rotation @ feet_body[index]
                      for index, leg in enumerate(LEG_ORDER)}
        for index, leg in enumerate(LEG_ORDER):
            if supports[index] and not self._previous_support[leg]:
                self._anchors_world[leg] = feet_world[leg].copy()
            self._previous_support[leg] = bool(supports[index])
        return {
            "desired_feet_world": feet_world,
            "anchors_world": {leg: self._anchors_world.get(leg, feet_world[leg]).copy()
                              for leg in LEG_ORDER},
            "planned_support": {leg: bool(supports[index]) for index, leg in enumerate(LEG_ORDER)},
            "cycle_phase": int(self._last_stance_group),
            "completed_cycles": len(self.full_cycle_durations_s),
        }


def pcr_default_joint_angles(config):
    """Convert the local named stand reference to PCR's expected key dictionary."""
    reference = config["stand_reference_rad"]
    angles = {"j_{}_thigh".format(leg): float(reference["thigh_by_leg"][leg])
              for leg in LEG_ORDER}
    angles.update({
        "j_{}_{}".format(leg, joint): float(reference[joint])
        for leg in LEG_ORDER for joint in ("knee", "ankle")})
    return angles
