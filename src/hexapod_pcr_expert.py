"""Lazy standalone adapter for the source-frozen PCR ExpertGround controller."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace


LEG_ORDER = ("lb", "lf", "lm", "rb", "rf", "rm")
JOINT_TYPES = ("thigh", "knee", "ankle")
JOINT_ORDER = tuple("j_{}_{}".format(leg, joint) for leg in LEG_ORDER for joint in JOINT_TYPES)


class PcrExpertAdapter:
    """Map named local simulator state to the original batched Torch controller."""

    def __init__(self, device, default_joint_angles, link_lengths_m, body_half_shape_m):
        # Isaac Gym must initialize before Torch in the installed runtime.
        project_root = str(Path(__file__).resolve().parents[1])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        import torch
        from third_party.pcr_expert.expert import ExpertGround

        self.torch = torch
        self.device = torch.device(device)
        cfg = SimpleNamespace(
            asset=SimpleNamespace(
                links=SimpleNamespace(l1=float(link_lengths_m[0]), l2=float(link_lengths_m[1]),
                                      l3=float(link_lengths_m[2])),
                body_shape=SimpleNamespace(x=float(body_half_shape_m["x"]),
                                           y=float(body_half_shape_m["y"]))),
            init_state=SimpleNamespace(default_joint_angles=dict(default_joint_angles)))
        self.expert = ExpertGround(cfg, self.device, 1)
        self.reset_sent = False
        self._last_stance_group = 0
        self.tripod_switch_count = 0
        self._previous_support = {leg: False for leg in LEG_ORDER}
        self._anchors_world = {}

    def step(self, q_by_name, dq_by_name, vx_m_s, vy_m_s, yaw_rad_s):
        if set(q_by_name) != set(JOINT_ORDER) or set(dq_by_name) != set(JOINT_ORDER):
            raise ValueError("PCR expert inputs must contain every named canonical joint")
        torch = self.torch
        q = torch.tensor([[q_by_name[name] for name in JOINT_ORDER]], dtype=torch.float32, device=self.device)
        dq = torch.tensor([[dq_by_name[name] for name in JOINT_ORDER]], dtype=torch.float32, device=self.device)
        reset = not self.reset_sent
        command = torch.tensor([[float(reset), float(vx_m_s), float(vy_m_s), 0.0,
                                 float(yaw_rad_s)]], dtype=torch.float32, device=self.device)
        target = self.expert.ProcessCommand(command, q, dq)
        self.reset_sent = True
        group = int(self.expert.stance_group_index[0].item())
        if group != self._last_stance_group:
            self.tripod_switch_count += 1
            self._last_stance_group = group
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
            "completed_cycles": self.tripod_switch_count // 2,
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
