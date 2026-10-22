"""Source parity and adapter contract for the standalone PCR ExpertGround backend."""
import hashlib
import json
import math
import os
from pathlib import Path
import types
import unittest

import numpy as np

from hexapod_pcr_expert import JOINT_ORDER, PcrExpertAdapter, pcr_default_joint_angles
from hexapod_sim import PROJECT_ROOT, load_config

UPSTREAM_ROOT = Path(os.environ.get("PCRNET_SOURCE_ROOT", "/home/artrc/Pcrnet_hexapod"))
UPSTREAM_EXPERT = UPSTREAM_ROOT / "legged_gym/envs/hex_v4/expert.py"
UPSTREAM_KINEMATIC = UPSTREAM_ROOT / "legged_gym/utils/kinematic.py"


def _stub_config(defaults):
    from types import SimpleNamespace

    return SimpleNamespace(
        asset=SimpleNamespace(links=SimpleNamespace(l1=0.072, l2=0.13, l3=0.17),
                              body_shape=SimpleNamespace(x=0.1, y=0.22)),
        init_state=SimpleNamespace(default_joint_angles=dict(defaults)))


def _load_upstream_oracle():
    source = UPSTREAM_EXPERT.read_text()
    original_imports = (
        "from .hex_climb_config import HexClimbCfg\n"
        "from .hex_ground_config import HexGroundCfg\n"
        "from legged_gym.utils.kinematic import Kinematic")
    injected_imports = (
        "HexClimbCfg = object\nHexGroundCfg = object\n"
        "from third_party.pcr_expert.kinematic import Kinematic")
    if original_imports not in source:
        raise AssertionError("upstream import block differs from audited snapshot")
    module = types.ModuleType("upstream_expert_oracle")
    exec(compile(source.replace(original_imports, injected_imports), str(UPSTREAM_EXPERT), "exec"),
         module.__dict__)
    return module


class PcrExpertTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(PROJECT_ROOT / "configs/pcr_expert_pilot.json")
        self._torch_threads = None

    def tearDown(self):
        if self._torch_threads is not None:
            import torch
            torch.set_num_threads(self._torch_threads)

    def test_pilot_selects_frozen_profile_and_preserves_actuator_limits(self):
        self.assertEqual(self.config["walk_controller"], "pcr_expert")
        self.assertEqual(self.config["walk"]["speed_m_s"], 0.5)
        self.assertEqual(self.config["walk"]["speed_ramp_s"], 2.0)
        self.assertEqual(self.config["duration_s"], 20.0)
        self.assertEqual(self.config["terrain"], {"kind": "flat"})
        self.assertIs(self.config["stop_on_violation"], False)
        self.assertNotIn("command_velocity_limit_rad_s", self.config["control"])
        self.assertEqual(self.config["control"]["joint_velocity_limit_rad_s"], 5.0)
        self.assertEqual(self.config["control"]["effort_limit_nm"], 27.0)
        self.assertEqual(self.config["actuator"]["velocity_reference"], "zero")
        self.assertEqual(self.config["sensing"]["kind"], "legacy_simulator")
        self.assertEqual(self.config["pcr_expert"]["swing_speed_scale"], 1.0)
        self.assertEqual(self.config["pcr_expert"]["swing_apex_z_m"], 0.02)

    def test_vendored_controller_hashes_match_provenance_without_upstream_checkout(self):
        local = (PROJECT_ROOT / "third_party/pcr_expert/expert.py").read_text()
        provenance = json.loads((PROJECT_ROOT / "third_party/pcr_expert/provenance.json").read_text())
        self.assertEqual(hashlib.sha256(local.encode()).hexdigest(),
                         provenance["files"]["expert.py"]["local_sha256"])
        local_kinematic = (PROJECT_ROOT / "third_party/pcr_expert/kinematic.py").read_bytes()
        self.assertEqual(hashlib.sha256(local_kinematic).hexdigest(),
                         provenance["files"]["kinematic.py"]["local_sha256"])
        geometry = provenance["algorithm_geometry"]
        self.assertEqual(geometry["link_lengths_m"], [0.072, 0.13, 0.17])
        self.assertEqual(geometry["local_urdf_comparison"]["joint_origin_lengths_m"],
                         [0.072, 0.13, 0.1855])
        self.assertEqual(geometry["local_urdf_comparison"]["source_minus_local_last_link_m"], -0.0155)

    @unittest.skipUnless(UPSTREAM_EXPERT.is_file() and UPSTREAM_KINEMATIC.is_file(),
                         "upstream checkout is optional outside local parity validation")
    def test_vendored_controller_differs_from_upstream_only_at_imports(self):
        local = (PROJECT_ROOT / "third_party/pcr_expert/expert.py").read_text()
        upstream = UPSTREAM_EXPERT.read_text()
        local_imports = (
            "HexClimbCfg = object  # runtime config classes are supplied by the standalone adapter\n"
            "HexGroundCfg = object\nfrom .kinematic import Kinematic")
        upstream_imports = (
            "from .hex_climb_config import HexClimbCfg\n"
            "from .hex_ground_config import HexGroundCfg\n"
            "from legged_gym.utils.kinematic import Kinematic")
        self.assertEqual(local.replace(local_imports, upstream_imports), upstream)
        self.assertEqual((PROJECT_ROOT / "third_party/pcr_expert/kinematic.py").read_bytes(),
                         UPSTREAM_KINEMATIC.read_bytes())

    def test_backend_dispatch_does_not_fall_through_to_project_tripod(self):
        source = (PROJECT_ROOT / "src/hexapod_sim.py").read_text()
        gate_start = source.index('if walk_backend == "openshc":', source.index('if gate_result == "ready":'))
        gate_end = source.index("gait_start_time = t", gate_start)
        gate_branch = source[gate_start:gate_end]
        self.assertLess(gate_branch.index('elif walk_backend == "python_tripod":'),
                        gate_branch.index('elif walk_backend == "pcr_expert":'))
        self.assertNotIn("else:", gate_branch)
        control_start = source.index('if walk_backend == "openshc" and openshc_bridge is not None')
        control_end = source.index("except (ValueError, RuntimeError", control_start)
        control_branch = source[control_start:control_end]
        self.assertLess(control_branch.index('elif walk_backend == "python_tripod"'),
                        control_branch.index('walk_backend == "pcr_expert"'))
        self.assertIn("walk_started and not stand_only", control_branch)
        self.assertLess(source.index("foot_contacts = {leg:"),
                        source.index("controller_contacts = foot_contacts"))

    def test_adapter_uses_named_canonical_order_and_initializes_only_once(self):
        defaults = pcr_default_joint_angles(self.config)
        adapter = PcrExpertAdapter("cpu", defaults, [0.072, 0.13, 0.17], {"x": 0.1, "y": 0.22})
        q = dict(defaults)
        dq = {name: 0.0 for name in JOINT_ORDER}
        first = adapter.step(q, dq, 0.0, 0.0, 0.0)
        self.assertEqual(tuple(first), JOINT_ORDER)
        self.assertTrue(adapter.reset_sent)
        adapter.step(q, dq, 0.0, 0.1, 0.0)
        self.assertTrue(adapter.reset_sent)
        self.assertEqual(adapter.tripod_switch_count, 0)
        self.assertEqual(adapter._activity_tick, 2)

    def test_swing_parameters_change_only_local_swing_waypoint_and_swing_motion(self):
        defaults = pcr_default_joint_angles(self.config)
        dq = {name: 0.0 for name in JOINT_ORDER}
        base = PcrExpertAdapter("cpu", defaults, [0.072, 0.13, 0.17], {"x": 0.1, "y": 0.22})
        faster = PcrExpertAdapter("cpu", defaults, [0.072, 0.13, 0.17], {"x": 0.1, "y": 0.22},
                                  swing_speed_scale=2.0)
        higher = PcrExpertAdapter("cpu", defaults, [0.072, 0.13, 0.17], {"x": 0.1, "y": 0.22},
                                  swing_apex_z_m=0.04)
        q = dict(defaults)
        for adapter in (base, faster, higher):
            adapter.step(q, dq, 0.2, 0.3, 0.1)

        self.assertEqual(base.swing_speed_scale, 1.0)
        self.assertEqual(base.swing_apex_z_m, 0.02)
        self.assertGreater(0.02, float(base.expert.B_e_init[:, 2].max().item()))
        np.testing.assert_array_equal(base.expert.R1_T_R.cpu().numpy(), faster.expert.R1_T_R.cpu().numpy())
        np.testing.assert_array_equal(base.expert.R1_T_R_swing.cpu().numpy(),
                                      faster.expert.R1_T_R_swing.cpu().numpy())
        self.assertGreater(float(np.max(np.abs(base.expert.B_e_des.cpu().numpy() -
                                               faster.expert.B_e_des.cpu().numpy()))), 1e-6)
        base_waypoint = base.expert.swing_init_point.cpu().numpy()
        higher_waypoint = higher.expert.swing_init_point.cpu().numpy()
        np.testing.assert_array_equal(base_waypoint[:, :2], higher_waypoint[:, :2])
        np.testing.assert_array_equal(base_waypoint[:, 2], np.full(6, 0.02, dtype=np.float32))
        np.testing.assert_array_equal(higher_waypoint[:, 2], np.full(6, 0.04, dtype=np.float32))
        self.assertTrue(bool(higher.expert.set_init_done[0].item()))
        higher.step(q, dq, 0.2, 0.3, 0.1)
        self.assertTrue(higher.reset_sent)
        self.assertTrue(bool(higher.expert.set_init_done[0].item()))
        np.testing.assert_array_equal(higher.expert.swing_init_point.cpu().numpy()[:, 2],
                                      np.full(6, 0.04, dtype=np.float32))

    def test_body_correction_matches_rigidly_transformed_dls_target_and_restores_planner(self):
        import torch
        from hexapod_body_compensation import SETTINGS

        self._torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        defaults = pcr_default_joint_angles(self.config)
        dq = {name: 0.0 for name in JOINT_ORDER}
        corrected = PcrExpertAdapter("cpu", defaults, [0.072, 0.13, 0.17], {"x": 0.1, "y": 0.22})
        baseline = PcrExpertAdapter("cpu", defaults, [0.072, 0.13, 0.17], {"x": 0.1, "y": 0.22})
        q = dict(defaults)
        correction = {"roll_rad": 0.02, "pitch_rad": -0.015, "height_m": 0.006}
        actual = corrected.step(q, dq, 0.0, 0.3, 0.0, body_correction=correction)
        baseline.step(q, dq, 0.0, 0.3, 0.0)

        expert = corrected.expert
        np.testing.assert_array_equal(expert.B_e_des.cpu().numpy(), baseline.expert.B_e_des.cpu().numpy())
        np.testing.assert_array_equal(expert.gaits.cpu().numpy(), baseline.expert.gaits.cpu().numpy())
        np.testing.assert_array_equal(expert.swing_reach_point.cpu().numpy(),
                                      baseline.expert.swing_reach_point.cpu().numpy())
        np.testing.assert_array_equal(expert.R1_T_R.cpu().numpy(), baseline.expert.R1_T_R.cpu().numpy())
        self.assertIs(expert.B_e_des_flat._base, expert.B_e_des)
        self.assertIsNone(expert.body_correction)

        roll, pitch, height = (correction["roll_rad"], correction["pitch_rad"], correction["height_m"])
        cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
        rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        ry = torch.tensor([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        rotation = ry @ rx
        feet = expert._B2R(expert.B_e_des).clone()
        feet[..., :3] = (feet[..., :3] - torch.tensor([0.0, 0.0, height])) @ rotation
        transformed = expert._R2B(feet)
        q_tensor = torch.tensor([[q[name] for name in JOINT_ORDER]], dtype=torch.float32)
        q_flat = q_tensor.view(-1, 3)
        damped_inverse = expert.kin.DampInvJac(q_flat)
        position_error = (transformed.view(-1, 4)[..., :3] - expert.B_e_cur_flat[..., :3]).unsqueeze(-1)
        expected = q_flat + 32 * (damped_inverse @ position_error).squeeze(-1) * expert.dt
        np.testing.assert_allclose([actual[name] for name in JOINT_ORDER], expected.reshape(18).numpy(),
                                   rtol=1e-5, atol=1e-6)

        actual_next = corrected.step(q, dq, 0.0, 0.3, 0.0)
        baseline_next = baseline.step(q, dq, 0.0, 0.3, 0.0, body_correction={
            "roll_rad": 0.0, "pitch_rad": 0.0, "height_m": 0.0})
        np.testing.assert_array_equal([actual_next[name] for name in JOINT_ORDER],
                                      [baseline_next[name] for name in JOINT_ORDER])
        self.assertLessEqual(corrected.max_body_angle_rad, SETTINGS["max_angle_rad"])
        self.assertLessEqual(corrected.max_body_height_m, SETTINGS["max_height_m"])

    def test_body_correction_rejects_values_outside_existing_bounds(self):
        defaults = pcr_default_joint_angles(self.config)
        adapter = PcrExpertAdapter("cpu", defaults, [0.072, 0.13, 0.17], {"x": 0.1, "y": 0.22})
        for correction in (
                {"roll_rad": 0.051, "pitch_rad": 0.0, "height_m": 0.0},
                {"roll_rad": 0.0, "pitch_rad": -0.051, "height_m": 0.0},
                {"roll_rad": 0.0, "pitch_rad": 0.0, "height_m": 0.0101}):
            with self.assertRaises(ValueError):
                adapter._validated_body_correction(correction)
        self.assertIsNone(adapter._validated_body_correction(
            {"roll_rad": 0.0, "pitch_rad": 0.0, "height_m": 0.0}))
        with self.assertRaises(ValueError):
            PcrExpertAdapter("cpu", defaults, [0.072, 0.13, 0.17], {"x": 0.1, "y": 0.22},
                             swing_apex_z_m=0.1001)

    def test_switch_event_timing_distinguishes_half_switches_from_full_cycles(self):
        defaults = pcr_default_joint_angles(self.config)
        adapter = PcrExpertAdapter("cpu", defaults, [0.072, 0.13, 0.17], {"x": 0.1, "y": 0.22})
        adapter._record_tripod_switch(1, 0.5)
        adapter._record_tripod_switch(0, 1.0)
        self.assertEqual(adapter.full_cycle_durations_s, [1.0])
        adapter._record_tripod_switch(1, 1.5)
        self.assertEqual(adapter.full_cycle_durations_s, [1.0])
        adapter._record_tripod_switch(0, 2.5)
        self.assertEqual(adapter.tripod_switch_times_s, [0.5, 1.0, 1.5, 2.5])
        self.assertEqual(adapter.tripod_switch_groups, ["B", "A", "B", "A"])
        self.assertEqual(adapter.full_cycle_durations_s, [1.0, 1.5])
        self.assertAlmostEqual(adapter.observed_full_cycle_frequency_hz, 0.8)

    @unittest.skipUnless(UPSTREAM_EXPERT.is_file() and UPSTREAM_KINEMATIC.is_file(),
                         "upstream checkout is optional outside local parity validation")
    def test_full_controller_and_persistent_state_match_upstream_oracle(self):
        import torch

        self._torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        defaults = pcr_default_joint_angles(self.config)
        adapter = PcrExpertAdapter("cpu", defaults, [0.072, 0.13, 0.17], {"x": 0.1, "y": 0.22})
        oracle_class = _load_upstream_oracle().ExpertGround
        oracle = oracle_class(_stub_config(defaults), torch.device("cpu"), 1)
        q = {name: defaults[name] for name in JOINT_ORDER}
        dq = {name: 0.0 for name in JOINT_ORDER}
        phases = ((0, 420, 0.0, 0.30, 0.0),
                  (420, 780, 0.08, 0.12, 0.12),
                  (780, 1180, -0.08, -0.28, -0.20),
                  (1180, 1420, 0.0, 0.0, 0.0))
        switches = 0
        prior_group = 0
        for tick in range(1420):
            vx, vy, yaw = next((x, y, w) for start, end, x, y, w in phases if start <= tick < end)
            reset = tick == 0
            actual = adapter.step(q, dq, vx, vy, yaw)
            command = torch.tensor([[float(reset), vx, vy, 0.0, yaw]], dtype=torch.float32)
            q_tensor = torch.tensor([[q[name] for name in JOINT_ORDER]], dtype=torch.float32)
            dq_tensor = torch.tensor([[dq[name] for name in JOINT_ORDER]], dtype=torch.float32)
            expected = oracle.ProcessCommand(command, q_tensor, dq_tensor).detach().numpy().reshape(18)
            np.testing.assert_array_equal(np.array([actual[name] for name in JOINT_ORDER]), expected)
            group = int(oracle.stance_group_index[0].item())
            switches += int(group != prior_group)
            prior_group = group
            np.testing.assert_array_equal(adapter.expert.gaits.cpu().numpy(), oracle.gaits.cpu().numpy())
            np.testing.assert_array_equal(adapter.expert.stance_group_index.cpu().numpy(),
                                          oracle.stance_group_index.cpu().numpy())
            np.testing.assert_array_equal(adapter.expert.swing_reach_point.cpu().numpy(),
                                          oracle.swing_reach_point.cpu().numpy())
            np.testing.assert_array_equal(adapter.expert.B_e_des.cpu().numpy(), oracle.B_e_des.cpu().numpy())
            np.testing.assert_array_equal(adapter.expert.set_init_done.cpu().numpy(),
                                          oracle.set_init_done.cpu().numpy())
            next_q = dict(actual)
            dq = {name: (next_q[name] - q[name]) / 0.02 for name in JOINT_ORDER}
            q = next_q
        self.assertEqual(adapter.tripod_switch_count, switches)
        self.assertGreaterEqual(switches, 2, "sequence should exercise event-driven tripod transitions")
        self.assertTrue(adapter.reset_sent)


if __name__ == "__main__":
    unittest.main()
