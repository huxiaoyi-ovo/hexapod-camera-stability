"""Offline checks for the independent local-asset simulator foundation."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from hexapod_robot import (JOINT_NAMES, LEG_ORDER, load_robot_model, lowest_stand_height,
                           map_named_values, pd_torque, stand_target_by_name)
from hexapod_sim import (PROJECT_ROOT, _StandWindowTracker, _runtime_provenance,
                         _limit_joint_position_step, _trajectory_velocity_reference, load_config)


class StandaloneHexapodTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(PROJECT_ROOT / "configs/flat_stand.json")
        self.model = load_robot_model(PROJECT_ROOT / self.config["robot_urdf"])
        self.target = stand_target_by_name(self.model, self.config["stand_reference_rad"])

    def test_local_asset_closure_and_urdf_metadata(self):
        urdf = self.model.path
        self.assertEqual(self.model.root_link, "body")
        self.assertEqual(set(self.model.limits), set(JOINT_NAMES))
        self.assertEqual(len(self.model.toe_link_names), 6)
        self.assertAlmostEqual(self.model.toe_radius_m, 0.009)
        self.assertAlmostEqual(self.model.limits["j_lb_thigh"].effort_nm, 27.0)
        self.assertAlmostEqual(self.model.limits["j_lb_thigh"].velocity_rad_s, 5.0)
        self.assertAlmostEqual(lowest_stand_height(self.model, self.target, 0.002), 0.11561747553310604, places=10)
        copied = [urdf]
        copied.extend(sorted((PROJECT_ROOT / "assets/hexapod/two_set_meshes/visual").glob("*.dae")))
        copied.extend(sorted((PROJECT_ROOT / "assets/hexapod/two_set_meshes/collision").glob("*.obj")))
        self.assertEqual(len(copied), 14)
        provenance = (PROJECT_ROOT / "assets/hexapod/PROVENANCE.md").read_text()
        for path in copied:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertIn(digest, provenance)
        self.assertTrue((PROJECT_ROOT / "assets/hexapod/UPSTREAM_LICENSE").is_file())

    def test_mapping_uses_simulator_reported_name_order(self):
        permutation = list(reversed(JOINT_NAMES))
        values = {name: index / 10.0 for index, name in enumerate(JOINT_NAMES)}
        mapped = map_named_values(values, permutation)
        np.testing.assert_allclose(mapped, [values[name] for name in permutation])
        with self.assertRaises(ValueError):
            map_named_values(values, permutation[:-1])

    def test_configured_reference_is_deterministic_and_urdf_bounded(self):
        second = stand_target_by_name(self.model, self.config["stand_reference_rad"])
        self.assertEqual(self.target, second)
        self.assertEqual(len(self.target), 18)
        for name, value in self.target.items():
            limits = self.model.limits[name]
            self.assertGreaterEqual(value, limits.lower_rad - 1e-4)
            self.assertLessEqual(value, limits.upper_rad + 1e-4)
        self.assertEqual(LEG_ORDER, tuple(self.config["leg_order"]))
        with self.assertRaises(ValueError):
            stand_target_by_name(self.model, {"thigh_by_leg": {}, "knee": 0.67, "ankle": -2.2})

    def test_pd_torque_clips_and_rejects_invalid_inputs(self):
        tau, saturated = pd_torque(np.array([0.0, 0.0]), np.array([0.0, 0.0]),
                                   np.array([1.0, -1.0]), 50.0, 1.5, np.array([27.0, 27.0]))
        np.testing.assert_allclose(tau, [27.0, -27.0])
        self.assertTrue(saturated.all())
        with self.assertRaises(ValueError):
            pd_torque(np.zeros(1), np.zeros(1), np.ones(1), float("nan"), 1.5, np.ones(1))
        with self.assertRaises(ValueError):
            pd_torque(np.zeros(1), np.zeros(2), np.ones(1), 50, 1.5, np.ones(1))

    def test_standalone_modules_have_no_upstream_python_imports(self):
        for relative in ("src/hexapod_robot.py", "src/hexapod_sim.py", "scripts/run_hexapod_pilot.py"):
            tree = ast.parse((PROJECT_ROOT / relative).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported = [node.module or ""]
                else:
                    continue
                self.assertFalse(any(name.startswith(("legged_gym", "pcr_", "integration.pcrnet")) for name in imported), relative)

    def test_runtime_config_has_no_upstream_paths(self):
        serialized = json.dumps(self.config)
        self.assertNotIn("Pcrnet_hexapod", serialized)
        self.assertNotIn("legged_gym", serialized)
        self.assertIs(self.config["asset_import"]["flip_visual_attachments"], True)

    def test_stop_on_violation_defaults_strict_and_requires_boolean(self):
        self.assertIs(self.config["stop_on_violation"], True)
        config = copy.deepcopy(self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config["stop_on_violation"] = False
            path.write_text(json.dumps(config))
            self.assertIs(load_config(path)["stop_on_violation"], False)
            config["stop_on_violation"] = "false"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "stop_on_violation"):
                load_config(path)

    def test_openshc_profile_uses_exact_upper_and_physics_cadence(self):
        config = load_config(PROJECT_ROOT / "configs/flat_walk_openshc.json")
        self.assertEqual(config["walk_controller"], "openshc")
        self.assertEqual(config["physics"]["dt_s"], 0.0025)
        self.assertEqual(config["control_period_s"], 0.02)
        self.assertEqual(config["walk"]["duty_factor"], 0.5)
        self.assertEqual(config["walk"]["frequency_hz"], 1.5)

    def test_trajectory_velocity_reference_updates_only_on_ticks_and_clips(self):
        limit = np.array([2.0, 5.0])
        held, previous = _trajectory_velocity_reference(
            np.array([0.0, 0.0]), None, np.zeros(2), True, 0.02, limit)
        np.testing.assert_array_equal(held, [0.0, 0.0])
        np.testing.assert_array_equal(previous, [0.0, 0.0])

        held, previous = _trajectory_velocity_reference(
            np.array([0.1, -0.02]), previous, held, True, 0.02, limit)
        np.testing.assert_allclose(held, [2.0, -1.0])
        held_during_physics, previous_during_physics = _trajectory_velocity_reference(
            np.array([9.0, 9.0]), previous, held, False, 0.02, limit)
        np.testing.assert_array_equal(held_during_physics, held)
        np.testing.assert_array_equal(previous_during_physics, previous)

        held, previous = _trajectory_velocity_reference(
            np.array([0.11, -0.04]), previous_during_physics, held_during_physics,
            True, 0.02, limit)
        np.testing.assert_allclose(held, [0.5, -1.0])
        np.testing.assert_array_equal(previous, [0.11, -0.04])

    def test_command_velocity_operating_cap_limits_qref_without_loosening_hard_cap(self):
        hard_velocity = np.array([5.0, 5.0])
        qref, operating_velocity, clamped = _limit_joint_position_step(
            np.array([1.0, -0.03]), np.zeros(2), hard_velocity, 4.0, 0.02)
        np.testing.assert_allclose(qref, [0.08, -0.03])
        np.testing.assert_array_equal(operating_velocity, [4.0, 4.0])
        np.testing.assert_array_equal(clamped, [True, False])
        np.testing.assert_array_equal(hard_velocity, [5.0, 5.0])

    def test_stand_window_uses_contiguous_elapsed_time_at_sample_resolution(self):
        for dt, window in ((0.003, 0.5), (0.0025, 0.501)):
            tracker = _StandWindowTracker(window)
            passing_samples = int(np.ceil(window / dt)) + 1
            results = [tracker.update((index + 1) * dt, True) for index in range(passing_samples)]
            self.assertFalse(any(results[:-1]))
            self.assertTrue(results[-1])

        tracker = _StandWindowTracker(0.5)
        self.assertFalse(tracker.update(1.0, True))
        self.assertFalse(tracker.update(1.3, False))
        self.assertFalse(tracker.update(1.6, True))
        self.assertTrue(tracker.update(2.1, True))
        self.assertFalse(tracker.update(2.2, False))
        self.assertFalse(tracker.update(2.7, True))

    def test_config_rejects_invalid_json_types_before_runtime(self):
        cases = []
        for key, value in (("substeps", True), ("substeps", 1.5), ("substeps", "1"),
                           ("position_iterations", False), ("velocity_iterations", "0")):
            invalid = copy.deepcopy(self.config)
            invalid["physics"][key] = value
            cases.append(invalid)
        for value in (True, 0, 7, 3.5, "3"):
            invalid = copy.deepcopy(self.config)
            invalid["gate"]["min_contacts"] = value
            cases.append(invalid)
        for key in ("fixed_base", "collapse_fixed_joints", "self_collision", "gpu_pipeline", "gpu_physics"):
            invalid = copy.deepcopy(self.config)
            invalid["physics"][key] = "false"
            cases.append(invalid)
        invalid = copy.deepcopy(self.config)
        invalid["asset_import"]["flip_visual_attachments"] = "false"
        cases.append(invalid)
        invalid = copy.deepcopy(self.config)
        invalid["physics"]["dt_s"] = True
        cases.append(invalid)
        invalid = copy.deepcopy(self.config)
        invalid["physics"]["dt_s"] = float("nan")
        cases.append(invalid)
        invalid = copy.deepcopy(self.config)
        del invalid["physics"]["substeps"]
        cases.append(invalid)
        invalid = copy.deepcopy(self.config)
        del invalid["gate"]["min_contacts"]
        cases.append(invalid)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for config in cases:
                path.write_text(json.dumps(config))
                with self.subTest(config=config), self.assertRaises(ValueError):
                    load_config(path)

    def test_runtime_provenance_reports_git_state_and_live_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("src/hexapod_sim.py", "src/hexapod_robot.py", "scripts/run_hexapod_pilot.py",
                             "src/hexapod_walk.py", "src/hexapod_actuator.py", "src/hexapod_openshc.py",
                             "src/hexapod_sensing.py", "src/hexapod_terrain.py",
                             "src/hexapod_body_compensation.py"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative)

            clean = [subprocess.CompletedProcess([], 0, "abc123\n", ""),
                     subprocess.CompletedProcess([], 0, "", "")]
            with mock.patch("hexapod_sim.subprocess.run", side_effect=clean):
                provenance = _runtime_provenance(root)
            self.assertEqual(provenance["git"], {"available": True, "revision": "abc123", "dirty": False,
                                                  "status": "available"})

            dirty = [subprocess.CompletedProcess([], 0, "abc123\n", ""),
                     subprocess.CompletedProcess([], 0, " M src/hexapod_sim.py\n", "")]
            with mock.patch("hexapod_sim.subprocess.run", side_effect=dirty):
                provenance = _runtime_provenance(root)
            self.assertEqual(provenance["git"], {"available": True, "revision": "abc123", "dirty": True,
                                                  "status": "available"})

            with mock.patch("hexapod_sim.subprocess.run", side_effect=FileNotFoundError("git unavailable")):
                provenance = _runtime_provenance(root)
            self.assertEqual(provenance["git"], {"available": False, "revision": None, "dirty": None,
                                                  "status": "unavailable_or_timed_out"})
            with mock.patch("hexapod_sim.subprocess.run",
                            side_effect=subprocess.TimeoutExpired("git", 2.0)):
                provenance = _runtime_provenance(root)
            self.assertEqual(provenance["git"], {"available": False, "revision": None, "dirty": None,
                                                  "status": "unavailable_or_timed_out"})
            self.assertEqual(set(provenance["runtime_dependency_sha256"]), {
                "src/hexapod_sim.py", "src/hexapod_robot.py", "scripts/run_hexapod_pilot.py",
                "src/hexapod_walk.py", "src/hexapod_actuator.py", "src/hexapod_openshc.py",
                "src/hexapod_sensing.py", "src/hexapod_terrain.py", "src/hexapod_body_compensation.py"})
            original = provenance["runtime_dependency_sha256"]["src/hexapod_sim.py"]
            (root / "src/hexapod_sim.py").write_text("changed runtime")
            changed = _runtime_provenance(root)["runtime_dependency_sha256"]["src/hexapod_sim.py"]
            self.assertNotEqual(original, changed)
            self.assertEqual(changed, hashlib.sha256(b"changed runtime").hexdigest())


if __name__ == "__main__":
    unittest.main()
