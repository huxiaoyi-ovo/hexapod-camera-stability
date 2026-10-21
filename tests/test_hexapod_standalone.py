"""Offline checks for the independent local-asset simulator foundation."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from hexapod_robot import (JOINT_NAMES, LEG_ORDER, load_robot_model, lowest_stand_height,
                           map_named_values, pd_torque, stand_target_by_name)
from hexapod_sim import PROJECT_ROOT, load_config


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


if __name__ == "__main__":
    unittest.main()
