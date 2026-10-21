"""URDF mapping and native-protocol checks for the vendored OpenSHC core."""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hexapod_robot import JOINT_TYPES, LEG_ORDER, lowest_stand_height, load_robot_model, stand_target_by_name, toe_positions_base
from hexapod_sim import PROJECT_ROOT, _openshc_gait_state, _openshc_velocity_command, load_config
from hexapod_openshc import (OpenShcBridge, UPSTREAM_LEG_ORDER, derive_initial_legs,
                             openshc_source_provenance, resolve_bridge_executable)


def _dh_transform(row):
    a, theta, d, alpha = map(float, row)
    ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def _openshc_fk(leg_data):
    transform = np.eye(4)
    for index, row in enumerate(leg_data["dh_rows"]):
        adjusted = list(row)
        if index:
            adjusted[1] += leg_data["q_rad"][index - 1]
        transform = transform @ _dh_transform(adjusted)
    return transform[:3, 3]


class OpenShcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(PROJECT_ROOT / "configs/flat_walk_openshc.json")
        cls.model = load_robot_model(PROJECT_ROOT / cls.config["robot_urdf"])

    def _pose(self, wide=False, yaw=0.0):
        reference = dict(self.config["stand_reference_rad"])
        reference["thigh_by_leg"] = dict(reference["thigh_by_leg"])
        if wide:
            reference["knee"] = 0.0
            reference["ankle"] = -1.0
        q_by_name = stand_target_by_name(self.model, reference)
        feet_base = toe_positions_base(self.model, q_by_name)
        body_xyz = np.array([0.13, -0.08, lowest_stand_height(self.model, q_by_name, 0.002)])
        half = yaw * 0.5
        quat = np.array([0.0, 0.0, math.sin(half), math.cos(half)])
        rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                             [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        feet_world = {leg: body_xyz + rotation @ feet_base["l_{}_toe".format(leg)] for leg in LEG_ORDER}
        return q_by_name, feet_world, body_xyz, quat

    def test_full_body_dh_matches_local_urdf_for_both_postures_and_yaw(self):
        for wide, yaw in ((False, 0.0), (True, 0.0), (False, 0.73), (True, -0.41)):
            q, feet, body, quat = self._pose(wide, yaw)
            mapped = derive_initial_legs(self.model, q, feet, body, quat)
            self.assertEqual(tuple(mapped), UPSTREAM_LEG_ORDER)
            for leg in UPSTREAM_LEG_ORDER:
                with self.subTest(wide=wide, yaw=yaw, leg=leg):
                    np.testing.assert_allclose(_openshc_fk(mapped[leg]), mapped[leg]["measured_foot_body_m"],
                                               atol=2e-10, rtol=0.0)
                    self.assertEqual(len(mapped[leg]["limits"]), 3)
                    for index, joint in enumerate(JOINT_TYPES):
                        name = "j_{}_{}".format(leg, joint)
                        self.assertAlmostEqual(mapped[leg]["q_rad"][index], q[name])

    def test_native_odometry_targets_map_through_initial_yaw(self):
        yaw = math.pi / 2.0
        start = np.array([1.0, -2.0, 0.4])
        response = {"planned_body_pose_odom": {"xyz_m": [0.2, 0.3, 0.0]}, "completed_cycles": 0,
                    "accepted_velocity_xy_m_s": [0.0, 0.0],
                    "legs": {leg: {"phase": 0, "state": 1, "phase_progress": 0.0,
                                   "stance_progress": 0.0, "swing_progress": 0.0,
                                   "foot_target_odom_m": [0.4, -0.1, 0.0],
                                   "q_target_rad": [0.0, 0.0, 0.0], "dq_target_rad_s": [0.0, 0.0, 0.0],
                                   "joint_velocity_limit_active": [False] * 3,
                                   "joint_position_limit_active": [False] * 3,
                                   "planned_support": True, "ik_residual_m": 0.0, "ik_quality": 1.0}
                              for leg in LEG_ORDER}}
        bridge = type("BridgeStub", (), {"legs": {leg: {"plane_to_body_toe_z_m": -0.1} for leg in LEG_ORDER}})()
        state = _openshc_gait_state(response, bridge, start, yaw)
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(state["planned_body_xyz_m"], start + rotation @ [0.2, 0.3, 0.0])
        np.testing.assert_allclose(state["desired_feet_world"]["lb"], start + rotation @ [0.4, -0.1, -0.1])

    def test_forward_command_is_not_rotated_twice_for_nonzero_initial_yaw(self):
        for yaw in (0.0, math.pi / 2.0, -0.63):
            command_native = _openshc_velocity_command(0.1)
            rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                                 [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
            expected_world = np.array([-math.sin(yaw) * 0.1, math.cos(yaw) * 0.1, 0.0])
            np.testing.assert_allclose(rotation @ np.r_[command_native, 0.0], expected_world, atol=1e-12)


    def test_native_bridge_seed_workspace_forward_turn_stop_and_provenance(self):
        try:
            executable = resolve_bridge_executable(PROJECT_ROOT)
        except FileNotFoundError:
            self.skipTest("OpenSHC bridge is optional in offline CI; run scripts/build_openshc.sh to enable")
        q, feet, body, quat = self._pose()
        with tempfile.TemporaryDirectory() as directory:
            bridge = OpenShcBridge(executable, self.model, q, feet, body, quat,
                                   self.config["walk"], Path(directory) / "bridge.log")
            try:
                init = bridge.initialization
                self.assertLess(init["max_fk_error_m"], 2e-5)
                self.assertGreater(init["forward_speed_limit_m_s"], self.config["walk"]["speed_m_s"])
                self.assertEqual(init["effective_period_control_ticks"], 32)
                self.assertAlmostEqual(init["effective_frequency_hz"], 1.5625)

                forward = bridge.step([0.1, 0.0], 0.0, q)
                self.assertGreater(np.linalg.norm(forward["accepted_velocity_xy_m_s"]), 0.0)
                self.assertLessEqual(np.linalg.norm(forward["accepted_velocity_xy_m_s"]),
                                     init["forward_speed_limit_m_s"] + 1e-9)
                for leg in LEG_ORDER:
                    for index, joint in enumerate(JOINT_TYPES):
                        name = "j_{}_{}".format(leg, joint)
                        limits = self.model.limits[name]
                        target = forward["q_target_by_name_rad"][name]
                        velocity = forward["dq_target_by_name_rad_s"][name]
                        self.assertGreaterEqual(target, limits.lower_rad - 1e-7)
                        self.assertLessEqual(target, limits.upper_rad + 1e-7)
                        self.assertLessEqual(abs(velocity), limits.velocity_rad_s + 1e-7)
                turning = bridge.step([0.05, 0.0], 0.1, q)
                self.assertTrue(np.all(np.isfinite(turning["accepted_velocity_xy_m_s"])))
                self.assertLessEqual(abs(turning["accepted_angular_velocity_rad_s"]), 0.5 + 1e-9)
                stopped = bridge.step([0.0, 0.0], 0.0, q)
                self.assertTrue(np.all(np.isfinite(stopped["planned_body_pose_odom"]["xyz_m"])))
                provenance = openshc_source_provenance(PROJECT_ROOT, executable)
                self.assertEqual(provenance["commit"], "48ffa8eb25c98a08d3e8b10f00df7fba2d952921")
                self.assertEqual(len(provenance["bridge_sha256"]), 64)
                self.assertGreaterEqual(len(provenance["files_sha256"]), 20)
            finally:
                bridge.close()

    def test_twenty_second_ideal_follow_keeps_targets_periodic_and_stance_feet_fixed(self):
        try:
            executable = resolve_bridge_executable(PROJECT_ROOT)
        except FileNotFoundError:
            self.skipTest("OpenSHC bridge is optional in offline CI; run scripts/build_openshc.sh to enable")
        q, feet, body, quat = self._pose()
        with tempfile.TemporaryDirectory() as directory:
            bridge = OpenShcBridge(executable, self.model, q, feet, body, quat,
                                   self.config["walk"], Path(directory) / "bridge.log")
            try:
                responses = []
                max_stance_world_step = 0.0
                previous_stance_world = {}
                q_measured = dict(q)
                for _ in range(1000):
                    response = bridge.step([0.0, 0.1], 0.0, q_measured)
                    responses.append(response)
                    q_measured = dict(response["q_target_by_name_rad"])
                    for leg in LEG_ORDER:
                        native = response["legs"][leg]
                        if native["planned_support"]:
                            toe_z = bridge.legs[leg]["plane_to_body_toe_z_m"]
                            world_target = np.asarray(native["foot_target_odom_m"], dtype=float) + [0, 0, toe_z]
                            if leg in previous_stance_world:
                                max_stance_world_step = max(
                                    max_stance_world_step,
                                    float(np.linalg.norm(world_target - previous_stance_world[leg])))
                            previous_stance_world[leg] = world_target
                        else:
                            previous_stance_world.pop(leg, None)
                    for name, value in response["q_target_by_name_rad"].items():
                        limit = self.model.limits[name]
                        self.assertGreaterEqual(value, limit.lower_rad - 1e-7)
                        self.assertLessEqual(value, limit.upper_rad + 1e-7)

                self.assertGreater(responses[-1]["planned_body_pose_odom"]["xyz_m"][1], 1.0)
                self.assertLess(max_stance_world_step, 0.01)
                earlier = responses[-33]["q_target_by_name_rad"]
                later = responses[-1]["q_target_by_name_rad"]
                max_cycle_difference = max(abs(later[name] - earlier[name]) for name in later)
                self.assertLess(max_cycle_difference, 0.05)
                self.assertLessEqual(bridge.completed_cycles, 32)
            finally:
                bridge.close()


if __name__ == "__main__":
    unittest.main()
