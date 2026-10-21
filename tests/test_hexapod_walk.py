"""Offline checks for the independent local-URDF walking primitives."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hexapod_robot import (JOINT_NAMES, JOINT_TYPES, LEG_ORDER, load_robot_model,
                           stand_target_by_name, toe_positions_base)
from hexapod_actuator import MitPdActuator
from hexapod_sim import load_config
from hexapod_walk import (ControlCadence, StandStartGate, TRIPOD_A, TRIPOD_B,
                          TripodGait, create_gait_after_stand_gate, leg_fk_jacobian,
                          leg_inverse_kinematics, walk_progress_failure)

ROOT = Path(__file__).resolve().parents[1]


class HexapodWalkTests(unittest.TestCase):
    def setUp(self):
        self.model = load_robot_model(ROOT / "assets/hexapod/urdf/hex_ground.urdf")
        config = json.loads((ROOT / "configs/flat_stand.json").read_text())
        self.q = stand_target_by_name(self.model, config["stand_reference_rad"])
        self.feet = toe_positions_base(self.model, self.q)

    def test_leg_fk_jacobian_matches_tree_fk_and_finite_difference(self):
        rng = np.random.RandomState(37)
        for sample in range(3):
            q = dict(self.q)
            for name in q:
                limit = self.model.limits[name]
                span = min(limit.upper_rad - limit.lower_rad, 0.18)
                q[name] += float(rng.uniform(-span, span))
            generic = toe_positions_base(self.model, q)
            for leg in LEG_ORDER:
                point, jacobian = leg_fk_jacobian(self.model, leg, q)
                np.testing.assert_allclose(point, generic["l_{}_toe".format(leg)], atol=1e-12)
                self.assertEqual(jacobian.shape, (3, 3))
                for index, kind in enumerate(JOINT_TYPES):
                    name = "j_{}_{}".format(leg, kind)
                    perturbed = dict(q)
                    perturbed[name] += 1e-6
                    next_point = toe_positions_base(self.model, perturbed)["l_{}_toe".format(leg)]
                    np.testing.assert_allclose((next_point - point) / 1e-6, jacobian[:, index], atol=2e-6)

    def test_bounded_ik_roundtrip_and_unreachable_rejection(self):
        for leg in LEG_ORDER:
            target_q = dict(self.q)
            names = ["j_{}_{}".format(leg, kind) for kind in JOINT_TYPES]
            for name, delta in zip(names, (0.035, -0.045, 0.025)):
                target_q[name] += delta
            target, _ = leg_fk_jacobian(self.model, leg, target_q)
            solved, residual = leg_inverse_kinematics(self.model, leg, target, self.q)
            self.assertLessEqual(residual, 2e-5)
            solved_full = dict(self.q)
            solved_full.update(solved)
            solved_point, _ = leg_fk_jacobian(self.model, leg, solved_full)
            np.testing.assert_allclose(solved_point, target, atol=2e-5)
            for name in names:
                limit = self.model.limits[name]
                self.assertGreaterEqual(solved[name], limit.lower_rad)
                self.assertLessEqual(solved[name], limit.upper_rad)
        with self.assertRaisesRegex(ValueError, "unreachable"):
            leg_inverse_kinematics(self.model, "lb", np.array([2.0, 2.0, 2.0]), self.q)

    def test_control_cadence_is_eight_physics_steps_with_held_command(self):
        cadence = ControlCadence(0.0025, 0.02)
        self.assertEqual(cadence.steps_per_update, 8)
        held = 0
        updates = []
        values = []
        for step in range(32):
            if cadence.is_tick(step):
                held = cadence.tick_index(step) + 10
                updates.append(step)
            values.append(held)
        self.assertEqual(updates, [0, 8, 16, 24])
        self.assertEqual(values, [10] * 8 + [11] * 8 + [12] * 8 + [13] * 8)
        with self.assertRaises(ValueError):
            ControlCadence(0.0025, 0.019)

    def test_mit_actuator_delay_saturation_and_analytic_torque_response(self):
        actuator = MitPdActuator(0.0025, 0.0025, 0.005, 10.0, 2.0, 0.0,
                                 np.array([1.0]), np.array([0.0]))
        samples = []
        for step in range(4):
            target = np.array([0.5 if step >= 0 else 0.0])
            samples.append(actuator.step(np.array([0.0]), np.array([0.0]), target))
        self.assertEqual([sample["delayed_position_rad"][0] for sample in samples], [0.0, 0.0, 0.5, 0.5])
        self.assertAlmostEqual(samples[2]["raw_torque_nm"][0], 1.0)
        self.assertTrue(samples[2]["torque_saturated"][0])
        alpha = 1.0 - np.exp(-2.0 * np.pi * 10.0 * 0.0025)
        self.assertAlmostEqual(samples[2]["applied_model_torque_nm"][0], alpha, places=7)
        self.assertAlmostEqual(samples[3]["applied_model_torque_nm"][0], 1.0 - (1.0 - alpha) ** 2, places=7)
        self.assertLessEqual(abs(samples[-1]["applied_model_torque_nm"][0]), 1.0)
        urdf_caps = np.array([min(self.model.limits[name].effort_nm, 27.0) for name in JOINT_NAMES])
        robot_actuator = MitPdActuator(0.0025, 0.0025, 0.0, 80.0, 50.0, 1.5, urdf_caps, np.zeros(len(JOINT_NAMES)))
        capped = robot_actuator.step(np.zeros(len(JOINT_NAMES)), np.zeros(len(JOINT_NAMES)),
                                     np.full(len(JOINT_NAMES), 10.0))
        np.testing.assert_allclose(capped["requested_torque_nm"], urdf_caps)
        self.assertTrue(np.all(capped["torque_saturated"]))

    def test_mit_actuator_explicit_initial_stand_seed_and_upper_target_zoh(self):
        cadence = ControlCadence(0.0025, 0.02)
        initial = np.array([0.2])
        actuator = MitPdActuator(0.0025, 0.0025, 0.02, 80.0, 50.0, 1.5,
                                 np.array([2.0]), initial)
        held = initial.copy()
        delayed_targets = []
        for step in range(18):
            if cadence.is_tick(step):
                held = np.array([0.4 if step >= 8 else 0.2])
            sample = actuator.step(initial, np.array([0.0]), held)
            delayed_targets.append(float(sample["delayed_position_rad"][0]))
        self.assertEqual(delayed_targets[:8], [0.2] * 8)
        self.assertEqual(delayed_targets[8:16], [0.2] * 8)
        self.assertEqual(delayed_targets[16:], [0.4] * 2)
        actuator.reset(initial, np.zeros(1), np.zeros(1))
        first = actuator.step(initial, np.zeros(1), np.array([0.9]))
        self.assertEqual(float(first["delayed_position_rad"][0]), 0.2)

    def test_walk_config_requires_explicit_finite_mit_actuator_contract(self):
        valid_path = ROOT / "configs/flat_walk.json"
        valid = load_config(valid_path)
        self.assertEqual(valid["actuator"]["kind"], "MIT_PD")
        self.assertEqual(valid["actuator"]["torque_feedforward_nm"], 0.0)
        cases = []
        for key, value in (("kind", "position"), ("command_delay_s", True),
                           ("command_delay_s", 0.001), ("torque_bandwidth_hz", float("nan")),
                           ("torque_feedforward_nm", 0.1)):
            invalid = json.loads(json.dumps(valid))
            invalid["actuator"][key] = value
            cases.append(invalid)
        invalid = json.loads(json.dumps(valid))
        del invalid["actuator"]
        cases.append(invalid)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "walk.json"
            for config in cases:
                path.write_text(json.dumps(config))
                with self.subTest(actuator=config.get("actuator")), self.assertRaises(ValueError):
                    load_config(path)

    def test_tripod_phases_and_world_anchors_remain_fixed_through_stance(self):
        settings = {"speed_m_s": 0.02, "frequency_hz": 0.55, "duty_factor": 0.67,
                    "swing_height_m": 0.015, "speed_ramp_s": 2.0,
                    "touchdown_timeout_s": 0.12, "max_stance_slip_m": 0.04}
        body = np.array([0.0, 0.0, 0.116])
        nominal = {leg: self.feet["l_{}_toe".format(leg)] for leg in LEG_ORDER}
        feet = {leg: point + body for leg, point in nominal.items()}
        contacts = {leg: True for leg in LEG_ORDER}
        with self.assertRaisesRegex(ValueError, "before a verified stand gate"):
            create_gait_after_stand_gate(False, nominal, feet, body, 0.0, settings, contacts)
        gait, initial = create_gait_after_stand_gate(True, nominal, feet, body, 0.0, settings, contacts)
        self.assertEqual(set(leg for leg in LEG_ORDER if initial["planned_support"][leg]), set(LEG_ORDER))
        anchors = {leg: value.copy() for leg, value in initial["anchors_world"].items()}

        gait.update(0.30, feet, contacts)
        before_liftoff = gait.update(0.308, feet, contacts)
        contacts.update({leg: False for leg in TRIPOD_B})
        liftoff = gait.update(0.310, feet, contacts)
        self.assertTrue(all(liftoff["planned_support"][leg] for leg in TRIPOD_A))
        self.assertTrue(all(not liftoff["planned_support"][leg] for leg in TRIPOD_B))
        for leg in TRIPOD_A:
            np.testing.assert_array_equal(liftoff["anchors_world"][leg], anchors[leg])
        for leg in TRIPOD_B:
            np.testing.assert_allclose(liftoff["desired_feet_world"][leg], anchors[leg], atol=1e-6)

        swing = gait.update(0.60, feet, contacts)
        for leg in TRIPOD_A:
            np.testing.assert_array_equal(swing["anchors_world"][leg], anchors[leg])
        self.assertTrue(any(swing["desired_feet_world"][leg][2] > anchors[leg][2] for leg in TRIPOD_B))

        shifted = {leg: value.copy() for leg, value in feet.items()}
        for leg in TRIPOD_B:
            shifted[leg][1] += 0.002
        contacts.update({leg: True for leg in TRIPOD_B})
        touchdown = gait.update(0.92, shifted, contacts)
        for leg in TRIPOD_B:
            np.testing.assert_array_equal(touchdown["anchors_world"][leg], shifted[leg])
        for leg in TRIPOD_A:
            np.testing.assert_array_equal(touchdown["anchors_world"][leg], anchors[leg])
        self.assertGreater(touchdown["planned_body_xyz_m"][1], body[1])

    def test_frequency_changes_preserve_accumulated_phase_and_cycle_count(self):
        settings = {"speed_m_s": 0.02, "frequency_hz": 0.55, "duty_factor": 0.67,
                    "swing_height_m": 0.015, "speed_ramp_s": 2.0,
                    "touchdown_timeout_s": 0.12, "max_stance_slip_m": 0.04}
        body = np.array([0.0, 0.0, 0.116])
        nominal = {leg: self.feet["l_{}_toe".format(leg)] for leg in LEG_ORDER}
        feet = {leg: point + body for leg, point in nominal.items()}
        contacts = {leg: True for leg in LEG_ORDER}

        gait = TripodGait(nominal, feet, body, 0.0, settings)
        gait.update(0.0, feet, contacts)
        before_change = gait.update(0.4, feet, contacts)
        gait.set_parameters(0.4, frequency_hz=0.8)
        at_change = gait.update(0.4, feet, contacts)
        self.assertAlmostEqual(at_change["cycle_phase"], before_change["cycle_phase"])
        after_change = gait.update(0.6, feet, contacts)
        self.assertAlmostEqual(after_change["cycle_phase"], (0.4 * 0.55 + 0.2 * 0.8) % 1.0)
        self.assertEqual(after_change["completed_cycles"], 0)
        crossed_cycle = gait.update(1.4, feet, contacts)
        self.assertAlmostEqual(crossed_cycle["cycle_phase"], (0.4 * 0.55 + 1.0 * 0.8) % 1.0)
        self.assertEqual(crossed_cycle["completed_cycles"], 1)

        fixed = TripodGait(nominal, feet, body, 0.0, settings)
        for elapsed_s in (0.0, 0.4, 1.4, 2.0):
            state = fixed.update(elapsed_s, feet, contacts)
            phase = elapsed_s * settings["frequency_hz"]
            self.assertAlmostEqual(state["cycle_phase"], phase % 1.0)
            self.assertEqual(state["completed_cycles"], int(phase))

    def test_gait_start_requires_verified_continuous_stand(self):
        gate = StandStartGate(window_s=0.5, timeout_s=3.0)
        self.assertEqual(gate.update(0.0, True), "waiting")
        self.assertEqual(gate.update(0.3, False), "waiting")
        self.assertEqual(gate.update(0.6, True), "waiting")
        self.assertEqual(gate.update(1.1, True), "ready")
        self.assertEqual(gate.update(1.2, False), "ready")

        failed = StandStartGate(window_s=0.5, timeout_s=1.0)
        self.assertEqual(failed.update(0.4, False), "waiting")
        self.assertEqual(failed.update(1.0, False), "failed_start")

    def test_walk_acceptance_requires_net_and_repeated_cycle_progress(self):
        thresholds = (5, 0.10, 0.005)
        self.assertIsNone(walk_progress_failure([0.012, 0.02, 0.02, 0.02, 0.02], 0.30, *thresholds))
        self.assertEqual(walk_progress_failure([0.012] + [0.0] * 4, 0.20, *thresholds),
                         "sustained_cycle_progress_not_reached")
        self.assertEqual(walk_progress_failure([0.012, 0.02, -0.001, 0.02, 0.02], 0.20, *thresholds),
                         "sustained_cycle_progress_not_reached")
        self.assertEqual(walk_progress_failure([0.10], 0.10, *thresholds),
                         "minimum_cycles_not_reached")
        self.assertEqual(walk_progress_failure([0.02] * 5, 0.04, *thresholds),
                         "minimum_net_forward_displacement_not_reached")

    def test_tripod_membership_is_canonical_and_disjoint(self):
        self.assertEqual(set(TRIPOD_A), {"lb", "lf", "rm"})
        self.assertEqual(set(TRIPOD_B), {"lm", "rb", "rf"})
        self.assertFalse(set(TRIPOD_A) & set(TRIPOD_B))
        self.assertEqual(set(TRIPOD_A) | set(TRIPOD_B), set(LEG_ORDER))


if __name__ == "__main__":
    unittest.main()
