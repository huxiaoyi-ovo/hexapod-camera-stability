"""Focused checks for the proprioceptive Phase 1 walk prototype."""
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from hexapod_robot import LEG_ORDER, load_robot_model, stand_target_by_name
from hexapod_sensing import ProprioceptiveContactOdometry, damped_tip_force_estimate
from hexapod_terrain import build_terrain
from hexapod_walk import TripodGait


ROOT = Path(__file__).resolve().parents[1]


def _gait_settings():
    return {"speed_m_s": 0.1, "frequency_hz": 1.0, "duty_factor": 0.7,
            "swing_height_m": 0.04, "speed_ramp_s": 2.0, "touchdown_timeout_s": 0.4,
            "max_stance_slip_m": 0.04,
            "terrain_adaptation": {"probe_speed_m_s": 0.04, "probe_max_depth_m": 0.015,
                                   "probe_timeout_s": 0.4, "liftoff_timeout_s": 0.4,
                                   "body_z_rate_m_s": 0.02, "toe_radius_m": 0.009}}


def _gait_frames():
    nominal = {leg: np.array([index * 0.1, 0.0, -0.1046])
               for index, leg in enumerate(LEG_ORDER)}
    world = {leg: np.array([index * 0.1, 0.0, 0.009])
             for index, leg in enumerate(LEG_ORDER)}
    return nominal, world


class TerrainWalkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_model(ROOT / "assets/hexapod/urdf/hex_ground.urdf")
        cls.q = stand_target_by_name(cls.model, {"thigh_by_leg": {"lb": 0.5, "lf": -0.5,
                                                                   "lm": 0.0, "rb": -0.5,
                                                                   "rf": 0.5, "rm": 0.0},
                                                 "knee": 0.67, "ankle": -2.2})

    def test_damped_tip_force_uses_the_declared_virtual_work_direction(self):
        jacobian = np.array([[0.2, 0.0, 0.1], [0.0, 0.3, 0.0], [0.0, 0.0, 0.25]])
        torque = np.array([0.0, 0.0, -1.0])
        expected = -np.linalg.solve(jacobian @ jacobian.T + 0.005**2 * np.eye(3),
                                    jacobian @ torque)
        np.testing.assert_allclose(damped_tip_force_estimate(jacobian, torque), expected)

    def test_force_hysteresis_confirms_and_clears_load_after_dwell(self):
        initial_xyz = np.array([0.0, 0.0, 0.1156])
        q_by_name = dict(self.q)
        estimate = ProprioceptiveContactOdometry(self.model, initial_xyz, np.array([0, 0, 0, 1]),
                                                 q_by_name, self.model.toe_radius_m,
                                                 force_filter_s=0.02, confirmation_s=0.04,
                                                 minimum_support=3)
        torques = {name: 0.0 for name in self.q}
        for leg in LEG_ORDER:
            torques["j_{}_ankle".format(leg)] = -10.0
        with mock.patch("hexapod_sensing.leg_fk_jacobian",
                        return_value=(np.zeros(3), np.eye(3))):
            for _ in range(4):
                state = estimate.update(q_by_name, np.array([0, 0, 0, 1]), torques, 0.01)
            self.assertFalse(any(state["contact_confirmed"].values()))
            state = estimate.update(q_by_name, np.array([0, 0, 0, 1]), torques, 0.01)
            self.assertTrue(all(state["contact_confirmed"].values()))
            zero_torque = {name: 0.0 for name in self.q}
            for _ in range(20):
                state = estimate.update(q_by_name, np.array([0, 0, 0, 1]), zero_torque, 0.01)
            self.assertFalse(any(state["contact_confirmed"].values()))
            self.assertEqual(state["support_count"], 6)  # confirmed stance prior survives unloading

    def test_confirmed_anchor_odometry_uses_only_the_support_mask(self):
        body = np.array([0.03, -0.02, 0.1156])
        yaw = 0.3
        quat = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
        estimate = ProprioceptiveContactOdometry(self.model, body, quat, self.q,
                                                 self.model.toe_radius_m, minimum_support=3)
        support = {leg: leg in ("lb", "lf", "rm") for leg in LEG_ORDER}
        estimate.set_support_mask(support, estimate.snapshot()["feet_world_m"])
        torque = {name: 0.0 for name in self.q}
        state = estimate.update(self.q, quat, torque, 0.01)
        anchor_consistent = state["body_xyz_m"].copy()
        for _ in range(4):
            state = estimate.update(self.q, quat, torque, 0.01)
        np.testing.assert_allclose(state["body_xyz_m"], anchor_consistent, atol=1e-10)
        self.assertEqual(state["support_count"], 3)

    def test_early_contact_waits_until_late_swing_and_is_held_as_stance(self):
        nominal, feet = _gait_frames()
        gait = TripodGait(nominal, feet, np.array([0.0, 0.0, 0.12]), 0.0, _gait_settings())
        contacts = {leg: True for leg in LEG_ORDER}
        gait.update(0.0, feet, contacts)
        contacts["rb"] = False  # reliable release arms this swing for landing
        state = gait.update(0.21, feet, contacts)
        self.assertTrue(state["release_seen"]["rb"])
        contacts["rb"] = True
        # A retrigger in the first swing half is ignored.
        state = gait.update(0.24, feet, contacts)
        self.assertFalse(state["confirmed_support"]["rb"])
        feet["rb"][2] = gait.swing_end["rb"][2] + 0.005
        state = gait.update(0.36, feet, contacts)
        self.assertTrue(state["confirmed_support"]["rb"])
        state = gait.update(0.37, feet, contacts)
        self.assertTrue(state["planned_support"]["rb"])

    def test_stale_apex_contact_cannot_latch_and_fails_bounded_probe(self):
        nominal, feet = _gait_frames()
        gait = TripodGait(nominal, feet, np.array([0.0, 0.0, 0.12]), 0.0, _gait_settings())
        contacts = {leg: True for leg in LEG_ORDER}
        gait.update(0.0, feet, contacts)
        gait.update(0.21, feet, contacts)  # old stance load flag remains true through liftoff
        feet["rb"][2] = gait.swing_end["rb"][2] + 0.04
        state = gait.update(0.36, feet, contacts)  # stale high contact at late swing apex
        self.assertFalse(state["confirmed_support"]["rb"])
        self.assertFalse(state["release_seen"]["rb"])
        gait.update(0.5, feet, contacts)
        with self.assertRaisesRegex(RuntimeError, "bounded touchdown probe"):
            gait.update(0.91, feet, contacts)

    def test_missing_touchdown_probes_only_to_limit_then_fails(self):
        nominal, feet = _gait_frames()
        gait = TripodGait(nominal, feet, np.array([0.0, 0.0, 0.12]), 0.0, _gait_settings())
        contacts = {leg: True for leg in LEG_ORDER}
        gait.update(0.0, feet, contacts)
        contacts["rb"] = False
        gait.update(0.21, feet, contacts)
        gait.update(0.5, feet, contacts)
        probe = gait.update(0.85, feet, contacts)
        self.assertAlmostEqual(probe["desired_feet_world"]["rb"][2],
                               gait.swing_end["rb"][2] - 0.014, places=8)
        with self.assertRaisesRegex(RuntimeError, "bounded touchdown probe"):
            gait.update(0.91, feet, contacts)

    def test_scheduled_late_touchdown_does_not_skip_the_next_swing(self):
        nominal, feet = _gait_frames()
        gait = TripodGait(nominal, feet, np.array([0.0, 0.0, 0.12]), 0.0, _gait_settings())
        contacts = {leg: True for leg in LEG_ORDER}
        gait.update(0.0, feet, contacts)
        contacts["rb"] = False
        gait.update(0.21, feet, contacts)
        gait.update(0.5, feet, contacts)  # scheduled landing starts finite probe
        contacts["rb"] = True
        state = gait.update(0.6, feet, contacts)  # delayed, but still in the scheduled stance interval
        self.assertTrue(state["confirmed_support"]["rb"])
        self.assertNotIn("rb", gait.early_landing_hold_cycle)

    def test_touchdown_at_floating_cycle_boundary_keeps_next_scheduled_swing(self):
        nominal, feet = _gait_frames()
        gait = TripodGait(nominal, feet, np.array([0.0, 0.0, 0.12]), 0.0, _gait_settings())
        contacts = {leg: True for leg in LEG_ORDER}
        gait.update(0.0, feet, contacts)
        gait._unwrapped_phase = 1.5 - 2e-16  # rb phase is 1.9999999999999998
        gait._phase_time_s = 0.0
        gait.stance["rb"] = False
        gait.confirmed_support["rb"] = False
        gait.release_seen["rb"] = True
        gait.swing_end["rb"] = feet["rb"].copy()
        self.assertEqual(gait.phase_for_leg("rb", 0.0), 0.0)

        state = gait.update(0.0, feet, contacts)
        self.assertTrue(state["confirmed_support"]["rb"])
        self.assertNotIn("rb", gait.early_landing_hold_cycle)

        state = gait.update(0.700000001, feet, contacts)
        self.assertFalse(state["planned_support"]["rb"])
        self.assertFalse(gait.stance["rb"])

    def test_static_terrain_profiles_are_bounded_and_reproducible(self):
        step_vertices, step_faces, step = build_terrain({"kind": "low_step"})
        self.assertGreater(step_faces.shape[0], 2)
        self.assertIn(0.0, step_vertices[:, 2])
        self.assertAlmostEqual(float(np.max(step_vertices[:, 2])), 0.005, places=7)
        front_wall_normals = []
        for face in step_faces:
            points = step_vertices[face]
            if np.allclose(points[:, 1], 0.55) and np.ptp(points[:, 2]) > 0:
                front_wall_normals.append(np.cross(points[1] - points[0], points[2] - points[0]))
        self.assertTrue(front_wall_normals)
        self.assertTrue(all(normal[1] < 0 for normal in front_wall_normals))
        ramp_vertices, _, ramp = build_terrain({"kind": "ramp", "height_m": 0.015})
        self.assertAlmostEqual(float(np.max(ramp_vertices[:, 2])), 0.015, places=7)
        _, bars_faces, bars = build_terrain({"kind": "bars"})
        self.assertEqual(bars["triangles"], 36)
        self.assertEqual(bars_faces.shape[0], 36)
        rough_a = build_terrain({"kind": "rough", "seed": 42})[2]
        rough_b = build_terrain({"kind": "rough", "seed": 42})[2]
        self.assertEqual(rough_a["sha256"], rough_b["sha256"])
        with self.assertRaises(ValueError):
            build_terrain({"kind": "rough", "seed": True})


if __name__ == "__main__":
    unittest.main()
