"""Focused checks for bounded RPZ compensation and its simulated camera frame."""
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from camera_pose_recorder import CameraPoseRecorder
from hexapod_body_compensation import BoundedBodyCompensator
from hexapod_camera_compensation import (CameraSpaceCompensator, camera_residual_twist,
                                         generalized_mass_matrix, uncontrollable_projection)
from hexapod_robot import LEG_ORDER, load_robot_model, stand_target_by_name, toe_positions_base
from hexapod_sim import PROJECT_ROOT, _walk_control_targets, load_config
from hexapod_walk import TripodGait, feet_world
from run_compensation_pilot import _camera_metrics_for_trace, nominal_camera_twist
from trace_metrics import camera_scene_from_json, image_motion_metrics, read_trace


class BodyCompensationTests(unittest.TestCase):
    def test_pilot_config_enables_only_supported_compensation_profile(self):
        config = load_config(ROOT / "configs/compensation_pilot.json")
        self.assertEqual(config["body_compensation"], {"enabled": False})
        self.assertEqual(config["terrain"]["kind"], "ramp")
        self.assertEqual(config["walk"]["speed_m_s"], 0.1)
        invalid = json.loads(json.dumps(config))
        invalid["body_compensation"]["enabled"] = "false"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_feedback_sign_slew_and_absolute_limits(self):
        quat = Rotation.from_euler("xyz", [0.2, -0.2, 0.0]).as_quat()
        controller = BoundedBodyCompensator(True, 0.02)
        corrections = controller.update(quat, np.zeros(3), np.array([0.0, 0.0, 0.1]),
                                        np.zeros(3), 0.2)
        self.assertLess(corrections["roll_rad"], 0.0)
        self.assertGreater(corrections["pitch_rad"], 0.0)
        previous = corrections
        self.assertGreater(corrections["height_m"], 0.0)
        self.assertLessEqual(corrections["height_m"], 0.0006)
        for _ in range(30):
            corrections = controller.update(quat, np.zeros(3), np.array([0.0, 0.0, 0.1]),
                                            np.zeros(3), 0.2)
            self.assertLessEqual(abs(corrections["roll_rad"] - previous["roll_rad"]), 0.006)
            self.assertLessEqual(abs(corrections["pitch_rad"] - previous["pitch_rad"]), 0.006)
            self.assertLessEqual(abs(corrections["height_m"] - previous["height_m"]), 0.000600001)
            previous = corrections
        self.assertLessEqual(abs(corrections["roll_rad"]), 0.05)
        self.assertLessEqual(abs(corrections["pitch_rad"]), 0.05)
        self.assertLessEqual(abs(corrections["height_m"]), 0.01)

    def test_world_gyro_is_rotated_to_body_for_rate_damping(self):
        yaw = math.pi / 2.0
        rotation = Rotation.from_euler("z", yaw)
        quat = rotation.as_quat()
        gyro_world = rotation.apply([0.01, 0.0, 0.0])
        controller = BoundedBodyCompensator(True, 0.02)
        output = controller.update(quat, gyro_world, np.array([0.0, 0.0, 0.1]),
                                  np.zeros(3), 0.1)
        self.assertLess(output["roll_rad"], 0.0)
        self.assertGreater(output["roll_rad"], -0.0004)
        self.assertAlmostEqual(output["pitch_rad"], 0.0)

    def test_pose_p_ignores_rate_feedback_but_keeps_angle_feedback(self):
        quat = Rotation.from_euler("x", 0.02).as_quat()
        slow = BoundedBodyCompensator(True, rate_feedback=False).update(
            quat, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.1]), np.zeros(3), 0.1)
        fast = BoundedBodyCompensator(True, rate_feedback=False).update(
            quat, np.array([3.0, -2.0, 1.0]), np.array([0.0, 0.0, 0.1]), np.zeros(3), 0.1)
        self.assertEqual(slow, fast)
        self.assertLess(slow["roll_rad"], 0.0)

    def test_angle_and_consistent_gyro_are_attenuated_at_12_5_hz(self):
        amplitude_rad = 1e-4

        def normalized_correction_amplitude(frequency_hz):
            controller = BoundedBodyCompensator(True, 0.02)
            outputs = []
            times = np.arange(1000) * 0.02
            for t in times:
                angle = amplitude_rad * math.sin(2.0 * math.pi * frequency_hz * t)
                gyro_x = (2.0 * math.pi * frequency_hz * amplitude_rad *
                          math.cos(2.0 * math.pi * frequency_hz * t))
                correction = controller.update(
                    Rotation.from_euler("x", angle).as_quat(), np.array([gyro_x, 0.0, 0.0]),
                    np.array([0.0, 0.0, 0.1]), np.zeros(3), 0.1)
                outputs.append(correction["roll_rad"])
            steady_times = times[250:]
            steady = np.asarray(outputs)[250:]
            phase = 2.0 * math.pi * frequency_hz * steady_times
            design = np.column_stack((np.sin(phase), np.cos(phase), np.ones_like(phase)))
            sine_coeff, cosine_coeff, _ = np.linalg.lstsq(design, steady, rcond=None)[0]
            return float(math.hypot(sine_coeff, cosine_coeff) / amplitude_rad)

        gait_band_gain = normalized_correction_amplitude(0.8)
        chatter_band_gain = normalized_correction_amplitude(12.5)
        self.assertGreater(gait_band_gain, 0.0)
        self.assertLess(chatter_band_gain, 0.3 * gait_band_gain)

    def test_disabled_compensation_is_zero_and_keeps_original_ik_targets(self):
        config = load_config(PROJECT_ROOT / "configs/flat_walk.json")
        model = load_robot_model(PROJECT_ROOT / config["robot_urdf"])
        q = stand_target_by_name(model, config["stand_reference_rad"])
        body = np.array([0.0, 0.0, 0.116])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        actual_feet = feet_world(model, q, body, quat)
        nominal_feet = {leg: toe_positions_base(model, q)["l_{}_toe".format(leg)] for leg in LEG_ORDER}
        contacts = {leg: True for leg in LEG_ORDER}
        settings = {"speed_m_s": 0.02, "frequency_hz": 0.55, "duty_factor": 0.67,
                    "swing_height_m": 0.015, "speed_ramp_s": 2.0,
                    "touchdown_timeout_s": 0.12, "max_stance_slip_m": 0.04}

        def make_gait():
            return TripodGait(nominal_feet, actual_feet, body, 0.0, settings)

        baseline = _walk_control_targets(model, make_gait(), 0.0, q, body, quat,
                                         contacts, q, actual_feet)
        disabled = _walk_control_targets(model, make_gait(), 0.0, q, body, quat,
                                         contacts, q, actual_feet,
                                         body_compensator=BoundedBodyCompensator(False))
        self.assertEqual(baseline[0], disabled[0])
        self.assertEqual(disabled[1]["body_compensation"], {"roll_rad": 0.0,
                                                              "pitch_rad": 0.0,
                                                              "height_m": 0.0})

    def test_camera_pose_metric_cancels_nominal_plus_y_and_detects_pitch_motion(self):
        camera, scene = camera_scene_from_json(ROOT / "templates/camera_scene.example.json")
        camera_config = load_config(ROOT / "configs/compensation_pilot.json")["compensation_pilot"]["camera"]
        p_bc = np.asarray(camera_config["base_to_camera_position_m"])
        q_bc = np.asarray(camera_config["base_to_camera_quaternion_xyzw"])
        r_bc = Rotation.from_quat(q_bc)
        nominal_world = np.array([0.0, 0.1, 0.0])
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "straight.csv"
            with trace.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("time_s", "root_x_m", "root_y_m", "root_z_m",
                                                            "roll_rad", "pitch_rad", "yaw_rad"))
                writer.writeheader()
                for index in range(41):
                    t = index * 0.02
                    writer.writerow({"time_s": t, "root_x_m": 0.0, "root_y_m": 0.1 * t,
                                     "root_z_m": 0.12, "roll_rad": 0.0, "pitch_rad": 0.0,
                                     "yaw_rad": 0.0})
            metrics = _camera_metrics_for_trace(trace, Path(directory), 0.0, 0.0, 0.1, p_bc, q_bc)
            self.assertEqual(metrics["status"], "computed")
            self.assertLess(metrics["primary_J_jitter_px_rms_per_frame"], 1e-9)

            recorder = CameraPoseRecorder(p_bc, q_bc)
            for index in range(41):
                t = index * 0.02
                rotation_world_body = Rotation.from_euler("xyz", [0.0, 0.05 * t, 0.0])
                nominal = nominal_camera_twist(rotation_world_body, nominal_world, r_bc)
                recorder.append(t, [0.0, 0.1 * t, 0.12], rotation_world_body.as_quat(), nominal)
            output = recorder.save_twist_csv(Path(directory) / "pitched_optical_twist.csv")
            t, actual, reference = read_trace(output)
            projected = image_motion_metrics(t, actual - reference, camera, scene)
            self.assertGreater(projected.jitter_px_rms_per_frame, 1e-3)

    def test_camera_space_mapping_is_finite_and_nominal_motion_is_zero(self):
        config_path = ROOT / "configs/camera_space_pilot.json"
        config = load_config(config_path)
        self.assertEqual(config["body_compensation"], {"enabled": True, "kind": "camera_space"})
        model = load_robot_model(ROOT / config["robot_urdf"])
        q = stand_target_by_name(model, config["stand_reference_rad"])
        mass = generalized_mass_matrix(model, q)
        self.assertTrue(np.all(np.linalg.eigvalsh(mass) > 0.0))
        pilot = config["compensation_pilot"]
        scene_data = json.loads((ROOT / pilot["camera_scene_template"]).read_text())
        controller = CameraSpaceCompensator(model, q, pilot["camera"], scene_data["camera"],
                                            scene_data["scene"], config["control"],
                                            config["actuator"], config["control_period_s"],
                                            config["physics"]["dt_s"])
        body = np.array([0.0, 0.0, 0.116])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        anchors = feet_world(model, q, body, quat)
        velocity = np.array([0.0, 0.1, 0.0])
        kwargs = dict(q_by_name=q, support={leg: True for leg in LEG_ORDER},
                      planned_body=body, yaw=0.0, anchors=anchors,
                      estimated_xyz=body, estimated_velocity=velocity,
                      imu_quaternion=quat, gyro_world=np.zeros(3),
                      nominal_velocity_world=velocity,
                      previous_applied_torque_by_name={name: 0.0 for name in q})
        correction, diagnostic = controller.update(**kwargs)
        self.assertEqual(diagnostic["support_count"], 6)
        self.assertGreaterEqual(diagnostic["mapping_rank"], 2)
        self.assertLess(diagnostic["predicted_pixel_before"], 1e-9)
        self.assertTrue(np.all(np.isfinite(correction)))

        pitched = Rotation.from_euler("y", 0.02).as_quat()
        kwargs["imu_quaternion"] = pitched
        kwargs["gyro_world"] = np.array([0.0, 0.05, 0.0])
        _, disturbed = controller.update(**kwargs)
        self.assertGreater(disturbed["predicted_pixel_before"], 1e-6)
        self.assertTrue(np.isfinite(disturbed["predicted_pixel_after"]))
        self.assertLessEqual(disturbed["predicted_pixel_after"],
                             disturbed["predicted_pixel_before"] + 1e-12)
        self.assertLessEqual(abs(controller.correction[0]), 0.05)
        self.assertLessEqual(abs(controller.correction[1]), 0.05)
        self.assertLessEqual(abs(controller.correction[0]), 0.006)
        self.assertLessEqual(abs(controller.correction[1]), 0.006)

    def test_camera_lever_arm_and_uncontrollable_projection_are_explicit(self):
        rotation = np.eye(3)
        omega = np.array([0.0, 0.0, 1.0])
        with_lever = camera_residual_twist(rotation, rotation, np.array([0.0, 0.18, 0.04]),
                                           np.zeros(3), omega, np.zeros(3))
        at_origin = camera_residual_twist(rotation, rotation, np.zeros(3),
                                          np.zeros(3), omega, np.zeros(3))
        np.testing.assert_allclose(with_lever[:3], [-0.18, 0.0, 0.0])
        np.testing.assert_allclose(at_origin[:3], np.zeros(3))
        mapping = np.array([[1.0], [0.0]])
        remaining = uncontrollable_projection(mapping, np.array([2.0, 3.0]))
        np.testing.assert_allclose(remaining, [0.0, 3.0], atol=1e-12)

    def test_camera_response_includes_delay_and_is_not_instantaneous(self):
        config = load_config(ROOT / "configs/camera_space_pilot.json")
        model = load_robot_model(ROOT / config["robot_urdf"])
        q = stand_target_by_name(model, config["stand_reference_rad"])
        pilot = config["compensation_pilot"]
        scene = json.loads((ROOT / pilot["camera_scene_template"]).read_text())
        controller = CameraSpaceCompensator(model, q, pilot["camera"], scene["camera"],
                                            scene["scene"], config["control"], config["actuator"],
                                            config["control_period_s"], config["physics"]["dt_s"])
        body = np.array([0.0, 0.0, 0.116])
        anchors = feet_world(model, q, body, np.array([0.0, 0.0, 0.0, 1.0]))
        q_map, _ = controller._support_map(q, body, 0.0, anchors, {leg: True for leg in LEG_ORDER})
        transition, input_step = controller._discrete_response(q_map)
        equilibrium = np.array([0.001, -0.0015, 0.0002])
        equilibrium_state = np.r_[equilibrium, np.zeros(6)]
        stationary = (transition @ equilibrium_state +
                      input_step @ np.r_[equilibrium, np.zeros(3)])
        np.testing.assert_allclose(stationary, equilibrium_state, atol=1e-12)
        before_delay = controller._step_response(np.array([1.0, 0.0, 0.0]), transition,
                                                  input_step, 8)
        after_delay = controller._step_response(np.array([1.0, 0.0, 0.0]), transition,
                                                 input_step, 16)
        np.testing.assert_allclose(before_delay, np.zeros(3), atol=1e-15)
        self.assertGreater(float(np.linalg.norm(after_delay)), 0.0)


if __name__ == "__main__":
    unittest.main()
