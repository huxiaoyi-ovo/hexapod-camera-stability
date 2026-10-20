# PCR-Net Isaac Gym integration contract (current)

**Base repository:** https://github.com/huxiaoyi-ovo/Pcrnet_hexapod

Do **not** use the unrelated `Grasp_hexapod` climbing controller as the locomotion backend for this paper.

## Verified entry points (GitHub main inspected 2026-10-20)

- `legged_gym/envs/hex_v4/hex_ground.py::HexGround.step(actions)` consumes `(num_envs,18)` normalized actions and advances PhysX with actuator/torque response.
- `HexGround._compute_torques` maps actions to target position by `default_dof_pos + action * cfg.control.action_scale`; the motor model and torque limits act during simulation.
- `env.root_states` stores world root position, xyzw quaternion, linear and angular velocity; `env.dof_pos`, `env.dof_vel`, `env.torques`, `env.contact_forces`, `env.feet_indices` are available.
- `src_real/locomotion/curv_adapt/scripts/curv_adapt_multi.py` contains reusable tripod gait and leg kinematics, but climbing/suction coupling should not be transferred blindly.
- `legged_gym/envs/hex_v4/hex_ground_config.py` has a D435i-like camera block. The numerical mounting defaults are **not** a verified RGB optical-frame extrinsic calibration.

## Implemented in this repository

- `src/tripod_gait.py`: nominal foot-reference generator **only** (not IK, balance controller, or fully executed gait).
- `src/pcr_action_interface.py`: absolute joint-target to normalized 18D action mapping, joint position/rate preliminary screening.
- `src/pcr_sim_bridge.py`: simulator-state to CSV recorder; no action generation.
- `src/pcr_sim_windows.py`: motion metric labels from physics logs; requires a verified optical-frame calibration and a reproducible reference camera motion.
- `src/closed_loop_surrogate.py`: offline response predictor; must be fitted on real **simulation** rollouts before selecting candidates.

## Critical missing integration

1. Validate joint order, axes and nominal foot coordinates against the PCR-Net hexapod URDF/FK/IK.
2. Generate physically feasible contact-aware stance/swing joint targets under body Roll/Pitch/Z corrections.
3. Run compensated and uncompensated **independent** rollouts with matched initial seeds; record all failures.
4. Calibrate/verify the effective actuator response and camera optical-frame transform.
5. Save reproducible seed, terrain instance, controller gain and code revision with each rollout.

This file supersedes the former Grasp integration notes; the host integration protocol is in `docs/18_LOCAL_ISAAC_GYM_INTEGRATION_TASK.md`.
