# SuperMimic-inspired camera-stable hexapod policy: exploratory research contract (2026-10-23)

**Status:** research proposal, NOT an implemented or validated policy. Working branch: `research/supermimic-vision`. Existing `test` branch and `configs/pcr_expert_pd.json` remain untouched. Target venue is intentionally unfixed.

## Research question

Can an ExpertGround-prior hexapod policy learned via differentiable short-horizon world-model imagination reduce **unwanted image-space motion** during locomotion across terrain/contact changes, without sacrificing achieved speed, passage, actuator limits, or safety? Does explicit treatment of **contact-transition uncertainty and actuator lag** improve the reliability of world-model gradients compared with an otherwise matched ordinary SuperMimic-style learner and PPO?

This is a hypothesis, not a novelty or success claim. Simple reward shaping and transferring SuperMimic from G1 to a hexapod are not standalone contributions.

## Prior art / positioning

* Liao et al., *SuperMimic* (CoRL 2026), https://arxiv.org/abs/2610.28480: learned differentiable dynamics, k=8 autoregressive fit, H=32 actor-critic imagination; its motion tracking benchmark and GPU speed numbers do **not** transfer automatically to this application.
* Li et al., *Hold My Beer / SoFTA* (CoRL 2025), https://proceedings.mlr.press/v305/li25i.html: humanoid walking + end-effector/video stabilization by slow/fast RL, so 'RL camera stabilization' is not new.
* Lai et al., *World Model-based Perception for Visual Legged Locomotion*, https://arxiv.org/abs/2409.16784: already uses world models for visual locomotion.
* Schwarke et al., *Learning Deployable Locomotion Control via Differentiable Simulation*, https://proceedings.mlr.press/v305/schwarke25a.html: identifies contact gradient issues and hardware transfer.
* Hexapod CPG + RL: https://www.sciencedirect.com/science/article/pii/S092523122402099X and https://www.mdpi.com/2076-0825/15/1/33 .
* Unofficial SuperMimic reproductions (not official author releases; verify individual licensing/metrics): https://github.com/zita-ch/supermimic_vibed ; https://github.com/CuboiLeo/SuperMimic-Reproduced .

## Freeze execution and model boundaries

* Preserve original ExpertGround gait generation, leg-group switching and nominal 18-joint targets. Do not modify vendored Expert implementation to force a periodic CPG. Existing compensated body-correction adapter can be studied but must not be presumed validated.
* Preserve PD/MIT-style actuator model, command delay, 5 rad/s measured-joint-speed hard acceptance, joint position limits, torque limits and support rules. Record and diagnose 21 over-limit samples / 5.16 rad/s peak and transient support warnings reported from the *local*, not GitHub-committed, PD slope pilot. PD is NOT motor-calibrated. Do not silently re-label an unsafe rollout as safe.
* The existing GPU Isaac Gym simulator is a **black-box data generator**. SuperMimic gradients are through a **separate learned PyTorch world model**, not through PhysX. Existing Isaac Gym Python 3.8/Torch 1.8 must not be upgraded in place; keep new training dependencies in a separate environment.
* Collect complete simulator transitions with explicit controller state, gait stage/phase, executed rather than merely requested actions, measured q/dq, body pose/twist, contact events, actuator delayed reference and torque state, and terrain/friction metadata. Log camera extrinsics/calibration.
* At deployment, actor input can include only past/current onboard-observable quantities: joint proprioception, IMU/gyro, action history, estimated contact, commanded velocity, planned gait stage; no true friction, future contact forces, privileged terrain height or perfect root velocity. Simulator context may train the world model but is NOT leaked to the deployed actor.

## Minimal proposed control architecture

* ExpertGround reference: `q_E(t)`.
* Start with **bounded, low-dimensional residual actions** `a = [delta_roll, delta_pitch, delta_z]` routed through the existing body-correction/IK interface; compare against unchanged ExpertGround and a manually tuned pose-feedback controller. Add bounded swing-leg residual foot-height adjustments only **after** support/actuation feasibility is verified, never as an unbounded arbitrary 18-DoF torque controller.
* `q_command = IK_or_joint_adapter(q_E, a)` followed by the *same* PD, delay and physical simulator across all methods.
* World model `f_w(s_t, a_t, c_t, xi) -> (s_{t+1}, p(contact_{t+1}), p(failure))`. Predict normalized state increments / actual actuator response. `c_t` is the planned phase/support mask; actual contacts are training labels and only estimated current contacts enter the online actor.
* Start with the plain SuperMimic-style k=8 autoregressive world model + H=16/32 critic-bootstrapped BPTT. Contact-conditioned dynamics and actuator-state features are **candidate ablations**, not assumed novel. If contact crossings give unreliable gradients, study an uncertainty-aware horizon / ensemble disagreement penalty or confidence-gated policy update against a plain-world-model control.
* Do not assume H100 3-hour throughput. Benchmark training speed, GPU memory, data throughput and action feasibility on available devices.

## Vision task reward, not a generic torso smoothing reward

Use a calibrated rigid camera at known base-to-camera pose. For stationary virtual 3D scene points with *fixed* depths and intrinsics, derive differentiable projected image motion from predicted camera pose/twist. Subtract expected smooth commanded ego-motion (including forward parallax and commanded yaw); penalize only high-frequency/uncommanded residual flow:
`J_image = sqrt(mean_{frame, landmarks}( || u_actual - u_nominal ||_2^2 ))` in pixels/frame.
At training time this can be computed from simulated camera kinematics and known virtual landmark depths without RGB rendering every physics step. Maintain a separate secondary exposure-time smear proxy if needed; never call angular rate 'true image clarity'. At final evaluation, validate against rendered/real RGB static-feature reprojection error or residual optical flow, not solely the analytic proxy.

Train reward or loss with visible terms: camera residual, commanded forward/yaw velocity tracking, support/passage/fall, joint velocity/position bounds, torque saturation, foot slip, action smoothness and achieved speed. Crucially the agent must not get a low-camera-motion score just by stopping, crouching or evading obstacles; report performance at *matched achieved speed* and conditioned on passage.

## Execution gates (no architecture churn)

**Gate A: executable reference + observation contract.** Read local PD config and recent traces on the actual development machine, verify measured speed limits and contact semantics, export 50 Hz transitions with 400 Hz actuator diagnostics; keep failures.

**Gate B: is learning necessary?** Build same-action/same-reward reference baselines: untouched ExpertGround, Expert+bounded pose-feedback, Expert+PPO residual. Get a robust slope/rough-terrain replay. If PPO cannot reduce residual image motion at matched forward progress, do NOT launch expensive world-model runs before diagnosing action authority and reward definition.

**Gate C: world model feasibility.** Fit offline world model to train terrains / seeds. Report 1, 8, 16 and 32-step held-out errors for body angular motion, camera residual flow, joint state and contact-event timing; report joint/contact predictive uncertainty; compare contact-conditioned versus plain model. If horizon is inaccurate around contact changes, shorten or confidence-gate model rollouts; do not rely on spurious BPTT gradients.

**Gate D: train and compare.** Once Gate B and C pass, close the SuperMimic-style loop: simulator collect -> replay -> world-model update -> H-step BPTT policy+critic -> simulator evaluation. Only judge policies on real PhysX rollouts, not imagined score. Reuse exact same observation/action/reward/speed/terrain splits for all methods.

**Gate E: paper evidence.** Distinct independent terrain realizations/seeds; multiple training seeds; known train/validation/test split by terrain and seed; baseline fairness and safety; transfer across slopes, steps, obstacle bars, rough terrain and actuator delay/friction perturbations; one simple quantitative real-robot RGB sequence if feasible, with uncalibrated actuator limitations explicitly noted.

## Minimum paper comparison matrix

1. ExpertGround only.
2. ExpertGround + fixed feedback compensation.
3. ExpertGround + PPO residual (same action and reward).
4. ExpertGround + ordinary SuperMimic-style world-model residual policy.
5. Proposed contact-/actuator-aware world model with vision-space objective.
6. Ablations: remove vision-space objective, remove contact conditioning, remove actuator delay state, remove uncertainty guard (if implemented).

Report: RMS residual image displacement (px/frame), video/static-feature error, camera angular jitter as secondary, matched forward speed, success, slip, support/velocity/torque-limit violations, mean episode return, simulator steps to target score, wall-clock and GPU-hours. 0 runs is not a success.

## Immediate first deliverable

Do NOT refactor the existing simulator or change the committed `test` branch. Implement a *read-only log conversion and diagnostic* that ingests the local `pcr_expert_pd.json` and actual `trace.csv` and emits a time-aligned 50 Hz dataset with verified raw q/dq/contact/actuator traces, plus a reproducible image-space metric test on a synthetic stationary landmark set. Offline unit tests and numerical finite-difference checks must pass before collecting large datasets.

## Status

This file documents an **exploratory pivot**. No SuperMimic algorithm or hexapod RL strategy has yet been trained/evaluated here. Local machine files under `/home/artrc/...` are not visible via the current chat; GitHub `test` last checked at `92ccc4011bdc2a9b09984e7844b3fab287ba5585`.
