# Runtime Architecture Decision — Machines Camera-Stable Hexapod

Date: 2026-10-20
Status: ACCEPTED FOR INITIAL IMPLEMENTATION
Related platform: https://github.com/huxiaoyi-ovo/Pcrnet_hexapod

## Decision

**Keep Python/PyTorch + Isaac Gym GPU tensors for all simulation-first research and dataset generation. Implement a portable, deterministic control core; port the real-robot 50-Hz IK/gait/body-compensation segment to C++/Eigen only after the Python Isaac Gym closed-loop baseline and real-device timing profile are available.**

Do NOT delay actual PhysX gait rollouts to rewrite a working or incomplete research prototype in C++.

## Confirmed code boundaries

- PCR-Net `legged_gym/envs/hex_v4/hex_ground.py::step` consumes `(num_envs,18)` PyTorch actions and calls the actuator model and PhysX; the action/torque logic is already tensor-based.
- PCR-Net `src_real/interface/scripts/physical_robot/run_agent2.py` declares `LOWLEVEL_HZ=50.0` and uses Torch in a Python ROS callback. This is the **existing reference**, not a finished new gait controller.
- Existing `src/closed_loop_surrogate.py` uses a small, 22-column Ridge feature vector and independent regressors for raw cost, compensated cost and attained speed.
- Existing `src/pcr_sim_bridge.py` extracts individual Isaac Gym tensors using `detach().cpu().numpy()` and writes rows to CSV. This is acceptable for single-env correctness tests but may force repeated GPU/CPU synchronization during batched high-throughput rollouts; measure and refactor before scaling.
- The original PCR-Net project has a ROS C++ locomotion package (`src_real/locomotion/curv_adapt/`) whose kinematics and frame conventions can be referenced. Do not assume it implements this paper's new tripod/compensation loop.

## Deployment decomposition

### Simulation control path, Python/PyTorch

Physics state tensors on GPU -> batched gait phase and foot trajectory -> batched IK and stance/body correction -> action tensor [N,18] -> existing HexGround actuator+PhysX -> buffered GPU results -> infrequent batched host copies.

- Avoid per-environment Python loops in the high-frequency path.
- Keep image residual proxies on tensor arrays during broad parameter sweeps; render selected RGB validation sequences separately.
- Keep training and model-fitting in Python.
- Store configuration hash, candidate gait, terrain ID, seed, compensation flag, achieved speed and failure flag with each rollout.

### Real-robot control path, potentially C++/ROS Noetic

IMU / joint feedback -> deterministic phase accumulator -> stance/swing trajectories -> Roll/Pitch/Z compensation -> IK and workspace/safety checks -> existing hardware command interface.

The predictor is **inference-only** in deployment: train the Ridge models in Python, export normalization vectors and coefficients, and reproduce feature ordering and transforms in C++/Eigen. A small fixed-size numerical kernel need not embed Torch.

- Preserve the robot's emergency stop, watchdog, contact guard and safe-standing fallback.
- Never change the production PCR-Net Follow/Avoid pipeline for this paper.
- Validate output and timing equivalence on the same recorded input traces before actuating hardware.

## Control/update-rate assumptions (not benchmark results)

- 50 Hz low-level control: 20 ms period; reserve substantial headroom for sensing, ROS transport and actuator communication.
- 5–10 Hz gait selection: initial design choice, not a measured hardware capability. Keep phase continuous and switch only at safe gait phases.
- Camera: use measured timestamped D435i RGB/IMU data; nominal 30 Hz alone is not proof of synchronization.

## Benchmark and porting gate

Measure separately, both on the simulator GPU and on the actual Jetson/host deployment target:

1. Input acquisition, GPU->CPU sync and CSV writer cost (simulation).
2. Candidate feature construction + raw/closed-loop Ridge inference + sorting for representative candidate menu sizes.
3. Gait update + 18-joint IK + compensated reference generation.
4. Message serialization/publishing, feedback age, scheduling jitter and end-to-end command latency.
5. Runtime p50, p95, p99, worst case, missed deadlines, GPU throughput (env-steps/s) and memory usage.

First fix algorithmic complexity, unnecessary tensor synchronizations, and logging, then benchmark again. **Port a specific real-time function to C++ if its measured tail latency or jitter threatens its deadline**; do not rewrite based on language stereotypes.

Provisional acceptance targets requiring hardware validation:
- No missed 50-Hz control deadlines during trial sequences.
- Controller timing and peak scheduling jitter documented over representative runs.
- Inference output matches Python within agreed numeric tolerances, preserving the selected candidate and safety decisions.
- Safety fallback stays active even when the predictor returns NaN, no candidates remain, or an input sensor times out.

## Priority relative to paper submission

P0: One independent Isaac Gym gait rollout with actual 18DOF contact and motor dynamics.
P0: Fixed realistic body compensator and paired off/on rollouts.
P0: Truthful image-motion residual, actual-speed-matched controls and held-out predictor evaluation.
P1: Batched GPU trajectory buffer + throughput measurement.
P1: Real-time profiler + first frozen-controller RGB robot verification.
P2: Selective C++ port (IK/gait/compensator and, if needed, tiny Ridge inference), followed by equivalence and deterministic safety testing.

## Scientific guardrails

Improved code execution time is an engineering result, not a scientific novelty claim. The paper's novelty remains **predicting and selecting candidate gaits based on the residual image motion after achievable compensation**. Compare methods with the same gait menu, compensator, actuator model, achieved speed and compute budget.

Neither unit tests nor controller microbenchmarks establish gait performance or image stabilization. Only real PhysX rollouts and/or measured RGB/IMU evidence support Results.
