# Architecture and experiment contract

This is the active implementation contract for the camera-stability project. It separates imported robot description from project-owned runtime and control, and separates a working simulator diagnostic from the research method described in the manuscript. The PCR-Net repository is the source of the local robot asset bundle only. Runtime entry point: [the standalone simulator guide](24_STANDALONE_SIMULATOR.md). Manuscript rules: [the writing guide](26_WRITING_GUIDE.md). Current evidence: [the refactor verification record](27_REFACTOR_VERIFICATION.md).

## Four independence principles

1. **Robot-model independence.** Copy only the URDF and its necessary visual/collision mesh dependency closure. At runtime, read joint identity, order, limits, mass, and inertia from the local model. The project implements and validates its own motor response, torque limits, and delay model. Upstream values may inform an explicitly documented reference, but upstream model/config code and learned actuator networks are not runtime dependencies.
2. **Simulation independence.** Own the Isaac Gym/PhysX task and call the simulator API directly. Set and record time step, contacts, friction, terrain, and initial conditions explicitly. Do not use `HexGround`, `legged_gym`, or a PCR task registry.
3. **Controlled experiments.** Do not inherit curriculum, rewards, or implicit episode lifecycle. A trial has an explicit, logged initialization/reset at its start; the baseline has no random resets. Retain terminal, failed, and interrupted traces. Random terrain, friction, or actuator changes are useful only as separately configured, seeded, logged experiments, never as framework defaults.
4. **Control independence.** Gait generation, FK/IK, body compensation, prediction, online scheduling, and acquisition belong to this repository. PCR code is reference material only, not an imported runtime or implementation dependency.

## Executable boundary

The current executable is a single-robot flat-ground standing diagnostic. It loads the local URDF/mesh bundle, maps DOFs by simulator-reported names, initializes a fixed stand target, applies explicit effort PD, and records measured root, joint, and contact state plus commanded PD torques and command-clipping flags. These flags are not measured actuator torques or validated physical saturation. Its configuration and import path are documented in [document 24](24_STANDALONE_SIMULATOR.md).

The following research components are **not yet connected or validated**: alternating-tripod locomotion, foot-target generation into joint targets, contact-aware FK/IK walking, fixed body compensation, motor delay/identification, camera calibration, gait-response prediction, online gait scheduling, paired ON/OFF data collection, and multi-terrain evaluation. Existing prototypes or unit tests for an individual module do not change this execution boundary. Do not label stand diagnostics or synthetic tests as gait, compensation, or camera evidence.

## Trial initialization, reset, and controlled randomization

Each trial begins with an explicit reset/initialization operation. Before the first physics step, record the resolved base pose and quaternion, all linear/angular velocities, every named joint position and velocity, terrain identity/hash, and all applied physical parameters. When implemented, initialization must also cover and log gait phase, controller and integrator state, predictor history, actuator delay-buffer contents, and RNG state. Store the experiment configuration, code revision, local asset hashes, simulator/Python/library versions, seed, and output location beside the trace. A fresh, unique output directory is required for each execution.

The flat-stand baseline uses one fixed configuration and no random draws. For future controller-on/off pairs, run two independent executions from the same saved initial state, terrain realization, gait parameters, target speed, physical configuration, and seed, using the same documented preconditioning/settling protocol; the compensator switch is the intended condition difference. Do not silently retry a failed start or reseed it. Later trajectories may diverge, so match and report **achieved** speed and other outcomes rather than claiming samplewise-identical motion. Assign both members of a pair to the same train/validation/test partition. Group partitions by terrain realization and seed, not by frames.

Randomization is an explicit experiment, not an environment default. Its versioned configuration must name the parameters and ranges, sampling distributions, seed and generator, and the sampled values for each trial. Save generated terrain or an immutable terrain identifier plus content hash. Record the resulting friction and actuator parameters, delay, and complete initial state. A seed alone does not establish bitwise reproducibility: record backend/device/library versions and assess repeatability for the chosen simulator. Do not mix randomized trials into the deterministic baseline without labeling the condition.

Automatic reset after a fall, safety stop, timeout, viewer close, or other terminal event is prohibited. Stop the trial, preserve partial trace and status, and start a new uniquely identified trial only through an explicit new-run request. Do not concatenate pre- and post-reset states into a continuous trial or train on failed/standing diagnostic traces.

## Data and comparison guardrails

- Separate requested commands from measured state and achieved speed; record both.
- ON/OFF conditions use the same model, configuration, initial state, terrain, gait, command, and controller parameters except for the declared compensation switch. Report run failures and differences in achieved speed.
- Only pre-decision information may enter a future online selector. Future contacts, exact terrain ground truth unavailable to the robot, and outcomes from another rollout are offline labels or diagnostics, never decision-time features.
- Keep calibration, train, validation, and held-out evaluation data separate. Never split correlated frames randomly across those partitions.
- A diagnostic status is operational metadata, not a study result. Claims about walking, compensation, image motion, or hardware require the corresponding validated executable path and evidence.

## Near-term sequence

The next executable step is project-owned gait/stance/contact control and FK/IK connected to the local model. Then identify motor response and delay; implement a fixed compensator and verify its achieved response; calibrate camera extrinsics and image metrics; only then run controlled paired conditions and train/evaluate predictors. Keep each stage independently testable and retain failed runs.
