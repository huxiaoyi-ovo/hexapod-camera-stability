# Architecture and experiment contract

This is the active implementation contract for the camera-stability project. It separates imported robot description from project-owned runtime and control, and separates a working simulator diagnostic from the research method described in the manuscript. The PCR-Net repository is the source of the local robot asset bundle only. Runtime entry point: [the standalone simulator guide](24_STANDALONE_SIMULATOR.md). Manuscript rules: [the writing guide](26_WRITING_GUIDE.md). Current evidence: [the refactor verification record](27_REFACTOR_VERIFICATION.md).

## Four independence principles

1. **Robot-model independence.** Copy only the URDF and its necessary visual/collision mesh dependency closure. At runtime, read joint identity, order, limits, mass, and inertia from the local model. The project implements and validates its own motor response, torque limits, and delay model. Upstream values may inform an explicitly documented reference, but upstream model/config code and learned actuator networks are not runtime dependencies.
2. **Simulation independence.** Own the Isaac Gym/PhysX task and call the simulator API directly. Set and record time step, contacts, friction, terrain, and initial conditions explicitly. Do not use `HexGround`, `legged_gym`, or a PCR task registry.
3. **Controlled experiments.** Do not inherit curriculum, rewards, or implicit episode lifecycle. A trial has an explicit, logged initialization/reset at its start; the baseline has no random resets. Retain terminal, failed, and interrupted traces. Random terrain, friction, or actuator changes are useful only as separately configured, seeded, logged experiments, never as framework defaults.
4. **Control independence.** Keep PCR runtime code out of the executable path. Project-owned algorithms and explicitly reused generic controllers are permitted; preserve the reused source and its license.

## Executable boundary

The default executable remains the single-robot flat-ground stand diagnostic with its historical 400 Hz explicit PD path. A separate low-speed Python tripod profile has completed a multi-cycle flat-ground run. The OpenSHC gait backend is connected, but has not passed continuous physical acceptance; its safety-stopped attempts and traces remain failures. These outcomes are configuration-specific simulator evidence, not a general walking or camera-stability result.

Walk profiles update gait and joint references at 50 Hz and use an explicit MIT-style position-target actuator model at 400 Hz with 400 Hz PhysX. The modeled torque is not sensor-measured actuator torque. The local URDF remains authoritative for geometry, mass, inertia, and limits. Generic mature control code may be reused when its source and license are recorded; PCR runtime code, `HexGround`, `legged_gym`, upstream configurations, and learned actuator networks remain excluded.

Fixed body compensation, validated actuator calibration, camera calibration, predictor training, online camera-aware scheduling, paired compensation ON/OFF data collection, and multi-terrain evaluation are not connected or validated. Do not infer these results from stand or gait prototypes.

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

## Formal frozen scope (2026-10-21)

The research method has four modules. (1) A parameterized, time-driven tripod gait uses continuously accumulated phase, frequency, duty factor, stride length, and swing height; it does not add a CPG or reinforcement learning. (2) Terrain adaptation confirms touchdown, uses finite probing, protects support, and adjusts landing height; it remains separate from body compensation and does not attempt complex terrain planning. (3) Bounded roll, pitch, and height feedback compensation feeds the existing IK and 18-joint PD/MIT-style limited actuator model, including its explicit delay. Compensation has an independent ON/OFF switch; this module does not add MPC. (4) The paper's core contribution is constrained selection among candidate gait parameters to predict and reduce residual camera motion after compensation. Modules 1–3 are the control foundation for module 4.

Execution order is fixed:

1. **Reliable continuous walking.** Demonstrate repeatable continuous walking over four specified terrain classes: low steps, ramps, repeated obstacle bars, and randomized uneven ground. Specify terrain difficulty in advance and keep it within the available workspace, swing height, and actuator capability. Any online frequency adjustment must preserve continuous gait phase. The controller may use only information available to the robot, not privileged simulator state. Evaluate measured displacement, achieved speed, passage, and support; preserve failed and interrupted runs.
2. **Closed-loop compensation.** On the same terrain-adaptation policy, compare compensation ON and OFF. Record body and camera residual motion and passage outcomes under paired, otherwise matched conditions.
3. **Compensation-aware gait selection.** Add a lightweight predictor and constrained candidate search. Compare against fixed gait parameters and ordinary scheduling at matched achieved speeds; evaluate residual motion, prediction validity, and computation cost.

The order is reliable continuous walking → closed-loop compensation → compensation-aware scheduling. Compensation OFF keeps terrain adaptation enabled. Compensation ON receives no additional information, different low-level policy, or more permissive passage threshold. Report achieved speed and passage together so that apparent benefit from slowing down is visible.

Potential controller inputs are encoder joint state, IMU, estimated foot kinematics, valid D435i depth, and deployable estimators; listing them does not claim they are integrated or validated. Simulator contact forces and privileged terrain truth are diagnostics only and must not enter final control. OpenSHC is a control-design reference only; further ROS/C++ runtime-adapter expansion is out of scope. The current tree still contains an OpenSHC prototype and simulator-truth control paths. The first stage's four-terrain continuous-walking acceptance is not complete; frozen requirements must not be reported as completed work.

Reversal of candidate rankings is not a success condition; preserve a result with zero reversals. If rankings remain unchanged and the proposed selector cannot outperform simpler methods, narrow the core innovation claim accordingly. This is an interpretation rule for eventual results, not an additional current validation task.
