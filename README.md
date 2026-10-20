# Hexapod Camera Stability — Machines Research

**Working title:** *Compensation-Aware Gait Adaptation for Camera-Stable Hexapod Locomotion on Uneven Terrain*  
**Target journal:** Machines (MDPI) · **Research mode:** simulation-first, hardware-assisted  
**Research platform:** [huxiaoyi-ovo/Pcrnet_hexapod](https://github.com/huxiaoyi-ovo/Pcrnet_hexapod)

## Research question

For a rigidly mounted camera on an uneven-terrain hexapod, can gait parameters be selected using **predicted residual image motion after the response of an actual, fixed body-pose compensator**, rather than selecting the gait with the smallest raw body disturbance?

The two working technical contributions are (1) gait-conditioned prediction of disturbance and closed-loop compensation response, and (2) real-time constrained scheduling of a small candidate gait set using predicted *residual* image motion and locomotion constraints. Both are **hypotheses under study**, not verified performance claims.

## Scope

- Reuse the PCR-Net hexapod model, Isaac Gym/PhysX, and D435i/ROS hardware interface.
- The low-level gait generator may be redesigned; no need to retrain PCR-Net Follow/Avoid.
- Generate training/validation/test data **in simulation only**. Split by terrain instances and seeds, not neighboring frames.
- Use fixed body compensation controller with measured/simulated bandwidth, contact, joint, and motor constraints.
- Maintain comparable *achieved* speeds for all baselines. Compare residual image motion, not total scene optical flow.
- Use short hardware sequences only for frozen-policy execution and RGB/IMU validation.

## Current engineering status (2026-10-20)

**Available:** numeric proof-of-interface code for a parameterized tripod gait, pinhole image-motion residual calculation, candidate scoring, simulated state logging, trial-window construction, motor-action conversion, and lightweight response surrogate; LaTeX manuscript work draft; unit tests.

**New, unexecuted P0 simulator pilot:** URDF-matched toe IK, continuous-phase gait, bounded body-pose correction and single-/paired-run launcher are now implemented. They must still be exercised in the original PCR-Net Isaac Gym environment. See [P0 PhysX pilot instructions](docs/23_PCR_PHYSX_PILOT.md), [collector](integration/pcrnet/collect_camera_stable_gaits.py) and [paired launcher](scripts/run_pcr_paired_pilot.py). Latest GitHub Actions offline test suite passed (41 tests); it is not a physics validation.\n\n**Not yet available:** working PCR-Net gait/IK/stance compensation closed-loop rollout in Isaac Gym; validated contact/motor response calibration; learned surrogate trained on actual Isaac Gym rollouts; any study performance numbers or real-robot experiments.

The repository does **not** contain fabricated Results. A successful unit test is not evidence of gait stability or camera improvement.

## Start here

1. Read [simulation-first experiment protocol](docs/17_SIMULATION_FIRST_RESEARCH_PROTOCOL.md).
2. Read [PCR-Net local integration task](docs/18_LOCAL_ISAAC_GYM_INTEGRATION_TASK.md).
3. Read [revised core method](docs/16_REVISED_CORE_METHOD_AND_EXECUTION.md).
4. Implement gait/IK and a fixed compensation controller inside a separate experimental task on `Pcrnet_hexapod`. Do not modify the mature PCR-Net Follow/Avoid policy pipeline.
5. Record paired compensation-on/off trials, train and evaluate the residual response model, and run speed-matched baselines.

## Repository layout

```text
src/           Numerical algorithms, camera metrics and PCR simulation adapters
scripts/       Log processing and response-model evaluation
integration/   Adapter contracts for PCR-Net
paper/         LaTeX manuscript work draft (Results intentionally incomplete)
docs/          Research protocol, method, references, integration tasks
tests/         Offline numeric and interface tests
figures/       Method diagram assets, not experimental evidence
templates/     Trial, camera and paper metadata examples
```

## Run offline unit tests

```bash
python -m pip install -r requirements-core.txt
PYTHONPATH=src python -m pytest -q tests/
```

These tests do not require Isaac Gym. Running the dynamics integration requires installing and configuring the original PCR-Net environment separately.

## Data and paper ethics

- Do not commit robot credentials, API tokens, raw human videos, private identifying data or unpublished PCR-Net review material to a public repository.
- Keep model checkpoints and bulk motion/video datasets outside Git unless later deliberately versioned.
- Only quote measured/simulated results when complete reproducible data are available.
- No license is granted until the owner decides how the research code should be released.
