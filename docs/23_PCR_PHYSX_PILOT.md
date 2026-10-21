> **Historical execution note — superseded for current runs.** This document preserves the original PCR-Net integration instructions and results for provenance. The canonical local simulator is now the standalone flat-stand diagnostic in [docs/24_STANDALONE_SIMULATOR.md](24_STANDALONE_SIMULATOR.md). The PCR collector and `src/pcr_*` code remain historical/integration work; they are not the standalone runtime. Scientific evidence and failure-status cautions below remain relevant to their original context.

# P0: PCR-Net PhysX Single-Robot Gait Pilot

Date: 2026-10-21 · Code integrated with the existing PCR checkout; CPU PhysX startup and data capture were executed, but the standing gate failed before walking was validated.

## What was implemented

- `src/pcr_urdf_ik.py`: reads **the official `cfg.asset.file` URDF**, checks its yaw/pitch leg-chain assumptions, constructs exact toe FK/IK and maps the 18 real `dof_names` without assuming a sorting order. Explicitly distinguishes the URDF toe reach from old geometry configuration constants.
- `src/tripod_gait.py`: preserves old test API and adds an unwrapped phase accumulator for future online frequency changes.
- `src/pcr_trial_controller.py`: creates foot reference, bounded Roll/Pitch/Z correction, IK, and action conversion. It is a **minimal work-in-progress controller**; safety/performance must be established by physics.
- `integration/pcrnet/collect_camera_stable_gaits.py`: creates **one** PCR HexGround physics environment, commands a gentle forward ramp, runs one trial, and uses the existing `PcrSimTap` to record actual root/joint/torque/contact states. Stop and mark failed if the simulator terminates or constraints fail. `--controller off/on` selects the one fixed controller without touching the high-level PCR Follow/Avoid logic.
- `scripts/run_pcr_paired_pilot.py`: invokes separate PCR simulator instances with matching seed, gait and speed, once without and once with compensation. Only reports a validated pair after both traces succeed.

## Run from existing Isaac Gym environment

With `PCR_ROOT` pointing to your local **Pcrnet_hexapod** checkout and `RESEARCH_ROOT` to this repo:

```bash
export PCR_ROOT=/home/hxy/RL_GYM_PROJECTS/RL_hexapod_gym
export RESEARCH_ROOT=/path/to/hexapod-camera-stability
cd "$RESEARCH_ROOT"
# In a separate offline-test environment if needed (do not upgrade NumPy/Torch inside Isaac Gym blindly):
python -m pip install -r requirements-core.txt
python -m pytest -q tests/
# The PhysX pilot below must run in the *existing compatible Isaac Gym environment*.

# First inspect the official URDF and existing command interface. Then single run:
python integration/pcrnet/collect_camera_stable_gaits.py \
  --pcr-root "$PCR_ROOT" --research-root "$RESEARCH_ROOT" \
  --output-root "$RESEARCH_ROOT/data/raw/pilot01" \
  --controller off --pair-id pilot01 --headless --steps 400

# After checking off-run status (and with a NEW output path/pair ID), run a fresh pair:
python scripts/run_pcr_paired_pilot.py \
  --pcr-root "$PCR_ROOT" --research-root "$RESEARCH_ROOT" \
  --output-root "$RESEARCH_ROOT/data/raw/pilot02" \
  --pair-id pilot02 --headless --steps 400
```

For users preferring the requested script path in PCR-Net itself, copy
`integration/pcrnet/collect_camera_stable_gaits.py` to
`Pcrnet_hexapod/legged_gym/scripts/collect_camera_stable_gaits.py`;
`--research-root` is still required. The unmodified PCR-Net environment remains the runtime.

## Pre-Paper acceptance criteria

1. `status.json` reports a completed **PhysX** trial (not synthetic numeric tests) and CSV stores actual observed root poses, measured foot contact, 18 joints, actual torques, planned supports, velocity commands, gait ID, seed and controller status.
2. Robot does not reset unexpectedly; if it falls, terminates or violates safety checks, preserve the partial CSV and failed status.
3. Verify actual forward progress, no persistent toe slip and no actuator saturation on the flat trial. No stability claim is allowed before that.
4. Compare the paired off/on traces at **achieved** speed. A successful log does not imply compensation is helpful.
5. Make sure camera **optical-frame extrinsics** are verified before running `build_pcr_sim_windows.py`; the checked-in calibration example is explicitly UNVERIFIED. Never mark it verified just to let the script run.
6. Do not launch multi-terrain data collection until these checks pass, and do not train a surrogate on failed trials.

## Known limitations and next upgrades

The pilot is intentionally **NumPy/single-env**, incurs CPU↔GPU synchronization for each log step, does not yet anchor stance toes in world coordinates, uses a bounded proportional pose correction with no identified response bandwidth, and uses the original URDF geometry but still requires coordinate/foot-ID verification in Isaac Gym. This is a test harness, not a publishable complete gait controller.

The next engineering step after obtaining first real physics logs is: world-anchored stance + transition-safe contact control, followed by batched PyTorch GPU actions and buffered logging. A contact-specific speed/force guard and real actuator-response randomization must precede multi-terrain experiments.

## Local verification (2026-10-21)

The existing `/home/artrc/miniconda3/envs/grasp_hexapod` environment runs Python 3.8.20, Torch 1.8.1+cpu and NumPy 1.23.5. Reuse it without installing or upgrading packages. For gymtorch, add its existing `bin` directory to `PATH` and use a temporary extension cache; no `LD_LIBRARY_PATH` change was needed:

```bash
cd /home/artrc/hexapod-camera-stability
ENV=/home/artrc/miniconda3/envs/grasp_hexapod
export PATH="$ENV/bin:$PATH"
export TORCH_EXTENSIONS_DIR=/tmp/physx_pilot_torch_extensions

"$ENV/bin/python" -m pytest -q tests/  # 41 passed

# Use a new timestamped output directory for every run.
RUN_DIR=/home/artrc/hexapod-camera-stability/data/raw/physx_cpu_off_YYYYMMDDTHHMMSS
"$ENV/bin/python" integration/pcrnet/collect_camera_stable_gaits.py \
  --pcr-root /home/artrc/Pcrnet_hexapod \
  --research-root /home/artrc/hexapod-camera-stability \
  --output-root "$RUN_DIR" --controller off --pair-id pilot01 --seed 42 \
  --steps 400 --frequency-hz 1.1 --duty-factor 0.58 --swing-height 0.025 \
  --touchdown-softness 0.5 --forward-mps 0.08 --ramp-s 1.0 \
  --sim-device cpu --headless
```

The verified run at `data/raw/physx_cpu_off_20261021T1240b/` used CPU PhysX (GPU pipeline disabled) and preserved 301 stand samples over 6.02 s, but failed the settle gate before walking; it is not a validated gait trial. The GPU PhysX initialization path previously terminated with a segmentation fault. No OFF/ON pair has been run. Camera optical extrinsics remain uncalibrated, so optical response windows are not validated.
