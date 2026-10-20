# P0: PCR-Net PhysX Single-Robot Gait Pilot

Date: 2026-10-20 · Code ready for local PCR-Net integration, **not executed in Isaac Gym**.

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