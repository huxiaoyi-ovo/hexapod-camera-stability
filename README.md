# Hexapod Camera Stability Research

**Working title:** *Compensation-Aware Gait Adaptation for Camera-Stable Hexapod Locomotion on Uneven Terrain*  
**Target journal:** Machines (MDPI) · **Research mode:** simulation-first, hardware-assisted

This repository studies whether gait selection using predicted residual camera motion after a fixed body-pose controller can outperform selection by raw body disturbance. The scientific contributions remain hypotheses; no study performance result is claimed here.

## Current executable status

The canonical local runtime is a standalone Isaac Gym/PhysX diagnostic using a locally copied hexapod URDF and its active mesh dependencies. It creates one robot on a flat plane and commands a deterministic standing reference with explicit PD effort control. It logs measured root, joint, and foot-contact state plus commanded PD torques and command-clipping flags to a unique `data/raw/standalone_<timestamp>/` directory. It does not walk or validate gait stability, camera compensation, image motion, or hardware behavior. The PD gains are uncalibrated; torque commands and clipping flags are not measured actuator response.

The older PCR-Net collector and `src/pcr_*` modules remain in the worktree as historical, inactive integration paths. The standalone runner does not require or read a PCR-Net checkout, its `legged_gym` code, weights, or configuration. Walking, body compensation, motor-delay identification, camera calibration, trained models, and experimental comparisons remain future research work. A corrected 6 s GUI stand diagnostic passed its configured gate and was visually accepted; it is operational bring-up evidence only. Unit tests and this stand diagnostic do not establish walking stability or camera improvement.

## Start and run

Read [the standalone simulator guide](docs/24_STANDALONE_SIMULATOR.md) for the local Isaac Gym environment, physical settings, diagnostic gate, and output interpretation. The direct entry point is `scripts/run_hexapod_pilot.py`, with explicit parameters in `configs/flat_stand.json`. Run offline checks with:

```bash
PYTHONPATH=src python -m pytest -q tests/
```

The historical [PCR PhysX pilot note](docs/23_PCR_PHYSX_PILOT.md) is preserved for provenance and is superseded for current execution by document 24.

Follow the [architecture and experiment contract](docs/25_ARCHITECTURE_AND_EXPERIMENT_CONTRACT.md), [paper writing guide](docs/26_WRITING_GUIDE.md), and [refactor verification record](docs/27_REFACTOR_VERIFICATION.md) for the four independence principles, reproducibility, claim boundaries, and verified evidence.

## Research and evidence

This repository has no general software release license; the owner has not decided how the research code should be released. The robot asset provenance and upstream license copy are documented separately in `assets/hexapod/` and do not grant a license for unspecified CAD assets.

The existing numerical camera metrics, gait prototypes, PCR adapters, and manuscript draft are research work in progress. Keep each result tied to its actual evidence source: offline checks, simulator traces, or hardware observations. Preserve failed trials and status; do not train from failed trials or label them successful. Verify camera optical-frame extrinsics before generating response windows. Do not put raw/private videos, credentials, checkpoints, or bulk datasets in Git.
