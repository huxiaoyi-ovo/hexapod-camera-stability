# Repository Guidelines

## Runtime boundary

The canonical local simulator is the standalone flat-ground stand diagnostic in `scripts/run_hexapod_pilot.py`, using `configs/flat_stand.json` and local assets in `assets/hexapod/`. It must not import or read `Pcrnet_hexapod`, `legged_gym`, PCR model weights, or upstream runtime configuration. The legacy `integration/pcrnet/` and `src/pcr_*` paths are preserved historical/integration work and are not required by the standalone runner. Keep those research modules separate.

The current standalone runner demonstrates only deterministic standing in Isaac Gym/PhysX. Each run has an explicit recorded trial-start initialization; automatic resets, walking, training, rewards, curriculum, learned actuator models, and validated camera compensation are absent. Treat physics and camera claims as unverified until supported by preserved simulator or hardware evidence.

## Layout and commands

- `src/hexapod_robot.py`: pure NumPy/XML metadata, named joint mapping, FK, and torque calculations.
- `src/hexapod_sim.py`: one-environment standalone Isaac Gym execution and trace logging.
- `scripts/`, `configs/`, `assets/`, and ignored `data/raw/`: entry points, parameters, model dependencies, and diagnostic outputs.
- `docs/24_STANDALONE_SIMULATOR.md`: canonical runtime guide. `docs/23_PCR_PHYSX_PILOT.md` is preserved history, superseded for execution.
- Read [architecture and experiment contract](docs/25_ARCHITECTURE_AND_EXPERIMENT_CONTRACT.md), [writing guide](docs/26_WRITING_GUIDE.md), and [verification record](docs/27_REFACTOR_VERIFICATION.md) for current boundaries and evidence.

Run offline checks with `PYTHONPATH=src python -m pytest -q tests/`. For local simulation, use `/home/artrc/miniconda3/envs/grasp_hexapod/bin/python`, prepend its `bin` directory to `PATH`, set `TORCH_EXTENSIONS_DIR=/tmp/physx_pilot_torch_extensions`, and follow document 24. Do not install or upgrade Isaac Gym, NumPy, or Torch as part of a run.

## Invariants and validation

Preserve canonical leg order `lb lf lm rb rf rm`, names `j_<leg>_<thigh|knee|ankle>`, and map using Gym-reported names. Retain URDF mass, inertia, limits, and geometry. Use JSON as the numeric runtime authority. Preserve dirty work. Report exact tests and separate offline checks from physics evidence; never infer walking stability, camera improvement, or hardware behavior from unit tests. Do not commit raw diagnostic data, credentials, private media, checkpoints, or bulk datasets.

Use only the `test` and `main` branches. Do active work on `test`; promote to `main` only with explicit user authorization. Do not create additional branches. Use four spaces, `snake_case` functions/modules, and `PascalCase` classes. Read project guidance and `git status --short` before nontrivial batches; validate only affected tiers. Use concise imperative commit subjects with `docs:`, `test:`, or `feat:` prefixes. Pull requests should state scope, rationale, exact validation, and whether evidence is offline or physics-based; do not imply unrun experiments passed.
