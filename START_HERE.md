# Project start

1. **Current executable milestone:** run a standalone flat-ground Isaac Gym standing diagnostic with the locally packaged URDF and meshes. It is not a walking trial or a camera-stability result.
2. **Runtime guide:** follow [docs/24_STANDALONE_SIMULATOR.md](docs/24_STANDALONE_SIMULATOR.md). The supported entry point is [scripts/run_hexapod_pilot.py](scripts/run_hexapod_pilot.py), configured by [configs/flat_stand.json](configs/flat_stand.json).
3. **Offline checks:** `PYTHONPATH=src python -m pytest -q tests/`. These do not run physics.
4. **Architecture and experiments:** follow [docs/25_ARCHITECTURE_AND_EXPERIMENT_CONTRACT.md](docs/25_ARCHITECTURE_AND_EXPERIMENT_CONTRACT.md). The current executable is stand-only; walking, project-owned FK/IK integration, body compensation, motor-delay identification, camera calibration, and paired ON/OFF comparisons remain future steps. Trial-start initialization is explicit; automatic reset and default randomization are prohibited.
5. **Manuscript:** [paper/main.tex](paper/main.tex) is the only manuscript entry. Follow [docs/26_WRITING_GUIDE.md](docs/26_WRITING_GUIDE.md); see [docs/27_REFACTOR_VERIFICATION.md](docs/27_REFACTOR_VERIFICATION.md) for the accepted stand diagnostic and verification scope.
6. **Historical integration:** `integration/pcrnet/` and `src/pcr_*` are preserved work in progress. The older PCR task-integration plans and [docs/23_PCR_PHYSX_PILOT.md](docs/23_PCR_PHYSX_PILOT.md) are retained as history and superseded for execution by documents 24 and 25. The canonical runner has no PCR-Net runtime dependency.

[Simulation-first research protocol](docs/17_SIMULATION_FIRST_RESEARCH_PROTOCOL.md) · [Revised core method](docs/16_REVISED_CORE_METHOD_AND_EXECUTION.md)
