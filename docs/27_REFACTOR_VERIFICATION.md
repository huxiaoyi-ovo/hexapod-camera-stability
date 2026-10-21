# Refactor verification record

This record binds the standalone architecture change to observed validation. It distinguishes simulator bring-up from paper evidence. Raw outputs are local under ignored `data/raw/` and are not included in the source commit.

## Corrected visual and stand diagnostic

- **Run:** `data/raw/standalone_20261021T070714_881693Z/`
- **Status:** `settled`, diagnostic kind `stand_diagnostic`, `valid_training_data=false`; 2,400 rows over 6.0 s, six toe contacts at the final sample, all stand-gate rejection counts zero, and zero PD command-clipping flags. Torque columns are commanded efforts, not measured actuator torques; motor response and delay remain uncalibrated.
- **Backend:** Isaac Gym / PhysX, GPU physics on `cuda:0`, CPU tensor pipeline, Python 3.8.20, NumPy 1.23.5, Torch 1.8.1+cpu.
- **Importer:** `flip_visual_attachments=true`, `use_mesh_materials=true`; both options are recorded in `status.json`. The independent visual review accepted the corrected horizontal body and attached six-leg assembly.
- **Interpretation:** operational evidence that this configured flat-stand run completed and met its diagnostic gate. It does not validate walking, the research body compensator, image stability, real-camera behavior, or hardware performance. PD gains remain uncalibrated.

The earlier run `data/raw/standalone_20261021T070244_134566Z/` completed a numeric gate but was rejected on visual review because the assembly appeared vertical with detached legs. Its trace is retained as a rejected visual diagnostic, not accepted model evidence. Earlier GUI initialization failures are likewise not successful runs.

## Portable identifiers and hashes

These values were computed from the preserved local files after the accepted run; raw data remain outside Git.

| Item | SHA-256 |
|---|---|
| Local URDF `assets/hexapod/urdf/hex_ground.urdf` | `187f8281e1976e050c7a960c288e033368c5d5ded3932725e34942a5085215e6` |
| Runtime config `configs/flat_stand.json` | `a756082953794a448422e3896de6da0513082a0de1ac643c21c2090276025678` |
| Runtime `src/hexapod_sim.py` | `3f5f31220c6365a032c5bc6cd2e4b23d9a10921b0867e4aba74c5bce28c67bd7` |
| Model parser `src/hexapod_robot.py` | `d3fb81d0a14fb2c31f7bf9781b7396edeab400dc04c71f2017096dc55a8a10e4` |
| Launcher `scripts/run_hexapod_pilot.py` | `5a8a9fbce85da8ad4e614906bd522935e8844c887c4c5735521721bd9eb75acd` |
| Raw `status.json` | `cae30d345c2d2cd1a98b44defd665442a572162cbadbd5c48fccbd92fd817ce9` |
| Raw `trace.csv` | `df0bc075f82eb5c49e5ba459040eae6ad49180b52e3cc02e945bccc779899f38` |
| Raw `resolved_config.json` | `8fe01d25f927bd250c9b53be562874d58eb3efe85ef8136d7954c1ac59eb1d50` |

`resolved_config.json` contains the complete local asset hash map and FK-derived start height. `status.json` contains the observed initial state, final state, simulator settings, contact forces, joint/body names, gate criteria, and importer options. The configuration itself has no randomization; the explicit initial state is recorded before the first physics step.

## Software checks and scope limits

Before this documentation/manuscript batch, the local worktree's full offline suite passed 48 tests; after the visual-import correction, the standalone test file passed 6 tests, Python 3.8 byte-compilation and config loading passed, and `git diff --check` passed. The exact staged tree was exported outside the worktree and its offline suite passed 47 tests using the existing Gym Python 3.8 environment (NumPy 1.23.5). `latexmk` built the 12-page manuscript from `paper/main.tex`; rendered Methods and Experimental Setup pages were visually reviewed. The copied source CAD assets retain their original CRLF bytes; whitespace checking on the other staged files is clean.

The standalone implementation does not yet support gait execution, paired ON/OFF studies, delay or motor identification, randomized experiment configs, terrain generation, camera calibration, or a validated data-to-predictor pipeline. Earlier PCR integration files remain historical/inactive and are outside the standalone execution path. Any future protocol changes must retain raw failed runs and cannot convert these diagnostics into training or comparative data.
