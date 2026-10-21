# Engineering readiness

| Work item | Status |
|---|---|
| Standalone Isaac Gym/PhysX diagnostic | Implemented; corrected 6 s GUI stand run passed its diagnostic gate and visual review. This is operational evidence only. |
| Local robot model assets | URDF and required visual/collision mesh closure are vendored; provenance and asset hashes are recorded. |
| Full offline suite | Prior worktree result: 48 tests passed. Clean staged-snapshot validation is pending this documentation/manuscript batch. |
| Standalone targeted checks | Six standalone tests, Python 3.8 compilation, and JSON config loading passed after the visual importer correction. |
| Project-owned walking gait and FK/IK integration | Pending; existing gait/IK prototypes are not connected to the standalone simulator. |
| Motor-response and delay identification | Pending; current stand PD values are uncalibrated. |
| Fixed body-pose compensator and verified camera extrinsics | Pending. |
| Paired ON/OFF trials and image-motion measurements | Not yet executed; stand diagnostics are not training or comparison data. |
| Predictor, online scheduler, multi-terrain evaluation | Pending. |
| Qualified comparative walking/camera Results | Pending. |
| Real-robot RGB/IMU evidence | Pending. |

The active platform boundary and required trial logging are in [the architecture and experiment contract](25_ARCHITECTURE_AND_EXPERIMENT_CONTRACT.md). Manuscript status is governed by [the writing guide](26_WRITING_GUIDE.md); validation evidence and scope are summarized in [the verification record](27_REFACTOR_VERIFICATION.md). Keep each unvalidated component and claim explicitly marked as pending.
