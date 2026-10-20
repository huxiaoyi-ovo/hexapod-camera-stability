# Project start · 2026-10-20

1. **Platform:** `huxiaoyi-ovo/Pcrnet_hexapod` (not Grasp_hexapod).
2. **Target:** MDPI *Machines*; submission rush, with acceptance by November 20 a target, **not a guarantee**.
3. **Main technical unknown:** whether gait-dependent finite-bandwidth compensation response materially affects optimal image-stability gait ranking.
4. **First executable milestone:** reproduce one continuous physics-based Isaac Gym rollout of a tripod gait with body compensation off and on; record actual camera pose, angular speed, achieved speed, contact forces, 18 DOFs and actuator tracking.
5. **Second milestone:** terrain/seed grouped dataset; short-horizon surrogate; comparison with raw-disturbance ranking and historical residual-score tuning.
6. **Last milestone:** matched-speed multi-terrain baselines + limited real RGB video comparisons; fill paper Results only with backed-up results.

[Experiments](docs/17_SIMULATION_FIRST_RESEARCH_PROTOCOL.md) · [Integration](docs/18_LOCAL_ISAAC_GYM_INTEGRATION_TASK.md) · [Method](docs/16_REVISED_CORE_METHOD_AND_EXECUTION.md)
