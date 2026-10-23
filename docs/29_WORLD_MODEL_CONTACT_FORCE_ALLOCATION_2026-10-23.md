# Contact-force-aware world-model visual stabilization (2026-10-23)

**Status**: targeted research extension; architecture and offline allocator proof-of-concept, **not** a trained policy or validated force controller. This document supersedes the earlier pose-only action-space recommendation in `docs/28_SUPERMIMIC_VISION_PIVOT_2026-10-23.md` but retains the original ExpertGround and PD baseline unchanged. No target journal is frozen.

## Central hypothesis

Under identical gait reference, actual forward velocity and terrain, anticipation of stance changes, realized ground reaction force (GRF), and motor response enables a constrained policy to redistribute stance forces / corrective body wrench and reduce residual camera image motion more reliably than pose-only stabilization or fixed force QP. This **must be tested**, not presumed; force allocation, position-force learning, and image-stabilizing locomotion have already been published separately.

## Evidence / closest prior work

- SuperMimic, CoRL 2026, https://supermimic.github.io/: a *learned differentiable world model* whose gradients train a policy, never differentiating through a rigid-body simulator directly.
- *Blind Walking Control of a Heavy-Duty Hexapod Robot Based on Optimal Force Allocation for Uneven and Sloped Terrains*, Journal of Field Robotics, https://doi.org/10.1002/rob.70051: hexapod constrained optimal GRF distribution is prior art.
- *Learning a Unified Policy for Position and Force Control in Legged Loco-Manipulation*, CoRL 2025, https://proceedings.mlr.press/v305/zhi25a.html: sensorless learned force estimation + position/velocity adjustments is prior art.
- *Hold My Beer*, CoRL 2025, https://proceedings.mlr.press/v305/li25i.html: stable walking video / end-effector stabilization already exists.
- *Learning Deployable Locomotion Control via Differentiable Simulation*, CoRL 2025, https://proceedings.mlr.press/v305/schwarke25a.html: non-smooth contacts challenge valid policy gradients.
- *From Policy Actions to Whole-Body Dynamics: A Unified Projection-Based Framework for Versatile Legged Loco-Manipulation*, RA-L, 2026, https://doi.org/10.1109/LRA.2026.3734869: optimizer projection of policy actions to contact-consistent dynamics exists.
- *Design and Dynamic Locomotion Control of Quadruped Robot with Perception-Less Terrain Adaptation*, https://pmc.ncbi.nlm.nih.gov/articles/PMC9494723/: force QP + Jacobian torque in classical legged control.

Therefore novelty cannot be 'use GRF QP', 'feed predicted contact to control', 'SuperMimic applied to hexapods' or 'RL stabilizes a camera'. Distinct value, if demonstrated, is world-model **forecast-informed GRF redistribution** specifically targeted at *camera-space residual* under contact mode uncertainty, realistic actuator delay and deployment observations.

## Minimal technical design (separate three levels)

1. **Frozen gait generator**: original ExpertGround outputs `q_E` and intended support/swing stage. No modification of its phase transitions / reference paths.
2. **Learned world model and actor**: world model predicts short-horizon changes in body twist, joint states, *realized* contact force or contact probability, delayed motor torque, and 6D camera pose. Actor observes only proprioception, IMU, joint history, commanded velocity, planned gait stage and **estimated current contact**; simulator-ground-truth contact force/terrain/friction are training targets / privileged model fit context, never inputs to deployed actor. Actor emits bounded desired corrective body wrench `delta_w = [Fx,Fy,Fz,Mx,My,Mz]` and optionally conservative RPZ correction. The actor is trained through a differentiable learned model; it **does not** directly output unrestricted 18-DoF torques or 6x3 force vectors.
3. **Physics feasibility + force allocator**: QP projects desired wrench to actual planned+confirmed stance feet, with (estimated) foot locations, contact normals, conservative friction bounds and motor torque headroom. Non-stance feet exert zero optimized contact force. In a 3-foot tripod stance there are nominally 9 scalar GRFs for 6 floating-base wrench components; feasible redistribution remains conditional on geometry and friction, not guaranteed by DOF counting. Hard constraints and output validity are checked before use, and unavailable/infeasible support means fall back to original ExpertGround PD.

For foot i in contact, foot position relative to estimated CoM `r_i`, support GRF `f_i` and contact normal `n_i`, define:
`A_i = [I_3 ; skew(r_i)]`, and desired net contact wrench `w_d = w_nominal + delta_w`.
Solve the (convex) constrained QP for stance forces:
`min_f || W (A f - w_d) ||_2^2 + lambda_f ||f||_2^2 + lambda_df ||f-f_prev||_2^2`
subject to
`0 <= n_i^T f_i <= f_zi,max`; `|t1_i^T f_i| <= mu_i (n_i^T f_i)/sqrt(2)`;
`|t2_i^T f_i| <= mu_i (n_i^T f_i)/sqrt(2)`; and (when available)
`|tau_PD,i + tau_ff,i| <= tau_max,i`, with actuator and reference-rate limits.
These two tangential inequalities form a *conservative inner approximation* of the Coulomb friction cone. The optimized force is a desired contact force, **not** guaranteed actual force. If body wrench mismatch is large, report infeasible/insufficient authority. Smooth load transfer across touchdown/liftoff; do not switch to the next stance group based on future simulator contact truth.

The existing `src/hexapod_actuator.py` supports **per-joint feedforward torque vectors** via `command_feedforward_nm` and a delayed 400-Hz torque response. However, `src/hexapod_sim.py` currently sets *one constant scalar zero feedforward* and validates a zero baseline. Any force-to-torque implementation (`tau_ff ~ J_i(q)^T f_i`, with sign and coordinate conventions verified against the actual plant) requires an **explicit, independent optional runtime path**, preserving a zero-feedforward reference baseline and logged safety constraints. Do NOT claim force controllability under PD merely because the QP has a numerical solution. No torque/force sensing has been verified on the physical hexapod. A joint-torque estimate `tau_model` and `J` may provide a *noisy* contact-force proxy but does not uniquely identify true multi-foot contact forces without further assumptions.

## World-model contact/action loop

At simulator step t, state history `h_t`, commanded correction `delta_w_t`, and current planned/estimated support `c_t`:
- Actor proposes `delta_w_t`.
- Fast constrained allocator produces `f_des,t`; force-to-torque mapping and unchanged PD reference produce motor commands.
- PhysX advances; read true `f_contact` only as an *offline supervised target/diagnostic* and collect (q, dq, IMU, model motor state, camera pose).
- World model learns `p(delta_s, realized_GRFS, c_next | history, executed_actuator_command, contact_estimate)`. Evaluate one-step and multi-step prediction especially at lift-off and touchdown.
- Policy optimization uses a differentiable trained model and either a differentiable projected force solver or **a clearly marked approximate gradient route**. A non-differentiable SciPy QP cannot simply be inserted in a BPTT computational graph while claiming exact analytic gradients. Gradient-through-optimizer can later use implicit differentiation or a verified differentiable QP implementation; otherwise use model-free actor updates or differentiable surrogate with real-PhysX policy evaluation.
- Use predicted contact probabilities / uncertainty to reduce commanded wrench, shrink assumed contact availability, shorten gradient horizon, or fall back to conservative PD. Contact force uncertainty is measured against held-out *true* simulation contacts; online inputs remain sensor-reproducible.

## Prioritized engineering gates

**F0** Preserve current reported local PD slope reference (40s, ~0.12 m/s, 21 speed-limit exceedance samples, 5.16 rad/s peak, brief support warnings). Current GitHub `test` and this branch don't contain the local `configs/pcr_expert_pd.json` file/traces; do not pretend to have run them. No changes to active PD configuration before diagnosing actual recorded safety failures.

**F1** Offline wrench/force projection feasibility. Implement a small deterministic 3/4/6-contact allocator test, with ground-normal/friction/max-normal-force bounds, support masks, target net wrench matching, and reported residual/infeasibility. No motor commands, no learned world model yet.

**F2** In PhysX, enable a separate flag for dynamic **bounded 18-DoF feedforward torque** only after confirming Jacobian sign/coordinates, delay/torque saturation and support changes. Compare zero FF against a **classical PD + contact-force QP**. If the QP does not improve relevant body image residual or support in matched-speed real PhysX episodes, do not jump straight to model-based RL.

**F3** Train baseline bounded RPZ residual PPO, then actor + QP with identical observations/reward. Compare actual GRF tracking, not commanded GRF alone. No oracle contact-force feedback to policy.

**F4** Fit and validate world model on train/held-out terrain realizations; evaluate contact-transition timing, force prediction and residual camera flow. Then train SuperMimic-inspired imagination actor with QP or differentiable safety projection (verified gradients), comparing plain WM versus contact/actuator-aware and uncertainty-gated WM.

**F5** Ablations: Expert only; Expert+pose feedback; Expert+classical GRF QP; Expert+PPO RPZ; Expert+PPO+QP; Expert+plain differentiable WM+QP; proposed contact/actuator-aware WM+QP. Keep identical commanded/achieved speeds, terrain difficulty and safety criteria. Evaluate image residual px/frame (excluding nominal commanded motion), episode success, safety and stance contact quality, actual motor torque clipping, GRF prediction MAE, force-feasibility and actual achieved wrench residual, computation latency, sample efficiency, and training seeds. Test contact/friction and actuator delay domain shifts. Independent real camera RGB test is desirable but not yet promised.

## Immediate development discipline

This adds **one force-allocation path**, not a new gait generator, new default motor model, or large all-at-once rewrite. First deliver an offline force QP and tests; then physically validate force-to-torque before putting it in a world-model gradient loop. No venue lock-in.