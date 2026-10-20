# Research workflow

1. Maintain reproducible versions of controller, evaluation code, and experiment configuration.
2. Integrate into the existing PCR-Net Isaac Gym hexapod model without disrupting PCR-Net target following.
3. Collect simulator trajectories separately from source code, using terrain/seed grouped train/validation/test splits.
4. Run speed-matched comparisons with identical actuator and compensation constraints.
5. Generate figures from recorded experiment results rather than illustrative values.
6. Validate final control on a limited set of real RGB/IMU sequences only after simulator results are credible.

**Publication discipline:** No claim of improved camera stability before actual robot simulation or hardware measurements.
