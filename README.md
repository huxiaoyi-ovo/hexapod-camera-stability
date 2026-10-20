# Hexapod Camera Stability

Research workspace for camera-stable hexapod locomotion on uneven terrain.

- **Target journal:** *Machines* (MDPI)
- **Research mode:** simulation-first, with limited real-robot verification
- **Base platform:** [Pcrnet_hexapod](https://github.com/huxiaoyi-ovo/Pcrnet_hexapod)
- **Status:** active research; performance results not yet established

## Repository policy

This repository is currently **public**. Only a public-safe project scaffold is present.
Unpublished manuscript material, unvalidated algorithms, experimental traces, calibration data,
videos, and checkpoints are intentionally withheld until the owner reviews repository visibility.

## Planned directories

- `src/` — gait and compensation algorithms
- `integration/` — PCR-Net Isaac Gym integration
- `scripts/` — experiment and evaluation tooling
- `tests/` — offline and simulation tests
- `docs/` — protocols and development decisions
- `paper/` — manuscript sources
- `figures/` — reproducible figure generation

## Research integrity

Simulation claims must be supported by actual Isaac Gym rollouts. Do not treat unit-test results
as robot-level evidence. Comparisons will control achieved walking speed and use the same
compensation controller and candidate gait set.

See [repository status](docs/STATUS.md).
