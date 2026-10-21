# Standalone Isaac Gym flat-stand diagnostic

This is the canonical local simulator entry point. It loads the URDF and referenced mesh files from this repository, builds one flat-ground PhysX environment, holds one explicit six-leg joint reference, and records measured state. Torque columns are commanded PD efforts; their clipping flags describe software command limits, not measured actuator torque or validated physical saturation. It does not import, read, or require a PCR-Net checkout, `legged_gym`, upstream weights, or upstream configuration. The standalone modules are original project code; only the locally packaged robot description and its active mesh dependencies were copied. See [asset provenance](../assets/hexapod/PROVENANCE.md) for source revision, hashes, and license boundary.

For ownership, trial-start reset and explicit randomization rules, use the [architecture and experiment contract](25_ARCHITECTURE_AND_EXPERIMENT_CONTRACT.md). The baseline uses a fixed initial configuration; randomized conditions are future, separately configured experiments.

## Scope and interpretation

The pilot demonstrates a repeatable **standing diagnostic only**. It does not run gait cycles, body-pose compensation, camera projection, random resets, training, rewards, curriculum, learned actuator compensation, or hardware control. The PD gains are proposed simulation values and have not been calibrated against hardware. A `settled` status means the recorded run met the configured stand window; it is not a walking, camera-stability, or real-robot claim. Any failure leaves its trace and a failed status in the unique run directory.

## Supported environment and command

Use the already installed environment without package or driver changes: Isaac Gym at `/home/artrc/isaacgym`, Python 3.8 in `/home/artrc/miniconda3/envs/grasp_hexapod`, and Torch 1.8.1+cpu. The launcher prepends the active Python interpreter environment `bin` path and sets a temporary Torch extension cache before importing Gym. Gym imports before Torch/gymtorch. The simulation uses GPU PhysX with the CPU tensor pipeline; Torch CUDA operations are not used.

```bash
cd /home/artrc/hexapod-camera-stability
ENV=/home/artrc/miniconda3/envs/grasp_hexapod
PATH="$ENV/bin:$PATH" TORCH_EXTENSIONS_DIR=/tmp/physx_pilot_torch_extensions \
  "$ENV/bin/python" scripts/run_hexapod_pilot.py --headless --sim-device cuda:0
```

For a graphical run on the tested PRIME/Vulkan setup, select the NVIDIA ICD for this process only:

```bash
cd /home/artrc/hexapod-camera-stability
ENV=/home/artrc/miniconda3/envs/grasp_hexapod
ulimit -c 0
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
PATH="$ENV/bin:$PATH" TORCH_EXTENSIONS_DIR=/tmp/physx_pilot_torch_extensions \
  "$ENV/bin/python" scripts/run_hexapod_pilot.py --duration-s 6.0 --hold-viewer
```

Each run creates a new `data/raw/standalone_<UTC timestamp>/` directory. It will not reuse an existing directory. A `status.json` marker with status `started` is written before importing native Gym code, so a native crash still leaves a clearly incomplete run marker. To choose a different output parent, pass `--output-root`. `--duration-s` changes the diagnostic length; `--config` selects a JSON configuration. In GUI mode omit `--headless`; use `--hold-viewer` to leave the finished scene visible until the window closes (maximum 20 minutes). Closing the viewer during the run stops the run and preserves its current trace.

## Model and initial state

The locally packaged `assets/hexapod/urdf/hex_ground.urdf` is byte-identical to its source. Its 18 actuated joints keep their original names, limits, mass, inertia, and geometry; all six fixed-joint toe links and their 9 mm spherical collisions are retained. Gym-reported DOF names determine the mapping at runtime. The canonical leg order is `lb`, `lf`, `lm`, `rb`, `rf`, `rm`.

The JSON explicitly enables Isaac Gym's `flip_visual_attachments` importer option. HexGround's upstream Gym importer used this option for visual-axis conversion. It is an importer behavior, not a claim about the DAE metadata: the packaged Collada files declare `Z_UP`, and neither URDF nor mesh files were modified to address rendering.

All numeric run parameters are in [configs/flat_stand.json](../configs/flat_stand.json). The deterministic reference has thigh angles `lb=+0.5`, `lf=-0.5`, `lm=0`, `rb=-0.5`, `rf=+0.5`, `rm=0` radians; each knee is `+0.67` rad and each ankle is `-2.2` rad. This explicit reference was informed by the robot geometry and previous stance values; it is not loaded from an upstream config. The base starts at x/y zero and yaw zero, with zero configured linear and angular velocity. Its z is computed by URDF-tree forward kinematics to put the lowest toe sphere 2 mm above the plane before the first physics step. The configured derived base height is recorded separately from the measured first post-step state.

## Physics and control settings

The configuration specifies a 2.5 ms physics step, one substep, gravity `[0, 0, -9.81] m/s²`, PhysX TGS, eight position iterations, two velocity iterations, 2 mm contact offset, and zero rest offset. The ground uses unit static and dynamic friction and zero restitution. The base is free; fixed joints are not collapsed, and no mass or inertia overrides are applied. Self-collision is enabled through the actor collision filter setting. GPU physics is enabled while `use_gpu_pipeline` is false.

Every physics step applies explicit effort PD to the measured joint state:

`tau = clip(50 * (q_ref - q) - 1.5 * dq, -27, +27) Nm`.

The torque is capped by both the configured 27 Nm limit and each URDF effort limit. No hidden Gym stiffness/damping is active. Joint position checks use the URDF ranges with `1e-4 rad` tolerance; velocity checks use the URDF limit (5 rad/s here) with `1e-4 rad/s` tolerance. Tilt above 60 degrees or root z below 0.035 m stops the trial. A safety stop is logged as failure; there is no automatic reset.

## Stand gate

A stand window is evaluated only after 2.0 s of simulation and must pass continuously through the final sample for 0.5 s. Each sample must have at least three toe bodies with measured contact force greater than 1 N, absolute roll and pitch no greater than 0.15 rad, 3D root speed no greater than 0.12 m/s, angular speed no greater than 0.25 rad/s, absolute vertical speed no greater than 0.06 m/s, no measured joint limit violation, and no PD command clipping. The gate uses observed body/contact/joint states. Non-finite state or a non-normalized root quaternion stops the run and is recorded with the last measured state. Closing the viewer early records an interrupted status. `status.json` is written before the optional viewer hold and records per-criterion rejection counts over eligible samples, the status, and any failure reason. A run that ends without a passing final window is `failed_stand_gate`.

## Output files

- `trace.csv`: one row per recorded physics step; measured root pose and velocity, contact count and contacted leg names, every measured joint position and velocity, target, commanded torque, limit status, and PD command-clipping count.
- `status.json`: diagnostic status, requested duration, gate result and rejection counts, simulator backend/configuration, reported Gym body/DOF names, package versions, and exact configured pre-step state.
- `resolved_config.json`: full configuration, resolved local URDF path, asset SHA-256 hashes, simulator device, and FK-derived start height.

These are diagnostic traces, not training data. Preserve failed logs and status. Offline tests validate parsing, mapping, deterministic configuration, and numeric control calculations; only the simulator trace provides physics evidence, and neither one establishes walking or camera improvement.

## Offline checks

```bash
PYTHONPATH=src python -m pytest -q tests/
```

These checks require no Isaac Gym and do not simulate physics. The older [PCR PhysX pilot note](23_PCR_PHYSX_PILOT.md) is retained as a historical record; use this document for current execution.


## Recorded local verification (2026-10-21)

The offline suite passed: `PYTHONPATH=src /home/artrc/miniconda3/envs/grasp_hexapod/bin/python -m pytest -q tests/` (48 passed). One headless GPU PhysX run completed 6.0 s and wrote `data/raw/standalone_20261021T065926_687119Z/`; its diagnostic status was `settled`, with 2400 trace rows, six contacted toes, zero gate rejections, and zero PD command-clipping events. This establishes only that the configured stand gate passed for that local simulation run.

A GUI run using the command above reached the viewer and completed 2400 physics samples in `data/raw/standalone_20261021T070244_134566Z/`. The numerical status was `settled`, but visual review rejected the rendered assembly: the body appeared vertical with detached legs. Its status records `visual_assembly_accepted=false`; interpret the trace as a numerical diagnostic only, not accepted model or physics evidence. The importer option above was then added to correct the identified visual-import difference.

The corrected GUI run in `data/raw/standalone_20261021T070714_881693Z/` used the recorded `flip_visual_attachments=true` option, completed 2400 steps over 6.0 s, and had six contacted toes at the final sample, zero stand-gate rejections, and zero PD command-clipping events. Independent visual review accepted the rendered assembly after confirming a horizontal body with all six legs attached. This remains a stand-only, uncalibrated simulation result.
