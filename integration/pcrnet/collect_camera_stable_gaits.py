#!/usr/bin/env python3
"""PCR-Net independent, single-robot Isaac Gym gait + compensation pilot.

Install in Pcrnet_hexapod/legged_gym/scripts/ or call from this research repo.
Uses the ORIGINAL PCR HexGround simulator and actuator network: no invented
robot dynamics and no inference of paper performance from a dry-run.

Stage gate: SINGLE trial with controller on/off, not yet a large batch runner.
No exception may be reported as a successful trial. Diagnostics sidecar
records status and failure reason; failed trials are never used for training.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pcr-root', type=Path, required=True)
    p.add_argument('--research-root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--task', default='hex_debug_plane', choices=('hex_debug_plane', 'hex_debug_heightfield'))
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--controller', required=True, choices=('off','on'))
    p.add_argument('--pair-id', default='pilot01')
    p.add_argument('--terrain-instance', default='flat_pilot')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--frequency-hz', type=float, default=1.1)
    p.add_argument('--duty-factor', type=float, default=0.58)
    p.add_argument('--swing-height', type=float, default=0.025)
    p.add_argument('--touchdown-softness', type=float, default=0.5)
    p.add_argument('--forward-mps', type=float, default=0.08)
    p.add_argument('--ramp-s', type=float, default=1.0)
    p.add_argument('--sim-device', default='cuda:0')
    p.add_argument('--headless', action='store_true')
    return p


def config_checks(args):
    if args.steps < 3 or args.seed < 0 or args.forward_mps <= 0 or args.ramp_s <= 0:
        raise ValueError('Invalid trial steps/seed/speed/ramp')
    if not args.pcr_root.joinpath('legged_gym/envs/hex_v4/hex_ground.py').is_file():
        raise FileNotFoundError('--pcr-root must point to Pcrnet_hexapod root')
    if not args.research_root.joinpath('src/pcr_trial_controller.py').is_file():
        raise FileNotFoundError('--research-root must point to camera-stability workspace')
    if args.task != 'hex_debug_plane':
        raise NotImplementedError('Only verified flat pilot planned; heightfield disabled until first flat gait walk')


def run(args):
    config_checks(args)
    # Isaac Gym import MUST precede torch, mirroring original PCR scripts.
    sys.path.insert(0, str(args.pcr_root.resolve()))
    sys.path.insert(0, str(args.research_root.joinpath('src').resolve()))
    import isaacgym  # noqa: F401
    import torch
    import legged_gym.envs  # noqa: F401; side-effect registers tasks
    from legged_gym.utils import task_registry, get_args
    from pcr_urdf_ik import HexUrdfIK
    from pcr_trial_controller import PcrSingleTrialController
    from tripod_gait import TripodConfig
    from pcr_sim_bridge import PcrSimTap, TrialIdentity, canonical_toe_permutation

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], '--task', args.task, '--num_envs', '1',
                    '--seed', str(args.seed), '--sim_device', args.sim_device]
        if args.headless:
            sys.argv.append('--headless')
        isaac_args = get_args()
    finally:
        sys.argv = original_argv

    cfg, _ = task_registry.get_cfgs(args.task)
    cfg.seed = args.seed
    cfg.env.num_envs = 1
    cfg.terrain.curriculum = False
    cfg.commands.curriculum = False
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.push_robots = False
    cfg.sensor.depth_camera.enable = False
    env, _ = task_registry.make_env(args.task, args=isaac_args, env_cfg=cfg)
    try:
        urdf = Path(cfg.asset.file)
        kin = HexUrdfIK.from_urdf(urdf)
        obs, _ = env.reset()
        gait = TripodConfig(args.frequency_hz, args.duty_factor,
                            args.swing_height, args.touchdown_softness)
        ctrl = PcrSingleTrialController(env, kin, gait, compensate=(args.controller=='on'))
        gait_id = (f'f{args.frequency_hz:.3f}_b{args.duty_factor:.3f}_'
                   f'h{args.swing_height:.3f}_s{args.touchdown_softness:.3f}')
        trial_id = f'{args.pair_id}_{args.controller}'
        outfile = args.output_root / f'{trial_id}.csv'
        identity = TrialIdentity(trial_id, args.pair_id, args.terrain_instance,
                                 args.seed, args.controller=='on', gait_id)
        perm = canonical_toe_permutation(env, 0)
        print('[pilot] Verified toe contact permutation lb,lf,lm,rb,rf,rm:', perm.tolist())
        with PcrSimTap(outfile, identity, canonical_feet_permutation=perm) as tap:
            for i in range(args.steps):
                simtime = i * float(env.dt)
                ramp = min(1.0, (simtime / args.ramp_s))
                speed = args.forward_mps * ramp
                actions, stance = ctrl.action(speed)
                # PhysX calls the real network-driven actuator model, no fake motion.
                action_t = torch.as_tensor(actions, dtype=torch.float32, device=env.device)
                _, _, _, dones, _ = env.step(action_t)
                if bool(dones[0].item()):
                    raise RuntimeError(f'Environment terminated/reset after step {i+1}; aborted, no successful trial')
                tap.append(env, 0, (i+1)*float(env.dt), [0.,speed,0.], stance)
                if i == 0 or (i + 1) % 50 == 0:
                    print(f'[pilot] {i+1}/{args.steps} t={(i+1)*float(env.dt):.2f}s pose_z={float(env.root_states[0,2]):.3f}')
        return {'status':'completed_raw_physics_trial','samples':args.steps,'csv':str(outfile)}
    finally:
        if getattr(env,'viewer',None) is not None:
            env.gym.destroy_viewer(env.viewer)
        if getattr(env,'sim',None) is not None:
            env.gym.destroy_sim(env.sim)


def main():
    args = parser().parse_args()
    # Safe identity; never overwrite the previous trial's outcomes.
    args.output_root.mkdir(parents=True, exist_ok=True)
    sidecar = args.output_root / f'{args.pair_id}_{args.controller}.status.json'
    if sidecar.exists():
        raise FileExistsError(f'Status already exists: {sidecar}')
    info = {'status':'started','parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    try:
        result = run(args)
        info.update(result)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        info.update(status='failed', failure_type=type(exc).__name__, failure_reason=str(exc))
        traceback.print_exc()
        return 1
    finally:
        sidecar.write_text(json.dumps(info, indent=2, ensure_ascii=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    sys.exit(main())