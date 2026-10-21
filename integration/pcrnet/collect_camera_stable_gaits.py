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
import csv
import numpy as np
from scipy.spatial.transform import Rotation


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


def stand_window_is_stable(samples, window_samples):
    """Require the configured continuous settle window after 2 s standing."""
    if len(samples) < window_samples:
        return False
    states = samples[-window_samples:]
    for state in states:
        root = state['root']
        roll, pitch, _ = Rotation.from_quat(root[3:7]).as_euler('xyz')
        contacts = int(np.count_nonzero(np.linalg.norm(state['foot_forces'], axis=1) > 1.0))
        if (contacts < 3 or abs(roll) > .15 or abs(pitch) > .15
                or np.linalg.norm(root[7:10]) > .12 or np.linalg.norm(root[10:13]) > .25
                or abs(root[9]) > .06 or state['max_dq_ratio'] > 1.0
                or state['torque_saturation_fraction'] > 0.0):
            return False
    return True


def summarize_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    walking = [r for r in rows if r['phase'] == 'walking']
    if len(walking) < 3:
        return {'walking_samples': len(walking), 'summary_valid': False}
    times = np.array([float(r['time_s']) for r in walking])
    pos_x = np.array([float(r['root_px']) for r in walking])
    pos_y = np.array([float(r['root_py']) for r in walking])
    quat = np.array([[float(r[k]) for k in ('root_qx','root_qy','root_qz','root_qw')] for r in walking])
    initial_forward_world = Rotation.from_quat(quat[0]).apply([0.,1.,0.])
    displacement = np.array([pos_x[-1]-pos_x[0], pos_y[-1]-pos_y[0], 0.])
    signed_forward_progress = float(np.dot(displacement, initial_forward_world))
    euler = Rotation.from_quat(quat).as_euler('xyz')
    z = np.array([float(r['root_pz']) for r in walking])
    contacts = np.array([np.array(json.loads(r['measured_foot_contact_json']), bool) for r in walking])
    toes = np.array([json.loads(r['toe_world_positions_m_json']) for r in walking])
    dt = np.diff(times)
    continuous_speed, continuous_drift, touchdown_n = [], 0.0, 0
    for i in range(1, len(walking)):
        touchdown_n += int(np.count_nonzero(contacts[i] & ~contacts[i-1]))
        active = contacts[i] & contacts[i-1]
        displacement = np.linalg.norm(toes[i,:,:2]-toes[i-1,:,:2], axis=1)
        continuous_speed.extend((displacement[active]/(times[i]-times[i-1])).tolist())
        continuous_drift += float(displacement[active].sum())
    contact_counts = contacts.sum(axis=1)
    elapsed = float(times[-1]-times[0])
    slip = np.asarray(continuous_speed)
    summary = {
        'summary_valid': True, 'walking_samples': len(walking),
        'forward_progress_world_y_m': float(pos_y[-1]-pos_y[0]),
        'achieved_forward_speed_world_y_mps': float((pos_y[-1]-pos_y[0])/elapsed),
        'forward_progress_initial_heading_m': signed_forward_progress,
        'achieved_speed_initial_heading_mps': signed_forward_progress/elapsed,
        'roll_rms_rad': float(np.sqrt(np.mean(euler[:,0]**2))),
        'roll_peak_abs_rad': float(np.max(np.abs(euler[:,0]))),
        'pitch_rms_rad': float(np.sqrt(np.mean(euler[:,1]**2))),
        'pitch_peak_abs_rad': float(np.max(np.abs(euler[:,1]))),
        'root_z_rms_about_mean_m': float(np.sqrt(np.mean((z-z.mean())**2))),
        'min_urdf_position_margin_rad': min(float(r['min_position_margin_rad']) for r in walking),
        'max_dq_to_urdf_limit_ratio': max(float(r['max_dq_limit_ratio']) for r in walking),
        'torque_saturation_fraction_mean': float(np.mean([float(r['torque_saturation_fraction']) for r in walking])),
        'torque_saturation_fraction_peak': max(float(r['torque_saturation_fraction']) for r in walking),
        'contact_foot_count_distribution': {str(n): int(np.count_nonzero(contact_counts == n)) for n in range(7)},
        'continuous_contact_slip_speed_mean_mps': float(slip.mean()) if slip.size else None,
        'continuous_contact_slip_speed_peak_mps': float(slip.max()) if slip.size else None,
        'continuous_contact_cumulative_drift_m': continuous_drift,
        'touchdown_transitions': touchdown_n,
        'joint_position_limits_respected': min(float(r['min_position_margin_rad']) for r in walking) >= -1e-8,
        'joint_velocity_limits_respected': max(float(r['max_dq_limit_ratio']) for r in walking) <= 1.0+1e-8,
        'torque_saturation_observed': any(float(r['torque_saturation_fraction']) > 0 for r in walking),
    }
    summary['physics_joint_audit_valid'] = (summary['joint_position_limits_respected']
                                            and summary['joint_velocity_limits_respected'])
    summary['acceptance_status'] = ('joint_limits_within_measured_trace_review_required'
                                    if summary['physics_joint_audit_valid']
                                    else 'invalid_joint_limit_excursion')
    return summary


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
    from pcr_sim_bridge import (PcrSimTap, TrialIdentity, canonical_toe_permutation,
                                read_hexground_state)

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
        simtime = 0.0
        control_dt = float(env.dt)
        if not np.isfinite(control_dt) or control_dt <= 0:
            raise ValueError(f'Invalid control dt: {control_dt}')
        settle_window_samples = int(math.ceil(.5/control_dt))
        stand_states = []
        stand_started_s = simtime
        stable_since = None
        stand_samples = 0
        walk_samples = 0
        terminal_state_captured = False
        with PcrSimTap(outfile, identity, canonical_feet_permutation=perm) as tap:
            # Settle on the default PCR joint target; keep stand rows separate.
            while simtime - stand_started_s < 6.0:
                actions, stance = ctrl.action(0.0)
                action_t = torch.as_tensor(actions, dtype=torch.float32, device=env.device)
                _, _, _, dones, _ = env.step(action_t)
                simtime += float(env.dt)
                if bool(dones[0].item()):
                    raise RuntimeError(f'Environment terminated/reset during stand step {stand_samples+1}; terminal_state_captured={terminal_state_captured}')
                state = read_hexground_state(env, 0)
                stand_states.append(state)
                tap.append(env, 0, simtime, [0.,0.,0.], stance,
                           phase='stand', reference_limited=ctrl.reference_limited)
                stand_samples += 1
                if simtime >= 2.0 and stand_window_is_stable(stand_states, settle_window_samples):
                    stable_since = simtime - settle_window_samples*control_dt
                    break
            if stable_since is None:
                raise RuntimeError('Stand settle gate failed within 6 s; walking was not started')
            ctrl.start_walking()
            print(f'[pilot] stand gate passed: samples={stand_samples} stable_from={stable_since:.2f}s')
            for i in range(args.steps):
                walk_time = i * float(env.dt)
                ramp = min(1.0, walk_time / args.ramp_s)
                speed = args.forward_mps * ramp
                actions, stance = ctrl.action(speed)
                action_t = torch.as_tensor(actions, dtype=torch.float32, device=env.device)
                _, _, _, dones, _ = env.step(action_t)
                simtime += float(env.dt)
                if bool(dones[0].item()):
                    raise RuntimeError(f'Environment terminated/reset after walking step {i+1}; terminal_state_captured=false')
                tap.append(env, 0, simtime, [0.,speed,0.], stance,
                           phase='walking', reference_limited=ctrl.reference_limited)
                walk_samples += 1
                if i == 0 or (i + 1) % 50 == 0:
                    print(f'[pilot] walking {i+1}/{args.steps} t={walk_time:.2f}s pose_z={float(env.root_states[0,2]):.3f}')
        metrics = summarize_csv(outfile)
        walking_outfile = outfile.with_name(outfile.stem+'_walking.csv')
        with outfile.open(newline='', encoding='utf-8') as source, \
                walking_outfile.open('w', newline='', encoding='utf-8') as target:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row['phase'] == 'walking':
                    writer.writerow(row)
        return {'status':'completed_raw_physics_trial','backend':f'PhysX/{args.sim_device}',
                'stand_samples':stand_samples,'walking_samples':walk_samples,
                'terminal_state_captured':terminal_state_captured,
                'control_dt_s':control_dt,'stand_stable_window_samples':settle_window_samples,
                'csv':str(outfile),'walking_only_csv':str(walking_outfile),'metrics':metrics}
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
    info = {'status':'started','backend':f'PhysX/{args.sim_device}',
            'terminal_state_captured':False,
            'settle_gate':{'minimum_stand_s':2.0,'stable_window_s':0.5,
                           'max_settle_s':6.0,'minimum_contacts':3,
                           'max_abs_roll_pitch_rad':0.15,'max_linear_speed_mps':0.12,
                           'max_angular_speed_rad_s':0.25,'max_abs_vertical_speed_mps':0.06,
                           'max_dq_limit_ratio':1.0,'torque_saturation_allowed':False},
            'csv':str(args.output_root / f'{args.pair_id}_{args.controller}.csv'),
            'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    try:
        result = run(args)
        info.update(result)
        if result['metrics'].get('physics_joint_audit_valid') is not True:
            info['status'] = 'failed_physics_audit'
            info['completed_raw_trial'] = True
            info['failure_reason'] = 'Measured joint position/velocity limits were not validated'
            print(json.dumps(info, indent=2))
            return 2
        print(json.dumps(info, indent=2))
        return 0
    except Exception as exc:
        info.update(status='failed', failure_type=type(exc).__name__, failure_reason=str(exc))
        traceback.print_exc()
        return 1
    finally:
        sidecar.write_text(json.dumps(info, indent=2, ensure_ascii=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    sys.exit(main())
