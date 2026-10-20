#!/usr/bin/env python3
"""Launch two independent PhysX trials with identical settings, compensator off/on.

This orchestrates the P0 pilot only. It does not train a surrogate and does not
claim an outcome until both new simulator processes exit successfully.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def make_commands(python, pilot, shared_args):
    base = [str(python), str(pilot), *shared_args]
    return (base + ['--controller','off'], base + ['--controller','on'])


def validate_pair(output_root, pair_id):
    outputs=[]
    for condition in ('off','on'):
        status_path=output_root / f'{pair_id}_{condition}.status.json'
        payload=json.loads(status_path.read_text(encoding='utf-8'))
        if payload['status'] != 'completed_raw_physics_trial':
            raise RuntimeError(f'Failed/unfinished trial: {status_path}')
        trace=Path(payload['csv'])
        if not trace.is_file():
            raise FileNotFoundError(trace)
        outputs.append((payload,trace))
    left,right=outputs
    for field in ('pair_id','terrain_instance','seed','frequency_hz',
                  'duty_factor','swing_height','touchdown_softness','forward_mps'):
        if left[0]['parameters'][field]!=right[0]['parameters'][field]:
            raise RuntimeError(f'Mismatched paired trial configuration: {field}')
    return [str(v[1]) for v in outputs]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--pcr-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--pair-id',default='flat_pilot_001')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--steps',type=int,default=400)
    parser.add_argument('--frequency-hz',type=float,default=1.1)
    parser.add_argument('--duty-factor',type=float,default=.58)
    parser.add_argument('--swing-height',type=float,default=.025)
    parser.add_argument('--touchdown-softness',type=float,default=.5)
    parser.add_argument('--forward-mps',type=float,default=.08)
    parser.add_argument('--ramp-s',type=float,default=1.)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--sim-device',default='cuda:0')
    args=parser.parse_args(argv)
    pilot=args.research_root/'integration/pcrnet/collect_camera_stable_gaits.py'
    if not pilot.is_file():
        parser.error(f'Pilot script missing: {pilot}')
    common=['--research-root',str(args.research_root),'--pcr-root',str(args.pcr_root),
            '--output-root',str(args.output_root),'--pair-id',args.pair_id,
            '--seed',str(args.seed),'--steps',str(args.steps),
            '--frequency-hz',str(args.frequency_hz),'--duty-factor',str(args.duty_factor),
            '--swing-height',str(args.swing_height),'--touchdown-softness',str(args.touchdown_softness),
            '--forward-mps',str(args.forward_mps),'--ramp-s',str(args.ramp_s),
            '--sim-device',args.sim_device]
    if args.headless:common.append('--headless')
    commands=make_commands(sys.executable,pilot,common)
    for command in commands:
        print('RUN:', ' '.join(command),flush=True)
        subprocess.run(command,check=True)
    print(json.dumps({'status':'paired_raw_physx_complete',
                      'traces':validate_pair(args.output_root,args.pair_id)},indent=2))
    return 0


if __name__=='__main__':
    sys.exit(main())