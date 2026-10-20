#!/usr/bin/env python3
"""Compute reproducible projected camera-motion proxies from measured traces."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from trace_metrics import read_trace, image_motion_metrics, camera_scene_from_json

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace', required=True, help='CSV with actual/nominal optical twists')
    p.add_argument('--camera', required=True, help='JSON scene and camera calibration')
    p.add_argument('--output', required=True, help='Output JSON, never overwrites raw logs')
    args = p.parse_args()
    t, actual, nominal = read_trace(args.trace)
    camera, scene = camera_scene_from_json(args.camera)
    result = image_motion_metrics(t, actual-nominal, camera, scene)
    output = result.to_dict()
    output['source_trace'] = str(Path(args.trace).resolve())
    output['camera_configuration'] = str(Path(args.camera).resolve())
    output['interpretation'] = 'projected motion proxy only; NOT rendered or measured RGB blur'
    output['under_sampled_exposure'] = output['median_samples_per_exposure'] < 3
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    main()
