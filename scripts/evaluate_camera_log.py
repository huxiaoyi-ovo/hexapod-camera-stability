#!/usr/bin/env python3
"""Compute reproducible projected camera-motion proxies from measured traces."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from file_paths import atomic_output_path, validate_output_path
from trace_metrics import read_trace, image_motion_metrics, camera_scene_from_json

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace', type=Path, required=True, help='CSV with actual/nominal optical twists')
    p.add_argument('--camera', type=Path, required=True, help='JSON scene and camera calibration')
    p.add_argument('--output', type=Path, required=True, help='Output JSON, never overwrites inputs')
    args = p.parse_args()
    trace_path = args.trace.resolve()
    camera_path = args.camera.resolve()
    try:
        output_path = validate_output_path(args.output, (trace_path, camera_path))
    except ValueError as error:
        p.error(str(error))

    t, actual, nominal = read_trace(trace_path)
    camera, scene = camera_scene_from_json(camera_path)
    result = image_motion_metrics(t, actual-nominal, camera, scene)
    output = result.to_dict()
    output['source_trace'] = str(trace_path)
    output['camera_configuration'] = str(camera_path)
    output['interpretation'] = 'projected motion proxy only; NOT rendered or measured RGB blur'
    output['under_sampled_exposure'] = output['median_samples_per_exposure'] < 3
    contents = json.dumps(output, indent=2)
    with atomic_output_path(output_path) as temporary:
        temporary.write_text(contents, encoding='utf-8')
    print(contents)

if __name__ == '__main__':
    main()
