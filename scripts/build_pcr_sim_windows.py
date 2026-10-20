#!/usr/bin/env python3
"""Convert recorded PCR HexGround trial CSVs + predecision manifest to labels.

Usage: PYTHONPATH=src python scripts/build_pcr_sim_windows.py \\
   --manifest outputs/trial_manifest.json --calibration config/camera.json \\
   --output outputs/response_windows.csv
"""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcr_sim_windows import build_response_window,validate_pair_balance,write_response_windows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--calibration',required=True)
    parser.add_argument('--output',required=True)
    a=parser.parse_args()
    trial_items=json.loads(Path(a.manifest).read_text(encoding='utf-8'))['trials']
    if not trial_items:
        raise ValueError('Manifest has no trials')
    base=Path(a.manifest).resolve().parent
    rows=[]
    for i,item in enumerate(trial_items):
        log=Path(item['log_csv'])
        if not log.is_absolute():log=base/log
        result=build_response_window(log,item['predecision'],a.calibration,
                                     item.get('window_start_s'),item.get('window_end_s'))
        if result['trial_id']!=item['trial_id']:
            raise ValueError(f'Trial mismatch in entry {i}')
        rows.append(result)
    count=validate_pair_balance(rows)
    write_response_windows(a.output,rows)
    print(f'Wrote {len(rows)} window labels from {count} off/on pairs -> {a.output}')
    print('Physical metrics: camera-geometry proxy only. Not actual RGB sharpness.')
if __name__=='__main__':main()
