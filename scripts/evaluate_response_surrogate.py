#!/usr/bin/env python3
"""Train on REAL trial CSV; hold out complete terrain instances, print predictions only.

Usage: python scripts/evaluate_response_surrogate.py \
  --input path/to/real_response_windows.csv --test-terrains heldout_01 heldout_02
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from closed_loop_surrogate import read_trials_csv, FixedCompensatorResponseModel
from sim_split import split_trials


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--test-terrains', nargs='+', required=True)
    args=p.parse_args()
    try:
        train, test = split_trials(read_trials_csv(args.input), args.test_terrains)
    except ValueError as error:
        p.error(str(error))
    model=FixedCompensatorResponseModel(alpha=1.).fit(train)
    for enabled in (0,1):
        rows=[r for r in test if int(r['controller_on'])==enabled]
        if not rows:
            raise SystemExit(f'Held-out data lacks controller_on={enabled} samples')
        pred=np.array([model.predict(row).closed_loop_jitter_px if enabled else model.predict(row).raw_jitter_px for row in rows])
        true=np.array([float(row['measured_jitter_px_frame']) for row in rows])
        # Error is reported over WINDOW rows, NOT independent experiment trials.
        mae=float(np.mean(np.abs(pred-true)))
        print(f'condition={"on" if enabled else "off"} heldout_windows={len(rows)} MAE_px_frame={mae:.4f}')
    print('WARNING: window-level MAE is diagnostic only; use independent terrain/rollout units for inferential statistics.')

if __name__=='__main__':
    main()
