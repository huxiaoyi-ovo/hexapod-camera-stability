"""Small synthetic numerical checks only, NOT locomotion evidence."""
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from closed_loop_surrogate import (
    feature_vector, FixedCompensatorResponseModel, rank_candidates,
)


def row(i, on, terrain='TRAIN_A'):
    # Artificial signals to verify math plumbing; NEVER report as experiment.
    gait_i = i % 8
    gait_f = 1.0 + .12*gait_i
    b = 0.52 + .025*(i%4)
    return dict(
        frequency_hz=gait_f, duty_factor=b, step_length_m=.08,
        swing_height_m=.065, touchdown_softness=(i%5)/5.,
        desired_speed_mps=.15, roughness_est_m=.01*(i%6),
        slope_est_rad=.02*(i%5), planned_stance_count=3+(i%4),
        predicted_joint_headroom=.4 + .05*(i%6),
        phase_rad=(i%9)*.5, roll_rate_rad_s=.03, pitch_rate_rad_s=-.02,
        previous_frequency_hz=1.2, previous_duty_factor=.60,
        previous_touchdown_softness=.4, time_since_gait_change_s=.25,
        pair_id=f'p{i}', terrain_instance_id=terrain, seed=i, gait_id=f'g{i%8}', controller_on=on,
        measured_jitter_px_frame=3. + .5*gait_f + .4*(i%6) - (1. + .13*(i%5) if on else 0.),
        measured_speed_mps=.15,
        pre_screened_feasible=True,
    )


def test_features_only_predecision():
    r=row(0, 1)
    vec=feature_vector(r)
    assert vec.shape==(22,)
    assert np.all(np.isfinite(vec))
    assert not np.any(np.isnan(vec))


def test_never_predict_untrained():
    with pytest.raises(RuntimeError):
        FixedCompensatorResponseModel().predict(row(0,1))


def test_paired_data_required():
    with pytest.raises(ValueError, match='off/on'):
        FixedCompensatorResponseModel().fit([row(i, 1) for i in range(10)])


def test_predict_nonnegative_and_two_objectives():
    model=FixedCompensatorResponseModel().fit([row(i, on) for i in range(50) for on in (0,1)])
    r=model.predict(row(4, 1))
    assert r.raw_jitter_px>=0 and r.closed_loop_jitter_px>=0
    assert np.isfinite(r.effective_jitter_reduction_px)
    can=[row(i, 1) for i in range(4)]
    for objective in ('raw','closed_loop'):
        ans=rank_candidates(model, can, can[0], .15, objective=objective, min_speed_fraction=.80)
        assert ans and np.isfinite(ans[0].score)


def test_reject_unverified_candidates():
    model=FixedCompensatorResponseModel().fit([row(i, on) for i in range(50) for on in (0,1)])
    candidate=row(0,1); candidate['pre_screened_feasible']=False
    assert rank_candidates(model,[candidate],candidate,.15)==[]


def test_pair_rejects_different_initial_conditions():
    rows=[row(i,on) for i in range(5) for on in (0,1)]
    rows[1]['phase_rad'] += .2
    with pytest.raises(ValueError, match='predecision inputs differ'):
        FixedCompensatorResponseModel().fit(rows)


def test_pair_rejects_false_seed_pairing():
    rows=[row(i,on) for i in range(5) for on in (0,1)]
    rows[1]['seed'] = 9
    with pytest.raises(ValueError, match='off/on'):
        FixedCompensatorResponseModel().fit(rows)


@pytest.mark.parametrize(('heldout_pair', 'conditions', 'bad_jitter', 'error'), [
    (0, (0, 1), None, 'Paired trials leaked across partitions'),
    (5, (0, 1), None, None),
    (5, (0,), None, 'Missing off/on trial'),
    (5, (0, 1, 1), None, 'Duplicate controller_on=1 window'),
    (5, (0, 1), 'nan', 'Held-out measured jitter must be finite and nonnegative'),
    (5, (0, 1), 'inf', 'Held-out measured jitter must be finite and nonnegative'),
    (5, (0, 1), '-1', 'Held-out measured jitter must be finite and nonnegative'),
])
def test_evaluation_checks_heldout_rows(tmp_path, monkeypatch, heldout_pair, conditions, bad_jitter, error):
    monkeypatch.delenv('PYTHONPATH', raising=False)
    rows = [row(i, on) for i in range(5) for on in (0, 1)]
    rows.extend(row(heldout_pair, on, 'HELD_OUT') for on in conditions)
    if bad_jitter is not None:
        rows[-1]['measured_jitter_px_frame'] = bad_jitter
    path = tmp_path / 'windows.csv'
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    script = Path(__file__).resolve().parents[1] / 'scripts/evaluate_response_surrogate.py'
    result = subprocess.run(
        [sys.executable, str(script), '--input', str(path), '--test-terrains', 'HELD_OUT'],
        cwd=tmp_path, capture_output=True, text=True,
    )
    if error:
        assert result.returncode == 2
        assert error in result.stderr
        assert 'MAE_px_frame' not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.count('heldout_windows=1') == 2
