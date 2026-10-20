"""Small synthetic numerical checks only, NOT locomotion evidence."""
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
