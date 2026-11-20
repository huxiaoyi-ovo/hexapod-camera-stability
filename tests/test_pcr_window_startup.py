"""Check window labels for a simulator rollout with a zero-command start."""
import numpy as np
import pytest

from pcr_sim_bridge import PcrSimTap, TrialIdentity
from pcr_sim_windows import build_response_window
from test_pcr_sim_bridge import good_calibration, mock_env, predecision


@pytest.mark.parametrize('speed', [.15, 0.])
def test_window_labels_accept_zero_start_if_translation_follows(tmp_path, speed):
    calibration = tmp_path / 'camera.json'
    good_calibration(calibration)
    trace = tmp_path / 'rollout.csv'
    identity = TrialIdentity('ramp_off', 'ramp', 'flat', 1, False, 'gaitA')
    with PcrSimTap(trace, identity) as tap:
        for i, t in enumerate(np.linspace(0., .6, 31)):
            tap.append(mock_env(t, speed), 0, t, [0., speed if i else 0., 0.], [True] * 6)

    if not speed:
        with pytest.raises(ValueError, match='requires forward translation'):
            build_response_window(trace, predecision(), calibration)
    else:
        result = build_response_window(trace, predecision(), calibration)
        assert result['measured_speed_mps'] == pytest.approx(speed)
