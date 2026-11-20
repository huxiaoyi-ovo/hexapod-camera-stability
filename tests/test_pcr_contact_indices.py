"""Verify that raw force capture cannot alias one foot to another."""
import numpy as np
import pytest

from pcr_sim_bridge import PcrSimTap, TrialIdentity, read_hexground_state
from test_pcr_sim_bridge import mock_env


@pytest.mark.parametrize('last_index', [4, 5.5, np.nan])
def test_hexground_rejects_malformed_foot_indices(last_index):
    env = mock_env()
    env.feet_indices = np.array([0, 1, 2, 3, 4, last_index])
    with pytest.raises(ValueError, match='six distinct nonnegative integer foot indices'):
        read_hexground_state(env)


def test_tap_captures_a_fixed_canonical_permutation(tmp_path):
    identity = TrialIdentity('flat_on', 'flat', 'flat', 1, True, 'gaitA')
    permutation = np.arange(6)
    with PcrSimTap(tmp_path / 'trial.csv', identity,
                   canonical_feet_permutation=permutation) as tap:
        permutation[:] = permutation[::-1]
        for t in (0., .02, .04):
            tap.append(mock_env(t), 0, t, [0., .15, 0.], [True] * 6)
    np.testing.assert_array_equal(tap.feet_perm, np.arange(6))
