"""Check that short and failed captures preserve the useful raw evidence."""
import csv

import pytest

from pcr_sim_bridge import PcrSimTap, TrialIdentity
from test_pcr_sim_bridge import mock_env


@pytest.mark.parametrize('simulator_failed', [False, True])
def test_partial_rollout_preserved_without_masking_its_failure(tmp_path, simulator_failed):
    trace = tmp_path / 'partial.csv'
    identity = TrialIdentity('partial_on', 'partial', 'flat', 1, True, 'gaitA')
    error = RuntimeError if simulator_failed else ValueError
    message = 'test simulator failed' if simulator_failed else 'Only 1 samples'
    with pytest.raises(error, match=message), PcrSimTap(trace, identity) as tap:
        tap.append(mock_env(), 0, 0.0, [0., .15, 0.], [True] * 6)
        if simulator_failed:
            raise RuntimeError(message)

    assert tap.handle.closed
    with trace.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    assert [row['trial_id'] for row in rows] == ['partial_on']
