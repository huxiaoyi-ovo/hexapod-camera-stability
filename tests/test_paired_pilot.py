"""Offline checks for the paired pilot's status and trace contract."""
import json
from pathlib import Path
import shutil

import pytest


@pytest.fixture
def pair_validator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from run_pcr_paired_pilot import validate_pair

    return validate_pair


def write_status(directory, condition, **overrides):
    trace = directory / f'pair01_{condition}.csv'
    if not trace.exists():
        trace.touch()
    params = {
        'pair_id': 'pair01', 'controller': condition, 'terrain_instance': 'flat',
        'task': 'hex_debug_plane', 'seed': 42, 'steps': 400, 'frequency_hz': 1.1,
        'duty_factor': .58, 'swing_height': .025, 'touchdown_softness': .5,
        'forward_mps': .08, 'ramp_s': 1.0, 'sim_device': 'cuda:0',
    }
    params.update(overrides)
    status = {
        'status': 'completed_raw_physics_trial',
        'csv': str(trace),
        'parameters': params,
    }
    (directory / f'pair01_{condition}.status.json').write_text(json.dumps(status), encoding='utf-8')
    return trace


def test_pair_validation_accepts_distinct_matching_trials(tmp_path, pair_validator):
    traces = [write_status(tmp_path, condition) for condition in ('off', 'on')]
    assert pair_validator(tmp_path, 'pair01') == [str(trace) for trace in traces]


def test_pair_validation_uses_the_trace_next_to_the_archived_status(tmp_path, pair_validator):
    source = tmp_path / 'simulator'
    source.mkdir()
    traces = [write_status(source, condition) for condition in ('off', 'on')]
    archive = tmp_path / 'analysis'
    shutil.copytree(source, archive)

    expected = [str(archive / trace.name) for trace in traces]
    assert pair_validator(archive, 'pair01') == expected
    shutil.rmtree(source)
    assert pair_validator(archive, 'pair01') == expected


def test_pair_validation_rejects_status_pointing_to_a_different_trial(tmp_path, pair_validator):
    write_status(tmp_path, 'off')
    on = write_status(tmp_path, 'on')
    status = on.with_suffix('.status.json')
    payload = json.loads(status.read_text(encoding='utf-8'))
    wrong_trace = tmp_path / 'other_on.csv'
    wrong_trace.touch()
    payload['csv'] = str(wrong_trace)
    status.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(RuntimeError, match='Trace filename does not match'):
        pair_validator(tmp_path, 'pair01')


@pytest.mark.parametrize(('overrides', 'message'), [
    ({'controller': 'off'}, 'Trial identity'),
    ({'pair_id': 'other'}, 'Trial identity'),
    ({'ramp_s': 2.0}, 'configuration: ramp_s'),
    ({'steps': 800}, 'configuration: steps'),
])
def test_pair_validation_rejects_mismatched_trials(tmp_path, pair_validator, overrides, message):
    write_status(tmp_path, 'off')
    write_status(tmp_path, 'on', **overrides)
    with pytest.raises(RuntimeError, match=message):
        pair_validator(tmp_path, 'pair01')


def test_pair_validation_rejects_shared_trace(tmp_path, pair_validator):
    off_trace = write_status(tmp_path, 'off')
    on_trace = write_status(tmp_path, 'on')
    on_trace.unlink()
    on_trace.hardlink_to(off_trace)
    with pytest.raises(RuntimeError, match='separate trace files'):
        pair_validator(tmp_path, 'pair01')
