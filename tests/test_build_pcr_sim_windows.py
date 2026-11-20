"""Run the window-label CLI against temporary, synthetic rollover logs."""
import json
from pathlib import Path
import subprocess
import sys

from test_pcr_sim_bridge import good_calibration, predecision, write_trace


def test_window_builder_protects_its_input_files(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/build_pcr_sim_windows.py'
    data = tmp_path / 'data'
    data.mkdir()
    off_trace = write_trace(data / 'off.csv', 0)
    on_trace = write_trace(data / 'on.csv', 1)
    calibration = data / 'camera.json'
    good_calibration(calibration)
    manifest = data / 'manifest.json'
    manifest.write_text(json.dumps({'trials': [
        {'trial_id': 'rollout_off', 'log_csv': off_trace.name, 'predecision': predecision()},
        {'trial_id': 'rollout_on', 'log_csv': str(on_trace), 'predecision': predecision()},
    ]}), encoding='utf-8')

    alias = tmp_path / 'raw-alias.csv'
    alias.hardlink_to(off_trace)
    inputs = [manifest, calibration, off_trace, on_trace]
    contents = [path.read_bytes() for path in inputs]
    command = [
        sys.executable, str(script), '--manifest', str(manifest),
        '--calibration', str(calibration), '--output',
    ]
    for output in [manifest, calibration, off_trace, alias]:
        result = subprocess.run(command + [str(output)], cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 2
        assert 'Output would overwrite input' in result.stderr
        assert [path.read_bytes() for path in inputs] == contents

    output = tmp_path / 'results' / 'labels.csv'
    result = subprocess.run(command + [str(output)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Wrote 2 window labels from 1 off/on pairs' in result.stdout
    assert output.is_file()
