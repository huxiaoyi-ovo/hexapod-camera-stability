"""Check validation where timestamped poses enter the camera recorder."""
import numpy as np
import pytest

from camera_pose_recorder import CameraPoseRecorder


IDENTITY = [0, 0, 0, 1]


@pytest.mark.parametrize(('time', 'position', 'rotation', 'message'), [
    (np.nan, [0, 0, 0], IDENTITY, 'Sampling time must be finite'),
    (1.0, [0, np.inf, 0], IDENTITY, 'Bad base position'),
    (1.0, [0, 0, 0], [IDENTITY], 'Quaternion must be a finite'),
])
def test_invalid_pose_does_not_pollute_recording(time, position, rotation, message):
    recorder = CameraPoseRecorder([0, 0, 0], IDENTITY)
    recorder.append(0.0, [0, 0, 0], IDENTITY, [0] * 6)
    with pytest.raises(ValueError, match=message):
        recorder.append(time, position, rotation, [0] * 6)
    recorder.append(1.0, [0, 0, 0], IDENTITY, [0] * 6)
    assert [sample[0] for sample in recorder.samples] == [0.0, 1.0]


def test_camera_extrinsic_requires_single_quaternion():
    with pytest.raises(ValueError, match='Quaternion must be a finite'):
        CameraPoseRecorder([0, 0, 0], [IDENTITY])


def test_reused_input_buffers_do_not_change_calibration_or_old_samples():
    offset = np.array([.1, 0, 0])
    nominal = np.zeros(6)
    recorder = CameraPoseRecorder(offset, IDENTITY)
    recorder.append(0.0, [0, 0, 0], IDENTITY, nominal)
    offset[0] = .2
    nominal[0] = .3
    recorder.append(1.0, [0, 0, 0], IDENTITY, nominal)
    assert [sample[1][0] for sample in recorder.samples] == [.1, .1]
    assert [sample[3][0] for sample in recorder.samples] == [0, .3]


def test_twist_recorder_cannot_overwrite_existing_raw_log(tmp_path):
    recorder = CameraPoseRecorder([0, 0, 0], IDENTITY)
    for t in (0., .05, .10):
        recorder.append(t, [t, 0, 0], IDENTITY, [0] * 6)
    path = tmp_path / 'raw.csv'
    recorder.save_twist_csv(path)
    original = path.read_bytes()

    recorder.append(.15, [.3, 0, 0], IDENTITY, [0] * 6)
    with pytest.raises(FileExistsError):
        recorder.save_twist_csv(path)
    assert path.read_bytes() == original
