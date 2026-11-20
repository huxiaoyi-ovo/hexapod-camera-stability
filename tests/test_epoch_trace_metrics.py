"""Check that absolute clock offsets do not distort frame integration."""
import numpy as np
import pytest

from controller import Camera, TrackedScene
from trace_metrics import image_motion_metrics


@pytest.mark.parametrize(('duration', 'frames'), [(.10, 2), (.09, 1)])
def test_camera_integrates_complete_frames_with_unix_epoch_stamps(duration, frames):
    t = np.linspace(0., duration, 101)
    camera = Camera(100., 100., 50., 50., .02, .05)
    scene = TrackedScene(np.array([[50., 50.]]), np.array([2.]), np.array([1.]))
    residual = np.zeros((len(t), 6))
    residual[:, 0] = .1
    for times in (t, t + 1_700_000_000., t - 1_700_000_000.):
        metrics = image_motion_metrics(times, residual, camera, scene)
        assert metrics.n_frames == frames
        assert metrics.jitter_px_rms_per_frame == pytest.approx(.25, abs=1e-12)
        assert metrics.exposure_trace_px_rms == pytest.approx(.10, abs=1e-12)
