"""CPU-only, simulator-independent tests of the analytic camera proxy."""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from image_stability_objective import residual_pixel_motion_rms


def inputs(dtype=torch.float64):
    k = torch.tensor([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]], dtype=dtype)
    points = torch.tensor([[-0.4, 0.3, 2.0], [0.1, -0.2, 3.0],
                           [0.25, 0.2, 4.0]], dtype=dtype)
    eye = torch.eye(3, dtype=dtype).unsqueeze(0)
    zero = torch.zeros((1, 3), dtype=dtype)
    return points, k, eye, zero


def score(points, k, r0, p0, r1, p1, nr0, np0, nr1, np1):
    return residual_pixel_motion_rms(points, k, r0, p0, r1, p1,
                                     nr0, np0, nr1, np1)[0]


def test_zero_residual_for_identical_nominal_movement():
    points, k, eye, zero = inputs()
    next_position = torch.tensor([[0.0, 0.0, 0.13]], dtype=points.dtype)
    value = score(points, k, eye, zero, eye, next_position,
                  eye, zero, eye, next_position)
    assert value.item() < 2e-6


def test_lateral_camera_jitter_matches_pinhole_projection():
    points, k, eye, zero = inputs()
    shift = torch.tensor([[0.02, 0.0, 0.0]], dtype=points.dtype)
    value = score(points, k, eye, zero, eye, shift,
                  eye, zero, eye, zero)
    expected = math.sqrt(sum((500.0 * 0.02 / d) ** 2 for d in (2., 3., 4.)) / 3)
    assert abs(value.item() - expected) < 1e-8


def test_jitter_gradient_flows_into_pose_action():
    points, k, eye, zero = inputs()
    dx = torch.tensor(0.02, dtype=points.dtype, requires_grad=True)
    new_position = torch.stack((dx, dx * 0., dx * 0.)).unsqueeze(0)
    value = score(points, k, eye, zero, eye, new_position,
                  eye, zero, eye, zero)
    value.backward()
    assert dx.grad is not None and dx.grad.item() > 10.


def test_no_visible_landmarks_is_an_error():
    points, k, eye, zero = inputs()
    with pytest.raises(ValueError, match="no landmark"):
        score(points, k, eye, zero, eye, torch.tensor([[0., 0., 10.]], dtype=points.dtype),
              eye, zero, eye, zero)
