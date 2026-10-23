"""Offline proof-of-concept tests; no motor commands / no PhysX required."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stance_force_allocator import allocate_stance_forces


def setup():
    r = np.array([[0.30, 0.0, 0.0], [-0.15, 0.2598076, 0.0],
                  [-0.15, -0.2598076, 0.0]])
    normals = np.tile([0., 0., 1.], (3, 1))
    return r, normals


def test_symmetric_tripod_supports_weight_without_tilt():
    r, normals = setup()
    result = allocate_stance_forces(
        r, normals, [True]*3, [0., 0., 60., 0., 0., 0.],
        friction_coefficient=0.6, max_normal_force_n=40.)
    assert result["optimizer_success"], result["message"]
    assert result["meets_wrench"]
    assert np.allclose(result["forces_n"][:, 2], 20., atol=0.02)
    assert np.max(np.abs(result["forces_n"][:, :2])) < 1e-3
    assert np.linalg.norm(result["wrench_residual"]) < 0.5


def test_low_friction_cannot_fake_unavailable_horizontal_force():
    r, normals = setup()
    result = allocate_stance_forces(
        r, normals, [True]*3, [25., 0., 60., 0., 0., 0.],
        friction_coefficient=0.1, max_normal_force_n=40.)
    assert result["optimizer_success"]
    assert not result["meets_wrench"]
    for f in result["forces_n"]:
        assert abs(f[0]) <= 0.1 * f[2]/np.sqrt(2.) + 1e-5
        assert abs(f[1]) <= 0.1 * f[2]/np.sqrt(2.) + 1e-5


def test_inactive_foot_always_exerts_zero_desired_grf():
    r, normals = setup()
    result = allocate_stance_forces(
        r, normals, [True, False, True], [0., 0., 40., 0., 0., 0.],
        friction_coefficient=0.6, max_normal_force_n=40.)
    assert result["optimizer_success"]
    assert np.array_equal(result["forces_n"][1], np.zeros(3))


def test_no_support_cannot_be_marked_valid():
    r, normals = setup()
    result = allocate_stance_forces(
        r, normals, [False]*3, [0., 0., 60., 0., 0., 0.],
        friction_coefficient=0.6, max_normal_force_n=40.)
    assert not result["optimizer_success"]
    assert not result["meets_wrench"]
    assert np.count_nonzero(result["forces_n"]) == 0


def test_zero_contact_normal_rejected():
    r, normals = setup()
    normals[0] = 0.
    with pytest.raises(ValueError):
        allocate_stance_forces(
            r, normals, [True]*3, [0., 0., 60., 0., 0., 0.],
            friction_coefficient=0.6, max_normal_force_n=40.)
