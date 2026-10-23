"""Differentiable stationary-landmark image-motion residual for camera-stability RL.

World-to-image projection assumes rigid camera poses R_wc, t_wc; the world
points are stationary. Nominal ego-motion is subtracted before scoring. This
is an analytic proxy (px / frame), not a measurement of RGB image sharpness.
"""
from __future__ import annotations

import torch


def _batched_project(world_points, rotation_wc, translation_wc, intrinsics, min_depth=1e-4):
    """Project (N,3) fixed world landmarks for a batch of camera poses.

    Rotation has shape (B,3,3) and maps camera coordinates to world axes.
    Translation has shape (B,3); intrinsics is (3,3) or (B,3,3).
    """
    if world_points.ndim != 2 or world_points.shape[-1] != 3:
        raise ValueError("world_points must have shape (N,3)")
    if rotation_wc.ndim != 3 or rotation_wc.shape[-2:] != (3, 3):
        raise ValueError("rotation_wc must have shape (B,3,3)")
    batch = rotation_wc.shape[0]
    if translation_wc.shape != (batch, 3):
        raise ValueError("translation_wc must have shape (B,3)")
    if intrinsics.shape not in ((3, 3), (batch, 3, 3)):
        raise ValueError("intrinsics must have shape (3,3) or (B,3,3)")
    points_in_camera = torch.matmul(
        world_points.unsqueeze(0) - translation_wc.unsqueeze(1), rotation_wc
    )
    z = points_in_camera[..., 2]
    valid = z > float(min_depth)
    z_safe = torch.where(valid, z, torch.ones_like(z))
    xy = points_in_camera[..., :2] / z_safe.unsqueeze(-1)
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    if intrinsics.ndim == 2:
        pixels = torch.stack((xy[..., 0] * fx + cx, xy[..., 1] * fy + cy), dim=-1)
    else:
        pixels = torch.stack((xy[..., 0] * fx[:, None] + cx[:, None],
                              xy[..., 1] * fy[:, None] + cy[:, None]), dim=-1)
    return pixels, valid


def residual_pixel_motion_rms(world_points, intrinsics,
                              actual_rotation_before, actual_position_before,
                              actual_rotation_after, actual_position_after,
                              nominal_rotation_before, nominal_position_before,
                              nominal_rotation_after, nominal_position_after,
                              min_depth=1e-4):
    """Return per-batch RMS unwanted displacement, in pixels per frame.

    Computes actual flow minus nominal commanded flow on a shared stationary
    landmark set. Nominal camera translation or turning alone scores zero.
    """
    a0, v_a0 = _batched_project(world_points, actual_rotation_before,
                               actual_position_before, intrinsics, min_depth)
    a1, v_a1 = _batched_project(world_points, actual_rotation_after,
                               actual_position_after, intrinsics, min_depth)
    n0, v_n0 = _batched_project(world_points, nominal_rotation_before,
                               nominal_position_before, intrinsics, min_depth)
    n1, v_n1 = _batched_project(world_points, nominal_rotation_after,
                               nominal_position_after, intrinsics, min_depth)
    valid = v_a0 & v_a1 & v_n0 & v_n1
    flow_difference = (a1 - a0) - (n1 - n0)
    per_point_sq = (flow_difference * flow_difference).sum(dim=-1)
    count = valid.sum(dim=1)
    if bool(torch.any(count == 0)):
        raise ValueError("no landmark remains visible across all four camera poses")
    return torch.sqrt((per_point_sq * valid.to(per_point_sq.dtype)).sum(dim=1) /
                      count.to(per_point_sq.dtype) + 1e-12)
