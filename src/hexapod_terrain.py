"""Deterministic shallow static terrain meshes for bounded walking trials."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Mapping, Tuple

import numpy as np


def _mesh_from_grid(xs: np.ndarray, ys: np.ndarray, heights: np.ndarray):
    vertices = np.array([[x, y, heights[j, i]] for j, y in enumerate(ys)
                         for i, x in enumerate(xs)], dtype=np.float32)
    triangles = []
    nx = len(xs)
    for j in range(len(ys) - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b, c, d = a + 1, a + nx, a + nx + 1
            if (i + j) % 2:
                triangles.extend(((a, b, c), (b, d, c)))
            else:
                triangles.extend(((a, b, d), (a, d, c)))
    return vertices, np.asarray(triangles, dtype=np.int32)


def _add_boundary_walls(vertices: np.ndarray, triangles: np.ndarray,
                        xs: np.ndarray, ys: np.ndarray, heights: np.ndarray,
                        base_z: float = 0.0):
    """Close a gridded raised surface with a flat bottom and vertical sides."""
    verts = vertices.tolist()
    faces = triangles.tolist()
    nx, ny = len(xs), len(ys)
    bottom_offset = len(verts)
    verts.extend((float(point[0]), float(point[1]), base_z) for point in vertices)
    faces.extend((int(a) + bottom_offset, int(c) + bottom_offset, int(b) + bottom_offset)
                 for a, b, c in triangles)
    edges = ([(i, i + 1) for i in range(nx - 1)] +
             [((ny - 1) * nx + i + 1, (ny - 1) * nx + i) for i in range(nx - 1)] +
             [(j * nx + nx - 1, (j + 1) * nx + nx - 1) for j in range(ny - 1)] +
             [((j + 1) * nx, j * nx) for j in range(ny - 1)])
    for first, second in edges:
        low_first, low_second = first + bottom_offset, second + bottom_offset
        faces.extend(((first, low_second, second), (first, low_first, low_second)))
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def build_terrain(config: Mapping[str, object]) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Create a finite mesh patch; this module exposes no terrain-query API."""
    kind = str(config.get("kind", "flat"))
    def number(key: str, default: float) -> float:
        value = config.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError("terrain.{} must be a finite number".format(key))
        return float(value)

    start_y = number("start_y_m", 0.55)
    end_y = number("end_y_m", 0.85)
    width = number("width_m", 2.0)
    if kind not in ("flat", "low_step", "ramp", "bars", "rough"):
        raise ValueError("terrain kind must be flat, low_step, ramp, bars, or rough")
    if not (0 < start_y < end_y and width > 0):
        raise ValueError("terrain dimensions must define a forward patch with positive width")
    height = number("height_m", 0.005 if kind != "ramp" else 0.015)
    seed = config.get("seed", 42)
    if type(seed) is not int:
        raise ValueError("terrain.seed must be a JSON integer")
    cell = number("cell_size_m", 0.1)
    if height <= 0 or cell <= 0:
        raise ValueError("terrain height and cell size must be positive")

    if kind == "flat":
        vertices = np.empty((0, 3), dtype=np.float32)
        triangles = np.empty((0, 3), dtype=np.int32)
    elif kind == "low_step":
        xs = np.linspace(-width / 2.0, width / 2.0, max(2, int(round(width / 0.1)) + 1))
        ys = np.array([start_y, end_y])
        heights = np.full((2, len(xs)), height)
        vertices, triangles = _mesh_from_grid(xs, ys, heights)
        vertices, triangles = _add_boundary_walls(vertices, triangles, xs, ys, heights)
    elif kind == "ramp":
        xs = np.linspace(-width / 2.0, width / 2.0, max(2, int(round(width / 0.1)) + 1))
        ys = np.linspace(start_y, end_y, max(4, int(round((end_y - start_y) / 0.02)) + 1))
        length = end_y - start_y
        profile = []
        for y in ys:
            u = (y - start_y) / length
            profile.append(height * min(1.0, u / 0.4, (1.0 - u) / 0.4))
        vertices, triangles = _mesh_from_grid(xs, ys, np.repeat(np.asarray(profile)[:, None], len(xs), axis=1))
    elif kind == "bars":
        bar_width = number("bar_width_m", 0.03)
        if bar_width <= 0 or bar_width >= (end_y - start_y) / 3.0:
            raise ValueError("terrain.bar_width_m must fit inside the terrain patch")
        count = 3
        centers = np.linspace(start_y + 0.05, end_y - 0.05, count)
        vertices_list, triangles_list = [], []
        xs = (-width / 2.0, width / 2.0)
        for center in centers:
            y0, y1 = center - bar_width / 2.0, center + bar_width / 2.0
            base = len(vertices_list)
            vertices_list.extend(((xs[0], y0, 0.0), (xs[1], y0, 0.0),
                                  (xs[0], y1, 0.0), (xs[1], y1, 0.0),
                                  (xs[0], y0, height), (xs[1], y0, height),
                                  (xs[0], y1, height), (xs[1], y1, height)))
            triangles_list.extend(((base + 4, base + 5, base + 7), (base + 4, base + 7, base + 6),
                                   (base, base + 3, base + 1), (base, base + 2, base + 3),
                                   (base, base + 1, base + 5), (base, base + 5, base + 4),
                                   (base + 1, base + 3, base + 7), (base + 1, base + 7, base + 5),
                                   (base + 3, base + 2, base + 6), (base + 3, base + 6, base + 7),
                                   (base + 2, base, base + 4), (base + 2, base + 4, base + 6)))
        vertices = np.asarray(vertices_list, dtype=np.float32)
        triangles = np.asarray(triangles_list, dtype=np.int32)
    else:
        nx = max(2, int(round(width / cell)) + 1)
        ny = max(2, int(round((end_y - start_y) / cell)) + 1)
        xs, ys = np.linspace(-width / 2.0, width / 2.0, nx), np.linspace(start_y, end_y, ny)
        rng = np.random.default_rng(seed)
        heights = rng.uniform(0.0, height, size=(ny, nx))
        heights[0, :] = 0.0
        heights[-1, :] = 0.0
        vertices, triangles = _mesh_from_grid(xs, ys, heights)
        vertices, triangles = _add_boundary_walls(vertices, triangles, xs, ys, heights)

    metadata = {"kind": kind, "start_y_m": start_y, "end_y_m": end_y, "width_m": width,
                "height_m": height if kind != "rough" else height,
                "seed": seed if kind == "rough" else None,
                "cell_size_m": cell if kind == "rough" else None,
                "shape": "three_30mm_bars" if kind == "bars" else kind,
                "vertices": int(vertices.shape[0]), "triangles": int(triangles.shape[0])}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode() +
                             vertices.tobytes() + triangles.tobytes()).hexdigest()
    metadata["sha256"] = digest
    return vertices, triangles, metadata
