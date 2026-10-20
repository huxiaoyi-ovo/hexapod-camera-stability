"""Camera-induced image motion metrics from ACTUAL camera-twist logs.

Required input per row: t, vx,vy,vz,wx,wy,wz, nominal_vx,...,nominal_wz.
All velocities are expressed in CAMERA OPTICAL coordinates (m/s, rad/s).
The 6D nominal twist describes desired smooth task motion, not a filtered
version of the measured gait disturbances. A suitable nominal trajectory
must be supplied by the controller or reference planner.

Outputs are projected image-motion proxies, NOT rendered blur or measured
image sharpness. No synthetic experimental results are generated here.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import numpy as np
from controller import Camera, TrackedScene, image_interaction_matrix

TWIST = ('vx', 'vy', 'vz', 'wx', 'wy', 'wz')
NOMINAL = tuple('nominal_'+v for v in TWIST)

@dataclass(frozen=True)
class ImageMotionMetrics:
    n_samples: int
    n_frames: int
    duration_s: float
    jitter_px_rms_per_frame: float
    exposure_trace_px_rms: float
    exposure_net_displacement_px_rms: float
    peak_exposure_trace_px: float
    median_samples_per_exposure: float
    camera_frame: str = 'camera_optical'

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def read_trace(path: str | Path):
    """Read a synchronized physical/simulator camera-motion log.

    Never silently zero-fill missing nominal twists. That would confuse
    deliberate navigation motion with unwanted gait-induced jitter.
    """
    with open(path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        required = ('t',) + TWIST + NOMINAL
        missing = [x for x in required if x not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f'Missing mandatory CSV fields: {missing}')
        rows = list(reader)
    if len(rows) < 3:
        raise ValueError('At least three logged samples are required')
    t = np.array([float(r['t']) for r in rows], dtype=float)
    actual = np.array([[float(r[k]) for k in TWIST] for r in rows], dtype=float)
    nominal = np.array([[float(r[k]) for k in NOMINAL] for r in rows], dtype=float)
    if not np.isfinite(t).all() or not np.isfinite(actual).all() or not np.isfinite(nominal).all():
        raise ValueError('All timestamps and twists must be finite')
    if np.any(np.diff(t) <= 0):
        raise ValueError('Time must increase strictly monotonically')
    return t, actual, nominal


def _interval_samples(t: np.ndarray, flow: np.ndarray, start: float, stop: float):
    """Add interpolated boundary values around all available internal samples."""
    inside = (t > start) & (t < stop)
    ts = np.r_[start, t[inside], stop]
    values = np.empty((len(ts), flow.shape[1]), float)
    for j in range(flow.shape[1]):
        values[:, j] = np.interp(ts, t, flow[:, j])
    return ts, values


def image_motion_metrics(t: np.ndarray, residual_camera_twist: np.ndarray,
                         camera: Camera, scene: TrackedScene) -> ImageMotionMetrics:
    """Integrate projected STATIC-scene image motion, using all logged samples.

    Metrics require a smooth-task reference for residual_camera_twist.
    For each camera frame:
      jitter = net projected residual displacement over the frame;
      exposure_trace = projected path length DURING exposure;
      exposure_net = exposure start-to-end displacement (not the same as blur).
    Exposure trace cannot represent real PSF without photometric integration.
    """
    t = np.asarray(t, dtype=float)
    residual = np.asarray(residual_camera_twist, dtype=float)
    if t.ndim != 1 or residual.shape != (len(t), 6) or len(t) < 3:
        raise ValueError('t(N), residual(N,6), N>=3 required')
    if np.any(np.diff(t) <= 0) or not np.isfinite(residual).all():
        raise ValueError('Invalid time or residual arrays')
    if camera.exposure_s > camera.frame_interval_s + 1e-12:
        raise ValueError('Exposure cannot exceed frame period in this frame model')
    L = image_interaction_matrix(camera, scene)
    flow = residual @ L.T  # (T, 2*N), pixel/s
    n_features = len(scene.depth_m)
    w = scene.weights / np.sum(scene.weights)
    starts = np.arange(t[0], t[-1] - camera.frame_interval_s + 1e-12,
                       camera.frame_interval_s)
    if len(starts) < 1:
        raise ValueError('Trace duration must cover at least one camera frame')
    jitter2 = []; trace2 = []; net2 = []; peaks = []; dens = []
    for start in starts:
        end = start + camera.frame_interval_s
        tf, vf = _interval_samples(t, flow, start, end)
        frame_disp = np.trapezoid(vf, tf, axis=0).reshape(n_features, 2)
        jitter2.append(float(np.sum(w * np.sum(frame_disp**2, axis=1))))
        exp_end = start + camera.exposure_s
        te, ve = _interval_samples(t, flow, start, exp_end)
        feature_velocity = np.linalg.norm(ve.reshape(len(te), n_features, 2), axis=2)
        path_length = np.trapezoid(feature_velocity, te, axis=0)
        exp_disp = np.trapezoid(ve, te, axis=0).reshape(n_features, 2)
        trace2.append(float(np.sum(w * path_length**2)))
        net2.append(float(np.sum(w * np.sum(exp_disp**2, axis=1))))
        peaks.append(float(np.max(path_length)))
        dens.append(int(np.count_nonzero((t >= start) & (t <= exp_end))))
    return ImageMotionMetrics(
        n_samples=len(t), n_frames=len(starts), duration_s=float(t[-1]-t[0]),
        jitter_px_rms_per_frame=float(np.sqrt(np.mean(jitter2))),
        exposure_trace_px_rms=float(np.sqrt(np.mean(trace2))),
        exposure_net_displacement_px_rms=float(np.sqrt(np.mean(net2))),
        peak_exposure_trace_px=float(np.max(peaks)),
        median_samples_per_exposure=float(np.median(dens)))


def camera_scene_from_json(config_path: str | Path):
    with open(config_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    camera = Camera(**payload['camera'])
    scene = TrackedScene(
        uv=np.asarray(payload['scene']['uv'], dtype=float),
        depth_m=np.asarray(payload['scene']['depth_m'], dtype=float),
        weights=np.asarray(payload['scene']['weights'], dtype=float))
    scene.validate()
    return camera, scene
