"""LEGACY IDEALIZED NUMERICAL SELECTOR -- NOT THE PAPER V3 CONTROL METHOD.

The camera and scene dataclasses and image-interaction matrix remain reusable.
The select_gait() inner least-squares ideal velocity correction assumes an
instantaneously attainable correction budget and MUST NOT be reported as a
validated actuator-aware compensator. For V3 use closed_loop_surrogate.py
with empirically recorded compensation-on/off trajectories.

Former description: Compensation-aware hexapod gait selector (V0.1).

This is a platform-independent numerical core. It does NOT contain a robot model,
contact simulator, learned disturbance data, or camera renderer. Integrate these
through the defined inputs rather than silently substituting fabricated results.

Camera twist convention: [vx, vy, vz, wx, wy, wz], camera optical frame; SI.
The local interaction matrix models a stationary scene under camera motion.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class Gait:
    frequency_hz: float
    duty_factor: float
    swing_height_m: float
    touchdown_softness: float  # 0..1; maps to actual foot trajectory in host gait generator


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    exposure_s: float
    frame_interval_s: float


@dataclass(frozen=True)
class TrackedScene:
    uv: np.ndarray         # (N, 2) image points, pixels
    depth_m: np.ndarray    # (N,) positive camera-frame depth
    weights: np.ndarray    # (N,) positive feature weights

    def validate(self) -> None:
        if self.uv.ndim != 2 or self.uv.shape[1] != 2:
            raise ValueError('uv must have shape (N,2)')
        n = len(self.uv)
        if n == 0 or self.depth_m.shape != (n,) or self.weights.shape != (n,):
            raise ValueError('scene depths/weights must match nonempty uv')
        if not (np.all(np.isfinite(self.uv)) and np.all(np.isfinite(self.depth_m))
                and np.all(np.isfinite(self.weights))):
            raise ValueError('scene inputs must be finite')
        if np.any(self.depth_m <= 0) or np.any(self.weights <= 0):
            raise ValueError('depth and weights must be positive')


@dataclass(frozen=True)
class Candidate:
    gait: Gait
    disturbance_camera_twist: np.ndarray   # (H,6), relative to desired smooth camera motion
    correction_speed_limit: np.ndarray     # (H,6), achievable camera-twist correction budget
    expected_forward_speed_mps: float
    min_swing_clearance_m: float
    min_stability_margin_m: float
    feasible_kinematics: bool              # host IK & joint checks MUST set this


@dataclass(frozen=True)
class Settings:
    minimum_speed_mps: float
    minimum_clearance_m: float
    minimum_stability_margin_m: float
    max_correction_change: np.ndarray  # shape (6,), camera correction change per sample
    regularization: float = 1e-4
    smoothing: float = 1e-4
    gait_change_penalty: float = 0.01


@dataclass(frozen=True)
class Result:
    gait: Gait
    total_camera_correction: np.ndarray  # (H,6) incl. reserved gaze correction
    rms_frame_jitter_px: float            # first-order projected residual per camera frame
    rms_exposure_motion_px: float         # first-order projected residual during exposure
    score: float


def image_interaction_matrix(camera: Camera, scene: TrackedScene) -> np.ndarray:
    """Return pixel flow Jacobian (2N x 6) from camera twist (m/s, rad/s).

    Pinhole small-motion approximation. Rolling shutter and nonrigid/dynamic
    scenes are NOT accounted for; these require separate validation.
    """
    scene.validate()
    if camera.fx <= 0 or camera.fy <= 0 or camera.exposure_s <= 0 or camera.frame_interval_s <= 0:
        raise ValueError('camera intrinsics/exposure/frame interval must be positive')
    x = (scene.uv[:, 0] - camera.cx) / camera.fx
    y = (scene.uv[:, 1] - camera.cy) / camera.fy
    z = scene.depth_m
    L = np.empty((len(z) * 2, 6), dtype=float)
    L[0::2, :] = np.stack((-1/z, np.zeros_like(z), x/z, x*y, -(1+x*x), y), axis=1) * camera.fx
    L[1::2, :] = np.stack((np.zeros_like(z), -1/z, y/z, (1+y*y), -x*y, -x), axis=1) * camera.fy
    return L


def _weighted_matrix(L: np.ndarray, scene: TrackedScene) -> np.ndarray:
    w = np.repeat(scene.weights / np.sum(scene.weights), 2)
    return np.sqrt(w)[:, None] * L


def _gait_distance(a: Gait, b: Gait) -> float:
    # Dimensionless, gentle penalty to discourage gratuitous switching.
    return ((a.frequency_hz-b.frequency_hz)/max(a.frequency_hz, 1e-3))**2 + \
           (a.duty_factor-b.duty_factor)**2 + \
           ((a.swing_height_m-b.swing_height_m)/max(a.swing_height_m, 0.01))**2 + \
           (a.touchdown_softness-b.touchdown_softness)**2


def select_gait(
    camera: Camera,
    scene: TrackedScene,
    candidates: Iterable[Candidate],
    settings: Settings,
    previous_gait: Gait,
    gaze_correction: Optional[np.ndarray] = None,  # (H,6), reserved target tracking command
    previous_stabilization: Optional[np.ndarray] = None, # (6,)
) -> Result:
    """Select gait using the camera-space disturbance LEFT AFTER bounded compensation.

    Per-candidate constrained bounded least-squares computes a feasible fast
    body compensation. Candidate scores use the optimal *residual* image motion,
    rather than uncompensated IMU motion. All trajectory/geometric/dynamic
    feasibility, including contact and touchdown, must be checked by host code.
    """
    L = _weighted_matrix(image_interaction_matrix(camera, scene), scene)
    maximum_change = np.asarray(settings.max_correction_change, dtype=float)
    if maximum_change.shape != (6,) or np.any(maximum_change <= 0):
        raise ValueError('max_correction_change must be positive (6,)')
    if settings.regularization < 0 or settings.smoothing < 0:
        raise ValueError('regularization and smoothing must be nonnegative')
    prior = np.zeros(6) if previous_stabilization is None else np.asarray(previous_stabilization, dtype=float)
    if prior.shape != (6,):
        raise ValueError('previous_stabilization must be (6,)')
    best: Optional[Result] = None
    for item in candidates:
        d = np.asarray(item.disturbance_camera_twist, dtype=float)
        limits = np.asarray(item.correction_speed_limit, dtype=float)
        if d.ndim != 2 or d.shape[1] != 6 or d.shape != limits.shape or len(d) == 0:
            raise ValueError('candidate disturbance and correction budget must match (H,6)')
        if np.any(limits < 0) or not np.all(np.isfinite(d)) or not np.all(np.isfinite(limits)):
            raise ValueError('candidate prediction and limits must be finite; budgets nonnegative')
        H = len(d)
        gaze = np.zeros_like(d) if gaze_correction is None else np.asarray(gaze_correction, dtype=float)
        if gaze.shape != d.shape:
            raise ValueError('gaze_correction must be (H,6)')
        if (not item.feasible_kinematics or
            item.expected_forward_speed_mps < settings.minimum_speed_mps or
            item.min_swing_clearance_m < settings.minimum_clearance_m or
            item.min_stability_margin_m < settings.minimum_stability_margin_m):
            continue
        x = prior.copy()
        correction = np.zeros_like(d)
        feasible = True
        image_velocity_sq = []
        for t in range(H):
            # Reserve available body correction authority for the visual gaze task.
            lower = np.maximum(-limits[t] - gaze[t], x - maximum_change)
            upper = np.minimum(limits[t] - gaze[t], x + maximum_change)
            if np.any(lower > upper + 1e-12):
                feasible = False
                break
            # Least-squares objective: image motion + control effort + action smoothness.
            A = np.concatenate((L, np.sqrt(settings.regularization)*np.eye(6),
                                np.sqrt(settings.smoothing)*np.eye(6)), axis=0)
            b = np.concatenate((-L@d[t], np.zeros(6), np.sqrt(settings.smoothing)*x))
            # SciPy rejects exactly equal upper/lower bounds; fix variables via tiny range.
            upper = np.maximum(upper, lower + 1e-10)
            opt = lsq_linear(A, b, bounds=(lower, upper), tol=1e-8, max_iter=100)
            if not opt.success:
                feasible = False
                break
            x = np.minimum(np.maximum(opt.x, lower), upper)
            correction[t] = x + gaze[t]
            pixel_flow = L@(d[t] + x)
            image_velocity_sq.append(float(pixel_flow@pixel_flow))
        if not feasible:
            continue
        rms_px_per_s = float(np.sqrt(np.mean(image_velocity_sq)))
        frame_jitter = rms_px_per_s * camera.frame_interval_s
        exposure_motion = rms_px_per_s * camera.exposure_s
        # The paper's image-space objective: predicted residual per-frame displacement.
        # Exposure motion is additionally reported; it is proportional under constant
        # frame interval and exposure and must NOT be claimed an independent term here.
        score = frame_jitter**2 + settings.gait_change_penalty*_gait_distance(item.gait, previous_gait)
        r = Result(item.gait, correction, frame_jitter, exposure_motion, score)
        if best is None or r.score < best.score:
            best = r
    if best is None:
        raise RuntimeError('No feasible candidate; keep the host controller safe gait / stop strategy')
    return best


# A minimal terrain- and phase-conditioned model for candidate disturbance traces.
# It requires measured/simulated data: NO dataset or trained coefficients are included.
class PhaseConditionedRidge:
    """Fit [gait, terrain, phase] -> CAMERA twist disturbance (6 outputs)."""
    def __init__(self, ridge: float = 1e-2):
        if ridge <= 0:
            raise ValueError('ridge must be positive')
        self.ridge = ridge
        self.coefficients: Optional[np.ndarray] = None

    @staticmethod
    def features(gait: Gait, terrain_roughness_m: float, slope_rad: float, phase_rad: float) -> np.ndarray:
        f, beta, height, soft = gait.frequency_hz, gait.duty_factor, gait.swing_height_m, gait.touchdown_softness
        s, c = np.sin(phase_rad), np.cos(phase_rad)
        return np.array([1.0, f, beta, height, soft, terrain_roughness_m, slope_rad,
                         s, c, np.sin(2*phase_rad), np.cos(2*phase_rad),
                         f*s, f*c, beta*s, beta*c, height*s, height*c,
                         soft*s, soft*c, terrain_roughness_m*s,
                         terrain_roughness_m*c, slope_rad*s, slope_rad*c], dtype=float)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'PhaseConditionedRidge':
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        if X.ndim != 2 or X.shape[1] != 23 or Y.shape != (len(X), 6) or len(X) < 23:
            raise ValueError('X must be (N,23), Y (N,6), N>=23')
        if not (np.isfinite(X).all() and np.isfinite(Y).all()):
            raise ValueError('training inputs must be finite')
        # Standardized ridge regression; bias column stays unregularized.
        self.mean = X[:,1:].mean(axis=0)
        self.scale = X[:,1:].std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        Z = np.column_stack([np.ones(len(X)), (X[:,1:]-self.mean)/self.scale])
        reg = self.ridge * np.eye(Z.shape[1]); reg[0,0] = 0
        self.coefficients = np.linalg.solve(Z.T@Z + reg, Z.T@Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError('Fit model on real simulator/robot samples first')
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[1] != 23:
            raise ValueError('X must be (N,23)')
        Z = np.column_stack([np.ones(len(X)), (X[:,1:]-self.mean)/self.scale])
        return Z @ self.coefficients
