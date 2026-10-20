"""Controller-conditioned *empirical* gait evaluator for Machines hexapod study.

The model is trained ONLY on observed/simulated trials with the actual fixed
body compensator enabled and disabled. It does not posit an ideal compensator.
All features of a candidate must be computable before its execution; never use
future contact forces, achieved speed, or true terrain state as a candidate
input. No training data or scientific results are bundled.

This is a deliberately simple, inspectable surrogate, not a replacement for
contact feasibility checking or the 50-Hz low-level position controller.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import csv
import numpy as np

# Fields supplied from *pre-decision* robot state, gait planner, and practical
# local estimates; NOT from oracle future simulator states.
INPUT_NAMES = (
    'frequency_hz', 'duty_factor', 'step_length_m', 'swing_height_m',
    'touchdown_softness', 'desired_speed_mps', 'roughness_est_m',
    'slope_est_rad', 'planned_stance_count', 'predicted_joint_headroom',
    'phase_rad', 'roll_rate_rad_s', 'pitch_rate_rad_s',
    'previous_frequency_hz', 'previous_duty_factor',
    'previous_touchdown_softness', 'time_since_gait_change_s',
)
REQUIRED = INPUT_NAMES + (
    'pair_id', 'terrain_instance_id', 'controller_on',
    'measured_jitter_px_frame', 'measured_speed_mps',
)


def _finite(row: Mapping[str, object], names: Sequence[str]) -> np.ndarray:
    values = np.array([float(row[n]) for n in names], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('All feature values must be finite')
    return values


def feature_vector(row: Mapping[str, object]) -> np.ndarray:
    v = _finite(row, INPUT_NAMES)
    (f, beta, stride, height, soft, speed, rough, slope, stance, headroom,
     phase, wr, wp, prev_f, prev_beta, prev_soft, since_change) = v
    if not (0.0 < beta < 1.0 and 0.0 <= soft <= 1.0 and f > 0.0
            and height > 0.0 and stride > 0.0 and speed > 0.0
            and 0 <= rough and 3 <= stance <= 6 and 0 <= headroom <= 1
            and prev_f > 0 and 0 < prev_beta < 1
            and 0 <= prev_soft <= 1 and since_change >= 0):
        raise ValueError('Candidate feature outside supported physical ranges')
    return np.array([
        1.0, f, beta, stride, height, soft, speed, rough, slope,
        stance / 6., headroom, np.sin(phase), np.cos(phase),
        wr, wp, f*rough, beta*stance/6., soft*rough,
        prev_f, prev_beta, prev_soft, since_change,
    ], dtype=float)


class StandardizedRidge:
    """Small deterministic regressor, fit on independent trial windows.

    log1p target prevents negative predicted jitter and treats large values
    more conservatively. No per-frame sample is counted as an independent
    replicate in the paper statistics.
    """
    def __init__(self, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        self.alpha = float(alpha)
        self.coef = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'StandardizedRidge':
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.ndim != 2 or x.shape[1] != 22 or x.shape[0] < 5:
            raise ValueError('Expected at least five (N,22) feature rows')
        if y.shape != (len(x),) or np.any(y < 0) or not np.isfinite(y).all():
            raise ValueError('Targets must be nonnegative and finite')
        if not np.isfinite(x).all() or not np.allclose(x[:, 0], 1.0):
            raise ValueError('Invalid feature matrix')
        self.mu = x[:, 1:].mean(axis=0)
        self.sd = x[:, 1:].std(axis=0)
        self.sd[self.sd < 1e-8] = 1.0
        z = np.column_stack([np.ones(len(x)), (x[:, 1:] - self.mu) / self.sd])
        reg = np.eye(z.shape[1]) * self.alpha
        reg[0, 0] = 0.0
        self.coef = np.linalg.solve(z.T @ z + reg, z.T @ np.log1p(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError('Fit on actual simulator or robot data first')
        x = np.asarray(x, float)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != 22 or not np.isfinite(x).all():
            raise ValueError('Expected finite features with shape (N,22)')
        z = np.column_stack([np.ones(len(x)), (x[:, 1:] - self.mu) / self.sd])
        return np.maximum(np.expm1(np.clip(z @ self.coef, -30., 20.)), 0.)


@dataclass(frozen=True)
class ResponseEstimate:
    raw_jitter_px: float
    closed_loop_jitter_px: float
    effective_jitter_reduction_px: float
    predicted_speed_mps: float


class FixedCompensatorResponseModel:
    """Predict both outcomes of the SAME fixed body-compensation controller.

    The difference of the two predicted costs is an *empirical capability
    proxy*, not a force feasibility polytope or a high-frequency Bode model.
    Use safety geometry/contact filters and measured response diagnostics too.
    """
    def __init__(self, alpha: float = 1.0):
        self.raw = StandardizedRidge(alpha)
        self.closed = StandardizedRidge(alpha)
        self.speed = StandardizedRidge(alpha)
        self.fitted = False

    def fit(self, rows: Sequence[Mapping[str, object]]) -> 'FixedCompensatorResponseModel':
        if not rows:
            raise ValueError('No recorded trials supplied')
        by_condition = {}
        entries = {0: [], 1: []}
        for row in rows:
            try:
                enabled = int(row['controller_on'])
                pair_id = str(row['pair_id'])
                terrain = str(row['terrain_instance_id'])
                seed = str(row['seed'])
                gait_id = str(row['gait_id'])
                jitter = float(row['measured_jitter_px_frame'])
                speed = float(row['measured_speed_mps'])
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError('Missing/invalid recorded trial field') from e
            if enabled not in (0, 1) or not pair_id or not terrain:
                raise ValueError('Bad controller condition or pairing metadata')
            if min(jitter, speed) < 0 or not np.isfinite(jitter + speed):
                raise ValueError('Invalid measured outcome')
            x = feature_vector(row)
            entries[enabled].append((x, jitter, speed))
            key = (terrain, seed, pair_id, gait_id)
            pair = by_condition.setdefault(key, {})
            if enabled in pair:
                raise ValueError('Duplicate controller condition in off/on pair')
            pair[enabled] = x
        if any(set(pair) != {0, 1} for pair in by_condition.values()):
            raise ValueError('Each (terrain,seed,pair_id,gait_id) needs off/on trials')
        for key, pair in by_condition.items():
            if not np.allclose(pair[0], pair[1], rtol=0, atol=1e-9):
                raise ValueError(f'Paired off/on predecision inputs differ: {key}')
        for enabled in (0, 1):
            if len(entries[enabled]) < 5:
                raise ValueError('At least five windows per controller condition')
        raw = np.stack([v[0] for v in entries[0]])
        closed = np.stack([v[0] for v in entries[1]])
        self.raw.fit(raw, np.array([v[1] for v in entries[0]]))
        self.closed.fit(closed, np.array([v[1] for v in entries[1]]))
        self.speed.fit(closed, np.array([v[2] for v in entries[1]]))
        self.fitted = True
        return self

    def predict(self, predecision_features: Mapping[str, object]) -> ResponseEstimate:
        if not self.fitted:
            raise RuntimeError('Fit on real data, do not invent response coefficients')
        v = feature_vector(predecision_features)
        a = float(self.raw.predict(v)[0]); b = float(self.closed.predict(v)[0]); s = float(self.speed.predict(v)[0])
        return ResponseEstimate(a, b, a-b, s)


@dataclass(frozen=True)
class RankedGait:
    candidate: Mapping[str, object]
    estimate: ResponseEstimate
    score: float


def rank_candidates(model: FixedCompensatorResponseModel,
                    candidates: Iterable[Mapping[str, object]],
                    current_gait: Mapping[str, float],
                    speed_command_mps: float,
                    min_speed_fraction: float = .92,
                    objective: str = 'closed_loop',
                    switch_weight: float = .015) -> list[RankedGait]:
    """Rank pre-screened feasible candidates. Safety checks belong to host.

    Objective 'raw' is the matched strong baseline with EXACTLY the same
    candidate menu, predictor features, and *executed* body compensator.
    """
    if objective not in ('closed_loop', 'raw'):
        raise ValueError('objective must be closed_loop or raw')
    if speed_command_mps <= 0 or not 0 < min_speed_fraction <= 1:
        raise ValueError('Invalid speed constraint')
    ranked = []
    for candidate in candidates:
        if not bool(candidate.get('pre_screened_feasible', False)):
            continue
        est = model.predict(candidate)
        if est.predicted_speed_mps < speed_command_mps * min_speed_fraction:
            continue
        motion = est.closed_loop_jitter_px if objective == 'closed_loop' else est.raw_jitter_px
        delta = sum(((float(candidate[n])-float(current_gait[n])) / scale)**2 for n,scale in
                    [('frequency_hz', 1.), ('duty_factor', .5), ('step_length_m', .1), ('touchdown_softness', 1.)])
        ranked.append(RankedGait(candidate, est, motion + switch_weight*delta))
    return sorted(ranked, key=lambda item: item.score)


def read_trials_csv(path: str) -> list[dict]:
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        missing = set(REQUIRED).difference(r.fieldnames or [])
        if missing:
            raise ValueError(f'Missing required columns: {sorted(missing)}')
        return list(r)
