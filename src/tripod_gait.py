"""Parameterized continuous tripod gait for hexapod locomotion experiments.

Coordinate convention checked against Pcrnet_hexapod: +x right, +y forward, +z up.
Leg order matches PCR CurvAdapt_multi: lb,lf,lm,rb,rf,rm.
The generator is still not verified on Isaac Gym; do not confuse interface
conventions with a completed real robot joint-axis check.
The output is an OPEN-LOOP body-frame foot-reference generator, not a
locomotion controller and not a replacement for contact-safe IK/WBC.

Foot trajectory uses Hermite swing/linear stance to make horizontal
velocity C1 at stance transitions; swing height is C1 and the kappa parameter
shifts height to have lower curvature at touchdown, while maintaining the
required clearance at mid-swing. Physical impact must be checked in dynamics.
"""
from dataclasses import dataclass
import numpy as np

LEG_ORDER = ('lb', 'lf', 'lm', 'rb', 'rf', 'rm')
# Left-back, left-front, right-middle vs left-middle, right-back, right-front
TRIPOD_PHASE = np.array([0.0, 0.0, 0.5, 0.5, 0.5, 0.0])

@dataclass(frozen=True)
class TripodConfig:
    frequency_hz: float
    duty_factor: float
    swing_height_m: float
    touchdown_softness: float

    def validate(self):
        if not (.05 <= self.frequency_hz <= 6):
            raise ValueError('frequency_hz outside tested parameter interface')
        if not (.50 <= self.duty_factor < .95):
            raise ValueError('duty factor must retain tripod support overlap')
        if not (0 < self.swing_height_m < .20):
            raise ValueError('swing height out of expected local range')
        if not (0 <= self.touchdown_softness <= 1):
            raise ValueError('touchdown_softness must lie in [0,1]')


def _hermite_return(s, width):
    """Cubic swing interpolation from -1/2 to +1/2, C1 at stance edges."""
    h00 = 2*s**3-3*s**2+1
    h10 = s**3-2*s**2+s
    h01 = -2*s**3+3*s**2
    h11 = s**3-s**2
    # stance slope: -1/beta in units of normalized stride/phase;
    # swing normalized slope equals -(1-beta)/beta.
    return -0.5*h00-width*h10+0.5*h01-width*h11


def foot_targets(t_s, nominal_feet_base, commanded_vx, commanded_vy,
                 commanded_yaw_rate, cfg: TripodConfig):
    """Return desired feet (6,3) and stance mask (6,) at simulation time t_s.

    `nominal_feet_base` must be from ACTUAL robot IK/FK; provided anchors
    should be those of a safe standing pose. No joint feasibility guarantee.
    Walking velocity is a *command*; achieved velocity comes from simulation.
    """
    return foot_targets_by_phase(cfg.frequency_hz * t_s, nominal_feet_base,
                                 commanded_vx, commanded_vy, commanded_yaw_rate, cfg)


def foot_targets_by_phase(phase_cycles, nominal_feet_base, commanded_vx, commanded_vy,
                          commanded_yaw_rate, cfg: TripodConfig):
    """Same gait but with a continuous accumulated cycle phase.

    Use this entrypoint when gait frequency changes online: frequency*t would
    cause a phase jump. Accumulated phase is supplied by ContinuousPhaseClock.
    """
    cfg.validate()
    feet = np.asarray(nominal_feet_base, dtype=float)
    if feet.shape != (6,3) or not np.isfinite(feet).all():
        raise ValueError('nominal_feet_base must be finite (6,3)')
    if not np.isfinite([phase_cycles,commanded_vx,commanded_vy,commanded_yaw_rate]).all():
        raise ValueError('time and commands must be finite')
    phase = (phase_cycles+TRIPOD_PHASE) % 1.0
    support = phase < cfg.duty_factor
    beta = cfg.duty_factor
    swing_norm = (phase-beta)/(1-beta)
    offset_frac = np.empty(6)
    offset_frac[support] = .5-phase[support]/beta
    ss = np.clip(swing_norm[~support],0,1)
    offset_frac[~support] = _hermite_return(ss,(1-beta)/beta)
    vx_local = commanded_vx - commanded_yaw_rate*feet[:,1]
    vy_local = commanded_vy + commanded_yaw_rate*feet[:,0]
    step_displacement = np.column_stack((vx_local,vy_local))*beta/cfg.frequency_hz
    out = feet.copy()
    out[:,:2] += step_displacement*offset_frac[:,None]
    # sin^2 at kappa=0, sin^4 at kappa=1. Both have zero vertical velocity
    # at liftoff and touchdown; sin^4 has zero vertical acceleration as well.
    if len(ss):
        soft_shape = np.sin(np.pi*ss)**(2+2*cfg.touchdown_softness)
        out[~support,2] += cfg.swing_height_m*soft_shape
    return out, support


@dataclass
class ContinuousPhaseClock:
    """Unwrapped phase, differentiable in time across frequency changes."""
    cycles: float = 0.0

    def advance(self, dt_s: float, frequency_hz: float) -> float:
        if not np.isfinite([dt_s, frequency_hz]).all() or dt_s <= 0 or frequency_hz <= 0:
            raise ValueError('Bad gait timestep/frequency')
        self.cycles += float(dt_s * frequency_hz)
        return self.cycles