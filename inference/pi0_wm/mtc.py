"""Synchronized WM position reference and gravity-relative torque feedforward."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MitReference:
    q: np.ndarray
    velocity: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    feedforward: np.ndarray
    transition_time_s: float = 0.0


class MtcController:
    def __init__(self, config, hardware, gravity, *, torque_only=False, require_tau=True):
        self.cfg = config
        self.hw = hardware
        self.gravity = gravity
        self.torque_only = torque_only
        # WM-backed MTC requires a synchronized tau prediction.  Pure Pi0
        # impedance tracking intentionally has no tau stream and uses only
        # analytical gravity feed-forward.
        self.require_tau = require_tau
        self.velocity = np.zeros(7)
        self.feedforward = None

    def reset(self, q, initial_torque=None):
        self.velocity = np.zeros(7)
        limit = np.asarray(self.cfg['total_torque_limit_nm'] if self.torque_only else self.cfg['feedforward_limit_nm'])
        # Seed from the previous mode's measured effort, not an instantaneous
        # jump to model gravity. No measurement means a zero-effort start.
        initial = np.zeros(7) if initial_torque is None else np.asarray(initial_torque, dtype=float)
        if initial.shape != (7,) or not np.isfinite(initial).all():
            raise ValueError('MTC initial torque must be a finite seven-vector')
        self.feedforward = np.clip(initial, -limit, limit)
        self.initial_feedforward = self.feedforward.copy()
        self.transition_time_s = 0.0

    def prepare(self, q_prediction, tau_prediction, state, held, dt):
        cfg = self.cfg
        if not np.isfinite(dt) or dt <= 0 or dt > cfg['watchdog_timeout_s']:
            raise RuntimeError('MTC control interval exceeded watchdog')
        def vector(value, label):
            value = np.asarray(value, dtype=float)
            if value.shape != (7,) or not np.isfinite(value).all():
                raise ValueError(f'MTC {label} must be a finite seven-vector')
            return value
        q, dq, held = (vector(state.q, 'measured q'), vector(state.dq, 'measured dq'), vector(held, 'held'))
        valid = q_prediction is not None
        target = vector(q_prediction, 'predicted q') if valid else held.copy()
        target = np.clip(target, self.hw['q_min'], self.hw['q_max'])
        if valid and tau_prediction is None and self.require_tau:
            raise ValueError('MTC predicted tau is required for WM tracking')
        tau = vector(tau_prediction, 'predicted tau') if valid and tau_prediction is not None else None
        vmax = np.asarray(cfg['maximum_velocity_rad_s'])
        amax = np.asarray(cfg['maximum_acceleration_rad_s2'])
        error = target - held
        # Leave enough distance to brake, including this discrete time step.
        braking_velocity = 2 * amax * np.abs(error) / (
            np.sqrt((amax * dt) ** 2 + 2 * amax * np.abs(error)) + amax * dt)
        desired_velocity = np.sign(error) * np.minimum(vmax, braking_velocity)
        velocity = np.clip(desired_velocity, self.velocity - amax * dt, self.velocity + amax * dt)
        if not valid:
            velocity = np.zeros(7)
        step_limit = np.asarray(self.hw['maximum_step_rad'])
        delta = np.clip(velocity * dt, -step_limit, step_limit)
        # A newly changed target can make acceleration and no-overshoot
        # constraints incompatible. Prioritize not crossing/moving away from
        # the target (also applies to an immediate hold).
        delta = np.clip(delta, np.minimum(error, 0), np.maximum(error, 0))
        command_q = np.clip(held + delta,
                            self.hw['q_min'], self.hw['q_max'])
        velocity = (command_q - held) / dt
        baseline = vector(self.gravity(q), 'gravity')
        if valid and tau is not None:
            # Reduce learned feedforward when the executed reference diverges
            # from the WM trajectory due to limiting/transition handling.
            deviation = np.abs(command_q - target)
            weight = np.clip(1 - deviation / cfg['prediction_deviation_rad'], 0, 1)
            residual = np.asarray(cfg['tau_scale']) * (tau - vector(self.gravity(target), 'reference gravity'))
            desired_ff = baseline + weight * residual
        else:
            desired_ff = baseline
        if self.torque_only:
            if np.any(np.abs(command_q - q) > np.asarray(cfg['maximum_tracking_error_rad'])):
                raise RuntimeError('TAU joint tracking error exceeded limit')
            # Compute PD in joint coordinates on the host. The motor receives
            # zero gains, so apply limits to the COMPLETE torque, not only FF.
            desired_ff = np.clip(desired_ff, -np.asarray(cfg['feedforward_limit_nm']),
                                 np.asarray(cfg['feedforward_limit_nm']))
            desired_ff = (np.asarray(cfg['kp']) * (command_q - q)
                          + np.asarray(cfg['kd']) * (velocity - dq) + desired_ff)
        transition_time = self.transition_time_s + dt
        blend = min(1.0, transition_time / cfg.get('startup_blend_s', 1.0))
        desired_ff = self.initial_feedforward + blend * (desired_ff - self.initial_feedforward)
        ff_limit = np.asarray(cfg['total_torque_limit_nm'] if self.torque_only else cfg['feedforward_limit_nm'])
        desired_ff = np.clip(desired_ff, -ff_limit, ff_limit)
        previous_ff = baseline if self.feedforward is None else self.feedforward
        ff_step = np.asarray(cfg['maximum_torque_rate_nm_s']) * dt
        feedforward = np.clip(np.clip(desired_ff, previous_ff - ff_step, previous_ff + ff_step), -ff_limit, ff_limit)
        if self.torque_only:
            return MitReference(command_q, velocity, np.zeros(7), np.zeros(7), feedforward, transition_time)
        kp, kd = np.asarray(cfg['kp']), np.asarray(cfg['kd'])
        pd = kp * (command_q - q) + kd * (velocity - dq)
        total_limit = np.asarray(cfg['total_torque_limit_nm'])
        # Reduce PD gains to fit the remaining instantaneous torque budget.
        # This bounds the estimate at this feedback sample, not future firmware
        # torque as the physical state changes between CAN updates.
        scale = np.ones(7)
        positive, negative = pd > 0, pd < 0
        scale[positive] = np.minimum(1, (total_limit[positive] - feedforward[positive]) / pd[positive])
        scale[negative] = np.minimum(1, (-total_limit[negative] - feedforward[negative]) / pd[negative])
        scale = np.clip(scale, 0, 1)
        return MitReference(command_q, velocity, kp * scale, kd * scale, feedforward, transition_time)

    def commit(self, reference):
        self.velocity = reference.velocity.copy()
        self.feedforward = reference.feedforward.copy()
        self.transition_time_s = reference.transition_time_s


def validate_mtc(config):
    config.setdefault('startup_blend_s', 1.0)
    config.setdefault('maximum_tracking_error_rad', 0.25)
    limits = {
        'kp': (0, 500), 'kd': (0, 5), 'tau_scale': (0, 1),
        'maximum_velocity_rad_s': (0, 45),
        'maximum_acceleration_rad_s2': (0, np.inf),
        'feedforward_limit_nm': (0, 16), 'total_torque_limit_nm': (0, 16),
        'maximum_torque_rate_nm_s': (0, np.inf),
        'maximum_tracking_error_rad': (0, np.inf),
    }
    for name, (low, high) in limits.items():
        value = np.asarray(config.get(name), dtype=float)
        if value.ndim == 0:
            value = np.full(7, value)
        allow_zero = name in ('kp', 'kd', 'tau_scale')
        if (value.shape != (7,) or not np.isfinite(value).all()
                or np.any(value < low) or (not allow_zero and np.any(value == 0)) or np.any(value > high)):
            raise ValueError(f'control.mtc.{name} has invalid seven-joint limits')
        config[name] = value.tolist()
    # Respect the tighter of this repository's limit and Nero joint limits.
    cap = np.array([16, 16, 16, 16, 8, 8, 8])
    ff, total = np.asarray(config['feedforward_limit_nm']), np.asarray(config['total_torque_limit_nm'])
    if np.any(ff > total) or np.any(total > cap):
        raise ValueError('MTC requires feedforward_limit <= total_torque_limit <= Nero joint limits')
    for name in ('prediction_deviation_rad', 'watchdog_timeout_s', 'startup_blend_s'):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'control.mtc.{name} must be positive and finite')
    if not config.get('urdf_path'):
        raise ValueError('control.mtc.urdf_path is required')
