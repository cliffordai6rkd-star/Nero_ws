from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nero_collection.config import CausalKalmanConfig


@dataclass(frozen=True)
class CausalJointState:
    timestamp_us: int
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray


class CausalJointKalmanFilter:
    """Variable-dt [q, dq, ddq] filter matching the PINN forward pass."""

    def __init__(self, config: CausalKalmanConfig, joint_count: int = 7) -> None:
        self.config = config
        self.joint_count = int(joint_count)
        if self.joint_count != 7:
            raise ValueError("Nero causal Kalman filter expects seven joints")
        self.position_variance = np.square(config.position_std)
        self.velocity_variance = np.square(config.velocity_std)
        self.jerk_variance = np.square(config.jerk_std)
        self.initial_covariance = np.stack(
            [
                np.diag(
                    np.square(
                        [
                            config.initial_position_std[joint],
                            config.initial_velocity_std[joint],
                            config.initial_acceleration_std[joint],
                        ]
                    )
                )
                for joint in range(self.joint_count)
            ],
            axis=0,
        )
        self.reset()

    def reset(self) -> None:
        self._timestamp_us: int | None = None
        self._state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None

    def update(self, timestamp_us: int, q: np.ndarray, dq: np.ndarray) -> CausalJointState:
        timestamp_us = int(timestamp_us)
        q = _finite_vector("q", q, self.joint_count)
        dq = _finite_vector("dq", dq, self.joint_count)
        if self._timestamp_us is not None and timestamp_us <= self._timestamp_us:
            raise ValueError(
                "causal Kalman timestamp must increase strictly: "
                f"{timestamp_us} <= {self._timestamp_us}"
            )

        reset = self._timestamp_us is None
        if self._timestamp_us is not None:
            dt = (timestamp_us - self._timestamp_us) * 1.0e-6
            reset = dt > self.config.max_gap_s
        if reset:
            self._state = np.stack((q, dq, np.zeros_like(q)), axis=-1)
            self._covariance = self.initial_covariance.copy()
        else:
            assert self._state is not None and self._covariance is not None
            transition = _transition(dt)
            for joint in range(self.joint_count):
                self._state[joint] = transition @ self._state[joint]
                self._covariance[joint] = _symmetrize(
                    transition @ self._covariance[joint] @ transition.T
                    + _white_jerk_covariance(dt, self.jerk_variance[joint])
                )

        assert self._state is not None and self._covariance is not None
        for joint in range(self.joint_count):
            self._state[joint], self._covariance[joint] = _measurement_update(
                self._state[joint],
                self._covariance[joint],
                q[joint],
                dq[joint],
                self.position_variance[joint],
                self.velocity_variance[joint],
            )
        self._timestamp_us = timestamp_us
        return CausalJointState(
            timestamp_us=timestamp_us,
            q=self._state[:, 0].copy(),
            dq=self._state[:, 1].copy(),
            ddq=self._state[:, 2].copy(),
        )


def _transition(dt: float) -> np.ndarray:
    return np.asarray(
        [
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _white_jerk_covariance(dt: float, spectral_density: float) -> np.ndarray:
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt
    return spectral_density * np.asarray(
        [
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0, dt2 / 2.0, dt],
        ],
        dtype=np.float64,
    )


def _measurement_update(
    state: np.ndarray,
    covariance: np.ndarray,
    q: float,
    dq: float,
    position_variance: float,
    velocity_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    measurement = np.asarray([q, dq], dtype=np.float64)
    noise = np.diag([position_variance, velocity_variance])
    innovation_covariance = h @ covariance @ h.T + noise
    gain = np.linalg.solve(innovation_covariance, h @ covariance).T
    updated_state = state + gain @ (measurement - h @ state)
    identity = np.eye(3, dtype=np.float64)
    correction = identity - gain @ h
    updated_covariance = (
        correction @ covariance @ correction.T + gain @ noise @ gain.T
    )
    return updated_state, _symmetrize(updated_covariance)


def _symmetrize(covariance: np.ndarray) -> np.ndarray:
    return 0.5 * (covariance + covariance.T)


def _finite_vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Kalman {name} must be a finite {size}-vector; got {vector}")
    return vector.copy()
