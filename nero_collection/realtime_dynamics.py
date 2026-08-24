from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Protocol

import numpy as np

from nero_collection.inverse_dynamics import JointTorqueResidualEstimate
from nero_collection.dynamics_processing import three_point_centered_sample


class JointTorqueResidualEstimator(Protocol):
    def estimate(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        tau_measured: np.ndarray,
    ) -> JointTorqueResidualEstimate:
        ...


class RealtimeVectorFilter(Protocol):
    def apply(self, value: np.ndarray, timestamp_us: int) -> np.ndarray:
        ...


@dataclass(frozen=True)
class RealtimeTorqueResidual:
    timestamp_us: int
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau: np.ndarray
    estimate: JointTorqueResidualEstimate


@dataclass(frozen=True)
class RealtimeJointState:
    timestamp_us: int
    sample_index: int
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau: np.ndarray


@dataclass(frozen=True)
class _RealtimeSample:
    timestamp_us: int
    q: np.ndarray
    tau: np.ndarray


class CenteredThreePointJointStateStream:
    """Emit the center state once its following sample has arrived."""

    def __init__(
        self,
        dof: int = 7,
        tau_filter: RealtimeVectorFilter | None = None,
    ) -> None:
        self.dof = int(dof)
        self.tau_filter = tau_filter
        self._samples: deque[_RealtimeSample] = deque(maxlen=3)
        self._sample_count = 0

    def reset(self) -> None:
        self._samples.clear()
        self._sample_count = 0
        reset = getattr(self.tau_filter, "reset", None)
        if callable(reset):
            reset()

    def append(
        self,
        timestamp_us: int,
        q: np.ndarray,
        tau: np.ndarray,
    ) -> RealtimeJointState | None:
        sample = _RealtimeSample(
            timestamp_us=int(timestamp_us),
            q=_finite_vector("q", q, self.dof),
            tau=_finite_vector("tau", tau, self.dof),
        )
        if self._samples and sample.timestamp_us <= self._samples[-1].timestamp_us:
            raise RuntimeError(
                "Three-point state estimation requires strictly increasing "
                f"timestamps: {sample.timestamp_us} <= {self._samples[-1].timestamp_us}"
            )
        self._samples.append(sample)
        if len(self._samples) < 3:
            return None
        samples = tuple(self._samples)
        center_timestamp_us, q_center, dq, ddq = three_point_centered_sample(
            tuple(item.timestamp_us for item in samples),
            tuple(item.q for item in samples),
        )
        center_tau = samples[1].tau
        tau_filtered = (
            self.tau_filter.apply(center_tau, center_timestamp_us)
            if self.tau_filter is not None
            else center_tau.copy()
        )
        sample_index = self._sample_count + 1
        self._sample_count += 1
        return RealtimeJointState(
            timestamp_us=center_timestamp_us,
            sample_index=sample_index,
            q=q_center.copy(),
            dq=dq.copy(),
            ddq=ddq.copy(),
            tau=np.asarray(tau_filtered, dtype=np.float64).copy(),
        )


class CenteredThreePointTorqueResidualStream:
    def __init__(
        self,
        estimator: JointTorqueResidualEstimator,
        dof: int = 7,
        tau_filter: RealtimeVectorFilter | None = None,
        tau_id_filter: RealtimeVectorFilter | None = None,
    ) -> None:
        self.estimator = estimator
        self.dof = int(dof)
        self.state_stream = CenteredThreePointJointStateStream(
            dof=self.dof,
            tau_filter=tau_filter,
        )
        self.tau_id_filter = tau_id_filter

    def append(
        self,
        timestamp_us: int,
        q: np.ndarray,
        tau: np.ndarray,
    ) -> RealtimeTorqueResidual | None:
        state = self.state_stream.append(timestamp_us, q, tau)
        if state is None:
            return None
        estimate = self.estimator.estimate(state.q, state.dq, state.ddq, state.tau)
        if self.tau_id_filter is not None:
            tau_id = self.tau_id_filter.apply(estimate.tau_id, state.timestamp_us)
            tau_model = tau_id + estimate.tau_friction + estimate.tau_bias
            estimate = JointTorqueResidualEstimate(
                tau_id=tau_id,
                tau_friction=estimate.tau_friction.copy(),
                tau_bias=estimate.tau_bias.copy(),
                tau_model=tau_model,
                tau_residual=tau_model - state.tau,
            )
        return RealtimeTorqueResidual(
            timestamp_us=state.timestamp_us,
            q=state.q,
            dq=state.dq,
            ddq=state.ddq,
            tau=state.tau,
            estimate=estimate,
        )


def _finite_vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.isfinite(vector).all():
        raise RuntimeError(f"Realtime tau_other requires a finite {size}D {name} vector; got {vector}")
    return vector.copy()
