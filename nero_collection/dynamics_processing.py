from __future__ import annotations

from collections import deque

import numpy as np


def reconstruct_state_from_positions(
    timestamp_us: np.ndarray,
    q: np.ndarray,
    *,
    state_method: str,
    spline_smoothing_rad2: float,
    fourier_fundamental_hz: float,
    fourier_harmonics: int,
    evaluation_timestamp_us: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if state_method == "finite_difference":
        return finite_difference_state(timestamp_us, q)
    source_t, source_q = _unique_samples(timestamp_us, q, minimum=4)
    evaluation_t = (
        source_t
        if evaluation_timestamp_us is None
        else _timestamp_vector(evaluation_timestamp_us, "evaluation_timestamp_us")
    )
    origin_us = source_t[0]
    source_time_s = (source_t - origin_us).astype(np.float64) * 1e-6
    evaluation_t = np.clip(evaluation_t, source_t[0], source_t[-1])
    evaluation_time_s = (evaluation_t - origin_us).astype(np.float64) * 1e-6
    if state_method == "spline":
        return _spline_state(
            source_time_s,
            source_q,
            evaluation_time_s,
            spline_smoothing_rad2,
        )
    if state_method == "fourier":
        return _fourier_state(
            source_time_s,
            source_q,
            evaluation_time_s,
            fourier_fundamental_hz,
            fourier_harmonics,
        )
    raise ValueError(f"unsupported state reconstruction method: {state_method!r}")


def resample_columns(
    source_timestamp_us: np.ndarray,
    values: np.ndarray,
    target_timestamp_us: np.ndarray,
    *,
    fallback_source_timestamp_us: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("resampling values must be a finite (N, D) array")
    target = _timestamp_vector(target_timestamp_us, "target_timestamp_us")
    source = np.asarray(source_timestamp_us, dtype=np.int64)
    if source.ndim == 1:
        source = np.repeat(source[:, None], values.shape[1], axis=1)
    if source.shape != values.shape:
        raise ValueError(
            f"source timestamps must have shape {(values.shape[0],)} or {values.shape}; got {source.shape}"
        )
    result = np.empty((target.size, values.shape[1]), dtype=np.float64)
    target_float = target.astype(np.float64)
    fallback = None
    if fallback_source_timestamp_us is not None:
        fallback = np.asarray(fallback_source_timestamp_us, dtype=np.int64)
        if fallback.ndim == 1:
            fallback = np.repeat(fallback[:, None], values.shape[1], axis=1)
        if fallback.shape != values.shape:
            raise ValueError(
                f"fallback source timestamps must have shape {values.shape}; got {fallback.shape}"
            )
    for column in range(values.shape[1]):
        try:
            column_t, column_values = _unique_samples(
                source[:, column],
                values[:, column],
                minimum=2,
            )
        except ValueError:
            if fallback is None:
                raise
            column_t, column_values = _unique_samples(
                fallback[:, column],
                values[:, column],
                minimum=2,
            )
        result[:, column] = np.interp(
            target_float,
            column_t.astype(np.float64),
            column_values,
        )
    return result


def select_source_timestamps(
    primary_timestamp_us: np.ndarray,
    acquired_timestamp_us: np.ndarray,
    *,
    minimum_unique: int,
) -> tuple[np.ndarray, bool]:
    primary = np.asarray(primary_timestamp_us, dtype=np.int64).reshape(-1)
    acquired = _timestamp_vector(acquired_timestamp_us, "acquired_timestamp_us")
    if primary.size != acquired.size:
        raise ValueError("primary and acquired timestamp series must have the same length")
    primary_valid = (
        np.all(primary > 0)
        and np.all(np.diff(primary) >= 0)
        and np.unique(primary).size >= minimum_unique
    )
    return (primary.copy(), False) if primary_valid else (acquired.copy(), True)


def filter_torque(
    timestamp_us: np.ndarray,
    tau: np.ndarray,
    *,
    median_window: int,
    lowpass_hz: float,
) -> np.ndarray:
    from scipy.signal import butter, medfilt, sosfiltfilt

    timestamp_us = _timestamp_vector(timestamp_us, "timestamp_us")
    tau = np.asarray(tau, dtype=np.float64)
    if tau.ndim != 2 or tau.shape[0] != timestamp_us.size or not np.isfinite(tau).all():
        raise ValueError("torque filtering requires finite tau with shape (N, D)")
    if median_window < 1 or median_window % 2 == 0:
        raise ValueError("median_window must be a positive odd integer")
    filtered = np.column_stack(
        [medfilt(tau[:, joint], kernel_size=median_window) for joint in range(tau.shape[1])]
    )
    dt = np.diff(timestamp_us).astype(np.float64) * 1e-6
    if np.any(dt <= 0):
        raise ValueError("torque filtering requires strictly increasing timestamps")
    sample_rate = 1.0 / float(np.median(dt))
    nyquist = 0.5 * sample_rate
    if lowpass_hz >= nyquist * 0.95:
        return filtered
    sos = butter(4, lowpass_hz / nyquist, btype="low", output="sos")
    return sosfiltfilt(sos, filtered, axis=0)


def finite_difference_state(
    timestamp_us: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one-frame-delayed three-point derivatives on the center timeline.

    For input samples ``(t0, q0), (t1, q1), (t2, q2)``, the output is
    aligned to ``t1``.  The implementation uses the two measured time
    intervals directly, so non-uniform sampling does not shift the velocity
    estimate away from the center sample.
    """
    timestamp_us = _timestamp_vector(timestamp_us, "timestamp_us")
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != timestamp_us.size or not np.isfinite(q).all():
        raise ValueError("finite differencing requires finite q with shape (N, D)")
    if timestamp_us.size < 3:
        empty = np.empty((0, q.shape[1]), dtype=np.float64)
        return empty.copy(), empty.copy(), empty.copy()
    dt = np.diff(timestamp_us).astype(np.float64) * 1e-6
    if np.any(dt <= 0):
        raise ValueError("finite differencing requires strictly increasing timestamps")
    h0 = dt[:-1, None]
    h1 = dt[1:, None]
    slope_before = (q[1:-1] - q[:-2]) / h0
    slope_after = (q[2:] - q[1:-1]) / h1
    interval = h0 + h1
    dq = (h1 * slope_before + h0 * slope_after) / interval
    ddq = 2.0 * (slope_after - slope_before) / interval
    return q[1:-1].copy(), dq, ddq


def three_point_centered_sample(
    timestamp_us: tuple[int, int, int] | np.ndarray,
    q: tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Compute q/dq/ddq at the middle of exactly three timestamped samples."""
    timestamps = np.asarray(timestamp_us, dtype=np.int64).reshape(-1)
    positions = np.asarray(q, dtype=np.float64)
    if timestamps.shape != (3,):
        raise ValueError(f"three-point state estimation requires 3 timestamps; got {timestamps.shape}")
    if positions.ndim != 2 or positions.shape[0] != 3 or not np.isfinite(positions).all():
        raise ValueError("three-point state estimation requires finite q with shape (3, D)")
    if np.any(timestamps <= 0) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("three-point state timestamps must be positive and strictly increasing")
    q_center, dq, ddq = finite_difference_state(timestamps, positions)
    return int(timestamps[1]), q_center[0], dq[0], ddq[0]


def filter_signal_causal(
    timestamp_us: np.ndarray,
    values: np.ndarray,
    *,
    median_window: int,
    lowpass_hz: float,
    name: str = "signal",
) -> np.ndarray:
    timestamp_us = _timestamp_vector(timestamp_us, "timestamp_us")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != timestamp_us.size or not np.isfinite(values).all():
        raise ValueError(f"causal {name} filtering requires finite values with shape (N, D)")
    if median_window < 1 or median_window % 2 == 0:
        raise ValueError("median_window must be a positive odd integer")
    if not np.isfinite(lowpass_hz) or lowpass_hz <= 0:
        raise ValueError("lowpass_hz must be positive and finite")
    history: deque[np.ndarray] = deque()
    filtered = np.empty_like(values)
    state: np.ndarray | None = None
    previous_timestamp_us: int | None = None
    for index, (timestamp, value) in enumerate(zip(timestamp_us, values)):
        history.append(value.copy())
        while len(history) > median_window:
            history.popleft()
        samples = list(history)
        samples = [samples[0]] * (median_window - len(samples)) + samples
        median_value = np.median(np.stack(samples, axis=0), axis=0)
        if state is None:
            state = median_value.copy()
        else:
            assert previous_timestamp_us is not None
            dt = (int(timestamp) - previous_timestamp_us) * 1e-6
            if dt <= 0:
                raise ValueError(
                    f"causal {name} filtering requires strictly increasing timestamps"
                )
            alpha = 1.0 - np.exp(-2.0 * np.pi * lowpass_hz * dt)
            state = alpha * median_value + (1.0 - alpha) * state
        filtered[index] = state
        previous_timestamp_us = int(timestamp)
    return filtered


def filter_torque_causal(
    timestamp_us: np.ndarray,
    tau: np.ndarray,
    *,
    median_window: int,
    lowpass_hz: float,
) -> np.ndarray:
    return filter_signal_causal(
        timestamp_us,
        tau,
        median_window=median_window,
        lowpass_hz=lowpass_hz,
        name="torque",
    )


def _spline_state(time_s, q, evaluation_time_s, smoothing_rad2):
    from scipy.interpolate import UnivariateSpline

    shape = (evaluation_time_s.size, q.shape[1])
    q_fit = np.empty(shape, dtype=np.float64)
    dq = np.empty(shape, dtype=np.float64)
    ddq = np.empty(shape, dtype=np.float64)
    smoothing = float(smoothing_rad2) * time_s.size
    for joint in range(q.shape[1]):
        spline = UnivariateSpline(time_s, q[:, joint], k=3, s=smoothing)
        q_fit[:, joint] = spline(evaluation_time_s)
        dq[:, joint] = spline.derivative(1)(evaluation_time_s)
        ddq[:, joint] = spline.derivative(2)(evaluation_time_s)
    return q_fit, dq, ddq


def _fourier_state(time_s, q, evaluation_time_s, fundamental_hz, harmonics):
    omega = 2.0 * np.pi * float(fundamental_hz) * np.arange(1, harmonics + 1)
    source_phase = time_s[:, None] * omega[None, :]
    source_sin = np.sin(source_phase)
    source_cos = np.cos(source_phase)
    design = np.column_stack((np.ones(time_s.size), source_sin, source_cos))
    coefficients, _, _, _ = np.linalg.lstsq(design, q, rcond=None)
    sin_coefficients = coefficients[1 : harmonics + 1]
    cos_coefficients = coefficients[harmonics + 1 :]

    evaluation_phase = evaluation_time_s[:, None] * omega[None, :]
    sin_phase = np.sin(evaluation_phase)
    cos_phase = np.cos(evaluation_phase)
    evaluation_design = np.column_stack(
        (np.ones(evaluation_time_s.size), sin_phase, cos_phase)
    )
    q_fit = evaluation_design @ coefficients
    dq = cos_phase @ (sin_coefficients * omega[:, None]) - sin_phase @ (
        cos_coefficients * omega[:, None]
    )
    ddq = -sin_phase @ (sin_coefficients * omega[:, None] ** 2) - cos_phase @ (
        cos_coefficients * omega[:, None] ** 2
    )
    return q_fit, dq, ddq


def _unique_samples(timestamp_us, values, minimum):
    timestamp = _timestamp_vector(timestamp_us, "timestamp_us", strictly_increasing=False)
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != timestamp.size or not np.isfinite(values).all():
        raise ValueError("sample values must be finite and match the timestamp length")
    if np.any(np.diff(timestamp) < 0):
        raise ValueError("sample timestamps must be non-decreasing")
    unique, first, counts = np.unique(timestamp, return_index=True, return_counts=True)
    last = first + counts - 1
    if unique.size < minimum:
        raise ValueError(f"at least {minimum} unique timestamped samples are required")
    return unique, values[last]


def _timestamp_vector(value, name, *, strictly_increasing=True):
    timestamp = np.asarray(value, dtype=np.int64).reshape(-1)
    if timestamp.size == 0 or np.any(timestamp <= 0):
        raise ValueError(f"{name} must contain positive timestamps")
    differences = np.diff(timestamp)
    if strictly_increasing and np.any(differences <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return timestamp
