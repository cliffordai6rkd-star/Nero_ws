from pathlib import Path

import numpy as np
import pytest

from nero_collection.config import (
    CollectionConfig,
    DynamicsProcessingConfig,
    OutputConfig,
    StateParamConfig,
    TeleopConfig,
    _parse_state_param,
)
from nero_collection.filters import OnePoleLowPass
from nero_collection.h5_writer import EpisodeBuffer
from nero_collection.dynamics_processing import (
    finite_difference_state,
    resample_columns,
    three_point_centered_sample,
)


def test_three_point_state_is_exact_for_quadratic_with_timestamp_jitter() -> None:
    timestamp_us = np.asarray([1_000_000, 1_007_000, 1_021_000], dtype=np.int64)
    time_s = (timestamp_us - timestamp_us[0]) * 1e-6
    q = (3.0 * time_s**2 + 2.0 * time_s + 0.4)[:, None]

    timestamp, q_center, dq, ddq = three_point_centered_sample(timestamp_us, q)

    assert timestamp == timestamp_us[1]
    assert q_center == pytest.approx(q[1])
    assert dq == pytest.approx([6.0 * time_s[1] + 2.0])
    assert ddq == pytest.approx([6.0])


def test_batch_finite_difference_returns_only_center_timeline() -> None:
    timestamp_us = np.asarray(
        [1_000_000, 1_007_000, 1_021_000, 1_030_000], dtype=np.int64
    )
    time_s = (timestamp_us - timestamp_us[0]) * 1e-6
    q = (time_s**2)[:, None]

    q_center, dq, ddq = finite_difference_state(timestamp_us, q)

    assert q_center == pytest.approx(q[1:-1])
    assert dq[:, 0] == pytest.approx(2.0 * time_s[1:-1])
    assert ddq[:, 0] == pytest.approx([2.0, 2.0])


def test_causal_median_rejects_single_sample_spike_before_iir() -> None:
    filt = OnePoleLowPass(cutoff_hz=10.0, median_window=3)

    first = filt.apply(np.array([1.0]), timestamp_us=1_000_000)
    spike = filt.apply(np.array([18.0]), timestamp_us=1_010_000)
    recovered = filt.apply(np.array([1.1]), timestamp_us=1_020_000)

    assert first == pytest.approx([1.0])
    assert spike == pytest.approx([1.0])
    assert 1.0 < recovered[0] < 1.1


def test_iir_alpha_uses_actual_timestamp_interval() -> None:
    filt = OnePoleLowPass(cutoff_hz=10.0, median_window=1)
    filt.apply(np.array([0.0]), timestamp_us=1_000_000)

    result = filt.apply(np.array([1.0]), timestamp_us=1_010_000)

    expected_alpha = 1.0 - np.exp(-2.0 * np.pi * 10.0 * 0.01)
    assert result == pytest.approx([expected_alpha])


def test_one_pole_reset_starts_a_fresh_episode() -> None:
    filt = OnePoleLowPass(cutoff_hz=10.0, median_window=3)
    filt.apply(np.array([1.0]), timestamp_us=1_000_000)
    filt.apply(np.array([2.0]), timestamp_us=1_010_000)

    filt.reset()
    result = filt.apply(np.array([9.0]), timestamp_us=2_000_000)

    assert result == pytest.approx([9.0])
    assert list(filt.history) == pytest.approx([np.array([9.0])])


@pytest.mark.parametrize("window", [0, 2, 4])
def test_state_config_rejects_invalid_median_window(window: int) -> None:
    with pytest.raises(ValueError, match="positive odd"):
        _parse_state_param({"median_window": window})


def test_episode_buffer_drops_acceleration_inputs() -> None:
    config = CollectionConfig(
        teleop=TeleopConfig(),
        output=OutputConfig(directory=Path(".")),
        robot_states={
            "velocity": StateParamConfig(enabled=True, lowpass=False),
        },
    )
    buffer = EpisodeBuffer(config=config, arm_names=("main",))

    first = buffer.append_teleop(
        1_000_000,
        {
            "q_follower": ("q", np.array([0.0])),
            "dq_follower": ("velocity", np.array([999.0])),
            "ddq_follower": ("acceleration", np.array([999.0])),
        },
    )
    second = buffer.append_teleop(
        1_010_000,
        {
            "q_follower": ("q", np.array([0.0001])),
            "dq_follower": ("velocity", np.array([999.0])),
            "ddq_follower": ("acceleration", np.array([999.0])),
        },
    )
    accepted = buffer.append_teleop(
        1_020_000,
        {
            "q_follower": ("q", np.array([0.0004])),
            "dq_follower": ("velocity", np.array([999.0])),
            "ddq_follower": ("acceleration", np.array([999.0])),
        },
    )

    assert first is not None
    assert second is not None
    assert accepted is not None
    assert accepted.timestamp_us == 1_020_000
    assert accepted.values["q_follower"][1] == pytest.approx([0.0004])
    assert accepted.values["dq_follower"][1] == pytest.approx([999.0])
    assert "ddq_follower" not in accepted.values
    assert buffer.teleop_timestamps_us == [1_000_000, 1_010_000, 1_020_000]


def test_finalize_teleop_data_preserves_aligned_derivatives_without_h5py(
    tmp_path: Path,
) -> None:
    config = CollectionConfig(
        teleop=TeleopConfig(),
        output=OutputConfig(directory=tmp_path),
        dynamics_processing=DynamicsProcessingConfig(
            enabled=True,
            state_method="finite_difference",
            torque_lowpass_hz=5.0,
            torque_median_window=1,
            min_samples=3,
        ),
        robot_states={
            "q": StateParamConfig(enabled=True),
            "velocity": StateParamConfig(enabled=True, lowpass=True, lowpass_cutoff_hz=5.0),
            "tau_id": StateParamConfig(enabled=True, lowpass=True, lowpass_cutoff_hz=5.0),
            "torque": StateParamConfig(enabled=True),
        },
    )
    buffer = EpisodeBuffer(config=config, arm_names=("main",))
    timestamps = np.asarray(
        [1_000_000, 1_010_000, 1_020_000, 1_030_000, 1_040_000], dtype=np.int64
    )
    q_scalar = np.asarray([0.0, 0.01, 0.03, 0.06, 0.10], dtype=np.float64)
    q = np.repeat(q_scalar[:, None], 7, axis=1)
    dq = np.repeat(np.arange(5, dtype=np.float64)[:, None], 7, axis=1)
    tau = np.repeat(
        np.asarray([[0.0], [1.0], [0.0], [0.0], [0.0]], dtype=np.float64),
        7,
        axis=1,
    )
    for index, timestamp_us in enumerate(timestamps):
        buffer.append_teleop(
            int(timestamp_us),
                {
                    "q_follower": ("q", q[index]),
                    "dq_follower": ("velocity", dq[index]),
                    "tau_follower": ("torque", tau[index]),
                },
            )

    data, state_names, attrs = buffer._finalize_teleop_data()

    assert state_names["dq_follower"] == "velocity"
    assert attrs["dq_follower"]["derivative_method"] == (
        "sign_corrected_official_motor_velocity_unfiltered"
    )
    assert attrs["dq_follower"]["coordinate_sign_correction_json"] == (
        "[-1, -1, -1, -1, -1, 1, -1]"
    )
    assert attrs["dq_follower"]["coordinate_frame"] == (
        "nero_joint_position_coordinates"
    )
    assert data["q_follower"] == pytest.approx(q)
    assert data["dq_follower"] == pytest.approx(dq)
    assert "ddq_follower" not in data
    assert "tau_f" not in data


def test_episode_save_preserves_raw_aligned_state_and_torque(
    tmp_path: Path,
) -> None:
    try:
        import h5py
    except Exception as exc:
        pytest.skip(f"h5py is unavailable: {exc}")

    config = CollectionConfig(
        teleop=TeleopConfig(),
        output=OutputConfig(directory=tmp_path),
        dynamics_processing=DynamicsProcessingConfig(
            enabled=True,
            state_method="finite_difference",
            spline_smoothing_rad2=0.0,
            torque_lowpass_hz=20.0,
            torque_median_window=1,
            min_samples=3,
        ),
        robot_states={
            "q": StateParamConfig(enabled=True),
            "velocity": StateParamConfig(enabled=True),
            "torque": StateParamConfig(enabled=True),
        },
    )
    buffer = EpisodeBuffer(config=config, arm_names=("main",))
    time_s = np.arange(100, dtype=np.float64) * 0.01
    timestamps = (1_000_000 + time_s * 1e6).astype(np.int64)
    phases = np.linspace(0.0, 0.6, 7)
    q = np.sin(time_s[:, None] + phases[None, :])
    tau = 0.2 * np.cos(time_s[:, None] + phases[None, :])
    tau[50, 3] += 5.0
    for index, timestamp_us in enumerate(timestamps):
        buffer.append_teleop(
            int(timestamp_us),
            {
                "q_follower": ("q", q[index]),
                "q_timestamp_follower_us": ("timestamp", np.asarray([timestamp_us])),
                "q_acquired_timestamp_follower_us": (
                    "timestamp",
                    np.asarray([timestamp_us]),
                ),
                "dq_follower": ("velocity", np.zeros(7)),
                "ddq_follower": ("acceleration", np.full(7, 999.0)),
                "tau_follower": ("torque", tau[index]),
                "motor_timestamp_follower_us": (
                    "timestamp",
                    np.full(7, timestamp_us, dtype=np.int64),
                ),
                "motor_acquired_timestamp_follower_us": (
                    "timestamp",
                    np.full(7, timestamp_us, dtype=np.int64),
                ),
            },
        )

    output = buffer.save(tmp_path / "episode.h5")

    with h5py.File(output, "r") as h5:
        teleop = h5["teleop"]
        assert h5.attrs["format"] == "factr_multimodal_episode/v11"
        assert np.allclose(teleop["dq_follower"][:], 0.0)
        assert "ddq_follower" not in teleop
        assert "dq_follower_firmware_raw" not in teleop
        assert "ddq_follower_adapter_raw" not in teleop
        assert "q_follower_raw" not in teleop
        assert "tau_follower_raw" not in teleop
        np.testing.assert_allclose(teleop["tau_follower"][:], tau)
        assert not bool(teleop["tau_follower"].attrs["lowpass"])
        assert teleop["tau_follower"].attrs["median_window"] == 1
        assert "tau_ext_follower" not in teleop
        assert teleop["q_follower"].shape == (100, 7)
        assert "tau_f" not in teleop
        assert np.allclose(teleop["timestamp_us"][:], timestamps)
        assert teleop["dq_follower"].attrs["derivative_method"] == (
            "sign_corrected_official_motor_velocity_unfiltered"
        )
        assert teleop["dq_follower"].attrs["timestamp_path"] == "teleop/timestamp_us"
        assert bool(teleop["tau_follower"].attrs["zero_phase"]) is False

def test_resampling_falls_back_when_sdk_motor_timestamp_is_stale() -> None:
    timeline = np.arange(20, dtype=np.int64) * 10_000 + 1_000_000
    stale_motor_timestamp = np.full((20, 2), 900_000, dtype=np.int64)
    values = np.column_stack((np.arange(20), np.arange(20) * 2)).astype(np.float64)

    result = resample_columns(
        stale_motor_timestamp,
        values,
        timeline,
        fallback_source_timestamp_us=timeline,
    )

    assert result == pytest.approx(values)
