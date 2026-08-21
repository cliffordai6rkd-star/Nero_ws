from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from nero_collection.config import (
    CollectionConfig,
    OutputConfig,
    StateParamConfig,
    TauExtInferenceConfig,
    TeleopConfig,
)
from nero_collection.h5_writer import EpisodeBuffer
from nero_collection.tau_ext_inference import (
    OnlineTauExtResult,
    SequenceCheckpointMetadata,
    TauExtInferenceMetadata,
)


def _model_metadata(path: Path, output_key: str) -> SequenceCheckpointMetadata:
    filters = {
        "tau": {
            "enabled": True,
            "operations": [{"type": "lowpass", "cutoff_hz": 10.0}],
        }
    }
    return SequenceCheckpointMetadata(
        checkpoint_path=path,
        horizon=50,
        input_keys=("q", "dq", "delta_q"),
        input_dims={"q": 7, "dq": 7, "delta_q": 7},
        output_key=output_key,
        output_dim=7,
        architecture="lstm",
        normalize_mode="quantile" if output_key == "tau_f" else "gaussian",
        sample_rate_hz=50.0,
        dataloader_filters=filters,
        target_contract=(
            "matched_causal_torque_filter_v1" if output_key == "tau_f" else None
        ),
        target_filter_cutoff_hz=10.0 if output_key == "tau_f" else None,
        target_filter_median_window=1 if output_key == "tau_f" else None,
    )


class _Inference:
    def __init__(self, tmp_path: Path) -> None:
        self.metadata = TauExtInferenceMetadata(
            tau_f=_model_metadata(tmp_path / "tau_f.pt", "tau_f"),
            tau_next=_model_metadata(tmp_path / "tau_next.pt", "tau"),
            input_keys=("q", "dq", "delta_q"),
        )
        self.calls = 0
        self.taus = []

    def estimate_aligned(self, timestamp_us, q, dq, tau, q_cmd):
        self.calls += 1
        q = np.asarray(q, dtype=np.float64)
        dq = np.asarray(dq, dtype=np.float64)
        tau = np.asarray(tau, dtype=np.float64)
        self.taus.append(tau.copy())
        tau_id = np.full(7, 2.0)
        tau_f = np.full(7, 0.5)
        tau_next = np.full(7, 4.0)
        return OnlineTauExtResult(
            timestamp_us=int(timestamp_us),
            q=q.copy(),
            dq=dq.copy(),
            ddq_kf_causal=np.full(7, 3.0),
            tau=tau.copy(),
            tau_id=tau_id,
            tau_id_filtered=tau_id.copy(),
            tau_f_pred=tau_f,
            tau_next_pred=tau_next,
            tau_ext_cal=tau_id + tau_f - tau,
            tau_ext_pred=tau_next - tau,
        )

    def warm_up(self):
        pass

    def reset_episode(self):
        pass


def _config(
    tmp_path: Path,
    feedback_source: str = "tau_free",
) -> CollectionConfig:
    states = {
        name: StateParamConfig(enabled=True, lowpass=False)
        for name in (
            "q",
            "velocity",
            "acceleration",
            "ee_pose",
            "torque",
            "tau_id",
            "current",
        )
    }
    return CollectionConfig(
        teleop=TeleopConfig(),
        output=OutputConfig(directory=tmp_path),
        tau_ext_inference=TauExtInferenceConfig(
            enabled=True,
            feedback_source=feedback_source,
        ),
        robot_states=states,
    )


def test_episode_buffer_can_disable_configured_online_inference(tmp_path: Path) -> None:
    buffer = EpisodeBuffer(
        config=_config(tmp_path),
        arm_names=("main",),
        enable_online_tau_ext=False,
    )

    assert buffer.online_tau_ext is None
    assert not buffer.warm_up_online_inference()


def test_episode_buffer_skips_inference_when_all_checkpoints_are_empty(
    tmp_path: Path,
) -> None:
    buffer = EpisodeBuffer(config=_config(tmp_path), arm_names=("main",))

    assert buffer.online_tau_ext is None
    assert not buffer.warm_up_online_inference()


def test_episode_buffer_passes_raw_tau_to_online_filter_once(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.robot_states["torque"] = StateParamConfig(
        enabled=True,
        lowpass=True,
        lowpass_cutoff_hz=10.0,
        median_window=1,
    )
    inference = _Inference(tmp_path)
    buffer = EpisodeBuffer(
        config=config,
        arm_names=("main",),
        online_tau_ext=inference,
    )
    zeros = np.zeros(7)
    for index, tau_value in enumerate((0.0, 10.0)):
        buffer.append_teleop(
            1_000_000 + index * 10_000,
            {
                "q_follower": ("q", zeros),
                "dq_follower": ("velocity", zeros),
                "tau_follower": ("torque", np.full(7, tau_value)),
                "q_cmd": ("q", zeros),
            },
        )

    np.testing.assert_allclose(inference.taus[1], 10.0)
    np.testing.assert_allclose(buffer.teleop_data["tau_follower"][1], 10.0)


@pytest.mark.parametrize("feedback_source", ("tau_f", "tau_free"))
def test_h5_v12_saves_matched_filter_dual_tau_ext_schema(
    tmp_path: Path,
    feedback_source: str,
) -> None:
    try:
        import h5py
    except (ImportError, ValueError) as exc:
        pytest.skip(f"h5py is unavailable or ABI-incompatible: {exc}")

    inference = _Inference(tmp_path)
    buffer = EpisodeBuffer(
        config=_config(tmp_path, feedback_source),
        arm_names=("main",),
        online_tau_ext=inference,
    )
    for index in range(3):
        q = np.full(7, index * 0.1)
        buffer.append_teleop(
            1_000_000 + index * 10_000,
            {
                "q_follower": ("q", q),
                "dq_follower": ("velocity", np.full(7, 0.2)),
                "ddq_follower": ("acceleration", np.full(7, 99.0)),
                "tau_follower": ("torque", np.full(7, 10.0)),
                "q_cmd": ("q", q + 0.1),
            },
        )

    output = buffer.save(tmp_path / "episode.h5")

    assert inference.calls == 3
    new_fields = {
        "tau_id",
        "tau_id_filtered",
        "tau_f_pred",
        "tau_next_pred",
        "tau_ext_cal_raw",
        "tau_ext_pred_raw",
        "tau_ext_cal",
        "tau_ext_pred",
    }
    removed_fields = {
        "tau_f_cal",
        "tau_bg_pred",
        "tau_ext_raw",
        "tau_ext_filtered",
        "tau_ext",
        "wrench_ext",
    }
    with h5py.File(output, "r") as h5:
        teleop = h5["teleop"]
        assert h5.attrs["format"] == "factr_multimodal_episode/v12"
        assert new_fields <= set(teleop)
        assert removed_fields.isdisjoint(teleop)
        assert "ddq_kf_causal" not in teleop
        assert "ddq_follower" not in teleop
        np.testing.assert_allclose(teleop["tau_ext_cal_raw"], -7.5)
        np.testing.assert_allclose(teleop["tau_ext_pred_raw"], -6.0)
        np.testing.assert_allclose(teleop["tau_ext_cal"], -7.5)
        np.testing.assert_allclose(teleop["tau_ext_pred"], -6.0)
        assert (
            teleop["tau_ext_cal"].attrs["definition"]
            == "tau_ext_filter(tau_id_filtered + tau_f_pred - tau_follower)"
        )
        assert (
            teleop["tau_ext_pred"].attrs["definition"]
            == "tau_ext_filter(tau_next_pred - checkpoint_causal_filter(tau_follower))"
        )
        assert bool(teleop["tau_ext_cal"].attrs["feedback_source"]) == (
            feedback_source == "tau_f"
        )
        assert bool(teleop["tau_ext_pred"].attrs["feedback_source"]) == (
            feedback_source == "tau_free"
        )
        assert (
            teleop["tau_ext_cal"].attrs["configured_force_feedback_source"]
            == feedback_source
        )
        assert teleop["tau_next_pred"].attrs["history_warmup_samples"] == 50
        assert teleop["tau_ext_pred"].attrs["history_warmup_output"] == "zeros"
        assert not bool(teleop["tau_f_pred"].attrs["lowpass"])
        assert not bool(teleop["q_follower"].attrs["lowpass"])
        assert not bool(teleop["dq_follower"].attrs["lowpass"])
        assert not bool(teleop["tau_follower"].attrs["lowpass"])
        assert teleop["tau_follower"].attrs["processing_method"] == (
            "nearest_motor_sample_unfiltered"
        )
        assert bool(teleop["tau_id_filtered"].attrs["lowpass"])
        assert teleop["tau_id_filtered"].attrs["lowpass_cutoff_hz"] == 10.0
        assert bool(teleop["tau_ext_cal"].attrs["lowpass"])
        assert (
            teleop["tau_ext_cal"].attrs["processing_method"]
            == "causal_hampel_then_butterworth_sos"
        )
        assert teleop["tau_ext_cal"].attrs["hampel_window"] == 5
        assert teleop["tau_ext_cal"].attrs["butterworth_order"] == 4
        assert teleop["tau_ext_cal"].attrs["lowpass_cutoff_hz"] == 8.0
        assert teleop["tau_ext_cal"].attrs["moving_average_window"] == 1
        assert not bool(teleop["tau_ext_cal_raw"].attrs["lowpass"])


def test_precomputed_dual_results_are_not_inferred_twice(tmp_path: Path) -> None:
    inference = _Inference(tmp_path)
    buffer = EpisodeBuffer(
        config=_config(tmp_path),
        arm_names=("main",),
        online_tau_ext=inference,
    )
    zeros = np.zeros(7)
    values = {
        "q_follower": ("q", zeros),
        "dq_follower": ("velocity", zeros),
        "tau_follower": ("torque", zeros),
        "q_cmd": ("q", zeros),
        "ddq_kf_causal": ("acceleration", zeros),
        "tau_id": ("torque", zeros),
        "tau_f_pred": ("torque", zeros),
        "tau_next_pred": ("torque", zeros),
        "tau_ext_cal": ("torque", zeros),
        "tau_ext_pred": ("torque", zeros),
    }

    accepted = buffer.append_teleop(1_000_000, values)

    assert accepted is not None
    assert inference.calls == 0
    assert set(values) - {"ddq_kf_causal"} <= set(accepted.values)
    assert "ddq_kf_causal" not in accepted.values


def test_robot_state_datasets_share_the_frame_timestamp(tmp_path: Path) -> None:
    try:
        import h5py
    except (ImportError, ValueError) as exc:
        pytest.skip(f"h5py is unavailable or ABI-incompatible: {exc}")

    buffer = EpisodeBuffer(
        config=_config(tmp_path),
        arm_names=("main",),
        enable_online_tau_ext=False,
    )
    q = np.arange(7, dtype=np.float64)
    buffer.append_teleop(
        999,
        {
            "q_follower": ("q", q),
            "tau_follower": ("torque", q + 10),
        },
    )

    output = buffer.save(tmp_path / "raw-timestamps.h5")

    with h5py.File(output, "r") as h5:
        teleop = h5["teleop"]
        np.testing.assert_array_equal(teleop["timestamp_us"][:], [999])
        assert "q_follower_timestamp_us" not in teleop
        assert "tau_follower_timestamp_us" not in teleop
        assert (
            teleop["q_follower"].attrs["timestamp_path"]
            == "teleop/timestamp_us"
        )
        assert (
            teleop["tau_follower"].attrs["timestamp_path"]
            == "teleop/timestamp_us"
        )


def test_q_cmd_is_saved_without_measured_q_and_has_causal_zoh_semantics(
    tmp_path: Path,
) -> None:
    try:
        import h5py
    except (ImportError, ValueError) as exc:
        pytest.skip(f"h5py is unavailable or ABI-incompatible: {exc}")

    config = _config(tmp_path)
    config.robot_states["q"] = StateParamConfig(enabled=False)
    buffer = EpisodeBuffer(
        config=config,
        arm_names=("main",),
        enable_online_tau_ext=False,
    )
    q_cmd = np.linspace(0.1, 0.7, 7)
    buffer.append_teleop(1_000_000, {"q_cmd": ("q", q_cmd)})

    output = buffer.save(tmp_path / "q-command.h5")

    with h5py.File(output, "r") as h5:
        teleop = h5["teleop"]
        assert "q_follower" not in teleop
        np.testing.assert_allclose(teleop["q_cmd"][0], q_cmd)
        assert teleop["q_cmd"].attrs["source"] == "actual_follower_command"
        assert (
            teleop["q_cmd"].attrs["command_semantics"]
            == "causal_zoh_at_state_sample"
        )
