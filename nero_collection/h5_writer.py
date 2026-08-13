from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nero_collection.config import CollectionConfig
from nero_collection.coordinates import NERO_V120_MOTOR_VELOCITY_TO_JOINT_SIGN
from nero_collection.contact_wrench import (
    PinocchioContactWrenchEstimator,
)
from nero_collection.tau_ext_inference import OnlineTauExtInference
from nero_collection.time_utils import now_us


FORMAT_VERSION = "factr_multimodal_episode/v9"

FOLLOWER_TELEOP_DATASETS = frozenset(
    {
        "q_follower",
        "q_leader",
        "q_cmd",
        "dq_follower",
        "dq_leader",
        "ddq_follower",
        "ddq_leader",
        "ee_pose_follower",
        "tau_follower",
        "tau_leader",
        "current_follower",
        "current_leader",
        "gripper_follower",
        "gripper_cmd",
        "ddq_kf_causal",
        "tau_id",
        "tau_id_filtered",
        "tau_f_pred",
        "tau_next_pred",
        "tau_ext_cal_raw",
        "tau_ext_pred_raw",
        "tau_ext_cal",
        "tau_ext_pred",
        "wrench_cal_raw",
        "wrench_pred_raw",
        "wrench_cal",
        "wrench_pred",
        "control_timestamp_us",
        "state_acquired_timestamp_follower_us",
        "q_source_timestamp_follower_us",
        "motor_source_timestamp_follower_us",
        "state_source_skew_follower_us",
        "model_observation_updated",
        "model_observation_timestamp_us",
        "model_prediction_age_us",
    }
)

_TIMING_DATASET_SOURCES = {
    "control_timestamp_us": "teleop_control_loop",
    "state_acquired_timestamp_follower_us": "host_state_read",
    "q_source_timestamp_follower_us": "pyagx_can_joint_group_frames",
    "motor_source_timestamp_follower_us": "pyagx_can_motor_state_frames",
    "state_source_skew_follower_us": "max_minus_min_q_and_motor_source_timestamp",
    "model_observation_timestamp_us": "causal_real_observation_selector",
    "model_prediction_age_us": "source_timestamp_minus_model_observation_timestamp",
}


@dataclass(frozen=True)
class AcceptedTeleopFrame:
    timestamp_us: int
    values: dict[str, tuple[str, np.ndarray]]


@dataclass
class EpisodeBuffer:
    config: CollectionConfig
    arm_names: tuple[str, ...]
    teleop_timestamps_us: list[int] = field(default_factory=list)
    teleop_data: dict[str, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    teleop_state_names: dict[str, str] = field(default_factory=dict)
    camera_frames: dict[str, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    camera_timestamps_us: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    episode_metadata: dict[str, Any] = field(default_factory=dict)
    last_input_timestamp_us: int | None = field(init=False, default=None)
    input_frame_count: int = field(init=False, default=0)
    duplicate_input_frame_count: int = field(init=False, default=0)
    online_tau_ext: OnlineTauExtInference | None = None
    enable_online_tau_ext: bool | None = None

    def __post_init__(self) -> None:
        inference_enabled = (
            self.config.tau_ext_inference.enabled
            if self.enable_online_tau_ext is None
            else self.enable_online_tau_ext
        )
        if self.online_tau_ext is None:
            inference_enabled = inference_enabled and any(
                branch.checkpoint_path is not None
                for branch in (
                    self.config.tau_ext_inference.tau_f,
                    self.config.tau_ext_inference.tau_next,
                )
            )
        if not inference_enabled:
            self.online_tau_ext = None
        elif self.online_tau_ext is None:
            self.online_tau_ext = OnlineTauExtInference(
                self.config.tau_ext_inference,
                self.config.realtime_plot.inverse_dynamics,
                self.config.dynamics_processing,
                self.config.robot_states,
            )

    def append_teleop(
        self,
        timestamp_us: int,
        values: dict[str, tuple[str, np.ndarray]],
        *,
        store: bool = True,
    ) -> AcceptedTeleopFrame | None:
        timestamp_us = int(timestamp_us)
        self.input_frame_count += 1
        values = {
            name: (state_name, np.asarray(value).copy())
            for name, (state_name, value) in values.items()
            if name in FOLLOWER_TELEOP_DATASETS
        }
        if self.last_input_timestamp_us is not None:
            if timestamp_us < self.last_input_timestamp_us:
                raise ValueError(
                    "teleop observation timestamp moved backwards: "
                    f"{timestamp_us} < {self.last_input_timestamp_us}"
                )
            if timestamp_us == self.last_input_timestamp_us:
                self.duplicate_input_frame_count += 1
                return None
        self.last_input_timestamp_us = timestamp_us
        if store:
            self.teleop_timestamps_us.append(timestamp_us)

        processed_values: dict[str, tuple[str, np.ndarray]] = {}
        for dataset_name, (state_name, value) in values.items():
            processed = self._process_center_value(
                dataset_name,
                state_name,
                np.asarray(value),
                timestamp_us,
            )
            if store:
                self.teleop_data[dataset_name].append(processed)
                self.teleop_state_names[dataset_name] = state_name
            processed_values[dataset_name] = (state_name, np.asarray(processed).copy())

        has_precomputed_tau_ext = "tau_ext_cal" in processed_values
        if self.online_tau_ext is not None and not has_precomputed_tau_ext:
            required = ["q_follower", "dq_follower", "tau_follower", "q_cmd"]
            missing = [name for name in required if name not in processed_values]
            if missing:
                raise RuntimeError(
                    f"tau_ext inference is missing follower datasets: {missing}"
                )
            result = self.online_tau_ext.estimate_aligned(
                timestamp_us,
                processed_values["q_follower"][1],
                processed_values["dq_follower"][1],
                np.asarray(values["tau_follower"][1], dtype=np.float64),
                processed_values["q_cmd"][1],
            )
            for name, value in (
                ("ddq_kf_causal", result.ddq_kf_causal),
                ("tau_id", result.tau_id),
                ("tau_id_filtered", result.tau_id_filtered),
                ("tau_f_pred", result.tau_f_pred),
                ("tau_next_pred", result.tau_next_pred),
                (
                    "tau_ext_cal_raw",
                    result.tau_ext_cal
                    if result.tau_ext_cal_raw is None
                    else result.tau_ext_cal_raw,
                ),
                (
                    "tau_ext_pred_raw",
                    result.tau_ext_pred
                    if result.tau_ext_pred_raw is None
                    else result.tau_ext_pred_raw,
                ),
                ("tau_ext_cal", result.tau_ext_cal),
                ("tau_ext_pred", result.tau_ext_pred),
                (
                    "model_observation_updated",
                    np.asarray(result.observation_updated, dtype=np.uint8),
                ),
                (
                    "model_observation_timestamp_us",
                    np.asarray(result.observation_timestamp_us, dtype=np.int64),
                ),
                (
                    "model_prediction_age_us",
                    np.asarray(result.prediction_age_us, dtype=np.int64),
                ),
            ):
                if name == "ddq_kf_causal":
                    state_name = "acceleration"
                elif name.endswith("timestamp_us"):
                    state_name = "timestamp"
                elif name.endswith("age_us"):
                    state_name = "duration"
                elif name == "model_observation_updated":
                    state_name = "flag"
                else:
                    state_name = "torque"
                if store:
                    self.teleop_data[name].append(value.copy())
                    self.teleop_state_names[name] = state_name
                processed_values[name] = (state_name, value.copy())
        if not store:
            return None
        return AcceptedTeleopFrame(timestamp_us, processed_values)

    def _process_center_value(
        self,
        dataset_name: str,
        state_name: str,
        value: np.ndarray,
        timestamp_us: int,
    ) -> np.ndarray:
        del dataset_name, state_name, timestamp_us
        # Persist aligned raw samples. Training/checkpoint filters operate on
        # independent copies and never mutate the recorded source signal.
        return value.copy()

    def append_camera(self, camera_name: str, timestamp_us: int, frame: np.ndarray) -> None:
        self.camera_timestamps_us[camera_name].append(int(timestamp_us))
        self.camera_frames[camera_name].append(np.asarray(frame, dtype=np.uint8))

    def warm_up_online_inference(self) -> bool:
        if self.online_tau_ext is None:
            return False
        self.online_tau_ext.warm_up()
        return True

    def reset_online_inference(self) -> bool:
        if self.online_tau_ext is None:
            return False
        self.online_tau_ext.reset_episode()
        return True

    @property
    def sample_count(self) -> int:
        return len(self.teleop_timestamps_us)

    def save(self, path: str | Path) -> Path:
        try:
            import h5py
        except Exception as exc:
            raise RuntimeError(
                "Failed to import h5py. Reinstall compatible numpy/h5py versions, for example: "
                'python -m pip install --upgrade --force-reinstall "numpy>=1.23,<3" "h5py>=3.11"'
            ) from exc

        out_path = Path(path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        finalized_data, finalized_state_names, finalized_attrs = self._finalize_teleop_data()

        with h5py.File(tmp_path, "w") as h5:
            h5.attrs["format"] = FORMAT_VERSION
            h5.attrs["saved_at_us"] = now_us()
            h5.create_dataset("config_yaml", data=self.config.raw_yaml, dtype=string_dtype)

            teleop = h5.create_group("teleop")
            teleop.attrs["arm_names"] = np.asarray(self.arm_names, dtype=string_dtype)
            teleop.attrs["data_role"] = "bilateral"
            teleop.attrs["input_frame_count"] = self.input_frame_count
            teleop.attrs["duplicate_input_frame_count"] = self.duplicate_input_frame_count
            teleop.attrs["command_datasets"] = np.asarray(
                tuple(
                    name
                    for name in ("q_cmd", "gripper_cmd")
                    if name in finalized_data
                ),
                dtype=string_dtype,
            )
            teleop.attrs["joint_layout"] = (
                "follower joint vectors are concatenated in arm_names order"
            )
            teleop.attrs["pose_layout"] = (
                "single follower: (N,4,4); multi follower: (N,A,4,4), "
                "A follows arm_names"
            )
            teleop_timestamp = teleop.create_dataset(
                "timestamp_us",
                data=np.asarray(self.teleop_timestamps_us, dtype=np.int64),
            )
            teleop_timestamp.attrs["source"] = "event_driven_can_state_watermark"
            teleop_timestamp.attrs["clock"] = "unix_epoch"
            teleop_timestamp.attrs["unit"] = "us"
            for name, data in sorted(finalized_data.items()):
                dataset = teleop.create_dataset(name, data=data, compression=_compression_for(data))
                state_name = finalized_state_names.get(name, "")
                dataset.attrs["state_name"] = state_name
                # Collected signals are persisted exactly as received. Derived
                # online inference outputs override these defaults below.
                dataset.attrs["lowpass"] = False
                dataset.attrs["median_window"] = 1
                if name == "ee_pose_follower":
                    dataset.attrs["frame_name"] = "tcp"
                    dataset.attrs["frame_type"] = "end_effector"
                if state_name == "timestamp":
                    dataset.attrs["clock"] = (
                        "monotonic" if name == "control_timestamp_us" else "unix_epoch"
                    )
                    dataset.attrs["unit"] = "us"
                if state_name == "duration":
                    dataset.attrs["unit"] = "us"
                if name in _TIMING_DATASET_SOURCES:
                    dataset.attrs["source"] = _TIMING_DATASET_SOURCES[name]
                    dataset.attrs["timestamp_path"] = "teleop/timestamp_us"
                for key, value in finalized_attrs.get(name, {}).items():
                    dataset.attrs[key] = value

            if self.camera_frames:
                cameras = h5.create_group("cameras")
                for camera_name, frames in sorted(self.camera_frames.items()):
                    if not frames:
                        continue
                    group = cameras.create_group(camera_name)
                    stacked_frames = np.stack(frames, axis=0)
                    group.create_dataset("frames", data=stacked_frames, compression="gzip", compression_opts=4)
                    group.create_dataset(
                        "timestamp_us",
                        data=np.asarray(self.camera_timestamps_us[camera_name], dtype=np.int64),
                    )
                    group.attrs["timeline"] = f"cameras/{camera_name}/timestamp_us"

            meta = h5.create_group("metadata")
            meta.create_dataset("arm_names_json", data=json.dumps(list(self.arm_names)), dtype=string_dtype)
            meta.create_dataset(
                "episode_json",
                data=json.dumps(self.episode_metadata, sort_keys=True),
                dtype=string_dtype,
            )

        tmp_path.replace(out_path)
        return out_path

    def _finalize_teleop_data(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, dict[str, object]]]:
        data = {name: _stack(values) for name, values in self.teleop_data.items()}
        state_names = dict(self.teleop_state_names)
        attrs: dict[str, dict[str, object]] = defaultdict(dict)
        timeline = np.asarray(self.teleop_timestamps_us, dtype=np.int64)
        if timeline.size and np.any(np.diff(timeline) <= 0):
            raise RuntimeError("Teleop acquisition timestamps must be strictly increasing")
        self._apply_dynamics_metadata(data, state_names, attrs, timeline)
        self._append_online_tau_ext_attrs(attrs)
        self._append_wrenches(data, state_names, attrs, timeline)
        return data, state_names, attrs

    def _append_online_tau_ext_attrs(self, attrs) -> None:
        if self.online_tau_ext is None:
            return
        metadata = self.online_tau_ext.metadata
        kalman = self.config.tau_ext_inference.state_estimator
        tau_ext_filter = metadata.tau_ext_filter
        source_filter = self.config.tau_ext_inference.source_butterworth_filter
        sample_rates = {
            name: model_metadata.sample_rate_hz
            for name, model_metadata in (
                ("tau_f", metadata.tau_f),
                ("tau_next", metadata.tau_next),
            )
            if model_metadata is not None
            and model_metadata.sample_rate_hz is not None
        }
        common = {
            "timestamp_path": "teleop/timestamp_us",
            "causal": True,
            "model_observation_policy": "per_model_fixed_rate_observation",
            "fixed_observation_interval": True,
            "active_model_sample_rates_hz_json": json.dumps(sample_rates),
            "observation_grid_phase": "per_model",
            "observation_gap_warning_s": (
                self.config.tau_ext_inference.observation_gap_warning_s
            ),
            "source_butterworth_filter_enabled": source_filter.enabled,
            "source_butterworth_filter_features_json": json.dumps(
                ["q", "dq", "tau"] if source_filter.enabled else []
            ),
            "source_butterworth_filter_cutoff_hz": source_filter.cutoff_hz,
            "source_butterworth_filter_order": source_filter.order,
            "source_butterworth_filter_causal": True,
            "source_butterworth_filter_variable_dt": True,
            "source_butterworth_filter_timestamp_path": "teleop/timestamp_us",
            "source_butterworth_filter_discretization": "trapezoidal_tustin",
            "source_butterworth_filter_initialization": "steady_first_sample",
            "dq_coordinate_sign_correction_json": json.dumps(
                NERO_V120_MOTOR_VELOCITY_TO_JOINT_SIGN
            ),
        }
        if metadata.tau_f is not None:
            attrs["ddq_kf_causal"].update(
                {
                **common,
                "first_valid_sample_index": 0,
                "definition": "causal Kalman acceleration state",
                "processing_method": "variable_dt_constant_acceleration_kalman_filter",
                "lowpass": False,
                "zero_phase": False,
                "measurement_datasets_json": json.dumps(
                    ["teleop/q_follower", "teleop/dq_follower"]
                ),
                "process_noise_model": "continuous_white_jerk",
                "position_std_json": json.dumps(list(kalman.position_std)),
                "velocity_std_json": json.dumps(list(kalman.velocity_std)),
                "jerk_std_json": json.dumps(list(kalman.jerk_std)),
                "initial_position_std_json": json.dumps(
                    list(kalman.initial_position_std)
                ),
                "initial_velocity_std_json": json.dumps(
                    list(kalman.initial_velocity_std)
                ),
                "initial_acceleration_std_json": json.dumps(
                    list(kalman.initial_acceleration_std)
                ),
                "max_gap_s": kalman.max_gap_s,
                }
            )
            attrs["tau_id"].update(
                {
                **common,
                "first_valid_sample_index": 0,
                "definition": "RNEA(q_follower, dq_follower, ddq_kf_causal)",
                "processing_method": "online_pinocchio_rnea",
                "lowpass": False,
                "zero_phase": False,
                "q_source_dataset": "teleop/q_follower",
                "dq_source_dataset": "teleop/dq_follower",
                "ddq_source_dataset": "teleop/ddq_kf_causal",
                "model_urdf": str(
                    self.config.realtime_plot.inverse_dynamics.urdf_path
                ),
                "model_manifest": str(
                    self.config.realtime_plot.inverse_dynamics.manifest_path or ""
                ),
                }
            )
        else:
            disabled_dynamics = {
                **common,
                "definition": "disabled because tau_f checkpoint is empty",
                "processing_method": "disabled_zero_output",
                "lowpass": False,
                "zero_phase": False,
            }
            attrs["ddq_kf_causal"].update(disabled_dynamics)
            attrs["tau_id"].update(disabled_dynamics)
        if metadata.tau_f is not None:
            attrs["tau_id_filtered"].update(
                {
                **common,
                "first_valid_sample_index": 0,
                "definition": "checkpoint_causal_filter(tau_id)",
                "processing_method": "checkpoint_dataloader_filter_pipeline",
                "lowpass": _pipeline_has_operation(
                    metadata.tau_f.dataloader_filters,
                    "tau",
                    "lowpass",
                ),
                "median_window": _pipeline_window(
                    metadata.tau_f.dataloader_filters,
                    "tau",
                    "median",
                    fallback=metadata.tau_f.target_filter_median_window or 1,
                ),
                "filter_operations_json": json.dumps(
                    _metadata_filter_operations(metadata.tau_f, "tau")
                ),
                "zero_phase": False,
                "filter_timeline": "teleop/timestamp_us",
                "filter_initialization": "first_sample",
                "source_dataset": "teleop/tau_id",
                }
            )
            lowpass_cutoff_hz = _pipeline_cutoff_hz(
                metadata.tau_f.dataloader_filters,
                "tau",
                fallback=metadata.tau_f.target_filter_cutoff_hz,
            )
            if lowpass_cutoff_hz is not None:
                attrs["tau_id_filtered"]["lowpass_cutoff_hz"] = lowpass_cutoff_hz
        else:
            attrs["tau_id_filtered"].update(
                {
                    **common,
                    "definition": "disabled because tau_f checkpoint is empty",
                    "processing_method": "disabled_zero_output",
                    "lowpass": False,
                    "zero_phase": False,
                }
            )
        for dataset_name, model_metadata, definition in (
            ("tau_f_pred", metadata.tau_f, "checkpoint prediction of tau_f"),
            ("tau_next_pred", metadata.tau_next, "checkpoint prediction of tau"),
        ):
            if model_metadata is None:
                attrs[dataset_name].update(
                    {
                        **common,
                        "definition": "disabled because checkpoint_path is empty",
                        "processing_method": "disabled_zero_output",
                        "lowpass": False,
                        "zero_phase": False,
                        "model_checkpoint": "",
                        "history_warmup_samples": 0,
                    }
                )
                continue
            attrs[dataset_name].update(
                {
                    **common,
                    "definition": definition,
                    "processing_method": "online_fixed_window",
                    "lowpass": False,
                    "zero_phase": False,
                    "model_checkpoint": str(model_metadata.checkpoint_path),
                    "model_architecture": model_metadata.architecture,
                    "model_horizon": model_metadata.horizon,
                    "history_warmup_samples": model_metadata.horizon,
                    "history_warmup_output": "zeros",
                    "model_inference_mode": model_metadata.inference_mode,
                    "model_input_keys_json": json.dumps(
                        list(model_metadata.input_keys)
                    ),
                    "model_input_dims_json": json.dumps(
                        model_metadata.input_dims,
                        sort_keys=True,
                    ),
                    "model_output_key": model_metadata.output_key,
                    "model_normalize_mode": model_metadata.normalize_mode,
                    "model_sample_rate_hz": (
                        model_metadata.sample_rate_hz
                        if model_metadata.sample_rate_hz is not None
                        else float("nan")
                    ),
                    "observation_sample_rate_hz": (
                        self.online_tau_ext.observation_sample_rate_hz(
                            "tau_f" if dataset_name == "tau_f_pred" else "tau_next"
                        )
                        if hasattr(
                            self.online_tau_ext,
                            "observation_sample_rate_hz",
                        )
                        else model_metadata.sample_rate_hz
                    ),
                    "observation_selection_policy": (
                        "fixed_phase_latest_complete_at_or_before_tick"
                        if dataset_name == "tau_next_pred"
                        else "episode_phase_absolute_nearest_complete_observation"
                    ),
                    "observation_phase_reference": (
                        "unix_epoch"
                        if dataset_name == "tau_next_pred"
                        else "first_complete_observation"
                    ),
                    "model_dataloader_filters_json": json.dumps(
                        model_metadata.dataloader_filters,
                        sort_keys=True,
                    ),
                    "model_target_contract": model_metadata.target_contract or "",
                    "model_target_filter_enabled": (
                        model_metadata.target_filter_enabled
                    ),
                    "model_target_filter_cutoff_hz": (
                        model_metadata.target_filter_cutoff_hz
                        if model_metadata.target_filter_cutoff_hz is not None
                        else float("nan")
                    ),
                    "model_target_filter_moving_average_window": (
                        model_metadata.target_filter_moving_average_window
                        if model_metadata.target_filter_moving_average_window
                        is not None
                        else -1
                    ),
                    "model_target_filter_median_window": (
                        model_metadata.target_filter_median_window
                        if model_metadata.target_filter_median_window is not None
                        else -1
                    ),
                    "model_target_filter_apply_additional_lowpass": (
                        model_metadata.target_filter_apply_additional_lowpass
                    ),
                }
            )
        tau_source_lowpass = bool(
            metadata.tau_f is not None
            and (
                _pipeline_has_operation(
                    metadata.tau_f.dataloader_filters,
                    "tau",
                    "lowpass",
                )
                or metadata.tau_f.target_filter_cutoff_hz
            )
        )
        tau_next_target = "tau_follower"
        if metadata.tau_next is not None and metadata.tau_next.dataloader_filters:
            operations = _metadata_filter_operations(metadata.tau_next, "tau")
            if operations:
                tau_next_target = "checkpoint_causal_filter(tau_follower)"
        elif metadata.tau_next is not None and metadata.tau_next.target_filter_enabled:
            moving_average_window = (
                metadata.tau_next.target_filter_moving_average_window
            )
            if (
                metadata.tau_next.target_filter_apply_additional_lowpass
                and moving_average_window is not None
            ):
                tau_next_target = "target_moving_average_then_lowpass(tau_follower)"
            elif metadata.tau_next.target_filter_apply_additional_lowpass:
                tau_next_target = "target_median_then_lowpass(tau_follower)"
            elif (moving_average_window or 1) > 1:
                tau_next_target = "target_trailing_moving_average(tau_follower)"
            elif (metadata.tau_next.target_filter_median_window or 1) > 1:
                tau_next_target = "target_trailing_median(tau_follower)"
        for raw_name, filtered_name, residual_definition, residual_method, feedback, model_metadata in (
            (
                "tau_ext_cal_raw",
                "tau_ext_cal",
                "tau_id_filtered + tau_f_pred - tau_follower",
                "online_inverse_dynamics_residual",
                False,
                metadata.tau_f,
            ),
            (
                "tau_ext_pred_raw",
                "tau_ext_pred",
                f"tau_next_pred - {tau_next_target}",
                "online_free_space_torque_residual",
                True,
                metadata.tau_next,
            ),
        ):
            if model_metadata is None:
                disabled = {
                    **common,
                    "definition": "disabled because checkpoint_path is empty",
                    "processing_method": "disabled_zero_output",
                    "lowpass": False,
                    "zero_phase": False,
                    "feedback_source": False,
                    "history_warmup_samples": 0,
                    "history_warmup_output": "zeros",
                }
                attrs[raw_name].update(disabled)
                attrs[filtered_name].update(disabled)
                continue
            horizon = model_metadata.horizon
            attrs[raw_name].update(
                {
                    **common,
                    "definition": residual_definition,
                    "processing_method": residual_method,
                    "lowpass": False,
                    "zero_phase": False,
                    "tau_source_lowpass": tau_source_lowpass,
                    "tau_ext_post_filter_applied": False,
                    "feedback_source": False,
                    "history_warmup_samples": horizon,
                    "history_warmup_output": "zeros",
                }
            )
            attrs[filtered_name].update(
                {
                    **common,
                    "definition": (
                        f"tau_ext_filter({residual_definition})"
                        if tau_ext_filter.enabled
                        else residual_definition
                    ),
                    "processing_method": (
                        "causal_hampel_then_butterworth_sos"
                        if tau_ext_filter.enabled
                        and tau_ext_filter.mode == "hampel_butterworth"
                        else f"causal_{tau_ext_filter.mode}_then_one_pole_iir"
                        if tau_ext_filter.enabled
                        else residual_method
                    ),
                    "lowpass": tau_ext_filter.enabled,
                    "median_window": (
                        tau_ext_filter.window
                        if tau_ext_filter.enabled
                        and tau_ext_filter.mode == "median"
                        else 1
                    ),
                    "moving_average_window": (
                        tau_ext_filter.window
                        if tau_ext_filter.enabled
                        and tau_ext_filter.mode == "moving_average"
                        else 1
                    ),
                    "zero_phase": False,
                    "tau_source_lowpass": tau_source_lowpass,
                    "tau_ext_post_filter_applied": tau_ext_filter.enabled,
                    "tau_ext_filter_mode": tau_ext_filter.mode,
                    "filter_initialization": "first_real_sample_padded",
                    "filter_timeline": "teleop/timestamp_us",
                    "source_dataset": f"teleop/{raw_name}",
                    "feedback_source": feedback,
                    "history_warmup_samples": horizon,
                    "history_warmup_output": "zeros",
                }
            )
            if tau_ext_filter.enabled:
                attrs[filtered_name]["lowpass_cutoff_hz"] = (
                    tau_ext_filter.cutoff_hz
                )
                if tau_ext_filter.mode == "hampel_butterworth":
                    attrs[filtered_name].update(
                        {
                            "hampel_window": tau_ext_filter.window,
                            "hampel_n_sigma": tau_ext_filter.hampel_n_sigma,
                            "butterworth_order": tau_ext_filter.order,
                            "filter_sample_rate_hz": tau_ext_filter.sample_rate_hz,
                            "filter_initialization": "steady_first_real_sample",
                        }
                    )

    def _append_wrenches(self, data, state_names, attrs, timeline) -> None:
        if "q_follower" not in data:
            return
        q = np.asarray(data["q_follower"], dtype=np.float64)
        expected_joint_shape = (timeline.size, 7)
        if q.shape != expected_joint_shape:
            raise RuntimeError(
                f"Cannot compute external wrenches from q_follower shape {q.shape}; "
                f"expected {expected_joint_shape}"
            )
        mapper = PinocchioContactWrenchEstimator(
            self.config.realtime_plot.wrench_mapping
        )
        mapping = self.config.realtime_plot.wrench_mapping
        for tau_name, wrench_name in (
            ("tau_ext_cal_raw", "wrench_cal_raw"),
            ("tau_ext_pred_raw", "wrench_pred_raw"),
            ("tau_ext_cal", "wrench_cal"),
            ("tau_ext_pred", "wrench_pred"),
        ):
            if tau_name not in data:
                continue
            tau_ext = np.asarray(data[tau_name], dtype=np.float64)
            if tau_ext.shape != expected_joint_shape:
                raise RuntimeError(
                    f"Cannot compute {wrench_name} from {tau_name} shape "
                    f"{tau_ext.shape}; expected {expected_joint_shape}"
                )
            wrench = np.empty((timeline.size, 6), dtype=np.float64)
            for index, (q_value, tau_value) in enumerate(zip(q, tau_ext)):
                wrench[index] = mapper.map_joint_torque(q_value, tau_value).wrench
            data[wrench_name] = wrench
            state_names[wrench_name] = "wrench"
            attrs[wrench_name].update(
                {
                    "definition": (
                        f"damped least-squares solution of {tau_name} = "
                        f"J(q)^T {wrench_name}"
                    ),
                    "processing_method": (
                        "pinocchio_frame_jacobian_damped_least_squares"
                    ),
                    "timestamp_path": "teleop/timestamp_us",
                    "q_source_dataset": "teleop/q_follower",
                    "tau_source_dataset": f"teleop/{tau_name}",
                    "tau_ext_post_filter_applied": bool(
                        attrs[tau_name].get("tau_ext_post_filter_applied", False)
                    ),
                    "components_json": json.dumps(
                        ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
                    ),
                    "component_units_json": json.dumps(
                        ["N", "N", "N", "N.m", "N.m", "N.m"]
                    ),
                    "frame_name": mapping.frame_name,
                    "frame_type": "end_effector",
                    "reference_frame": mapping.reference_frame,
                    "wrench_convention": "environment_on_tool",
                    "damping": mapping.damping,
                    "model_urdf": str(mapping.urdf_path),
                }
            )

    def _apply_dynamics_metadata(self, data, state_names, attrs, timeline) -> None:
        for role in ("leader", "follower"):
            q_name = f"q_{role}"
            dq_name = f"dq_{role}"
            ddq_name = f"ddq_{role}"
            if q_name in data:
                q_values = np.asarray(data[q_name], dtype=np.float64)
                if q_values.ndim != 2 or q_values.shape[0] != timeline.size:
                    raise RuntimeError(f"Cannot save {q_name} with shape {q_values.shape}")
                state_names[q_name] = "q"
                if dq_name in data:
                    if np.asarray(data[dq_name]).shape != q_values.shape:
                        raise RuntimeError(
                            f"Centered {dq_name} must match {q_name}; got {np.asarray(data[dq_name]).shape}"
                        )
                    state_names[dq_name] = "velocity"
                if ddq_name in data:
                    if np.asarray(data[ddq_name]).shape != q_values.shape:
                        raise RuntimeError(
                            f"Centered {ddq_name} must match {q_name}; got {np.asarray(data[ddq_name]).shape}"
                        )
                    state_names[ddq_name] = "acceleration"
                attrs[q_name].update(
                    {
                        "processing_method": "nearest_complete_joint_group_state",
                        "timestamp_path": "teleop/timestamp_us",
                        "lowpass": False,
                        "median_window": 1,
                        "derivative_q_mean_window_samples": 1,
                        "derivative_q_mean_window_enabled": False,
                        "derivative_q_lowpass": False,
                    }
                )
                if dq_name in data:
                    attrs[dq_name].update(
                        {
                            "derived_from": "motor_state_velocity",
                            "derivative_method": (
                                "sign_corrected_official_motor_velocity_unfiltered"
                            ),
                            "timestamp_path": "teleop/timestamp_us",
                            "first_valid_sample_index": 0,
                            "uses_official_firmware_velocity": True,
                            "formula": "dq_raw[k]=motor_state_velocity[k]*joint_sign",
                            "coordinate_sign_correction_json": json.dumps(
                                NERO_V120_MOTOR_VELOCITY_TO_JOINT_SIGN
                            ),
                            "coordinate_frame": "nero_joint_position_coordinates",
                            "lowpass": False,
                            "median_window": 1,
                        }
                    )
                if ddq_name in data:
                    attrs[ddq_name].update(
                        {
                            "derived_from_json": json.dumps([dq_name]),
                            "derivative_method": (
                                "causal_first_derivative_of_sign_corrected_raw_official_velocity"
                            ),
                            "timestamp_path": "teleop/timestamp_us",
                            "first_valid_sample_index": 0,
                            "uses_measured_intervals": True,
                            "formula": "ddq_raw[k]=(dq[k]-dq[k-1])/measured_dt",
                            "lowpass": False,
                            "median_window": 1,
                        }
                    )

            tau_name = f"tau_{role}"
            if tau_name not in data:
                continue
            tau_values = np.asarray(data[tau_name], dtype=np.float64)
            if tau_values.ndim != 2 or tau_values.shape[0] != timeline.size:
                raise RuntimeError(f"Cannot save {tau_name} with shape {tau_values.shape}")
            state_names[tau_name] = "torque"
            attrs[tau_name].update(
                {
                    "processing_method": "nearest_motor_sample_unfiltered",
                    "timestamp_path": "teleop/timestamp_us",
                    "median_window": 1,
                    "zero_phase": False,
                    "causal": True,
                    "lowpass": False,
                }
            )


def _metadata_filter_operations(metadata, key: str) -> list[dict[str, object]]:
    spec = metadata.dataloader_filters.get(key) or {}
    if bool(spec.get("enabled", False)):
        return [dict(operation) for operation in spec.get("operations", ())]
    operations = []
    if metadata.target_filter_median_window is not None:
        operations.append(
            {"type": "median", "window": metadata.target_filter_median_window}
        )
    if metadata.target_filter_cutoff_hz is not None:
        operations.append(
            {"type": "lowpass", "cutoff_hz": metadata.target_filter_cutoff_hz}
        )
    return operations


def _pipeline_has_operation(filters, key: str, operation_type: str) -> bool:
    spec = filters.get(key) or {}
    return bool(spec.get("enabled", False)) and any(
        operation.get("type") == operation_type
        for operation in spec.get("operations", ())
    )


def _pipeline_window(filters, key: str, operation_type: str, *, fallback: int) -> int:
    spec = filters.get(key) or {}
    for operation in spec.get("operations", ()):
        if operation.get("type") == operation_type:
            return int(operation["window"])
    return int(fallback)


def _pipeline_cutoff_hz(filters, key: str, *, fallback):
    spec = filters.get(key) or {}
    for operation in spec.get("operations", ()):
        if operation.get("type") == "lowpass":
            return float(operation["cutoff_hz"])
    return None if fallback is None else float(fallback)

def _stack(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float64)
    return np.stack(values, axis=0)

def _compression_for(data: np.ndarray) -> str | None:
    if data.dtype == np.uint8 or data.size > 2048:
        return "gzip"
    return None
