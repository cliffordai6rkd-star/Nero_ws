from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nero_collection.config import CollectionConfig
from nero_collection.coordinates import NERO_V120_MOTOR_VELOCITY_TO_JOINT_SIGN
from nero_collection.tau_ext_inference import OnlineTauExtInference
from nero_collection.time_utils import now_us


FORMAT_VERSION = "factr_multimodal_episode/v12"

FOLLOWER_TELEOP_DATASETS = frozenset(
    {
        "q_follower",
        "q_leader",
        "q_cmd",
        "dq_cmd",
        "dq_follower",
        "dq_leader",
        "ee_pose_follower",
        "tau_follower",
        "tau_leader",
        "current_follower",
        "current_leader",
        "gripper_follower",
        "gripper_cmd",
        "tau_id",
        "tau_id_filtered",
        "tau_f_pred",
        "tau_next_pred",
        "tau_ext_cal_raw",
        "tau_ext_pred_raw",
        "tau_ext_cal",
        "tau_ext_pred",
        "model_observation_updated",
        "model_observation_timestamp_us",
        "model_prediction_age_us",
    }
)

_TIMING_DATASET_SOURCES = {
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
                self.config.tau_ext_inference.inverse_dynamics,
                self.config.dynamics_processing,
                self.config.robot_states,
                source_sample_rate_hz=self.config.teleop.command.sample_rate_hz,
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
                if name.endswith("timestamp_us"):
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
                    for name in ("q_cmd", "dq_cmd", "gripper_cmd")
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
            teleop_timestamp.attrs["source"] = "fixed_rate_robot_state_sample"
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
                if name == "q_cmd":
                    dataset.attrs["source"] = "actual_follower_command"
                    dataset.attrs["definition"] = (
                        "latest successfully issued follower joint-position "
                        "target effective at the robot-state sample"
                    )
                    dataset.attrs["command_semantics"] = "causal_zoh_at_state_sample"
                if name == "dq_cmd":
                    dataset.attrs["source"] = "actual_follower_command"
                    dataset.attrs["definition"] = (
                        "follower joint-velocity target issued with q_cmd"
                    )
                    dataset.attrs["command_semantics"] = "same_mit_command_as_q_cmd"
                if state_name == "timestamp":
                    dataset.attrs["clock"] = "unix_epoch"
                    dataset.attrs["unit"] = "us"
                if state_name == "duration":
                    dataset.attrs["unit"] = "us"
                if name in _TIMING_DATASET_SOURCES:
                    dataset.attrs["source"] = _TIMING_DATASET_SOURCES[name]
                    dataset.attrs["timestamp_path"] = "teleop/timestamp_us"
                if state_name not in {"timestamp", "duration"}:
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
        return data, state_names, attrs

    def _append_online_tau_ext_attrs(self, attrs) -> None:
        if self.online_tau_ext is None:
            return
        metadata = self.online_tau_ext.metadata
        tau_ext_filter = metadata.tau_ext_filter
        feedback_source = self.config.tau_ext_inference.feedback_source
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
        source_sample_rate_hz = getattr(
            self.online_tau_ext,
            "source_sample_rate_hz",
            self.config.teleop.command.sample_rate_hz,
        )
        common = {
            "timestamp_path": "teleop/timestamp_us",
            "causal": True,
            "model_observation_policy": "per_model_filtered_source_frame_stride",
            "fixed_observation_interval": False,
            "active_model_sample_rates_hz_json": json.dumps(sample_rates),
            "source_sample_rate_hz": source_sample_rate_hz,
            "observation_grid_phase": "episode_source_frame_zero",
            "configured_force_feedback_source": feedback_source,
            "observation_gap_warning_s": (
                self.config.tau_ext_inference.observation_gap_warning_s
            ),
            "source_butterworth_filter_enabled": source_filter.enabled,
            "source_butterworth_filter_features_json": json.dumps(
                ["q", "dq", "tau", "q_cmd"] if source_filter.enabled else []
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
            attrs["tau_id"].update(
                {
                **common,
                "first_valid_sample_index": 0,
                "definition": "RNEA using internal causal Kalman acceleration",
                "processing_method": "online_pinocchio_rnea",
                "lowpass": False,
                "zero_phase": False,
                "q_source_dataset": "teleop/q_follower",
                "dq_source_dataset": "teleop/dq_follower",
                "ddq_source": "internal_causal_kalman_not_persisted",
                "model_urdf": str(
                    self.config.tau_ext_inference.inverse_dynamics.urdf_path
                ),
                "model_manifest": str(
                    self.config.tau_ext_inference.inverse_dynamics.manifest_path or ""
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
            model_name = "tau_f" if dataset_name == "tau_f_pred" else "tau_next"
            observation_sample_rate_hz = (
                self.online_tau_ext.observation_sample_rate_hz(model_name)
                if hasattr(self.online_tau_ext, "observation_sample_rate_hz")
                else model_metadata.sample_rate_hz
            )
            if hasattr(self.online_tau_ext, "observation_stride_frames"):
                observation_stride_frames = (
                    self.online_tau_ext.observation_stride_frames(model_name)
                )
            elif source_sample_rate_hz and observation_sample_rate_hz:
                ratio = float(source_sample_rate_hz) / float(observation_sample_rate_hz)
                rounded_ratio = int(round(ratio))
                observation_stride_frames = (
                    rounded_ratio
                    if rounded_ratio >= 1 and np.isclose(ratio, rounded_ratio)
                    else None
                )
            else:
                observation_stride_frames = None
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
                    "observation_sample_rate_hz": observation_sample_rate_hz,
                    "observation_selection_policy": (
                        "first_filtered_source_frame_then_every_nth_frame"
                    ),
                    "observation_phase_reference": "first_complete_source_frame",
                    "observation_stride_frames": observation_stride_frames,
                    "model_dataloader_filters_json": json.dumps(
                        model_metadata.dataloader_filters,
                        sort_keys=True,
                    ),
                    "model_derived_target_config_json": json.dumps(
                        model_metadata.derived_target_config,
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
                feedback_source == "tau_f",
                metadata.tau_f,
            ),
            (
                "tau_ext_pred_raw",
                "tau_ext_pred",
                f"tau_next_pred - {tau_next_target}",
                "online_free_space_torque_residual",
                feedback_source == "tau_free",
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

    def _apply_dynamics_metadata(self, data, state_names, attrs, timeline) -> None:
        for role in ("leader", "follower"):
            q_name = f"q_{role}"
            dq_name = f"dq_{role}"
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
