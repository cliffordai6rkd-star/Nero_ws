from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nero_collection.config import CollectionConfig, StateParamConfig
from nero_collection.coordinates import NERO_V120_MOTOR_VELOCITY_TO_JOINT_SIGN
from nero_collection.contact_wrench import (
    PinocchioContactWrenchEstimator,
)
from nero_collection.filters import (
    DatasetFilterBank,
    OnePoleLowPass,
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
    }
)

_MEASURED_TORQUE_DATASETS = frozenset({"tau_leader", "tau_follower"})
_INFERENCE_TORQUE_DATASETS = frozenset(
    {
        "tau_id",
        "tau_id_filtered",
        "tau_f_pred",
        "tau_next_pred",
        "tau_ext_cal_raw",
        "tau_ext_pred_raw",
        "tau_ext_cal",
        "tau_ext_pred",
    }
)

_TIMING_DATASET_SOURCES = {
    "control_timestamp_us": "teleop_control_loop",
    "state_acquired_timestamp_follower_us": "host_state_read",
    "q_source_timestamp_follower_us": "pyagx_can_joint_group_frames",
    "motor_source_timestamp_follower_us": "pyagx_can_motor_state_frames",
    "state_source_skew_follower_us": "max_minus_min_q_and_motor_source_timestamp",
}


@dataclass(frozen=True)
class AcceptedTeleopFrame:
    timestamp_us: int
    values: dict[str, tuple[str, np.ndarray]]


@dataclass
class EpisodeBuffer:
    config: CollectionConfig
    arm_names: tuple[str, ...]
    sample_rate_hz: float
    teleop_timestamps_us: list[int] = field(default_factory=list)
    teleop_data: dict[str, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    teleop_state_names: dict[str, str] = field(default_factory=dict)
    camera_frames: dict[str, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    camera_timestamps_us: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    episode_metadata: dict[str, Any] = field(default_factory=dict)
    measured_torque_filters: dict[str, OnePoleLowPass] = field(
        init=False, default_factory=dict
    )
    last_input_timestamp_us: int | None = field(init=False, default=None)
    input_frame_count: int = field(init=False, default=0)
    duplicate_input_frame_count: int = field(init=False, default=0)
    online_tau_ext: OnlineTauExtInference | None = None

    def __post_init__(self) -> None:
        self.filter_bank = DatasetFilterBank(self.config.robot_states)
        if self.online_tau_ext is None and self.config.tau_ext_inference.enabled:
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
            ):
                state_name = "acceleration" if name == "ddq_kf_causal" else "torque"
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
        if dataset_name in _INFERENCE_TORQUE_DATASETS:
            return value.copy()
        if state_name in {"q", "velocity", "acceleration"}:
            return value.copy()
        if dataset_name in _MEASURED_TORQUE_DATASETS and self.config.dynamics_processing.enabled:
            filt = self.measured_torque_filters.get(dataset_name)
            if filt is None:
                processing = self.config.dynamics_processing
                filt = OnePoleLowPass(
                    processing.torque_lowpass_hz,
                    processing.torque_median_window,
                )
                self.measured_torque_filters[dataset_name] = filt
            return filt.apply(value, timestamp_us)
        return self.filter_bank.apply(dataset_name, state_name, value, timestamp_us)

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
            teleop_timestamp.attrs["source"] = "delayed_nearest_joint_group_state"
            teleop_timestamp.attrs["clock"] = "unix_epoch"
            teleop_timestamp.attrs["unit"] = "us"
            for name, data in sorted(finalized_data.items()):
                dataset = teleop.create_dataset(name, data=data, compression=_compression_for(data))
                state_name = finalized_state_names.get(name, "")
                dataset.attrs["state_name"] = state_name
                state_config = self.config.robot_states.get(state_name)
                saved_value_is_lowpass = bool(
                    state_config and state_config.lowpass and state_name != "q"
                )
                dataset.attrs["lowpass"] = saved_value_is_lowpass
                dataset.attrs["median_window"] = state_config.median_window if state_config else 1
                if state_config and saved_value_is_lowpass:
                    dataset.attrs["lowpass_cutoff_hz"] = state_config.lowpass_cutoff_hz
                    dataset.attrs["filter_timeline"] = "teleop/timestamp_us"
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
        processing = self.config.dynamics_processing
        if timeline.size and np.any(np.diff(timeline) <= 0):
            raise RuntimeError("Teleop acquisition timestamps must be strictly increasing")
        self._apply_dynamics_metadata(data, state_names, attrs, timeline)
        if processing.enabled and timeline.size < processing.min_samples:
            raise RuntimeError(
                "Dynamics-aware episode saving requires at least "
                f"{processing.min_samples} samples; got {timeline.size}"
            )
        if processing.enabled and processing.state_method != "finite_difference":
            raise RuntimeError(
                "H5 episode saving only supports causal finite_difference processing; "
                f"got {processing.state_method!r}"
            )
        self._append_online_tau_ext_attrs(attrs)
        self._append_wrenches(data, state_names, attrs, timeline)
        return data, state_names, attrs

    def _append_online_tau_ext_attrs(self, attrs) -> None:
        if self.online_tau_ext is None:
            return
        metadata = self.online_tau_ext.metadata
        kalman = self.config.tau_ext_inference.state_estimator
        tau_ext_filter = metadata.tau_ext_filter
        common = {
            "timestamp_path": "teleop/timestamp_us",
            "causal": True,
            "dq_coordinate_sign_correction_json": json.dumps(
                NERO_V120_MOTOR_VELOCITY_TO_JOINT_SIGN
            ),
        }
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
        attrs["tau_id_filtered"].update(
            {
                **common,
                "first_valid_sample_index": 0,
                "definition": "causal_lowpass(tau_id)",
                "processing_method": "causal_median_then_one_pole_iir",
                "lowpass": True,
                "lowpass_cutoff_hz": metadata.tau_f.target_filter_cutoff_hz,
                "median_window": metadata.tau_f.target_filter_median_window,
                "zero_phase": False,
                "filter_timeline": "teleop/timestamp_us",
                "filter_initialization": "first_sample",
                "source_dataset": "teleop/tau_id",
            }
        )
        for dataset_name, model_metadata, definition in (
            ("tau_f_pred", metadata.tau_f, "checkpoint prediction of tau_f"),
            ("tau_next_pred", metadata.tau_next, "checkpoint prediction of tau"),
        ):
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
                    "model_target_contract": model_metadata.target_contract or "",
                    "model_target_filter_enabled": (
                        model_metadata.target_filter_enabled
                    ),
                    "model_target_filter_cutoff_hz": (
                        model_metadata.target_filter_cutoff_hz
                        if model_metadata.target_filter_cutoff_hz is not None
                        else float("nan")
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
            self.config.dynamics_processing.enabled
            or self.config.robot_states.get("torque", StateParamConfig()).lowpass
        )
        tau_next_target = "tau_follower"
        if metadata.tau_next.target_filter_enabled:
            if metadata.tau_next.target_filter_apply_additional_lowpass:
                tau_next_target = "target_median_then_lowpass(tau_follower)"
            elif (metadata.tau_next.target_filter_median_window or 1) > 1:
                tau_next_target = "target_trailing_median(tau_follower)"
        for raw_name, filtered_name, residual_definition, residual_method, feedback, horizon in (
            (
                "tau_ext_cal_raw",
                "tau_ext_cal",
                "tau_id_filtered + tau_f_pred - tau_follower",
                "online_inverse_dynamics_residual",
                False,
                metadata.tau_f.horizon,
            ),
            (
                "tau_ext_pred_raw",
                "tau_ext_pred",
                f"tau_next_pred - {tau_next_target}",
                "online_free_space_torque_residual",
                True,
                metadata.tau_next.horizon,
            ),
        ):
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
                        f"causal_{tau_ext_filter.mode}_then_one_pole_iir"
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
        processing = self.config.dynamics_processing
        q_state = self.config.robot_states["q"]
        dq_state = self.config.robot_states["velocity"]
        ddq_state = self.config.robot_states["acceleration"]
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
                        "derivative_q_lowpass": q_state.lowpass,
                        "derivative_q_lowpass_cutoff_hz": q_state.lowpass_cutoff_hz,
                    }
                )
                if dq_name in data:
                    attrs[dq_name].update(
                        {
                            "derived_from": "motor_state_velocity",
                            "derivative_method": (
                                "sign_corrected_official_motor_velocity_with_causal_lowpass"
                            ),
                            "timestamp_path": "teleop/timestamp_us",
                            "first_valid_sample_index": 0,
                            "uses_official_firmware_velocity": True,
                            "formula": "dq_raw[k]=motor_state_velocity[k]*joint_sign",
                            "coordinate_sign_correction_json": json.dumps(
                                NERO_V120_MOTOR_VELOCITY_TO_JOINT_SIGN
                            ),
                            "coordinate_frame": "nero_joint_position_coordinates",
                            "lowpass": dq_state.lowpass,
                            "lowpass_cutoff_hz": dq_state.lowpass_cutoff_hz,
                            "median_window": 1,
                        }
                    )
                if ddq_name in data:
                    attrs[ddq_name].update(
                        {
                            "derived_from_json": json.dumps([dq_name]),
                            "derivative_method": (
                                "causal_first_derivative_of_sign_corrected_filtered_official_velocity"
                            ),
                            "timestamp_path": "teleop/timestamp_us",
                            "first_valid_sample_index": 0,
                            "uses_measured_intervals": True,
                            "formula": "ddq_raw[k]=(dq[k]-dq[k-1])/measured_dt",
                            "lowpass": ddq_state.lowpass,
                            "lowpass_cutoff_hz": ddq_state.lowpass_cutoff_hz,
                            "median_window": 1,
                        }
                    )

            tau_name = f"tau_{role}"
            if tau_name not in data:
                continue
            tau_values = np.asarray(data[tau_name], dtype=np.float64)
            if tau_values.ndim != 2 or tau_values.shape[0] != timeline.size:
                raise RuntimeError(f"Cannot save {tau_name} with shape {tau_values.shape}")
            torque_config = self.config.robot_states.get("torque", StateParamConfig())
            if processing.enabled:
                processing_method = "causal_median_then_one_pole_iir"
                lowpass_cutoff_hz = processing.torque_lowpass_hz
                median_window = processing.torque_median_window
                lowpass = True
            elif torque_config.lowpass:
                processing_method = "causal_median_then_one_pole_iir"
                lowpass_cutoff_hz = torque_config.lowpass_cutoff_hz
                median_window = torque_config.median_window
                lowpass = True
            else:
                processing_method = "nearest_motor_sample_unfiltered"
                lowpass_cutoff_hz = None
                median_window = 1
                lowpass = False
            state_names[tau_name] = "torque"
            attrs[tau_name].update(
                {
                    "processing_method": processing_method,
                    "timestamp_path": "teleop/timestamp_us",
                    "median_window": median_window,
                    "zero_phase": False,
                    "causal": True,
                    "lowpass": lowpass,
                }
            )
            if lowpass_cutoff_hz is not None:
                attrs[tau_name]["lowpass_cutoff_hz"] = lowpass_cutoff_hz

def _stack(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float64)
    return np.stack(values, axis=0)

def _compression_for(data: np.ndarray) -> str | None:
    if data.dtype == np.uint8 or data.size > 2048:
        return "gzip"
    return None
