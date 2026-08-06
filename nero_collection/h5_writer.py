from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nero_collection.config import CollectionConfig, StateParamConfig
from nero_collection.contact_wrench import (
    PinocchioContactWrenchEstimator,
    PinocchioJointTorqueResidualEstimator,
    wrench_ext_dataset_attrs,
)
from nero_collection.filters import (
    DatasetFilterBank,
    OnePoleLowPass,
)
from nero_collection.tau_f_inference import OnlineTauFInference
from nero_collection.time_utils import now_us


FORMAT_VERSION = "factr_multimodal_episode/v7"

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
        "tau_f_cal",
        "tau_f_pred",
        "tau_bg_pred",
        "tau_ext_raw",
        "tau_ext_filtered",
        "tau_ext",
    }
)

_MEASURED_TORQUE_DATASETS = frozenset({"tau_leader", "tau_follower"})
_INFERENCE_TORQUE_DATASETS = frozenset(
    {
        "tau_f_cal",
        "tau_f_pred",
        "tau_bg_pred",
        "tau_ext_raw",
        "tau_ext_filtered",
        "tau_ext",
    }
)


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
    online_tau_f: OnlineTauFInference | None = None

    def __post_init__(self) -> None:
        self.filter_bank = DatasetFilterBank(self.config.robot_states)
        if self.online_tau_f is None and (
            self.config.tau_f_inference.enabled
            or self.config.realtime_plot.enabled
        ):
            self.online_tau_f = OnlineTauFInference(
                self.config.tau_f_inference,
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

        has_precomputed_tau_ext = "tau_ext" in processed_values
        if self.online_tau_f is not None and not has_precomputed_tau_ext:
            required = ("q_follower", "dq_follower", "ddq_follower", "tau_follower")
            missing = [name for name in required if name not in processed_values]
            if missing:
                raise RuntimeError(
                    f"tau_f inference is missing follower datasets: {missing}"
                )
            result = self.online_tau_f.estimate_centered(
                timestamp_us,
                processed_values["q_follower"][1],
                processed_values["dq_follower"][1],
                processed_values["ddq_follower"][1],
                processed_values["tau_follower"][1],
            )
            prediction_name = (
                "tau_bg_pred"
                if self.config.tau_f_inference.mode == "tau_bg"
                else "tau_f_pred"
            )
            for name, value in (
                ("tau_f_cal", result.tau_f_cal),
                (prediction_name, result.model_prediction),
                (
                    "tau_ext_raw",
                    result.tau_ext_raw
                    if result.tau_ext_raw is not None
                    else result.tau_ext,
                ),
                (
                    "tau_ext_filtered",
                    result.tau_ext_filtered
                    if result.tau_ext_filtered is not None
                    else result.tau_ext,
                ),
                ("tau_ext", result.tau_ext),
            ):
                if store:
                    self.teleop_data[name].append(value.copy())
                    self.teleop_state_names[name] = "torque"
                processed_values[name] = ("torque", value.copy())
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
        if self.online_tau_f is None:
            return False
        self.online_tau_f.warm_up()
        return True

    def reset_online_inference(self) -> bool:
        if self.online_tau_f is None:
            return False
        self.online_tau_f.reset_episode()
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
                    dataset.attrs["clock"] = "unix_epoch"
                    dataset.attrs["unit"] = "us"
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
        if not processing.enabled:
            if self.online_tau_f is None:
                self._append_follower_tau_f(data, state_names, attrs, timeline)
            else:
                self._append_online_tau_f_attrs(attrs)
            self._append_wrench_ext(data, state_names, attrs, timeline)
            return data, state_names, attrs

        if timeline.size < processing.min_samples:
            raise RuntimeError(
                "Dynamics-aware episode saving requires at least "
                f"{processing.min_samples} samples; got {timeline.size}"
            )
        if processing.state_method != "finite_difference":
            raise RuntimeError(
                "H5 episode saving only supports causal finite_difference processing; "
                f"got {processing.state_method!r}"
            )
        if self.online_tau_f is None:
            self._append_follower_tau_f(data, state_names, attrs, timeline)
        else:
            self._append_online_tau_f_attrs(attrs)
        self._append_wrench_ext(data, state_names, attrs, timeline)
        return data, state_names, attrs

    def _append_online_tau_f_attrs(self, attrs) -> None:
        if self.online_tau_f is None:
            return
        metadata = self.online_tau_f.metadata
        inference = self.config.tau_f_inference
        mode = inference.mode
        prediction_name = "tau_bg_pred" if mode == "tau_bg" else "tau_f_pred"
        raw_definition = (
            "tau_follower - tau_bg_pred"
            if mode == "tau_bg"
            else "tau_f_cal - tau_f_pred"
        )
        q_state = self.config.robot_states["q"]
        dq_state = self.config.robot_states["velocity"]
        ddq_state = self.config.robot_states["acceleration"]
        common = {
            "timestamp_path": "teleop/timestamp_us",
            "causal": True,
            "first_valid_sample_index": 0,
            "state_derivative_method": "official_velocity_with_fused_acceleration",
            "q_mean_window_samples": 1,
            "q_mean_window_enabled": False,
            "q_lowpass_cutoff_hz": q_state.lowpass_cutoff_hz,
            "dq_lowpass_cutoff_hz": dq_state.lowpass_cutoff_hz,
            "ddq_lowpass_cutoff_hz": ddq_state.lowpass_cutoff_hz,
            "model_checkpoint": str(metadata.checkpoint_path),
            "model_architecture": metadata.architecture,
            "model_horizon": metadata.horizon,
            "model_training_horizon": metadata.horizon,
            "model_inference_mode": (
                "constant_zero"
                if metadata.architecture == "zeros"
                else "stateful_recurrent_step"
            ),
            "model_input_keys_json": json.dumps(list(metadata.input_keys)),
            "model_input_dims_json": json.dumps(metadata.input_dims, sort_keys=True),
            "model_output_key": metadata.output_key,
            "model_normalize_mode": metadata.normalize_mode,
            "external_torque_mode": mode,
        }
        attrs["tau_f_cal"].update(
            {
                **common,
                "definition": "tau_id - tau_follower",
                "processing_method": "online_rnea_minus_filtered_measured_torque",
            }
        )
        attrs[prediction_name].update(
            {
                **common,
                "definition": (
                    f"constant zero {prediction_name} compatibility value"
                    if metadata.architecture == "zeros"
                    else f"checkpoint prediction of {mode}"
                ),
                "processing_method": (
                    "constant_zero_when_tau_f_inference_disabled"
                    if metadata.architecture == "zeros"
                    else "online_stateful_recurrent_single_frame"
                ),
            }
        )
        attrs["tau_ext_raw"].update(
            {
                **common,
                "definition": raw_definition,
                "processing_method": "online_external_torque_residual_unfiltered",
                "lowpass": False,
                "gate_applied": False,
            }
        )
        filtered_attrs = {
            **common,
            "definition": f"lowpass({raw_definition})",
            "processing_method": (
                "causal_one_pole_iir"
                if inference.tau_ext_lowpass_hz is not None
                else "identity"
            ),
            "lowpass": inference.tau_ext_lowpass_hz is not None,
            "gate_applied": False,
        }
        final_attrs = {
            **common,
            "definition": raw_definition,
            "processing_method": "online_lowpass_then_hard_gate",
            "lowpass": inference.tau_ext_lowpass_hz is not None,
            "gate_applied": True,
            "gate_type": "per_joint_absolute_hard_threshold",
            "gate_threshold_nm_json": json.dumps(
                list(inference.tau_ext_gate_threshold_nm)
            ),
        }
        if inference.tau_ext_lowpass_hz is not None:
            filtered_attrs["lowpass_cutoff_hz"] = inference.tau_ext_lowpass_hz
            final_attrs["lowpass_cutoff_hz"] = inference.tau_ext_lowpass_hz
        attrs["tau_ext_filtered"].update(filtered_attrs)
        attrs["tau_ext"].update(final_attrs)

    def _append_wrench_ext(self, data, state_names, attrs, timeline) -> None:
        if "q_follower" not in data or "tau_ext" not in data:
            return
        q = np.asarray(data["q_follower"], dtype=np.float64)
        tau_ext = np.asarray(data["tau_ext"], dtype=np.float64)
        expected_joint_shape = (timeline.size, 7)
        if q.shape != expected_joint_shape or tau_ext.shape != expected_joint_shape:
            raise RuntimeError(
                "Cannot compute wrench_ext from q_follower/tau_ext shapes "
                f"{q.shape}/{tau_ext.shape}; expected {expected_joint_shape}"
            )
        mapper = PinocchioContactWrenchEstimator(
            self.config.realtime_plot.wrench_mapping
        )
        wrench = np.empty((timeline.size, 6), dtype=np.float64)
        for index, (q_value, tau_value) in enumerate(zip(q, tau_ext)):
            wrench[index] = mapper.map_joint_torque(q_value, tau_value).wrench
        data["wrench_ext"] = wrench
        state_names["wrench_ext"] = "wrench"
        attrs["wrench_ext"].update(
            wrench_ext_dataset_attrs(self.config.realtime_plot.wrench_mapping)
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
                            "derivative_method": "official_motor_velocity_with_causal_lowpass",
                            "timestamp_path": "teleop/timestamp_us",
                            "first_valid_sample_index": 0,
                            "uses_official_firmware_velocity": True,
                            "formula": "dq_raw[k]=motor_state_velocity[k]",
                            "lowpass": dq_state.lowpass,
                            "lowpass_cutoff_hz": dq_state.lowpass_cutoff_hz,
                            "median_window": 1,
                        }
                    )
                if ddq_name in data:
                    attrs[ddq_name].update(
                        {
                            "derived_from_json": json.dumps([dq_name, q_name]),
                            "derivative_method": "fused_dq_first_derivative_and_q_second_derivative",
                            "timestamp_path": "teleop/timestamp_us",
                            "first_valid_sample_index": 0,
                            "uses_measured_intervals": True,
                            "formula": "ddq_raw[k]=mean(d(dq_official)/dt, d2(q_lp)/dt2)",
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

    def _append_follower_tau_f(self, data, state_names, attrs, timeline) -> None:
        required = ("q_follower", "dq_follower", "ddq_follower", "tau_follower")
        if not all(name in data for name in required):
            return
        if timeline.size == 0:
            return
        q = np.asarray(data["q_follower"], dtype=np.float64)
        dq = np.asarray(data["dq_follower"], dtype=np.float64)
        ddq = np.asarray(data["ddq_follower"], dtype=np.float64)
        tau = np.asarray(data["tau_follower"], dtype=np.float64)
        expected_shape = (timeline.size, 7)
        if any(value.shape != expected_shape for value in (q, dq, ddq, tau)):
            return
        estimator = PinocchioJointTorqueResidualEstimator(
            self.config.realtime_plot.inverse_dynamics
        )
        tau_id_filter = _make_state_lowpass_filter(self.config.robot_states, "tau_id")
        residuals: list[np.ndarray] = []
        for timestamp_us, q_value, dq_value, ddq_value, tau_value in zip(
            timeline, q, dq, ddq, tau
        ):
            estimate = estimator.estimate(q_value, dq_value, ddq_value, tau_value)
            tau_id = np.asarray(estimate.tau_id, dtype=np.float64)
            if tau_id_filter is not None:
                tau_id = tau_id_filter.apply(tau_id, int(timestamp_us))
            residuals.append(tau_id - tau_value)
        if not residuals:
            return
        data["tau_f"] = np.stack(residuals, axis=0)
        data["tau_f_timestamp_us"] = timeline.copy()
        state_names["tau_f"] = "torque"
        state_names["tau_f_timestamp_us"] = "timestamp"
        tau_id_config = self.config.robot_states.get("tau_id", StateParamConfig())
        torque_config = self.config.robot_states.get("torque", StateParamConfig())
        q_state = self.config.robot_states["q"]
        dq_state = self.config.robot_states["velocity"]
        ddq_state = self.config.robot_states["acceleration"]
        tau_id_lowpass = bool(tau_id_config.lowpass)
        tau_lowpass = bool(self.config.dynamics_processing.enabled or torque_config.lowpass)
        attrs["tau_f"].update(
            {
                "definition": "tau_id - tau_follower",
                "processing_method": "per_joint_group_state_rnea",
                "q_source_dataset": "teleop/q_follower",
                "tau_source_dataset": "teleop/tau_follower",
                "timestamp_path": "teleop/tau_f_timestamp_us",
                "dt_source": "native CAN joint-group timestamps",
                "first_valid_sample_index": 0,
                "state_derivative_method": "official_velocity_with_fused_acceleration",
                "q_mean_window_samples": 1,
                "q_mean_window_enabled": False,
                "q_lowpass_cutoff_hz": q_state.lowpass_cutoff_hz,
                "dq_lowpass_cutoff_hz": dq_state.lowpass_cutoff_hz,
                "ddq_lowpass_cutoff_hz": ddq_state.lowpass_cutoff_hz,
                "lowpass": tau_id_lowpass or tau_lowpass,
                "dq_lowpass": dq_state.lowpass,
                "ddq_lowpass": ddq_state.lowpass,
                "tau_id_lowpass": tau_id_lowpass,
                "tau_lowpass": tau_lowpass,
                "causal": True,
                "model_urdf": str(self.config.realtime_plot.inverse_dynamics.urdf_path),
                "model_manifest": str(
                    self.config.realtime_plot.inverse_dynamics.manifest_path or ""
                ),
            }
        )
        if tau_id_lowpass:
            attrs["tau_f"].update(
                {
                    "tau_id_filter_method": "causal_median_then_one_pole_iir",
                    "tau_id_lowpass_cutoff_hz": tau_id_config.lowpass_cutoff_hz,
                    "tau_id_median_window": tau_id_config.median_window,
                }
            )


def _make_state_lowpass_filter(
    robot_states: dict[str, StateParamConfig],
    state_name: str,
) -> OnePoleLowPass | None:
    param = robot_states.get(state_name)
    if param is None or not param.lowpass:
        return None
    return OnePoleLowPass(param.lowpass_cutoff_hz, param.median_window)


def _stack(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float64)
    return np.stack(values, axis=0)

def _compression_for(data: np.ndarray) -> str | None:
    if data.dtype == np.uint8 or data.size > 2048:
        return "gzip"
    return None
