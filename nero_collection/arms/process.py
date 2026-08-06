from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np

from nero_collection.arms.base import ArmState, GripperState
from nero_collection.config import ArmEndpointConfig
from nero_collection.time_utils import now_us


log = logging.getLogger(__name__)

_DOF = 7
_STATE_FLOAT_WIDTH = 51
_STATE_INT_WIDTH = 39
_STATE_PUBLISH_POLL_S = 0.001
_DEFAULT_HISTORY_SIZE = 4096
_RPC_TIMEOUT_S = 15.0


@dataclass(frozen=True)
class StateDrainResult:
    states: tuple[ArmState, ...]
    dropped: int


class SharedArmStateRing:
    """Single-writer, multi-process ring for complete seven-axis arm states."""

    def __init__(self, context: Any, capacity: int = _DEFAULT_HISTORY_SIZE) -> None:
        if capacity < 2:
            raise ValueError("arm state history capacity must be at least two")
        self.capacity = int(capacity)
        self._floats = context.RawArray("d", self.capacity * _STATE_FLOAT_WIDTH)
        self._ints = context.RawArray("q", self.capacity * _STATE_INT_WIDTH)
        self._sequence = context.Value("Q", 0, lock=True)

    def reset(self) -> None:
        with self._sequence.get_lock():
            self._sequence.value = 0

    def append(self, state: ArmState) -> int:
        float_row, int_row = _encode_arm_state(state)
        with self._sequence.get_lock():
            sequence = int(self._sequence.value) + 1
            index = (sequence - 1) % self.capacity
            floats = np.frombuffer(self._floats, dtype=np.float64).reshape(
                self.capacity, _STATE_FLOAT_WIDTH
            )
            ints = np.frombuffer(self._ints, dtype=np.int64).reshape(
                self.capacity, _STATE_INT_WIDTH
            )
            floats[index] = float_row
            ints[index] = int_row
            self._sequence.value = sequence
        return sequence

    def read_after(self, sequence: int) -> tuple[tuple[ArmState, ...], int, int]:
        with self._sequence.get_lock():
            current = int(self._sequence.value)
            if current <= 0:
                return (), 0, 0
            requested = int(sequence)
            if requested > current:
                requested = 0
            oldest = max(1, current - self.capacity + 1)
            start = max(requested + 1, oldest)
            dropped = max(0, oldest - (requested + 1))
            if start > current:
                return (), current, dropped
            indices = [
                (value - 1) % self.capacity for value in range(start, current + 1)
            ]
            floats = np.frombuffer(self._floats, dtype=np.float64).reshape(
                self.capacity, _STATE_FLOAT_WIDTH
            )[indices].copy()
            ints = np.frombuffer(self._ints, dtype=np.int64).reshape(
                self.capacity, _STATE_INT_WIDTH
            )[indices].copy()
        states = tuple(
            _decode_arm_state(float_row, int_row)
            for float_row, int_row in zip(floats, ints)
        )
        return states, current, dropped

    def latest(self) -> tuple[ArmState | None, int]:
        with self._sequence.get_lock():
            current = int(self._sequence.value)
            if current <= 0:
                return None, 0
            index = (current - 1) % self.capacity
            float_row = np.frombuffer(self._floats, dtype=np.float64).reshape(
                self.capacity, _STATE_FLOAT_WIDTH
            )[index].copy()
            int_row = np.frombuffer(self._ints, dtype=np.int64).reshape(
                self.capacity, _STATE_INT_WIDTH
            )[index].copy()
        return _decode_arm_state(float_row, int_row), current


class IsolatedArmProcess:
    """ArmInterface proxy whose SDK and CAN access live in a child process."""

    def __init__(
        self,
        config: ArmEndpointConfig,
        backend: str = "pyagxarm",
        *,
        history_size: int = _DEFAULT_HISTORY_SIZE,
    ) -> None:
        self.config = config
        self.backend = str(backend)
        self.name = config.name
        self.dof = _DOF
        self.history_size = int(history_size)
        self._context = mp.get_context("spawn")
        self._state_ring = SharedArmStateRing(self._context, self.history_size)
        self._requests = self._context.Queue(maxsize=256)
        self._responses = self._context.Queue(maxsize=256)
        self._faults = self._context.Queue(maxsize=8)
        self._process = None
        self._rpc_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._request_id = 0
        self._state_sequence = 0
        self._latest_state: ArmState | None = None
        self._history_drop_count = 0
        self._remote_fault: tuple[str, str] | None = None

    @property
    def history_drop_count(self) -> int:
        with self._state_lock:
            return self._history_drop_count

    def connect(self) -> None:
        if self._process is not None:
            return
        self._state_ring.reset()
        self._state_sequence = 0
        self._latest_state = None
        self._remote_fault = None
        process = self._context.Process(
            target=_arm_process_worker,
            args=(
                self.config,
                self.backend,
                self._state_ring,
                self._requests,
                self._responses,
                self._faults,
            ),
            name=f"nero-hardware-{self.name}",
            daemon=True,
        )
        process.start()
        self._process = process
        try:
            response = self._responses.get(timeout=_RPC_TIMEOUT_S)
        except queue.Empty as exc:
            self._terminate_worker()
            raise RuntimeError(
                f"hardware process for {self.name} did not start within {_RPC_TIMEOUT_S:.1f}s"
            ) from exc
        if response[0] != 0 or not response[1]:
            self._terminate_worker()
            raise RuntimeError(
                f"hardware process startup failed for {self.name}: {response[3]}\n{response[4]}"
            )
        log.info(
            "isolated hardware process ready arm=%s pid=%s history=%d",
            self.name,
            process.pid,
            self.history_size,
        )

    def disconnect(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            self._rpc("__shutdown__", timeout_s=5.0)
        except Exception as exc:
            log.warning("hardware process shutdown request failed arm=%s: %s", self.name, exc)
        process.join(timeout=5.0)
        if process.is_alive():
            log.error("hardware process did not stop arm=%s; terminating child", self.name)
            process.terminate()
            process.join(timeout=2.0)
        self._process = None
        self._remote_fault = None
        self._clear_ipc_queue(self._faults)
        self._clear_ipc_queue(self._responses)

    def enable(self) -> None:
        self._rpc("enable")

    def disable(self) -> None:
        self._rpc("disable")

    def set_leader_mode(self) -> None:
        self._state_reset_rpc("set_leader_mode")

    def set_follower_mode(self) -> None:
        self._state_reset_rpc("set_follower_mode")

    def set_normal_mode(self) -> None:
        self._state_reset_rpc("set_normal_mode")

    def read_control_role(self, refresh: bool = False) -> str | None:
        return self._rpc("read_control_role", bool(refresh))

    def configure_state_alignment(
        self,
        delay_s: float,
        output_rate_hz: float,
        q_mean_window_samples: int,
        q_lowpass_cutoff_hz: float | None,
        dq_lowpass_cutoff_hz: float | None,
        ddq_lowpass_cutoff_hz: float | None,
        maximum_input_gap_s: float = 0.03,
    ) -> None:
        self._state_reset_rpc(
            "configure_state_alignment",
            delay_s,
            output_rate_hz,
            q_mean_window_samples,
            q_lowpass_cutoff_hz,
            dq_lowpass_cutoff_hz,
            ddq_lowpass_cutoff_hz,
            maximum_input_gap_s,
        )

    def read_state(self) -> ArmState:
        result = self.drain_states()
        if result.states:
            return _copy_arm_state(result.states[-1])
        with self._state_lock:
            if self._latest_state is not None:
                return _copy_arm_state(self._latest_state)
        self._raise_remote_fault()
        return _empty_arm_state()

    def drain_states(self) -> StateDrainResult:
        self._ensure_worker_alive()
        self._raise_remote_fault()
        with self._state_lock:
            states, sequence, dropped = self._state_ring.read_after(self._state_sequence)
            self._state_sequence = sequence
            self._history_drop_count += dropped
            if states:
                self._latest_state = states[-1]
            return StateDrainResult(states=states, dropped=dropped)

    def peek_latest_state(self) -> ArmState:
        """Inspect producer freshness without advancing any consumer sequence."""
        self._ensure_worker_alive()
        self._raise_remote_fault()
        state, _ = self._state_ring.latest()
        return _copy_arm_state(state) if state is not None else _empty_arm_state()

    def read_leader_joint_positions(self) -> np.ndarray:
        return self.read_state().q.copy()

    def command_joint_positions(self, q: np.ndarray) -> None:
        self._rpc("command_joint_positions", np.asarray(q, dtype=np.float64))

    def command_joint_impedance(
        self,
        q: np.ndarray,
        v_des: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        t_ff: np.ndarray,
    ) -> None:
        self._rpc(
            "command_joint_impedance",
            np.asarray(q, dtype=np.float64),
            np.asarray(v_des, dtype=np.float64),
            np.asarray(kp, dtype=np.float64),
            np.asarray(kd, dtype=np.float64),
            np.asarray(t_ff, dtype=np.float64),
        )

    def validate_joint_impedance_support(self) -> None:
        self._rpc("validate_joint_impedance_support")

    def configure_joint_impedance_mode(self) -> None:
        self._state_reset_rpc("configure_joint_impedance_mode")

    def move_joints(self, q: np.ndarray) -> None:
        self._rpc("move_joints", np.asarray(q, dtype=np.float64))

    def wait_motion_done(self, timeout_s: float, poll_interval_s: float = 0.1) -> bool:
        return bool(
            self._rpc(
                "wait_motion_done",
                float(timeout_s),
                float(poll_interval_s),
                timeout_s=max(float(timeout_s) + 2.0, _RPC_TIMEOUT_S),
            )
        )

    def init_gripper(self, effector: str = "AGX_GRIPPER") -> None:
        self._rpc("init_gripper", str(effector))

    def read_gripper_state(self) -> GripperState:
        return self._rpc("read_gripper_state")

    def read_leader_gripper_state(self) -> GripperState:
        return self._rpc("read_leader_gripper_state")

    def disable_gripper(self) -> None:
        self._rpc("disable_gripper")

    def command_gripper(self, value: float, force_n: float, mode: str = "width") -> None:
        self._rpc("command_gripper", float(value), float(force_n), str(mode))

    def _state_reset_rpc(self, method: str, *args: Any) -> Any:
        result = self._rpc(method, *args)
        with self._state_lock:
            self._state_sequence = 0
            self._latest_state = None
            self._remote_fault = None
        return result

    def _rpc(self, method: str, *args: Any, timeout_s: float = _RPC_TIMEOUT_S) -> Any:
        self._ensure_worker_alive()
        with self._rpc_lock:
            self._request_id += 1
            request_id = self._request_id
            try:
                self._requests.put((request_id, method, args), timeout=1.0)
            except queue.Full as exc:
                raise RuntimeError(
                    f"hardware command queue is full arm={self.name} method={method}"
                ) from exc
            try:
                response = self._responses.get(timeout=float(timeout_s))
            except queue.Empty as exc:
                raise RuntimeError(
                    f"hardware command timed out arm={self.name} method={method} "
                    f"timeout={timeout_s:.1f}s"
                ) from exc
            if response[0] != request_id:
                raise RuntimeError(
                    f"hardware response order mismatch arm={self.name}: "
                    f"expected={request_id} got={response[0]}"
                )
            if not response[1]:
                raise RuntimeError(
                    f"hardware command failed arm={self.name} method={method}: "
                    f"{response[3]}\n{response[4]}"
                )
            return response[2]

    def _raise_remote_fault(self) -> None:
        latest = None
        while True:
            try:
                latest = self._faults.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            with self._state_lock:
                self._remote_fault = latest
        with self._state_lock:
            remote_fault = self._remote_fault
        if remote_fault is not None:
            raise RuntimeError(
                f"hardware state publisher failed arm={self.name}: "
                f"{remote_fault[0]}\n{remote_fault[1]}"
            )

    def _ensure_worker_alive(self) -> None:
        process = self._process
        if process is None:
            raise RuntimeError(f"hardware process for {self.name} is not connected")
        if not process.is_alive():
            raise RuntimeError(
                f"hardware process exited arm={self.name} exitcode={process.exitcode}"
            )

    def _terminate_worker(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        self._process = None

    @staticmethod
    def _clear_ipc_queue(value: Any) -> None:
        while True:
            try:
                value.get_nowait()
            except queue.Empty:
                return


def _arm_process_worker(
    config: ArmEndpointConfig,
    backend: str,
    state_ring: SharedArmStateRing,
    requests: Any,
    responses: Any,
    faults: Any,
) -> None:
    _configure_hardware_process_environment()
    arm = None
    publisher_thread = None
    publisher_stop = threading.Event()

    def stop_publisher() -> None:
        nonlocal publisher_thread
        publisher_stop.set()
        if publisher_thread is not None and publisher_thread.is_alive():
            publisher_thread.join(timeout=1.0)
            if publisher_thread.is_alive():
                raise RuntimeError(
                    f"state publisher for arm {config.name} did not stop"
                )
        publisher_thread = None

    def start_publisher() -> None:
        nonlocal publisher_thread
        if publisher_thread is not None and publisher_thread.is_alive():
            return
        publisher_stop.clear()
        publisher_thread = threading.Thread(
            target=_state_publisher_loop,
            args=(arm, state_ring, publisher_stop, faults),
            name=f"nero-state-publisher-{config.name}",
            daemon=True,
        )
        publisher_thread.start()

    try:
        arm = _build_worker_arm(config, backend)
        arm.connect()
        start_publisher()
        responses.put((0, True, None, None, None))
        while True:
            request_id, method, args = requests.get()
            try:
                if method == "__shutdown__":
                    stop_publisher()
                    arm.disconnect()
                    responses.put((request_id, True, None, None, None))
                    return
                if method in {
                    "set_leader_mode",
                    "set_follower_mode",
                    "set_normal_mode",
                    "configure_state_alignment",
                    "configure_joint_impedance_mode",
                }:
                    stop_publisher()
                    state_ring.reset()
                    _clear_queue(faults)
                result = getattr(arm, method)(*args)
                if method in {
                    "set_leader_mode",
                    "set_follower_mode",
                    "set_normal_mode",
                    "configure_state_alignment",
                    "configure_joint_impedance_mode",
                }:
                    start_publisher()
                responses.put((request_id, True, result, None, None))
            except BaseException as exc:
                responses.put(
                    (request_id, False, None, str(exc), traceback.format_exc())
                )
    except BaseException as exc:
        try:
            responses.put((0, False, None, str(exc), traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            stop_publisher()
        except Exception:
            pass
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass


def _state_publisher_loop(
    arm: Any,
    state_ring: SharedArmStateRing,
    stop_event: threading.Event,
    faults: Any,
) -> None:
    last_timestamp_us = 0
    try:
        while not stop_event.is_set():
            state = arm.read_state()
            timestamp_us = int(state.q_timestamp_us or state.timestamp_us)
            if timestamp_us > last_timestamp_us and _publishable_arm_state(state):
                state_ring.append(state)
                last_timestamp_us = timestamp_us
            stop_event.wait(_STATE_PUBLISH_POLL_S)
    except BaseException as exc:
        _put_latest(faults, (str(exc), traceback.format_exc()))
        stop_event.set()


def _build_worker_arm(config: ArmEndpointConfig, backend: str) -> Any:
    normalized = str(backend).lower().replace("-", "_")
    if normalized in {"mock", "sim", "simulation"}:
        from nero_collection.arms.mock import MockArm

        return MockArm(config)
    if normalized in {"pyagxarm", "py_agx_arm", "agx"}:
        from nero_collection.arms.pyagx import PyAgxArmAdapter

        return PyAgxArmAdapter(config)
    raise ValueError(f"Unsupported isolated arm backend {backend!r}")


def _configure_hardware_process_environment() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def _publishable_arm_state(state: ArmState) -> bool:
    vectors = (state.q, state.dq, state.ddq, state.torque, state.current)
    if not all(
        np.asarray(value, dtype=np.float64).shape == (_DOF,) for value in vectors
    ):
        return False
    required = (state.q, state.dq, state.torque, state.current)
    return all(
        np.all(np.isfinite(np.asarray(value, dtype=np.float64)))
        for value in required
    )


def _encode_arm_state(state: ArmState) -> tuple[np.ndarray, np.ndarray]:
    q = _vector(state.q, np.float64)
    dq = _vector(state.dq, np.float64)
    ddq = _vector(state.ddq, np.float64)
    torque = _vector(state.torque, np.float64)
    current = _vector(state.current, np.float64)
    ee_pose = np.asarray(state.ee_pose, dtype=np.float64)
    if ee_pose.shape != (4, 4) or not np.all(np.isfinite(ee_pose)):
        raise RuntimeError(f"arm state ee_pose must be finite 4x4; got {ee_pose}")
    floats = np.concatenate((q, dq, ddq, ee_pose.reshape(-1), torque, current))
    ints = np.concatenate(
        (
            np.asarray(
                [
                    state.timestamp_us,
                    state.acquired_timestamp_us,
                    state.q_timestamp_us,
                    state.q_acquired_timestamp_us,
                ],
                dtype=np.int64,
            ),
            _timestamp_vector(state.q_component_timestamp_us),
            _timestamp_vector(state.q_source_before_timestamp_us),
            _timestamp_vector(state.q_source_after_timestamp_us),
            _timestamp_vector(state.motor_timestamp_us),
            _timestamp_vector(state.motor_acquired_timestamp_us),
        )
    )
    return floats, ints


def _decode_arm_state(floats: np.ndarray, ints: np.ndarray) -> ArmState:
    return ArmState(
        q=floats[0:7].copy(),
        dq=floats[7:14].copy(),
        ddq=floats[14:21].copy(),
        ee_pose=floats[21:37].reshape(4, 4).copy(),
        torque=floats[37:44].copy(),
        current=floats[44:51].copy(),
        timestamp_us=int(ints[0]),
        acquired_timestamp_us=int(ints[1]),
        q_timestamp_us=int(ints[2]),
        q_acquired_timestamp_us=int(ints[3]),
        q_component_timestamp_us=ints[4:11].copy(),
        q_source_before_timestamp_us=ints[11:18].copy(),
        q_source_after_timestamp_us=ints[18:25].copy(),
        motor_timestamp_us=ints[25:32].copy(),
        motor_acquired_timestamp_us=ints[32:39].copy(),
    )


def _copy_arm_state(state: ArmState) -> ArmState:
    floats, ints = _encode_arm_state(state)
    return _decode_arm_state(floats, ints)


def _empty_arm_state() -> ArmState:
    timestamp_us = now_us()
    nan = np.full(_DOF, np.nan, dtype=np.float64)
    zeros = np.zeros(_DOF, dtype=np.int64)
    return ArmState(
        q=nan.copy(),
        dq=nan.copy(),
        ddq=nan.copy(),
        ee_pose=np.eye(4, dtype=np.float64),
        torque=nan.copy(),
        current=nan.copy(),
        timestamp_us=timestamp_us,
        acquired_timestamp_us=timestamp_us,
        q_timestamp_us=0,
        q_acquired_timestamp_us=timestamp_us,
        q_component_timestamp_us=zeros.copy(),
        q_source_before_timestamp_us=zeros.copy(),
        q_source_after_timestamp_us=zeros.copy(),
        motor_timestamp_us=zeros.copy(),
        motor_acquired_timestamp_us=zeros.copy(),
    )


def _vector(value: Any, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).reshape(-1)
    if result.shape != (_DOF,):
        raise RuntimeError(f"arm state vector must have shape ({_DOF},); got {result}")
    return result.copy()


def _timestamp_vector(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64).reshape(-1)
    if result.size == 0:
        return np.zeros(_DOF, dtype=np.int64)
    if result.shape != (_DOF,):
        raise RuntimeError(f"arm timestamp vector must have shape ({_DOF},); got {result}")
    return result.copy()


def _put_latest(target: Any, value: Any) -> None:
    try:
        target.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        target.get_nowait()
    except queue.Empty:
        pass
    try:
        target.put_nowait(value)
    except queue.Full:
        pass


def _clear_queue(target: Any) -> None:
    while True:
        try:
            target.get_nowait()
        except queue.Empty:
            return
