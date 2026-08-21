from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from typing import Any

import numpy as np

from nero_collection.arms.base import ArmState, GripperState
from nero_collection.config import ArmEndpointConfig


log = logging.getLogger(__name__)

_DOF = 7
_RPC_TIMEOUT_S = 15.0


class IsolatedArmProcess:
    """ArmInterface proxy whose SDK and CAN access live in a child process."""

    def __init__(
        self,
        config: ArmEndpointConfig,
        backend: str = "pyagxarm",
    ) -> None:
        self.config = config
        self.backend = str(backend)
        self.name = config.name
        self.dof = _DOF
        self._context = mp.get_context("spawn")
        self._requests = self._context.Queue(maxsize=256)
        self._responses = self._context.Queue(maxsize=256)
        self._process = None
        self._rpc_lock = threading.Lock()
        self._request_id = 0

    def connect(self) -> None:
        if self._process is not None:
            return
        process = self._context.Process(
            target=_arm_process_worker,
            args=(
                self.config,
                self.backend,
                self._requests,
                self._responses,
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
            "isolated hardware process ready arm=%s pid=%s",
            self.name,
            process.pid,
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

    def read_state(self) -> ArmState:
        return _copy_arm_state(self._rpc("read_state"))

    def peek_latest_state(self) -> ArmState:
        return self.read_state()

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

    def reset_gripper(self) -> bool:
        return bool(self._rpc("reset_gripper"))

    def read_gripper_state(self) -> GripperState:
        return self._rpc("read_gripper_state")

    def read_leader_gripper_state(self) -> GripperState:
        return self._rpc("read_leader_gripper_state")

    def disable_gripper(self) -> None:
        self._rpc("disable_gripper")

    def command_gripper(self, value: float, force_n: float, mode: str = "width") -> None:
        self._rpc("command_gripper", float(value), float(force_n), str(mode))

    def _state_reset_rpc(self, method: str, *args: Any) -> Any:
        return self._rpc(method, *args)

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
            deadline = time.monotonic() + float(timeout_s)
            while True:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    raise RuntimeError(
                        f"hardware command timed out arm={self.name} method={method} "
                        f"timeout={timeout_s:.1f}s"
                    )
                try:
                    response = self._responses.get(timeout=remaining_s)
                except queue.Empty as exc:
                    raise RuntimeError(
                        f"hardware command timed out arm={self.name} method={method} "
                        f"timeout={timeout_s:.1f}s"
                    ) from exc
                response_id = int(response[0])
                if response_id < request_id:
                    log.warning(
                        "discarding stale hardware response arm=%s method=%s "
                        "expected=%d got=%d",
                        self.name,
                        method,
                        request_id,
                        response_id,
                    )
                    continue
                if response_id > request_id:
                    raise RuntimeError(
                        f"hardware response order mismatch arm={self.name}: "
                        f"expected={request_id} got={response_id}"
                    )
                break
            if not response[1]:
                raise RuntimeError(
                    f"hardware command failed arm={self.name} method={method}: "
                    f"{response[3]}\n{response[4]}"
                )
            return response[2]

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
    requests: Any,
    responses: Any,
) -> None:
    _configure_hardware_process_environment()
    arm = None
    try:
        arm = _build_worker_arm(config, backend)
        arm.connect()
        responses.put((0, True, None, None, None))
        while True:
            request_id, method, args = requests.get()
            try:
                if method == "__shutdown__":
                    arm.disconnect()
                    responses.put((request_id, True, None, None, None))
                    return
                result = getattr(arm, method)(*args)
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
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass


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
