"""Shared robot-side primitives for the ICRA2027 Nero TA-VLA runners."""

from __future__ import annotations

import threading
import time
from typing import Any, Iterable

import numpy as np

from .tavla_client import JOINT_DIM


class CameraStream:
    """Continuously consume one OpenCV/V4L2 camera and expose fresh BGR frames."""

    def __init__(
        self,
        index: int,
        name: str,
        *,
        width: int = 640,
        height: int = 480,
        fps: float = 5.0,
        maximum_age_s: float = 0.5,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for the Nero TA-VLA camera streams"
            ) from exc

        if maximum_age_s <= 0.0:
            raise ValueError("maximum_age_s must be positive")
        self.name = str(name)
        self.maximum_age_s = float(maximum_age_s)
        self._cv2 = cv2
        self._capture = cv2.VideoCapture(int(index), cv2.CAP_V4L2)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"cannot open {self.name} camera {index}")

        # Two default 30 Hz YUYV streams can saturate the USB link.  Five FPS
        # matches the deployed client and is sufficient for remote inference.
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self._capture.set(cv2.CAP_PROP_FPS, float(fps))
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_time = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._read_loop,
            name=f"{self.name}-camera",
            daemon=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if ok:
                with self._lock:
                    self._frame = np.asarray(frame)
                    self._frame_time = time.monotonic()
            else:
                time.sleep(0.02)

    def latest(self, timeout_s: float = 0.0) -> np.ndarray:
        """Return a copied fresh frame, optionally waiting for initial capture."""
        if timeout_s < 0.0:
            raise ValueError("timeout_s must be non-negative")
        deadline = time.monotonic() + float(timeout_s)
        while True:
            with self._lock:
                frame = self._frame
                frame_time = self._frame_time
            age_s = time.monotonic() - frame_time
            if frame is not None and age_s <= self.maximum_age_s:
                return frame.copy()
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{self.name} camera has no fresh frame")
            time.sleep(0.02)

    def close(self) -> None:
        self._stop.set()
        self._capture.release()
        self._thread.join(timeout=1.0)


def create_nero_robot(can_channel: str, firmware: str = "V112") -> Any:
    """Create the raw pyAgxArm Nero object used by the deployment bundle."""
    try:
        from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
    except ImportError as exc:
        try:
            from PyAgxArm import (  # type: ignore[no-redef]
                AgxArmFactory,
                ArmModel,
                NeroFW,
                create_agx_arm_config,
            )
        except ImportError:
            raise RuntimeError(
                "pyAgxArm is required for physical Nero TA-VLA inference"
            ) from exc

    try:
        firmware_value = getattr(NeroFW, str(firmware).upper())
    except AttributeError as exc:
        available = sorted(name for name in dir(NeroFW) if name.startswith("V"))
        raise ValueError(
            f"unknown Nero firmware profile {firmware!r}; available={available}"
        ) from exc

    # Support both the V112 deployment SDK and the newer SDK signature used by
    # this workspace's pyAgxArm adapter.
    common = {
        "robot": ArmModel.NERO,
        "firmeware_version": firmware_value,
        "interface": "socketcan",
        "channel": str(can_channel),
    }
    try:
        config = create_agx_arm_config(**common)
    except TypeError:
        config = create_agx_arm_config(comm="can", bitrate=1_000_000, **common)
    return AgxArmFactory.create_arm(config)


def _message_vector(message: Any, name: str) -> np.ndarray:
    if message is None:
        raise RuntimeError(f"no Nero {name} feedback")
    value = getattr(message, "msg", message)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (JOINT_DIM,) or not np.all(np.isfinite(array)):
        raise RuntimeError(
            f"invalid Nero {name} feedback: expected finite ({JOINT_DIM},), got {array}"
        )
    return array


def read_joint_state(robot: Any) -> np.ndarray:
    return _message_vector(robot.get_joint_angles(), "joint-angle")


def read_joint_effort(robot: Any) -> np.ndarray:
    values: list[float] = []
    for joint_index in range(1, JOINT_DIM + 1):
        message = robot.get_motor_states(joint_index)
        if message is None:
            raise RuntimeError(
                f"incomplete Nero torque feedback at joint {joint_index}"
            )
        state = getattr(message, "msg", message)
        values.append(float(getattr(state, "torque")))
    effort = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(effort)):
        raise RuntimeError(f"Nero torque feedback contains NaN or Inf: {effort}")
    return effort


def status_and_enable(robot: Any) -> tuple[Any, list[bool]]:
    status = robot.get_arm_status()
    enabled = list(robot.get_joints_enable_status_list())
    if status is None:
        raise RuntimeError("no Nero arm-status feedback")
    return status, enabled


def require_normal(robot: Any, timeout_s: float = 2.0) -> Any:
    """Require fresh NORMAL feedback and all seven enabled joints."""
    deadline = time.monotonic() + float(timeout_s)
    last_status: Any = None
    last_enabled: list[bool] = []
    while time.monotonic() < deadline:
        last_status, last_enabled = status_and_enable(robot)
        if len(last_enabled) == JOINT_DIM:
            status_value = getattr(last_status.msg, "arm_status", None)
            if status_value != 0:
                raise RuntimeError(f"Nero is not NORMAL: {status_value}")
            if not all(last_enabled):
                raise RuntimeError(f"Nero is not fully enabled: {last_enabled}")
            return last_status
        time.sleep(0.05)
    raise RuntimeError(
        "Nero enable feedback did not arrive: "
        f"status={last_status!r}, enabled={last_enabled}"
    )


def clip_joint_target(
    target: np.ndarray | Iterable[float],
    current: np.ndarray | Iterable[float],
    maximum_step_rad: float,
) -> np.ndarray:
    """Clamp all seven Nero joint targets against the latest measured state."""
    maximum_step_rad = float(maximum_step_rad)
    if not np.isfinite(maximum_step_rad) or maximum_step_rad <= 0.0:
        raise ValueError("maximum_step_rad must be positive and finite")
    target_array = _finite_vector(target, "target")
    current_array = _finite_vector(current, "current")
    return np.clip(
        target_array,
        current_array - maximum_step_rad,
        current_array + maximum_step_rad,
    ).astype(np.float32)


def send_move_j(robot: Any, target: np.ndarray | Iterable[float]) -> None:
    command = _finite_vector(target, "target")
    result = robot.move_j(command.tolist())
    if result is False:
        raise RuntimeError("Nero move_j returned False")


def _finite_vector(value: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (JOINT_DIM,) or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be a finite ({JOINT_DIM},) vector, got {array}"
        )
    return array.copy()


__all__ = [
    "CameraStream",
    "clip_joint_target",
    "create_nero_robot",
    "read_joint_effort",
    "read_joint_state",
    "require_normal",
    "send_move_j",
    "status_and_enable",
]
