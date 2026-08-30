"""Robot-side WebSocket client for the ICRA2027 TA-VLA policies.

This module deliberately does not send commands to a robot.  It formats the
dual-camera, joint-state, and torque-history observation expected by the trained
policy and returns a 50-step, 7-D joint-action chunk.  Integrate the returned
chunk with the robot controller's own limits, watchdog, and emergency stop.
"""

from __future__ import annotations

import argparse
from collections import deque
import logging
from pathlib import Path
import threading
import time
from typing import Final, Iterable

import numpy as np

try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy
except ImportError:  # Robot/client dependency; keep format helpers importable for tests.
    image_tools = None
    websocket_client_policy = None


EFFORT_OFFSETS: Final[tuple[int, ...]] = (-50, -44, -39, -33, -28, -22, -17, -11, -6, 0)
HISTORY_FRAMES: Final[int] = 51
HISTORY_SECONDS: Final[float] = 2.0
EFFORT_SAMPLE_TOLERANCE_SECONDS: Final[float] = 0.06
JOINT_DIM: Final[int] = 7
ACTION_HORIZON: Final[int] = 50
TASK_PROMPTS: Final[dict[str, str]] = {
    "usb": "insert the USB plug into the port",
    "button": "press the button",
    "cucumber": "peel the cucumber",
}


def _vector7(value: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (JOINT_DIM,):
        raise ValueError(f"{name} must have shape ({JOINT_DIM},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _rgb_uint8(image: np.ndarray, *, bgr: bool) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"image must be HWC or CHW, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"image must have three color channels, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.all(np.isfinite(image)):
            raise ValueError("image contains NaN or Inf")
        # Camera APIs commonly expose either [0, 1] floats or [0, 255] floats.
        if image.size and float(np.max(image)) <= 1.0:
            image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    if bgr:
        image = image[..., ::-1]
    return np.ascontiguousarray(image)


class EffortHistoryBuffer:
    """Keeps timestamped torques and selects the training-time 25 Hz offsets."""

    def __init__(self) -> None:
        self._samples: deque[tuple[float, np.ndarray]] = deque(maxlen=2 * HISTORY_FRAMES)
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()

    def append(self, effort: np.ndarray | Iterable[float], *, timestamp: float | None = None) -> None:
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("effort timestamp must be finite")
        # Robot SDKs often reuse a mutable read buffer, so every history item
        # must own its memory.
        owned_effort = _vector7(effort, "effort").copy()
        with self._lock:
            if self._samples and timestamp <= self._samples[-1][0]:
                raise ValueError("effort timestamps must be strictly increasing")
            self._samples.append((timestamp, owned_effort))

    @property
    def ready(self) -> bool:
        with self._lock:
            if len(self._samples) < len(EFFORT_OFFSETS):
                return False
            span = self._samples[-1][0] - self._samples[0][0]
            return span >= HISTORY_SECONDS - EFFORT_SAMPLE_TOLERANCE_SECONDS

    @property
    def num_frames(self) -> int:
        with self._lock:
            return len(self._samples)

    def sampled(self) -> np.ndarray:
        with self._lock:
            samples = list(self._samples)
        if len(samples) < len(EFFORT_OFFSETS):
            raise RuntimeError("torque history is not ready: collect approximately 2 seconds at 25 Hz")

        timestamps = np.asarray([sample[0] for sample in samples], dtype=np.float64)
        efforts = [sample[1] for sample in samples]
        targets = timestamps[-1] + np.asarray(EFFORT_OFFSETS, dtype=np.float64) / 25.0
        indices = np.asarray([int(np.argmin(np.abs(timestamps - target))) for target in targets])
        errors = np.abs(timestamps[indices] - targets)
        if float(np.max(errors)) > EFFORT_SAMPLE_TOLERANCE_SECONDS:
            raise RuntimeError(
                "torque history has a gap or is shorter than 2 seconds; "
                f"largest sampling error is {float(np.max(errors)) * 1000.0:.1f} ms"
            )
        return np.stack([efforts[index] for index in indices]).astype(np.float32)


class TavlaObservationBuilder:
    def __init__(self, task: str, *, camera_color: str = "rgb", resize: int = 224) -> None:
        if task not in TASK_PROMPTS:
            raise ValueError(f"task must be one of {tuple(TASK_PROMPTS)}, got {task!r}")
        if camera_color not in ("rgb", "bgr"):
            raise ValueError("camera_color must be 'rgb' or 'bgr'")
        self.task = task
        self.prompt = TASK_PROMPTS[task]
        self.bgr = camera_color == "bgr"
        self.resize = resize

    def build(
        self,
        *,
        side_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray | Iterable[float],
        effort_history: np.ndarray,
        prompt: str | None = None,
    ) -> dict:
        side = _rgb_uint8(side_image, bgr=self.bgr)
        wrist = _rgb_uint8(wrist_image, bgr=self.bgr)
        if self.resize > 0:
            if image_tools is None:
                raise RuntimeError(
                    "openpi-client is required to resize TA-VLA images; "
                    "install the client bundle or construct with resize=0"
                )
            side = image_tools.resize_with_pad(side, self.resize, self.resize)
            wrist = image_tools.resize_with_pad(wrist, self.resize, self.resize)

        effort_history = np.asarray(effort_history, dtype=np.float32)
        expected_effort_shape = (len(EFFORT_OFFSETS), JOINT_DIM)
        if effort_history.shape != expected_effort_shape:
            raise ValueError(
                f"effort_history must have shape {expected_effort_shape}, got {effort_history.shape}"
            )
        if not np.all(np.isfinite(effort_history)):
            raise ValueError("effort_history contains NaN or Inf")

        return {
            "images": {
                # These names match Icra2027TavlaDataConfig in training/config.py.
                "cam_high": side,
                "cam_left_wrist": wrist,
            },
            "state": _vector7(state, "state"),
            "effort": effort_history,
            "prompt": prompt or self.prompt,
        }


class TavlaRemotePolicy:
    """Convenience wrapper used inside the robot's 25 Hz control loop."""

    def __init__(
        self,
        host: str,
        *,
        port: int = 8000,
        task: str = "usb",
        camera_color: str = "rgb",
        resize: int = 224,
    ) -> None:
        if websocket_client_policy is None:
            raise RuntimeError(
                "openpi-client is required for remote TA-VLA inference; "
                "install packages/openpi-client from the client bundle"
            )
        self.builder = TavlaObservationBuilder(task, camera_color=camera_color, resize=resize)
        self.effort_history = EffortHistoryBuffer()
        self.policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        logging.info("TA-VLA server metadata: %s", self.policy.get_server_metadata())

    @property
    def ready(self) -> bool:
        return self.effort_history.ready

    def observe_effort(
        self, effort: np.ndarray | Iterable[float], *, timestamp: float | None = None
    ) -> None:
        """Call once per 25 Hz robot cycle, including cycles without inference."""
        self.effort_history.append(effort, timestamp=timestamp)

    def make_observation(
        self,
        *,
        side_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray | Iterable[float],
        prompt: str | None = None,
    ) -> dict:
        """Snapshot a time-aligned request before dispatching it to a worker."""
        return self.builder.build(
            side_image=side_image,
            wrist_image=wrist_image,
            state=state,
            effort_history=self.effort_history.sampled(),
            prompt=prompt,
        )

    def infer_observation(
        self,
        observation: dict,
        *,
        current_state: np.ndarray | Iterable[float],
        max_joint_step: float | None = None,
    ) -> np.ndarray:
        """Run a prebuilt observation; call from one dedicated worker thread."""
        result = self.policy.infer(observation)
        if "actions" not in result:
            raise RuntimeError(f"server response has no 'actions' key: {tuple(result)}")

        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] != ACTION_HORIZON or actions.shape[1] < JOINT_DIM:
            raise RuntimeError(
                f"expected server actions shaped ({ACTION_HORIZON}, >= {JOINT_DIM}), got {actions.shape}"
            )
        actions = np.ascontiguousarray(actions[:, :JOINT_DIM])
        if not np.all(np.isfinite(actions)):
            raise RuntimeError("server returned NaN or Inf actions")

        # The ICRA2027 dataset stores seven Nero arm joints.  Although the
        # upstream TA-VLA transform applies its delta conversion only to the
        # first six dimensions, every returned dimension is a physical joint
        # target and therefore receives the same local safety limit.
        if max_joint_step is not None:
            if max_joint_step <= 0:
                raise ValueError("max_joint_step must be positive")
            current = _vector7(current_state, "current_state")
            actions = np.clip(
                actions,
                current[None, :] - max_joint_step,
                current[None, :] + max_joint_step,
            )
        return actions

    def infer(
        self,
        *,
        side_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray | Iterable[float],
        prompt: str | None = None,
        max_joint_step: float | None = None,
    ) -> np.ndarray:
        observation = self.make_observation(
            side_image=side_image,
            wrist_image=wrist_image,
            state=state,
            prompt=prompt,
        )
        return self.infer_observation(
            observation,
            current_state=state,
            max_joint_step=max_joint_step,
        )


def _history_from_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.shape == (len(EFFORT_OFFSETS), JOINT_DIM):
        return array
    if array.ndim == 2 and array.shape[1] == JOINT_DIM and array.shape[0] >= HISTORY_FRAMES:
        array = array[-HISTORY_FRAMES:]
        current = array.shape[0] - 1
        return np.stack([array[current + offset] for offset in EFFORT_OFFSETS])
    raise ValueError(
        "effort_history must be pre-sampled [10, 7] or contain at least 51 consecutive 25 Hz frames [N, 7]"
    )


def _self_test(task: str, camera_color: str) -> None:
    builder = TavlaObservationBuilder(task, camera_color=camera_color)
    history = EffortHistoryBuffer()
    for index in range(HISTORY_FRAMES):
        history.append(np.full((JOINT_DIM,), index, dtype=np.float32), timestamp=index / 25.0)
    sampled = history.sampled()
    observation = builder.build(
        side_image=np.zeros((480, 640, 3), dtype=np.uint8),
        wrist_image=np.zeros((3, 360, 640), dtype=np.uint8),
        state=np.zeros((JOINT_DIM,), dtype=np.float32),
        effort_history=sampled,
    )
    expected_samples = np.asarray([50 + offset for offset in EFFORT_OFFSETS], dtype=np.float32)
    np.testing.assert_array_equal(observation["effort"][:, 0], expected_samples)
    assert observation["images"]["cam_high"].shape == (224, 224, 3)
    assert observation["images"]["cam_left_wrist"].shape == (224, 224, 3)
    assert observation["state"].shape == (JOINT_DIM,)
    print("TA-VLA client self-test passed")
    print("observation shapes:")
    print("  side image:    ", observation["images"]["cam_high"].shape)
    print("  wrist image:   ", observation["images"]["cam_left_wrist"].shape)
    print("  state:         ", observation["state"].shape)
    print("  effort history:", observation["effort"].shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="TA-VLA server IP")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", choices=tuple(TASK_PROMPTS), default="usb")
    parser.add_argument("--camera-color", choices=("rgb", "bgr"), default="rgb")
    parser.add_argument("--input", type=Path, help="NPZ with side_image, wrist_image, state, effort_history")
    parser.add_argument("--output", type=Path, default=Path("tavla_actions.npy"))
    parser.add_argument("--self-test", action="store_true", help="validate formatting without connecting to a server")
    parser.add_argument("--max-joint-step", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.self_test:
        _self_test(args.task, args.camera_color)
        return
    if args.input is None:
        parser.error("--input is required unless --self-test is used")

    with np.load(args.input) as sample:
        required = {"side_image", "wrist_image", "state", "effort_history"}
        missing = required.difference(sample.files)
        if missing:
            raise ValueError(f"NPZ is missing keys: {sorted(missing)}")
        effort_history = _history_from_array(sample["effort_history"])
        builder = TavlaObservationBuilder(args.task, camera_color=args.camera_color)
        observation = builder.build(
            side_image=sample["side_image"],
            wrist_image=sample["wrist_image"],
            state=sample["state"],
            effort_history=effort_history,
        )
        state = np.asarray(sample["state"], dtype=np.float32)

    if websocket_client_policy is None:
        raise RuntimeError(
            "openpi-client is required for remote TA-VLA inference; "
            "install packages/openpi-client from the client bundle"
        )
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("TA-VLA server metadata: %s", policy.get_server_metadata())
    start = time.perf_counter()
    result = policy.infer(observation)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    actions = np.asarray(result["actions"], dtype=np.float32)[:, :JOINT_DIM]
    if actions.shape != (ACTION_HORIZON, JOINT_DIM):
        raise RuntimeError(f"expected actions shaped ({ACTION_HORIZON}, {JOINT_DIM}), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("server returned NaN or Inf actions")
    if args.max_joint_step is not None:
        if args.max_joint_step <= 0:
            raise ValueError("--max-joint-step must be positive")
        actions = np.clip(
            actions,
            state[None, :] - args.max_joint_step,
            state[None, :] + args.max_joint_step,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, actions)
    print(f"saved {actions.shape} actions to {args.output} ({elapsed_ms:.1f} ms)")


if __name__ == "__main__":
    main()
