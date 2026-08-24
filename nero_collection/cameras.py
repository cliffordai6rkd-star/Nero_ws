from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nero_collection.config import CameraConfig
from nero_collection.time_utils import now_us

log = logging.getLogger(__name__)

# Measure decoded frames over a full 10-second interval before deciding that
# the configured camera stream is persistently running below target rate.
_FPS_MEASUREMENT_WINDOW_S = 10.0
_FPS_WARNING_RATIO = 0.90


class CameraUnavailable(RuntimeError):
    pass


@dataclass
class _FrameRateMeasurement:
    window_s: float
    first_timestamp_s: float | None = None
    frame_count: int = 0
    complete: bool = False

    def observe(self, timestamp_s: float) -> tuple[float, float] | None:
        if self.complete:
            return None
        timestamp_s = float(timestamp_s)
        if self.first_timestamp_s is None:
            self.first_timestamp_s = timestamp_s
            self.frame_count = 1
            return None

        self.frame_count += 1
        elapsed_s = timestamp_s - self.first_timestamp_s
        if elapsed_s < self.window_s:
            return None
        self.complete = True
        measured_hz = (self.frame_count - 1) / elapsed_s
        return measured_hz, elapsed_s


@dataclass
class CameraFrame:
    camera_name: str
    timestamp_us: int
    frame: np.ndarray
    # Optional independent preview frame. ``frame`` remains the policy/data
    # path so existing callers keep the configured output_size contract.
    preview_frame: np.ndarray | None = None


class CameraVisualizer:
    """Display selected RGB streams in an isolated, lossy worker process.

    Acquisition only performs a non-blocking enqueue.  If rendering falls behind,
    old preview frames are discarded; control/state processing is never backpressured.
    """

    def __init__(self, camera_names: tuple[str, ...]) -> None:
        self.camera_names = frozenset(camera_names)
        self._queue = None
        self._process = None
        self._dispatch_queue: queue.Queue = queue.Queue(
            maxsize=max(2, 2 * len(self.camera_names))
        )
        self._dispatch_thread: threading.Thread | None = None
        self._failure_logged = False

    @classmethod
    def from_config(cls, configs: tuple[CameraConfig, ...]) -> "CameraVisualizer":
        return cls(tuple(config.name for config in configs if config.visualize))

    @property
    def process_queue(self):
        """Direct producer queue used by isolated acquisition processes."""
        return self._queue

    def start(self) -> None:
        if not self.camera_names or self._process is not None:
            return
        self._failure_logged = False
        process_queue = None
        process = None
        try:
            context = mp.get_context("spawn")
            # A preview is intentionally lossy. The producer-side queue is an
            # in-process handoff, so frame copies and multiprocessing serialization
            # happen outside the control/data-collection loop.
            process_queue = context.Queue(maxsize=max(2, 2 * len(self.camera_names)))
            process = context.Process(
                target=_camera_visualizer_worker,
                args=(tuple(sorted(self.camera_names)), process_queue),
                name="camera-visualizer",
                daemon=True,
            )
            process.start()
            self._queue = process_queue
            self._process = process
            self._dispatch_thread = threading.Thread(
                target=self._dispatch_loop,
                name="camera-visualizer-dispatch",
                daemon=True,
            )
            self._dispatch_thread.start()
        except Exception:
            try:
                if process is not None and process.is_alive():
                    process.terminate()
            except Exception:
                pass
            if process_queue is not None:
                process_queue.close()
            self._queue = None
            self._process = None
            self._dispatch_thread = None
            self._log_failure("failed to start camera visualization process")
            return
        log.info(
            "camera visualization process enabled pid=%s names=%s",
            process.pid,
            sorted(self.camera_names),
        )

    def stop(self) -> None:
        process = self._process
        frame_queue = self._queue
        _put_latest_camera_preview(self._dispatch_queue, None)
        dispatch_thread = self._dispatch_thread
        if dispatch_thread is not None:
            dispatch_thread.join(timeout=1.0)
        if frame_queue is not None:
            _put_latest_camera_preview(frame_queue, None)
        if process is not None:
            process.join(timeout=2.0)
            if process.is_alive():
                log.warning(
                    "camera visualization process did not stop within 2 seconds; "
                    "terminating it"
                )
                process.terminate()
                process.join(timeout=1.0)
        if frame_queue is not None:
            frame_queue.close()
        self._queue = None
        self._process = None
        self._dispatch_thread = None
        while True:
            try:
                self._dispatch_queue.get_nowait()
            except queue.Empty:
                break

    def submit(self, frame: CameraFrame) -> None:
        if frame.camera_name not in self.camera_names:
            return
        process = self._process
        if process is None or self._dispatch_thread is None:
            return
        try:
            if not process.is_alive():
                return
        except Exception:
            self._log_failure("camera visualization process status check failed")
            return
        # Never copy or pickle in the caller. CameraManager's caller owns the
        # frame for the remainder of this cycle and does not mutate it.
        _put_latest_camera_preview(self._dispatch_queue, frame)

    def _dispatch_loop(self) -> None:
        while True:
            try:
                item = self._dispatch_queue.get()
                if item is None:
                    return
                frame_queue = self._queue
                if frame_queue is None:
                    return
                preview_frame = (
                    item.preview_frame
                    if item.preview_frame is not None
                    else item.frame
                )
                preview = CameraFrame(
                    camera_name=item.camera_name,
                    timestamp_us=int(item.timestamp_us),
                    frame=np.asarray(preview_frame).copy(),
                )
                _put_latest_camera_preview(frame_queue, preview)
            except Exception:
                self._log_failure("camera visualization dispatch failed")
                return

    def _log_failure(self, message: str) -> None:
        if not self._failure_logged:
            self._failure_logged = True
            log.warning("%s; continuing without camera preview", message)


def _camera_visualizer_worker(
    camera_names: tuple[str, ...],
    frame_queue,
) -> None:
    """Own all GUI calls in a child process and render one labeled camera grid."""
    # OpenCV's Qt build can select Wayland even when only its xcb plugin is
    # installed.  Keep that GUI-only environment choice inside the child.
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    cv2 = None
    window_name = "Nero cameras"
    window_open = False
    latest_frames: dict[str, np.ndarray] = {}
    stop = False
    try:
        cv2 = _import_cv2()
        selected = frozenset(camera_names)
        while not stop:
            try:
                item = frame_queue.get(timeout=0.05)
                items = [item]
                while True:
                    try:
                        items.append(frame_queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                items = []

            for item in items:
                if item is None:
                    stop = True
                    break
                if isinstance(item, CameraFrame) and item.camera_name in selected:
                    latest_frames[item.camera_name] = np.asarray(
                        item.preview_frame
                        if item.preview_frame is not None
                        else item.frame
                    )
            if stop:
                break

            if latest_frames:
                preview = _compose_camera_preview(
                    camera_names,
                    latest_frames,
                    cv2,
                )
                if not window_open:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    height, width = preview.shape[:2]
                    scale = max(1.0, 512.0 / width, 384.0 / height)
                    cv2.resizeWindow(
                        window_name,
                        int(round(width * scale)),
                        int(round(height * scale)),
                    )
                    window_open = True
                cv2.imshow(window_name, preview)
            cv2.waitKey(1)
    except Exception:
        log.exception(
            "camera visualization process failed; acquisition and control continue"
        )
    finally:
        if window_open and cv2 is not None:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                log.debug("failed to destroy camera window %s", window_name)
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass


def _compose_camera_preview(
    camera_names: tuple[str, ...],
    latest_frames: dict[str, np.ndarray],
    cv2,
) -> np.ndarray:
    available = [
        np.asarray(latest_frames[name])
        for name in camera_names
        if name in latest_frames
    ]
    if not available:
        raise ValueError("camera preview requires at least one frame")
    tile_height = max(int(frame.shape[0]) for frame in available)
    tile_width = max(int(frame.shape[1]) for frame in available)
    tiles: list[np.ndarray] = []
    for camera_name in camera_names:
        rgb_frame = latest_frames.get(camera_name)
        if rgb_frame is None:
            tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
        else:
            tile = cv2.cvtColor(np.asarray(rgb_frame), cv2.COLOR_RGB2BGR)
            if tile.shape[:2] != (tile_height, tile_width):
                tile = cv2.resize(
                    tile,
                    (tile_width, tile_height),
                    interpolation=cv2.INTER_AREA,
                )
            tile = np.ascontiguousarray(tile, dtype=np.uint8)

        label = f"name: {camera_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )
        origin = (10, 10 + text_height)
        cv2.rectangle(
            tile,
            (6, 6),
            (14 + text_width, 14 + text_height + baseline),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            tile,
            label,
            origin,
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    return np.concatenate(tiles, axis=1)


def _put_latest_camera_preview(frame_queue, value) -> None:
    """Never block a producer on a slow or hidden preview window."""
    try:
        frame_queue.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        frame_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        frame_queue.put_nowait(value)
    except queue.Full:
        pass


class CameraSource:
    name: str

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def poll(self) -> CameraFrame | None:
        raise NotImplementedError


class ProcessCameraSource(CameraSource):
    """Camera proxy whose device acquisition runs continuously in a child process."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.name = config.name
        self._context = mp.get_context("spawn")
        self._process = None
        self._frames = None
        self._status = None
        self._faults = None
        self._stop_event = None
        self._preview_queue = None
        self._preview_direct = False
        self._delivered_timestamp_us = 0

    @property
    def preview_direct(self) -> bool:
        return self._preview_direct

    def attach_preview_queue(self, preview_queue) -> None:
        if self._process is not None:
            raise RuntimeError("camera preview queue must be attached before start")
        self._preview_queue = preview_queue if self.config.visualize else None
        self._preview_direct = self._preview_queue is not None

    def start(self) -> None:
        if self._process is not None:
            return
        self._frames = self._context.Queue(maxsize=2)
        self._status = self._context.Queue(maxsize=1)
        self._faults = self._context.Queue(maxsize=2)
        self._stop_event = self._context.Event()
        process = self._context.Process(
            target=_camera_acquisition_worker,
            args=(
                self.config,
                self._frames,
                self._status,
                self._faults,
                self._stop_event,
                self._preview_queue,
            ),
            name=f"nero-camera-{self.name}",
            daemon=True,
        )
        try:
            process.start()
        except Exception as exc:
            self._release_ipc()
            raise CameraUnavailable(
                f"failed to start camera process {self.name}: {exc}"
            ) from exc
        self._process = process
        startup_timeout_s = (
            2.0 * float(self.config.startup_timeout_s)
            + float(self.config.warmup_s)
            + 2.0
        )
        try:
            ready, detail, trace = self._status.get(timeout=startup_timeout_s)
        except queue.Empty as exc:
            self._terminate_process()
            raise CameraUnavailable(
                f"camera process {self.name} did not start within "
                f"{startup_timeout_s:.1f}s"
            ) from exc
        if not ready:
            self._terminate_process()
            raise CameraUnavailable(
                f"camera process {self.name} failed to start: {detail}\n{trace}"
            )
        log.info(
            "isolated camera acquisition ready name=%s pid=%s",
            self.name,
            process.pid,
        )

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        process.join(timeout=3.0)
        if process.is_alive():
            log.warning("camera process did not stop name=%s; terminating it", self.name)
            process.terminate()
            process.join(timeout=1.0)
        self._process = None
        self._release_ipc()
        self._delivered_timestamp_us = 0

    def poll(self) -> CameraFrame | None:
        process = self._process
        if process is None:
            raise CameraUnavailable(f"camera process {self.name} is not started")
        self._raise_worker_fault()
        latest = None
        while True:
            try:
                latest = self._frames.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            timestamp_us = int(latest.timestamp_us)
            if timestamp_us > self._delivered_timestamp_us:
                self._delivered_timestamp_us = timestamp_us
                return latest
        self._raise_worker_fault()
        if not process.is_alive():
            raise CameraUnavailable(
                f"camera process {self.name} exited unexpectedly "
                f"with code {process.exitcode}"
            )
        return None

    def _raise_worker_fault(self) -> None:
        if self._faults is None:
            return
        latest = None
        while True:
            try:
                latest = self._faults.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            raise CameraUnavailable(
                f"camera process {self.name} failed: {latest[0]}\n{latest[1]}"
            )

    def _terminate_process(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        self._process = None
        self._release_ipc()

    def _release_ipc(self) -> None:
        for value in (self._frames, self._status, self._faults):
            if value is not None:
                value.close()
        self._frames = None
        self._status = None
        self._faults = None
        self._stop_event = None


def _camera_acquisition_worker(
    config: CameraConfig,
    frame_queue,
    status_queue,
    fault_queue,
    stop_event,
    preview_queue,
) -> None:
    source = None
    ready = False
    try:
        # Frame delivery is explicitly lossy. Do not keep this process alive at
        # shutdown waiting for multiprocessing feeder threads to flush stale frames.
        for lossy_queue in (frame_queue, preview_queue):
            cancel_join = getattr(lossy_queue, "cancel_join_thread", None)
            if callable(cancel_join):
                cancel_join()
        try:
            os.nice(5)
        except OSError:
            log.warning(
                "could not lower camera worker scheduling priority name=%s",
                config.name,
                exc_info=True,
            )
        source = _build_camera(config)
        source.start()
        status_queue.put((True, None, None))
        ready = True
        while not stop_event.is_set():
            frame = source.poll()
            if frame is None:
                stop_event.wait(0.001)
                continue
            _put_latest_camera_preview(frame_queue, frame)
            if preview_queue is not None:
                _put_latest_camera_preview(preview_queue, frame)
    except BaseException as exc:
        detail = str(exc)
        trace = traceback.format_exc()
        if ready:
            _put_latest_camera_preview(fault_queue, (detail, trace))
        else:
            status_queue.put((False, detail, trace))
    finally:
        if source is not None:
            try:
                source.stop()
            except Exception:
                log.exception("camera source shutdown failed name=%s", config.name)


@dataclass
class MockCamera(CameraSource):
    config: CameraConfig
    name: str = field(init=False)
    _next_frame_t: float = field(default=0.0)
    _counter: int = 0

    def __post_init__(self) -> None:
        self.name = self.config.name

    def start(self) -> None:
        self._next_frame_t = time.monotonic()

    def stop(self) -> None:
        return None

    def poll(self) -> CameraFrame | None:
        now = time.monotonic()
        if now < self._next_frame_t:
            return None
        period = 1.0 / max(self.config.fps, 1.0)
        self._next_frame_t = now + period
        width, height = self.config.width, self.config.height
        yy, xx = np.mgrid[0:height, 0:width]
        preview = np.zeros((height, width, 3), dtype=np.uint8)
        preview[..., 0] = (xx + self._counter) % 255
        preview[..., 1] = (yy + 2 * self._counter) % 255
        preview[..., 2] = (40 + 3 * self._counter) % 255
        self._counter += 1
        policy = preview
        if self.config.output_size is not None:
            output_width, output_height = self.config.output_size
            y_index = np.minimum(
                (np.arange(output_height) * height / output_height).astype(int),
                height - 1,
            )
            x_index = np.minimum(
                (np.arange(output_width) * width / output_width).astype(int),
                width - 1,
            )
            policy = preview[y_index][:, x_index]
        if self.config.preview_output_size is not None:
            preview_width, preview_height = self.config.preview_output_size
            y_index = np.minimum(
                (np.arange(preview_height) * height / preview_height).astype(int),
                height - 1,
            )
            x_index = np.minimum(
                (np.arange(preview_width) * width / preview_width).astype(int),
                width - 1,
            )
            preview = preview[y_index][:, x_index]
        return CameraFrame(
            self.name,
            now_us(),
            policy,
            preview if self.config.visualize else None,
        )


@dataclass
class OrbbecDabaiCamera(CameraSource):
    config: CameraConfig
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.config.name
        module = _import_orbbec_module()
        if module is None:
            raise CameraUnavailable("Orbbec_DaBai_SDK is not installed")
        raise CameraUnavailable(
            "Orbbec_DaBai_SDK is installed, but this project needs the local camera API binding added in OrbbecDabaiCamera"
        )

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def poll(self) -> CameraFrame | None:
        return None


@dataclass
class V4L2Camera(CameraSource):
    config: CameraConfig
    name: str = field(init=False)
    _capture: Any = field(init=False, default=None)
    _reader_thread: threading.Thread | None = field(init=False, default=None)
    _stop_event: threading.Event = field(init=False, default_factory=threading.Event)
    _frame_lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _latest_frame: np.ndarray | None = field(init=False, default=None)
    _latest_preview_frame: np.ndarray | None = field(init=False, default=None)
    _latest_timestamp_us: int = field(init=False, default=0)
    _delivered_timestamp_us: int = field(init=False, default=0)
    _last_frame_monotonic_s: float = field(init=False, default=0.0)
    _reader_error: str | None = field(init=False, default=None)
    _driver_reported_fps: float = field(init=False, default=0.0)
    _fps_measurement_enabled: threading.Event = field(
        init=False, default_factory=threading.Event
    )

    def __post_init__(self) -> None:
        self.name = self.config.name
        if self.config.device is None and self.config.serial_number is None:
            raise CameraUnavailable(
                f"V4L2 camera {self.name} must define device or serial_number"
            )
        if self.config.depth:
            raise CameraUnavailable(f"V4L2 camera {self.name} does not support depth=true")

    def start(self) -> None:
        if self._capture is not None:
            return
        av = _import_av()
        cv2 = _import_cv2()
        device = (
            self.config.device
            if self.config.device is not None
            else _resolve_v4l2_device_by_serial(str(self.config.serial_number))
        )
        device = f"/dev/video{device}" if isinstance(device, int) else str(device)
        self._configure_device_controls(device)
        open_deadline = time.monotonic() + self.config.startup_timeout_s
        capture = None
        last_error: BaseException | None = None
        attempt = 0
        while time.monotonic() < open_deadline:
            attempt += 1
            try:
                capture = av.open(
                    device,
                    format="video4linux2",
                    options=_pyav_v4l2_options(self.config),
                )
                break
            except Exception as exc:
                last_error = exc
            if attempt == 1:
                log.warning(
                    "V4L2 camera %s could not open device %r; retrying for %.1fs",
                    self.name,
                    device,
                    self.config.startup_timeout_s,
                )
            time.sleep(0.2)
        if capture is None:
            detail = f": {last_error}" if last_error is not None else ""
            raise CameraUnavailable(
                f"failed to open V4L2 device {device!r} for camera {self.name}{detail}"
            )
        stream = capture.streams.video[0]
        _configure_pyav_decoder(stream)
        self._capture = capture
        self._driver_reported_fps = float(stream.average_rate or self.config.fps)
        with self._frame_lock:
            self._reader_error = None
            self._latest_frame = None
            self._latest_preview_frame = None
            self._latest_timestamp_us = 0
            self._delivered_timestamp_us = 0
            self._last_frame_monotonic_s = 0.0
        self._stop_event.clear()
        self._fps_measurement_enabled.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(cv2,),
            name=f"v4l2-{self.name}",
            daemon=True,
        )
        self._reader_thread.start()
        deadline = time.monotonic() + self.config.startup_timeout_s
        while time.monotonic() < deadline:
            with self._frame_lock:
                reader_error = self._reader_error
                if self._latest_frame is not None:
                    first_frame = self._latest_frame
                    break
            if reader_error is not None:
                self.stop()
                raise CameraUnavailable(reader_error)
            time.sleep(0.02)
        else:
            self.stop()
            raise CameraUnavailable(
                f"V4L2 camera {self.name} on {device!r} did not produce a frame within "
                f"{self.config.startup_timeout_s:.1f}s"
            )
        log.info(
            "V4L2 camera opened name=%s device=%s requested=%dx%d@%.1f format=%s "
            "driver=pyav/%dx%d@%.1f output=%s",
            self.name,
            device,
            self.config.width,
            self.config.height,
            self.config.fps,
            self.config.pixel_format,
            int(stream.width),
            int(stream.height),
            self._driver_reported_fps,
            first_frame.shape,
        )
        self._warm_up()
        self._fps_measurement_enabled.set()
        log.info(
            "V4L2 camera ready name=%s device=%s warmup=%.1fs",
            self.name,
            device,
            self.config.warmup_s,
        )

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._reader_thread
        if thread is not None:
            thread.join(timeout=1.0)
        capture = self._capture
        if thread is not None and thread.is_alive() and capture is not None:
            try:
                capture.close()
            except Exception:
                log.debug("failed to interrupt PyAV camera %s", self.name, exc_info=True)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        if capture is not None:
            try:
                capture.close()
            except Exception:
                log.debug("failed to close PyAV camera %s", self.name, exc_info=True)
        self._reader_thread = None
        self._capture = None
        self._fps_measurement_enabled.clear()

    def poll(self) -> CameraFrame | None:
        with self._frame_lock:
            if self._reader_error is not None:
                raise CameraUnavailable(self._reader_error)
            if (
                self._capture is not None
                and self._last_frame_monotonic_s > 0.0
                and time.monotonic() - self._last_frame_monotonic_s
                > self.config.frame_timeout_s
            ):
                raise CameraUnavailable(
                    f"V4L2 camera {self.name} frame timeout exceeded: "
                    f"no frame for more than {self.config.frame_timeout_s:.3f}s"
                )
            if (
                self._latest_frame is None
                or self._latest_timestamp_us <= self._delivered_timestamp_us
            ):
                return None
            timestamp_us = self._latest_timestamp_us
            frame = self._latest_frame.copy()
            preview_frame = (
                None
                if self._latest_preview_frame is None
                else self._latest_preview_frame.copy()
            )
            self._delivered_timestamp_us = timestamp_us
        return CameraFrame(self.name, timestamp_us, frame, preview_frame)

    def _configure_device_controls(self, device: str) -> None:
        if self.config.exposure is not None:
            _set_v4l2_control(device, "auto_exposure", 1)
            _set_v4l2_control(
                device,
                "exposure_time_absolute",
                int(self.config.exposure),
            )
        if self.config.exposure_dynamic_framerate is not None:
            _set_v4l2_boolean_control(
                device,
                "exposure_dynamic_framerate",
                self.config.exposure_dynamic_framerate,
            )

    def _reader_loop(self, cv2) -> None:
        fps_measurement = _FrameRateMeasurement(_FPS_MEASUREMENT_WINDOW_S)
        capture = self._capture
        if capture is None:
            return
        try:
            for decoded_frame in capture.decode(video=0):
                if self._stop_event.is_set():
                    return
                frame = decoded_frame.to_ndarray(format="bgr24")
                prepared, preview = _prepare_v4l2_frames(
                    frame,
                    self.config,
                    cv2,
                    include_preview=self.config.visualize,
                )
                self._store_frame(
                    prepared,
                    now_us(),
                    preview_frame=preview if self.config.visualize else None,
                )
                result = (
                    fps_measurement.observe(time.monotonic())
                    if self._fps_measurement_enabled.is_set()
                    else None
                )
                if result is not None:
                    measured_hz, window_s = result
                    log_fn = (
                        log.warning
                        if measured_hz < self.config.fps * _FPS_WARNING_RATIO
                        else log.info
                    )
                    log_fn(
                        "V4L2 camera fps name=%s requested=%.2f driver_reported=%.2f "
                        "measured=%.2f window=%.2fs decoder=pyav",
                        self.name,
                        self.config.fps,
                        self._driver_reported_fps,
                        measured_hz,
                        window_s,
                    )
            if not self._stop_event.is_set():
                raise CameraUnavailable(f"V4L2 camera {self.name} stream ended unexpectedly")
        except Exception as exc:
            if self._stop_event.is_set():
                return
            error = f"V4L2 camera {self.name} PyAV reader failed: {exc}"
            log.exception(error)
            self._set_reader_error(error)
            self._stop_event.set()

    def _warm_up(self) -> None:
        duration_s = self.config.warmup_s
        if duration_s <= 0.0:
            return
        with self._frame_lock:
            initial_timestamp_us = self._latest_timestamp_us
        log.info("warming up V4L2 camera name=%s duration=%.1fs", self.name, duration_s)
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            thread = self._reader_thread
            if thread is None or not thread.is_alive():
                self.stop()
                raise CameraUnavailable(
                    f"V4L2 camera {self.name} reader stopped during warmup"
                )
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        with self._frame_lock:
            latest_timestamp_us = self._latest_timestamp_us
        if latest_timestamp_us <= initial_timestamp_us:
            self.stop()
            raise CameraUnavailable(
                f"V4L2 camera {self.name} produced no new frame during "
                f"{duration_s:.1f}s warmup"
            )

    def _store_frame(
        self,
        frame: np.ndarray,
        timestamp_us: int,
        *,
        preview_frame: np.ndarray | None = None,
    ) -> None:
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_preview_frame = (
                None if preview_frame is None else np.asarray(preview_frame).copy()
            )
            self._latest_timestamp_us = int(timestamp_us)
            self._last_frame_monotonic_s = time.monotonic()

    def _set_reader_error(self, error: str) -> None:
        with self._frame_lock:
            self._reader_error = str(error)


def _resolve_v4l2_device_by_serial(
    serial_number: str,
    by_id_directory: str | Path = "/dev/v4l/by-id",
) -> str:
    serial_number = str(serial_number).strip()
    if not serial_number:
        raise CameraUnavailable("V4L2 camera serial_number must be non-empty")
    directory = Path(by_id_directory)
    if not directory.is_dir():
        raise CameraUnavailable(f"V4L2 stable-device directory does not exist: {directory}")
    matches = sorted(
        path
        for path in directory.iterdir()
        if serial_number in path.name and path.name.endswith("-video-index0") and path.exists()
    )
    if not matches:
        raise CameraUnavailable(
            f"No V4L2 capture device found for serial_number={serial_number!r} in {directory}"
        )
    if len(matches) > 1:
        raise CameraUnavailable(
            f"Multiple V4L2 capture devices match serial_number={serial_number!r}: {matches}"
        )
    return str(matches[0])


def _set_v4l2_boolean_control(
    device: str | int,
    control_name: str,
    enabled: bool,
) -> None:
    _set_v4l2_control(device, control_name, 1 if enabled else 0)


def _set_v4l2_control(
    device: str | int,
    control_name: str,
    expected: int,
) -> None:
    executable = shutil.which("v4l2-ctl")
    if executable is None:
        raise CameraUnavailable(
            f"camera control {control_name} requires v4l2-ctl; install v4l-utils"
        )

    device_arg = f"/dev/video{device}" if isinstance(device, int) else str(device)
    expected = int(expected)
    command = [
        executable,
        "-d",
        device_arg,
        f"--set-ctrl={control_name}={expected}",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise CameraUnavailable(
            f"failed to set V4L2 control {control_name}={expected} on {device_arg}: {detail}"
        )

    verify = subprocess.run(
        [executable, "-d", device_arg, f"--get-ctrl={control_name}"],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    if verify.returncode != 0:
        detail = (verify.stderr or verify.stdout).strip()
        raise CameraUnavailable(
            f"failed to verify V4L2 control {control_name} on {device_arg}: {detail}"
        )
    try:
        value_text = verify.stdout.rsplit(":", 1)[1].strip().split(maxsplit=1)[0]
        actual = int(value_text)
    except (IndexError, ValueError) as exc:
        raise CameraUnavailable(
            f"unexpected V4L2 control response for {control_name}: {verify.stdout.strip()}"
        ) from exc
    if actual != expected:
        raise CameraUnavailable(
            f"V4L2 control {control_name} on {device_arg} is {actual}, expected {expected}"
        )
    log.info("V4L2 camera control device=%s %s=%d", device_arg, control_name, actual)


class CameraManager:
    def __init__(
        self,
        cameras: list[CameraSource],
        visualizer: CameraVisualizer | None = None,
    ) -> None:
        self.cameras = cameras
        self.visualizer = visualizer
        self._visualizer_active = visualizer is not None
        self._visualizer_failure_logged = False

    @classmethod
    def from_config(
        cls,
        configs: tuple[CameraConfig, ...],
        visualizer: CameraVisualizer | None = None,
    ) -> "CameraManager":
        cameras: list[CameraSource] = []
        for camera_config in configs:
            try:
                backend = camera_config.backend.lower().replace("-", "_")
                if backend in {"v4l2", "opencv_v4l2", "opencv"}:
                    # Validate without opening the device; acquisition itself is
                    # created only inside ProcessCameraSource's child process.
                    V4L2Camera(camera_config)
                    camera = ProcessCameraSource(camera_config)
                else:
                    camera = _build_camera(camera_config)
            except CameraUnavailable as exc:
                log.warning("skip camera %s: %s", camera_config.name, exc)
                continue
            cameras.append(camera)
        return cls(cameras, visualizer=visualizer)

    def start(self) -> None:
        started: list[CameraSource] = []
        self._visualizer_active = self.visualizer is not None
        self._visualizer_failure_logged = False
        try:
            if self.visualizer is not None:
                try:
                    self.visualizer.start()
                except Exception:
                    self._visualizer_active = False
                    log.warning(
                        "camera visualization failed to start; continuing without preview",
                        exc_info=True,
                    )
            preview_queue = (
                getattr(self.visualizer, "process_queue", None)
                if self._visualizer_active and self.visualizer is not None
                else None
            )
            for camera in self.cameras:
                attach_preview = getattr(camera, "attach_preview_queue", None)
                if callable(attach_preview):
                    attach_preview(preview_queue)
                log.info("starting camera %s", camera.name)
                camera.start()
                started.append(camera)
        except Exception:
            for camera in reversed(started):
                camera.stop()
            if self.visualizer is not None:
                self.visualizer.stop()
            raise

    def stop(self) -> None:
        for camera in self.cameras:
            try:
                camera.stop()
            except Exception as exc:  # pragma: no cover - shutdown guard
                log.debug("camera stop failed for %s: %s", camera.name, exc)
        if self.visualizer is not None:
            try:
                self.visualizer.stop()
            except Exception as exc:  # pragma: no cover - shutdown guard
                log.debug("camera visualizer stop failed: %s", exc)

    def poll(self) -> list[CameraFrame]:
        frames: list[CameraFrame] = []
        for camera in self.cameras:
            frame = camera.poll()
            if frame is not None:
                frames.append(frame)
                if (
                    self._visualizer_active
                    and self.visualizer is not None
                    and not bool(getattr(camera, "preview_direct", False))
                ):
                    try:
                        self.visualizer.submit(frame)
                    except Exception:
                        self._visualizer_active = False
                        if not self._visualizer_failure_logged:
                            self._visualizer_failure_logged = True
                            log.warning(
                                "camera visualization failed during polling; "
                                "continuing without preview",
                                exc_info=True,
                            )
        return frames


def _build_camera(config: CameraConfig) -> CameraSource:
    backend = config.backend.lower().replace("-", "_")
    if backend in {"mock", "simulation"}:
        return MockCamera(config)
    if backend in {"orbbec", "orbbec_dabai", "orbbec_dabai_sdk"}:
        return OrbbecDabaiCamera(config)
    if backend in {"v4l2", "opencv_v4l2", "opencv"}:
        return V4L2Camera(config)
    raise CameraUnavailable(f"unsupported camera backend {config.backend!r}")


def _import_orbbec_module() -> object | None:
    for module_name in ("Orbbec_DaBai_SDK", "orbbec_dabai_sdk", "pyorbbecsdk"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


def _import_av():
    try:
        return importlib.import_module("av")
    except ImportError as exc:
        raise CameraUnavailable(
            "V4L2 camera backend requires PyAV; install av>=15,<16"
        ) from exc


def _import_cv2():
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise CameraUnavailable(
            "V4L2 camera backend requires OpenCV; install opencv-python>=4.9"
        ) from exc


def _pyav_v4l2_options(config: CameraConfig) -> dict[str, str]:
    input_formats = {
        "MJPG": "mjpeg",
        "YUYV": "yuyv422",
        "YUY2": "yuyv422",
        "RGB3": "rgb24",
        "BGR3": "bgr24",
        "GREY": "gray",
    }
    try:
        input_format = input_formats[config.pixel_format.upper()]
    except KeyError as exc:
        raise CameraUnavailable(
            f"PyAV V4L2 backend does not support pixel format {config.pixel_format!r}"
        ) from exc
    return {
        "video_size": f"{config.width}x{config.height}",
        "framerate": f"{config.fps:.12g}",
        "input_format": input_format,
    }


def _configure_pyav_decoder(stream) -> None:
    codec_context = stream.codec_context
    codec_context.thread_count = 1
    codec_context.thread_type = "NONE"


def _prepare_v4l2_frames(
    frame: np.ndarray,
    config: CameraConfig,
    cv2,
    *,
    include_preview: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return independent policy and preview RGB frames.

    ``output_size`` is intentionally applied only to the policy path.  The
    preview defaults to the cropped decoded resolution and can be resized
    independently through ``preview_output_size``.
    """
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(
            f"V4L2 camera {config.name} returned unsupported frame shape {frame.shape}"
        )
    height, width = frame.shape[:2]
    y0, y1, x0, x1 = config.crop
    y0 = 0 if y0 is None else y0
    y1 = height if y1 is None else y1
    x0 = 0 if x0 is None else x0
    x1 = width if x1 is None else x1
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise RuntimeError(
            f"camera {config.name} crop {config.crop} is outside frame {width}x{height}"
        )
    cropped = frame[y0:y1, x0:x1]
    preview = cropped if include_preview else None
    if include_preview and config.preview_output_size is not None:
        preview_width, preview_height = config.preview_output_size
        shrinking = preview_width < cropped.shape[1] or preview_height < cropped.shape[0]
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        preview = cv2.resize(
            cropped,
            (preview_width, preview_height),
            interpolation=interpolation,
        )
    if preview is not None:
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    policy = cropped
    if config.output_size is not None:
        output_width, output_height = config.output_size
        shrinking = output_width < policy.shape[1] or output_height < policy.shape[0]
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        policy = cv2.resize(
            policy,
            (output_width, output_height),
            interpolation=interpolation,
        )
    # OpenCV V4L2 decodes to BGR; datasets use conventional RGB channel order.
    policy = cv2.cvtColor(policy, cv2.COLOR_BGR2RGB)
    return (
        np.ascontiguousarray(policy, dtype=np.uint8),
        None
        if preview is None
        else np.ascontiguousarray(preview, dtype=np.uint8),
    )


def _prepare_v4l2_frame(frame: np.ndarray, config: CameraConfig, cv2) -> np.ndarray:
    """Backward-compatible policy-frame helper used by existing callers/tests."""
    policy, _preview = _prepare_v4l2_frames(
        frame,
        config,
        cv2,
        include_preview=False,
    )
    return policy
