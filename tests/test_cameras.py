from __future__ import annotations

from importlib.util import find_spec
import queue
import threading
import time

import numpy as np
import pytest

from nero_collection.cameras import (
    CameraFrame,
    CameraManager,
    CameraUnavailable,
    CameraVisualizer,
    ProcessCameraSource,
    V4L2Camera,
    _FrameRateMeasurement,
    _FPS_MEASUREMENT_WINDOW_S,
    _build_camera,
    _camera_acquisition_worker,
    _camera_visualizer_worker,
    _configure_pyav_decoder,
    _pyav_v4l2_options,
    _put_latest_camera_preview,
    _prepare_v4l2_frame,
    _resolve_v4l2_device_by_serial,
    _set_v4l2_boolean_control,
)
from nero_collection.config import CameraConfig, _parse_camera


def test_parse_v4l2_camera_config() -> None:
    config = _parse_camera(
        {
            "name": "wrist",
            "backend": "v4l2",
            "device": "/dev/video2",
            "pixel_format": "mjpg",
            "buffer_size": 1,
            "startup_timeout_s": 2.5,
            "warmup_s": 10.0,
            "frame_timeout_s": 0.5,
            "visualize": True,
            "width": 640,
            "height": 480,
            "fps": 30,
            "exposure_dynamic_framerate": False,
            "crop": [10, 470, 20, 620],
            "output_size": [256, 192],
        }
    )

    assert config.device == "/dev/video2"
    assert config.pixel_format == "MJPG"
    assert config.buffer_size == 1
    assert config.startup_timeout_s == pytest.approx(2.5)
    assert config.warmup_s == pytest.approx(10.0)
    assert config.frame_timeout_s == pytest.approx(0.5)
    assert config.visualize is True
    assert config.exposure_dynamic_framerate is False
    assert config.crop == (10, 470, 20, 620)
    assert config.output_size == (256, 192)
    assert isinstance(_build_camera(config), V4L2Camera)


def test_camera_manager_isolates_v4l2_acquisition_process() -> None:
    config = CameraConfig(
        name="wrist",
        backend="v4l2",
        device="/dev/video2",
    )

    manager = CameraManager.from_config((config,))

    assert len(manager.cameras) == 1
    assert isinstance(manager.cameras[0], ProcessCameraSource)


def test_process_camera_source_continuously_publishes_mock_frames() -> None:
    source = ProcessCameraSource(
        CameraConfig(
            name="process_mock",
            backend="mock",
            width=16,
            height=12,
            fps=50.0,
            startup_timeout_s=0.5,
        )
    )
    source.start()
    try:
        deadline = time.monotonic() + 2.0
        frame = None
        while time.monotonic() < deadline:
            frame = source.poll()
            if frame is not None:
                break
            time.sleep(0.01)
        assert frame is not None
        assert frame.camera_name == "process_mock"
        assert frame.frame.shape == (12, 16, 3)
    finally:
        source.stop()


def test_mock_camera_keeps_high_resolution_preview_when_policy_is_resized() -> None:
    source = _build_camera(
        CameraConfig(
            name="preview_mock",
            backend="mock",
            width=16,
            height=12,
            output_size=(4, 3),
            visualize=True,
        )
    )
    source.start()
    try:
        frame = source.poll()
        assert frame is not None
        assert frame.frame.shape == (3, 4, 3)
        assert frame.preview_frame is not None
        assert frame.preview_frame.shape == (12, 16, 3)
    finally:
        source.stop()


def test_camera_worker_does_not_flush_lossy_frame_queue_on_exit(monkeypatch) -> None:
    class Queue:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel_join_thread(self) -> None:
            self.cancelled = True

        def put(self, _value) -> None:
            return None

    class StopEvent:
        @staticmethod
        def is_set() -> bool:
            return True

    frame_queue = Queue()
    preview_queue = Queue()
    status_queue = Queue()
    fault_queue = Queue()
    monkeypatch.setattr("nero_collection.cameras.os.nice", lambda _increment: 0)

    _camera_acquisition_worker(
        CameraConfig(name="mock", backend="mock"),
        frame_queue,
        status_queue,
        fault_queue,
        StopEvent(),
        preview_queue,
    )

    assert frame_queue.cancelled is True
    assert preview_queue.cancelled is True


def test_parse_v4l2_camera_config_by_serial_number() -> None:
    config = _parse_camera(
        {"name": "wrist", "backend": "v4l2", "serial_number": "CC1WC520122"}
    )

    assert config.device is None
    assert config.serial_number == "CC1WC520122"
    assert isinstance(_build_camera(config), V4L2Camera)


@pytest.mark.parametrize(
    "data",
    [
        {"name": "camera", "backend": "v4l2"},
        {
            "name": "camera",
            "backend": "v4l2",
            "device": "/dev/video2",
            "serial_number": "CC1WC520122",
        },
        {"name": "camera", "backend": "v4l2", "device": "/dev/video2", "fps": 0},
        {"name": "camera", "backend": "v4l2", "device": "/dev/video2", "warmup_s": -1},
        {"name": "camera", "backend": "v4l2", "device": "/dev/video2", "frame_timeout_s": 0},
        {
            "name": "camera",
            "backend": "v4l2",
            "device": "/dev/video2",
            "visualize": "true",
        },
        {
            "name": "camera",
            "backend": "v4l2",
            "device": "/dev/video2",
            "exposure_dynamic_framerate": "false",
        },
        {"name": "camera", "backend": "v4l2", "device": "/dev/video2", "pixel_format": "MJPEG"},
        {"name": "camera", "backend": "v4l2", "device": "/dev/video2", "crop": [5, 4, 0, None]},
        {"name": "camera", "backend": "v4l2", "device": "/dev/video2", "output_size": [0, 192]},
    ],
)
def test_parse_v4l2_camera_rejects_invalid_settings(data) -> None:
    with pytest.raises(ValueError):
        _parse_camera(data)


@pytest.mark.skipif(find_spec("cv2") is None, reason="OpenCV is not installed")
def test_v4l2_preprocessing_crops_resizes_and_converts_to_rgb() -> None:
    import cv2

    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[..., 0] = 10
    frame[..., 1] = 20
    frame[..., 2] = 30
    config = CameraConfig(
        name="camera",
        backend="v4l2",
        device="/dev/video2",
        width=6,
        height=4,
        crop=(1, 3, 2, 6),
        output_size=(2, 1),
    )

    output = _prepare_v4l2_frame(frame, config, cv2)

    assert output.shape == (1, 2, 3)
    assert output.dtype == np.uint8
    assert output.flags.c_contiguous
    assert np.all(output == np.asarray([30, 20, 10], dtype=np.uint8))


@pytest.mark.skipif(find_spec("cv2") is None, reason="OpenCV is not installed")
def test_v4l2_preprocessing_returns_independent_preview_resolution() -> None:
    import cv2

    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    config = CameraConfig(
        name="camera",
        backend="v4l2",
        device="/dev/video2",
        width=6,
        height=4,
        output_size=(2, 1),
    )

    from nero_collection.cameras import _prepare_v4l2_frames

    policy, preview = _prepare_v4l2_frames(frame, config, cv2)
    assert policy.shape == (1, 2, 3)
    assert preview.shape == (4, 6, 3)


def test_v4l2_poll_returns_each_latest_frame_once() -> None:
    camera = V4L2Camera(CameraConfig(name="camera", backend="v4l2", device="/dev/video2"))
    first = np.full((2, 3, 3), 1, dtype=np.uint8)
    second = np.full((2, 3, 3), 2, dtype=np.uint8)

    camera._store_frame(first, 100)
    frame = camera.poll()
    assert frame is not None
    assert frame.timestamp_us == 100
    assert np.array_equal(frame.frame, first)
    assert camera.poll() is None

    camera._store_frame(second, 200)
    frame = camera.poll()
    assert frame is not None
    assert frame.timestamp_us == 200
    assert np.array_equal(frame.frame, second)


def test_v4l2_warmup_keeps_reader_running_until_duration_elapses(monkeypatch) -> None:
    camera = V4L2Camera(
        CameraConfig(
            name="external",
            backend="v4l2",
            device="/dev/video4",
            warmup_s=0.1,
        )
    )
    clock = {"now": 0.0}

    class ReaderThread:
        @staticmethod
        def is_alive() -> bool:
            return True

    def fake_sleep(duration_s: float) -> None:
        clock["now"] += duration_s
        camera._store_frame(np.zeros((2, 3, 3), dtype=np.uint8), 2)

    camera._reader_thread = ReaderThread()
    camera._latest_timestamp_us = 1
    monkeypatch.setattr("nero_collection.cameras.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr("nero_collection.cameras.time.sleep", fake_sleep)

    camera._warm_up()

    assert clock["now"] == pytest.approx(0.1)
    assert camera._latest_timestamp_us == 2


def test_v4l2_poll_raises_when_frames_become_stale(monkeypatch) -> None:
    camera = V4L2Camera(
        CameraConfig(
            name="wrist",
            backend="v4l2",
            device="/dev/video2",
            frame_timeout_s=0.5,
        )
    )
    clock = {"now": 1.0}
    monkeypatch.setattr("nero_collection.cameras.time.monotonic", lambda: clock["now"])
    camera._capture = object()
    camera._store_frame(np.zeros((2, 3, 3), dtype=np.uint8), 1)
    clock["now"] = 1.501

    with pytest.raises(CameraUnavailable, match="frame timeout"):
        camera.poll()


def test_pyav_v4l2_options_map_camera_mode() -> None:
    config = CameraConfig(
        name="wrist",
        backend="v4l2",
        device="/dev/video2",
        pixel_format="MJPG",
        width=640,
        height=480,
        fps=30.0,
    )

    assert _pyav_v4l2_options(config) == {
        "video_size": "640x480",
        "framerate": "30",
        "input_format": "mjpeg",
    }


def test_pyav_decoder_is_single_threaded() -> None:
    class CodecContext:
        thread_count = 0
        thread_type = "SLICE"

    class Stream:
        codec_context = CodecContext()

    stream = Stream()
    _configure_pyav_decoder(stream)

    assert stream.codec_context.thread_count == 1
    assert stream.codec_context.thread_type == "NONE"


@pytest.mark.skipif(find_spec("cv2") is None, reason="OpenCV is not installed")
def test_pyav_reader_decodes_and_publishes_rgb_frame() -> None:
    import cv2

    camera = V4L2Camera(
        CameraConfig(
            name="wrist",
            backend="v4l2",
            device="/dev/video2",
            width=3,
            height=2,
        )
    )
    bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    bgr[..., 0] = 10
    bgr[..., 1] = 20
    bgr[..., 2] = 30

    class DecodedFrame:
        def to_ndarray(self, *, format):
            assert format == "bgr24"
            return bgr

    class Capture:
        def decode(self, *, video):
            assert video == 0
            yield DecodedFrame()
            camera._stop_event.set()

    camera._capture = Capture()
    camera._reader_loop(cv2)

    frame = camera.poll()
    assert frame is not None
    assert frame.frame.shape == (2, 3, 3)
    assert np.all(frame.frame == np.asarray([30, 20, 10], dtype=np.uint8))


def test_frame_rate_measurement_reports_completed_window_once() -> None:
    measurement = _FrameRateMeasurement(window_s=2.0)

    assert measurement.observe(10.0) is None
    assert measurement.observe(10.5) is None
    assert measurement.observe(11.0) is None
    assert measurement.observe(11.5) is None
    result = measurement.observe(12.0)

    assert result == pytest.approx((2.0, 2.0))
    assert measurement.observe(12.5) is None


def test_camera_fps_monitor_uses_ten_second_window() -> None:
    assert _FPS_MEASUREMENT_WINDOW_S == pytest.approx(10.0)


def test_set_v4l2_boolean_control_sets_and_verifies(monkeypatch) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1].startswith("--get-ctrl"):
            return type(
                "Result",
                (),
                {
                    "returncode": 0,
                    "stdout": "exposure_dynamic_framerate: 0 (Disabled)\n",
                    "stderr": "",
                },
            )()
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("nero_collection.cameras.shutil.which", lambda _: "/usr/bin/v4l2-ctl")
    monkeypatch.setattr("nero_collection.cameras.subprocess.run", fake_run)

    _set_v4l2_boolean_control("/dev/video2", "exposure_dynamic_framerate", False)

    assert calls[0][-1] == "--set-ctrl=exposure_dynamic_framerate=0"
    assert calls[1][-1] == "--get-ctrl=exposure_dynamic_framerate"


def test_resolve_v4l2_device_by_serial_uses_capture_index(tmp_path) -> None:
    capture = tmp_path / "usb-Orbbec_Dabai_CC1WC520122-video-index0"
    metadata = tmp_path / "usb-Orbbec_Dabai_CC1WC520122-video-index1"
    capture.touch()
    metadata.touch()

    result = _resolve_v4l2_device_by_serial("CC1WC520122", tmp_path)

    assert result == str(capture)


def test_resolve_v4l2_device_by_serial_rejects_unknown_serial(tmp_path) -> None:
    with pytest.raises(CameraUnavailable, match="No V4L2 capture device"):
        _resolve_v4l2_device_by_serial("missing", tmp_path)


def test_camera_manager_stops_started_sources_after_start_failure() -> None:
    events: list[str] = []

    class Source:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def start(self) -> None:
            events.append(f"{self.name}:start")
            if self.fail:
                raise RuntimeError("start failed")

        def stop(self) -> None:
            events.append(f"{self.name}:stop")

    manager = CameraManager([Source("first"), Source("second", fail=True)])

    with pytest.raises(RuntimeError, match="start failed"):
        manager.start()

    assert events == ["first:start", "second:start", "first:stop"]


def test_camera_visualizer_start_failure_does_not_stop_camera(monkeypatch) -> None:
    events: list[str] = []

    class Source:
        name = "external"

        def start(self) -> None:
            events.append("camera:start")

        def stop(self) -> None:
            events.append("camera:stop")

        def poll(self):
            return None

    class Visualizer:
        def start(self) -> None:
            events.append("visualizer:start")
            raise RuntimeError("no display")

        def stop(self) -> None:
            events.append("visualizer:stop")

    manager = CameraManager([Source()], visualizer=Visualizer())
    # A user-supplied visualizer must be best effort just like the built-in one.
    monkeypatch.setattr(
        "nero_collection.cameras.log.warning",
        lambda *_args, **_kwargs: None,
    )

    # CameraManager's startup guard handles GUI failures independently of cameras.
    manager.start()
    assert events == ["visualizer:start", "camera:start"]
    manager.stop()
    assert events == ["visualizer:start", "camera:start", "camera:stop", "visualizer:stop"]


def test_camera_manager_forwards_frames_to_visualizer() -> None:
    events: list[str] = []
    expected = CameraFrame("external", 123, np.zeros((2, 3, 3), dtype=np.uint8))

    class Source:
        name = "external"

        def start(self) -> None:
            events.append("camera:start")

        def stop(self) -> None:
            events.append("camera:stop")

        def poll(self):
            return expected

    class Visualizer:
        def start(self) -> None:
            events.append("visualizer:start")

        def stop(self) -> None:
            events.append("visualizer:stop")

        def submit(self, frame: CameraFrame) -> None:
            assert frame is expected
            events.append("visualizer:submit")

    manager = CameraManager([Source()], visualizer=Visualizer())
    manager.start()
    assert manager.poll() == [expected]
    manager.stop()

    assert events == [
        "visualizer:start",
        "camera:start",
        "visualizer:submit",
        "camera:stop",
        "visualizer:stop",
    ]


def test_camera_visualizer_submit_does_not_copy_in_control_thread() -> None:
    class Process:
        @staticmethod
        def is_alive() -> bool:
            return True

    visualizer = CameraVisualizer(("external",))
    visualizer._process = Process()
    visualizer._dispatch_thread = object()
    frame = CameraFrame(
        "external",
        123,
        np.zeros((2, 3, 3), dtype=np.uint8),
    )

    visualizer.submit(frame)

    queued = visualizer._dispatch_queue.get_nowait()
    assert queued is frame


def test_camera_visualizer_worker_displays_submitted_frame(monkeypatch) -> None:
    displayed = threading.Event()
    calls: list[tuple[str, str]] = []

    class CV2:
        WINDOW_NORMAL = 1
        COLOR_RGB2BGR = 2
        FONT_HERSHEY_SIMPLEX = 3
        LINE_AA = 4
        INTER_AREA = 5

        @staticmethod
        def namedWindow(name, _mode):
            calls.append(("open", name))

        @staticmethod
        def resizeWindow(name, width, height):
            assert name == "Nero cameras"
            assert (width, height) == (1024, 384)

        @staticmethod
        def cvtColor(frame, _conversion):
            return frame

        @staticmethod
        def getTextSize(text, _font, _scale, _thickness):
            return (len(text) * 8, 16), 4

        @staticmethod
        def rectangle(_frame, _start, _end, _color, _thickness):
            return None

        @staticmethod
        def putText(_frame, text, *_args):
            calls.append(("label", text))

        @staticmethod
        def imshow(name, frame):
            assert name == "Nero cameras"
            assert frame.shape == (192, 512, 3)
            calls.append(("show", name))
            displayed.set()

        @staticmethod
        def waitKey(_delay):
            return -1

        @staticmethod
        def destroyWindow(name):
            calls.append(("close", name))

    monkeypatch.setattr("nero_collection.cameras._import_cv2", lambda: CV2)
    frame_queue: queue.Queue = queue.Queue(maxsize=4)
    _put_latest_camera_preview(
        frame_queue,
        CameraFrame("side", 122, np.zeros((192, 256, 3), dtype=np.uint8))
    )
    _put_latest_camera_preview(
        frame_queue,
        CameraFrame("wrist", 123, np.zeros((192, 256, 3), dtype=np.uint8))
    )
    worker = threading.Thread(
        target=_camera_visualizer_worker,
        args=(("side", "wrist"), frame_queue),
        name="camera-visualizer-worker-test",
    )
    worker.start()
    assert displayed.wait(timeout=1.0)
    _put_latest_camera_preview(frame_queue, None)
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert calls.count(("open", "Nero cameras")) == 1
    assert ("label", "name: side") in calls
    assert ("label", "name: wrist") in calls
    assert ("show", "Nero cameras") in calls
    assert ("close", "Nero cameras") in calls


def test_camera_visualizer_starts_spawn_process(monkeypatch) -> None:
    created = {}

    class FakeQueue:
        def close(self):
            created["queue_closed"] = True

    class FakeProcess:
        pid = 1234

        def __init__(self, **kwargs):
            created.update(kwargs)
            self.alive = False

        def start(self):
            created["started"] = True
            self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            created["join_timeout"] = timeout
            self.alive = False

        def terminate(self):
            created["terminated"] = True
            self.alive = False

    class FakeContext:
        @staticmethod
        def Queue(maxsize):
            created["queue_size"] = maxsize
            return FakeQueue()

        @staticmethod
        def Process(**kwargs):
            return FakeProcess(**kwargs)

    monkeypatch.setattr(
        "nero_collection.cameras.mp.get_context",
        lambda method: FakeContext() if method == "spawn" else None,
    )
    monkeypatch.setattr(
        "nero_collection.cameras._put_latest_camera_preview",
        lambda _queue, value: created.setdefault("stop_value", value),
    )

    visualizer = CameraVisualizer(("external",))
    visualizer.start()
    visualizer.stop()

    assert created["target"] is _camera_visualizer_worker
    assert created["args"][0] == ("external",)
    assert created["queue_size"] == 2
    assert created["started"] is True
    assert created["stop_value"] is None
    assert created["queue_closed"] is True
