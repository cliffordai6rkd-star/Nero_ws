"""Independent real-time kinematic visualization of sampled CARS-WM futures."""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


log = logging.getLogger(__name__)


@dataclass
class VisualizationPacket:
    timestamp: float
    observed_q: np.ndarray
    predicted_q: np.ndarray | None = None
    predicted_ee_position: np.ndarray | None = None
    prediction_id: int | None = None
    # Optional q to put on the displayed robot.  Online packets leave this
    # unset, so the display follows the latest measured ``observed_q``.  The
    # offline LeRobot replay sets it to the first WM future q, allowing the
    # kinematic model to play the same q that q-mode execution would consume
    # without changing the required observed_q diagnostic field.
    playback_q: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.timestamp = float(self.timestamp)
        if not np.isfinite(self.timestamp):
            raise ValueError("visualization packet timestamp must be finite")
        self.observed_q = _joint_vector(self.observed_q, "observed_q")
        if self.predicted_q is not None:
            values = np.asarray(self.predicted_q, dtype=np.float64)
            if values.ndim != 3 or values.shape[-1] != self.observed_q.size or not np.isfinite(values).all():
                raise ValueError("predicted_q must have shape [N,H,J] and be finite")
            self.predicted_q = values.copy()
        if self.predicted_ee_position is not None:
            values = np.asarray(self.predicted_ee_position, dtype=np.float64)
            if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
                raise ValueError("predicted_ee_position must have shape [N,H,3] and be finite")
            if self.predicted_q is not None and values.shape[:2] != self.predicted_q.shape[:2]:
                raise ValueError("predicted_ee_position and predicted_q must share [N,H]")
            self.predicted_ee_position = values.copy()
        if self.playback_q is not None:
            values = _joint_vector(self.playback_q, "playback_q")
            if values.size != self.observed_q.size:
                raise ValueError(
                    "playback_q must have the same joint dimension as observed_q"
                )
            self.playback_q = values


def _joint_vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.ndim != 1 or result.size < 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite joint vector")
    return result.copy()


def _put_latest(sample_queue: Any, value: Any) -> None:
    """Non-blocking bounded queue update; an old frame is disposable."""
    try:
        sample_queue.put_nowait(value)
        return
    except (queue.Full, OSError, ValueError):
        pass
    try:
        sample_queue.get_nowait()
    except (queue.Empty, OSError, ValueError):
        return
    try:
        sample_queue.put_nowait(value)
    except (queue.Full, OSError, ValueError):
        pass


class MujocoKinematicVisualizer:
    """Process-isolated latest-only visualizer.

    MuJoCo never steps dynamics here.  It only writes qpos and calls
    ``mj_forward`` for the displayed pose and scratch FK data.  Online
    packets display measured ``observed_q``; offline replay may set
    ``VisualizationPacket.playback_q`` to display a WM command instead.
    """

    def __init__(self, config: Any) -> None:
        if not bool(config.enabled):
            self.enabled = False
            return
        self.enabled = True
        self.config = config
        self._ctx = mp.get_context("spawn")
        self._queue = self._ctx.Queue(maxsize=1)
        self._process: mp.Process | None = None
        self._latest_observed = np.zeros(len(config.robot_joint_names), dtype=np.float64)
        self._latest_observed_timestamp = -np.inf
        self._latest_prediction: VisualizationPacket | None = None
        self._prediction_id = -1
        self._last_prediction_publish_s = -np.inf

    def start(self) -> None:
        if not self.enabled or self._process is not None:
            return
        self._process = self._ctx.Process(
            target=_visualizer_process,
            args=(self.config, self._queue),
            name="nero-mujoco-kinematic-visualizer",
            daemon=True,
        )
        self._process.start()

    def publish_observed(self, timestamp: float, observed_q: Any) -> None:
        if not self.enabled:
            return
        try:
            timestamp = float(timestamp)
            observed = _joint_vector(observed_q, "observed_q")
            if not np.isfinite(timestamp):
                raise ValueError("observed visualization timestamp must be finite")
            # Online callers use the monotonic runtime clock.  A delayed state
            # callback must not move the displayed robot back in time.
            if timestamp < self._latest_observed_timestamp:
                return
            self._latest_observed = observed
            self._latest_observed_timestamp = timestamp
            previous = self._latest_prediction
            packet = VisualizationPacket(
                timestamp,
                self._latest_observed,
                None if previous is None else previous.predicted_q,
                None if previous is None else previous.predicted_ee_position,
                None if previous is None else previous.prediction_id,
                None if previous is None else previous.playback_q,
            )
            _put_latest(self._queue, packet)
        except Exception:
            log.debug("dropping invalid observed visualization packet", exc_info=True)

    def publish_prediction(
        self,
        timestamp: float,
        observed_q: Any,
        predicted_q: Any,
        *,
        prediction_id: int | None = None,
        predicted_ee_position: Any | None = None,
        playback_q: Any | None = None,
    ) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        period = 1.0 / float(self.config.prediction_visualization_hz)
        if prediction_id is not None and prediction_id == self._prediction_id:
            return
        if now - self._last_prediction_publish_s < period:
            return
        try:
            timestamp = float(timestamp)
            if not np.isfinite(timestamp):
                raise ValueError("prediction visualization timestamp must be finite")
            predicted = np.asarray(predicted_q, dtype=np.float64)
            if self._latest_observed_timestamp == -np.inf:
                # This is only needed for offline callers that publish a
                # prediction before their first observed packet.
                self._latest_observed = _joint_vector(observed_q, "observed_q")
                self._latest_observed_timestamp = timestamp
            packet = VisualizationPacket(
                max(timestamp, self._latest_observed_timestamp),
                self._latest_observed.copy(),
                predicted,
                None if predicted_ee_position is None else np.asarray(predicted_ee_position, dtype=np.float64),
                prediction_id,
                None if playback_q is None else np.asarray(playback_q, dtype=np.float64),
            )
        except Exception:
            log.debug("dropping invalid prediction visualization packet", exc_info=True)
            return
        self._latest_prediction = packet
        self._prediction_id = -1 if prediction_id is None else int(prediction_id)
        self._last_prediction_publish_s = now
        _put_latest(self._queue, packet)

    def close(self) -> None:
        if not self.enabled:
            return
        _put_latest(self._queue, None)
        process = self._process
        self._process = None
        if process is not None:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        try:
            self._queue.close()
        except Exception:
            pass

    publish = publish_prediction
    publish_observation = publish_observed


class MujocoKinematicFK:
    """Synchronous FK helper used by tests and offline validation.

    ``display_data`` and ``scratch_data`` are intentionally separate; callers
    can update the observed pose without clobbering a prediction rollout.
    """

    def __init__(self, config: Any) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(config.mujoco_model_path).expanduser()))
        self.display_data = mujoco.MjData(self.model)
        self.scratch_data = mujoco.MjData(self.model)
        self.addresses, self.ee_id, self.ee_kind = _resolve_ids(mujoco, self.model, config)

    def set_observed_q(self, q: Any) -> np.ndarray:
        return _forward_position(self.mujoco, self.model, self.display_data, self.addresses, _joint_vector(q, "observed_q"), self.ee_id, self.ee_kind)

    def predicted_ee_positions(self, predicted_q: Any) -> np.ndarray:
        values = np.asarray(predicted_q, dtype=np.float64)
        if values.ndim != 3 or values.shape[-1] != self.addresses.size or not np.isfinite(values).all():
            raise ValueError(f"predicted_q must have shape [N,H,{self.addresses.size}]")
        return np.stack(
            [[_forward_position(self.mujoco, self.model, self.scratch_data, self.addresses, q, self.ee_id, self.ee_kind) for q in trajectory] for trajectory in values],
            axis=0,
        )


def _resolve_ids(mujoco: Any, model: Any, config: Any) -> tuple[np.ndarray, int, int]:
    addresses = []
    for name in config.robot_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if joint_id < 0:
            raise ValueError(f"MuJoCo model has no configured robot joint {name!r}")
        if int(model.jnt_type[joint_id]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError(f"visualization joint {name!r} must be hinge or slide")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    if len(set(addresses)) != len(addresses):
        raise ValueError("configured MuJoCo robot joints must map to unique qpos addresses")
    if config.ee_site_name:
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(config.ee_site_name))
        if ee_id < 0:
            raise ValueError(f"MuJoCo model has no configured ee site {config.ee_site_name!r}")
        return np.asarray(addresses, dtype=np.int64), ee_id, int(mujoco.mjtObj.mjOBJ_SITE)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(config.ee_body_name))
    if ee_id < 0:
        raise ValueError(f"MuJoCo model has no configured ee body {config.ee_body_name!r}")
    return np.asarray(addresses, dtype=np.int64), ee_id, int(mujoco.mjtObj.mjOBJ_BODY)


def _forward_position(mujoco: Any, model: Any, data: Any, addresses: np.ndarray, q: np.ndarray, ee_id: int, ee_kind: int) -> np.ndarray:
    if q.shape != addresses.shape:
        raise ValueError(f"q shape {q.shape} does not match configured joint mapping {addresses.shape}")
    data.qpos[addresses] = q
    mujoco.mj_forward(model, data)
    if ee_kind == int(mujoco.mjtObj.mjOBJ_SITE):
        return np.asarray(data.site_xpos[ee_id], dtype=np.float64).copy()
    return np.asarray(data.xpos[ee_id], dtype=np.float64).copy()


def _draw(viewer: Any, mujoco: Any, observed_position: np.ndarray, trajectories: np.ndarray, config: Any) -> None:
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        capacity = len(scene.geoms)
        identity = np.eye(3, dtype=np.float64).reshape(-1)
        # Use one shared translucent yellow for every predicted sample so the
        # cloud reads as a single WM prediction distribution.
        trajectory_color = np.asarray((1.0, 0.85, 0.05, 0.45), dtype=np.float32)
        for trajectory in trajectories:
            for step_index, position in enumerate(trajectory):
                if scene.ngeom >= capacity:
                    return
                rgba = trajectory_color
                size = np.full(3, float(config.point_size) * (1.0 - 0.35 * step_index / max(1, trajectory.shape[0] - 1)))
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, size, position, identity, rgba)
                scene.ngeom += 1
        if scene.ngeom < capacity:
            # Mark the current displayed TCP pose separately from the sampled
            # future points: a translucent red box remains easy to distinguish
            # while still showing the trajectory behind it.
            # Keep the current-observation marker visible when trajectory
            # points are configured very small.  ``point_size`` controls only
            # the predicted samples; the marker has a practical lower bound.
            marker_edge = max(float(config.point_size) * 2.5, 0.012)
            marker_size = np.full(3, marker_edge)
            marker_rgba = np.asarray((1.0, 0.05, 0.05, 0.55), dtype=np.float32)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_BOX,
                marker_size,
                observed_position,
                identity,
                marker_rgba,
            )
            scene.ngeom += 1


def _visualizer_process(config: Any, sample_queue: Any) -> None:
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(Path(config.mujoco_model_path).expanduser()))
        addresses, ee_id, ee_kind = _resolve_ids(mujoco, model, config)
        log.info("CARS-WM joint order -> MuJoCo qpos addresses: %s", dict(zip(config.robot_joint_names, addresses.tolist())))
        display_data = mujoco.MjData(model)
        scratch_data = mujoco.MjData(model)
        viewer = None
        if not bool(config.headless):
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(model, display_data)
        latest: VisualizationPacket | None = None
        latest_trajectory = np.empty((0, 0, 3), dtype=np.float64)
        computed_prediction_id: int | None = None
        latest_packet_timestamp = -np.inf
        timing_reported = False
        fk_timing_reported = False
        next_render = time.monotonic()
        while True:
            try:
                item = sample_queue.get(timeout=0.05)
                if item is None:
                    break
                if float(item.timestamp) < latest_packet_timestamp:
                    continue
                latest = item
                # Drain stale packets so rendering is always latest-only.
                while True:
                    try:
                        item = sample_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        return
                    if float(item.timestamp) < latest_packet_timestamp:
                        continue
                    latest = item
            except queue.Empty:
                pass
            if latest is None:
                continue
            latest_packet_timestamp = float(latest.timestamp)
            # ``playback_q`` is used only by offline simulation/replay.  The
            # real-time hardware path never sets it and therefore continues
            # to show the measured observed_q pose.
            display_q = latest.playback_q if latest.playback_q is not None else latest.observed_q
            _forward_position(mujoco, model, display_data, addresses, display_q, ee_id, ee_kind)
            if (
                latest.predicted_q is not None
                and latest.prediction_id is not None
                and latest.prediction_id != computed_prediction_id
            ):
                predicted = latest.predicted_q
                fk_started = time.perf_counter()
                # Always execute FK in this process from predicted q.  A packet
                # may carry a precomputed position for auditing, but it is not
                # used as a rendering shortcut: every q goes through the
                # independent scratch MjData and mj_forward contract.
                latest_trajectory = np.stack(
                    [[_forward_position(mujoco, model, scratch_data, addresses, q, ee_id, ee_kind) for q in trajectory] for trajectory in predicted],
                    axis=0,
                )
                latest.predicted_ee_position = latest_trajectory
                computed_prediction_id = latest.prediction_id
                fk_elapsed_ms = (time.perf_counter() - fk_started) * 1e3
                if not fk_timing_reported:
                    log.info("sampled futures FK: N=%s H_f=%s shape=%s elapsed_ms=%.3f", predicted.shape[0], predicted.shape[1], latest_trajectory.shape, fk_elapsed_ms)
                    fk_timing_reported = True
            if viewer is not None and time.monotonic() >= next_render:
                render_started = time.perf_counter()
                display_q = latest.playback_q if latest.playback_q is not None else latest.observed_q
                # Keep the reference marker tied to the measured/recorded
                # observation.  Offline replay may display a predicted
                # playback_q on the robot, but comparing the WM trajectory to
                # that moving pose would hide its offset from the true anchor.
                observed_position = _forward_position(
                    mujoco, model, scratch_data, addresses,
                    latest.observed_q, ee_id, ee_kind,
                )
                _draw(viewer, mujoco, observed_position, latest_trajectory, config)
                viewer.sync()
                render_elapsed_ms = (time.perf_counter() - render_started) * 1e3
                if not timing_reported:
                    log.info("MuJoCo render elapsed_ms=%.3f", render_elapsed_ms)
                    timing_reported = True
                next_render = time.monotonic() + 1.0 / float(config.render_fps)
                if not viewer.is_running():
                    break
    except Exception:
        log.exception("real-time kinematic visualization stopped")
    finally:
        if 'viewer' in locals() and viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass


MujocoRealtimeVisualizer = MujocoKinematicVisualizer

__all__ = ["MujocoKinematicFK", "MujocoKinematicVisualizer", "MujocoRealtimeVisualizer", "VisualizationPacket"]
