"""Full latest/executing futures with IDs, using the existing kinematic helpers."""
from __future__ import annotations

import logging
import queue
import time
from types import SimpleNamespace
import numpy as np

from inference.mujoco_visualization import MujocoKinematicVisualizer, MujocoKinematicFK, _put_latest, _draw

log = logging.getLogger(__name__)


class Visualizer(MujocoKinematicVisualizer):
    def __init__(self, config):
        super().__init__(SimpleNamespace(**config))

    def start(self):
        if not self.enabled or self._process is not None:
            return
        ready = self._ctx.Queue(maxsize=1)
        self._process = self._ctx.Process(target=_process, args=(self.config, self._queue, ready), daemon=True)
        self._process.start()
        try:
            error = ready.get(timeout=15)
            if error is not None:
                raise RuntimeError(f'MuJoCo initialization failed: {error}')
        except queue.Empty as exc:
            raise RuntimeError('MuJoCo initialization timed out before calibration') from exc
        finally:
            ready.close()

    def update(self, step, q, execution):
        if not self.enabled:
            return
        def pack(result):
            if result is None:
                return None
            return {'request_id': result.request.request_id, 'anchor': result.request.anchor,
                    'plans': result.request.plan_versions, 'q': result.value['q']}
        _put_latest(self._queue, {'step': step, 'q': np.asarray(q).copy(),
                                 'latest': pack(execution.latest), 'active': pack(execution.current),
                                 'wm_loop_id': execution.wm_loop_id, 'sample': execution.selected_sample})


def _marker(viewer, mj, position, size, color, label=''):
    scene = viewer.user_scn
    if scene.ngeom >= len(scene.geoms):
        return
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_SPHERE, np.full(3, size), position,
                   np.eye(3).reshape(-1), np.asarray(color, dtype=np.float32))
    geom.label = label
    scene.ngeom += 1


def _process(config, samples, ready):
    viewer = None
    try:
        fk = MujocoKinematicFK(config)
        mj = fk.mujoco
        if not config.headless:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(fk.model, fk.display_data)
        ready.put(None)
        cached = {}
        next_render = 0.0
        while True:
            try:
                packet = samples.get(timeout=0.1)
            except queue.Empty:
                continue
            if packet is None:
                return
            while True:
                try:
                    newer = samples.get_nowait()
                    if newer is None:
                        return
                    packet = newer
                except queue.Empty:
                    break
            observed = fk.set_observed_q(packet['q'])
            valid_ids = set()
            for key in ('latest', 'active'):
                result = packet[key]
                if result is not None:
                    ident = result['request_id']
                    valid_ids.add(ident)
                    if ident not in cached:
                        cached[ident] = fk.predicted_ee_positions(result['q'])
            cached = {k: v for k, v in cached.items() if k in valid_ids}
            if viewer is None or time.monotonic() < next_render:
                continue
            latest = packet['latest']
            trajectories = np.empty((0, 0, 3)) if latest is None else cached[latest['request_id']]
            _draw(viewer, mj, observed, trajectories, config)
            with viewer.lock():
                for key, color in [('latest', (0.2, 0.6, 1, 0.8)), ('active', (0.2, 1, 0.3, 0.85))]:
                    result = packet[key]
                    if result is None:
                        continue
                    trajectory = cached[result['request_id']][packet['sample']]
                    for point in trajectory:
                        _marker(viewer, mj, point, config.point_size, color)
                    label = f"{key} req={result['request_id']} sample={packet['sample']} plans={result['plans']}"
                    _marker(viewer, mj, trajectory[0], config.point_size * 1.5, color, label)
                active = packet['active']
                if active is not None:
                    index = packet['step'] - active['anchor']
                    trajectory = cached[active['request_id']][packet['sample']]
                    if 0 <= index < len(trajectory):
                        _marker(viewer, mj, trajectory[index], 0.008, (1, 0.8, 0, 1),
                                f"execute loop={packet['wm_loop_id']} point={index}")
            viewer.sync()
            next_render = time.monotonic() + 1 / config.render_fps
            if not viewer.is_running():
                return
    except Exception as exc:
        try:
            ready.put_nowait(str(exc))
        except queue.Full:
            pass
        log.exception('pi0_wm kinematic visualization stopped')
    finally:
        if viewer is not None:
            viewer.close()
