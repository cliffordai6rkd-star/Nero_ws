"""Official openpi client; the server owns training transforms and normalization."""
from __future__ import annotations

import logging
import numpy as np
from scipy.spatial.transform import Rotation

log = logging.getLogger(__name__)


def ee_pose(matrix):
    matrix = np.asarray(matrix)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('hardware EE pose must be a finite base-to-link7 matrix')
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    if quat[3] < 0:
        quat *= -1
    return np.r_[matrix[:3, 3], quat].astype(np.float32)


def put_nested(target, path, value):
    keys = path.split('/')
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


class PiClient:
    def __init__(self, config):
        self.config = config
        self.policy = None

    def infer(self, observation):
        # Both connection and blocking infer belong to the permanent pi worker.
        if self.policy is None:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
            self.policy = WebsocketClientPolicy(host=self.config['host'], port=self.config['port'])
            actual = self.policy.get_server_metadata().get('pi0_wm')
            expected = self.config['interface']
            if actual != expected:
                raise ValueError(f'pi0 server training interface mismatch: actual={actual}, expected={expected}')
            log.info('pi0 server interface verified: %s', actual)
        response = self.policy.infer(observation)
        actions = np.asarray(response[self.config['output_key']], dtype=np.float32).copy()
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError('server must return post-transform physical absolute EE actions [T,7]')
        # A raw model latent/padded output must never accidentally reach WM.
        norms = np.linalg.norm(actions[:, 3:], axis=-1)
        if np.any(np.abs(norms - 1) > self.config['quaternion_tolerance']):
            raise ValueError('pi0 quaternion norms outside physical pose tolerance; check output transform')
        actions[:, 3:] /= norms[:, None]
        actions[actions[:, 6] < 0, 3:] *= -1
        return actions


def observation(config, state, frames):
    result = {}
    put_nested(result, config['interface']['state_key'], ee_pose(state.ee_pose))
    put_nested(result, config['interface']['prompt_key'], config['prompt'])
    for name, key in config['interface']['images'].items():
        frame = frames[name].frame
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError(f'camera {name} must supply uint8 HWC RGB')
        put_nested(result, key, frame.copy())
    return result
