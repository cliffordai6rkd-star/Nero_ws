from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from inference.pi0_wm.config import load_config
from inference.pi0_wm.core import Plans
from inference.pi0_wm.ik import PoseIK
from inference.pi0_wm.runtime import Runtime


@pytest.fixture
def config():
    config = load_config(Path(__file__).parents[1] / 'inference/configs/pi0_wm.yaml')
    config['control']['mode'] = 'q'
    return config


def test_disabled_wm_needs_no_checkpoint_and_is_not_constructed(config, tmp_path, monkeypatch):
    import inference.pi0_wm.runtime as module
    config['wm'] = {'enable': False}
    config['mujoco']['enabled'] = False
    path = tmp_path / 'pure.yaml'
    path.write_text(yaml.safe_dump(config))
    parsed = load_config(path)
    monkeypatch.setattr(module, 'WMAdapter', lambda *a: pytest.fail('WM loaded'))
    monkeypatch.setattr(module, 'MockWM', lambda *a: pytest.fail('mock WM loaded'))
    rt = Runtime(parsed, mock=True)
    assert rt.ik is not None and rt.wm_worker is None
    seed = np.asarray(config['hardware']['endpoint']['rest_q'])
    output = rt.infer_pi(({}, seed))
    assert output.shape == (50, 7)
    np.testing.assert_allclose(output, np.tile(seed, (50, 1)), atol=1e-6)


def test_real_ik_roundtrip_and_continuous_chunk(config):
    ik = PoseIK(config['mujoco'], config['hardware'])
    seed = np.asarray(config['hardware']['endpoint']['rest_q'])
    qs = [seed + np.array([.01*i, .002*i, 0, -.002*i, 0, 0, -.002*i]) for i in range(1, 6)]
    targets = np.stack([ik.pose(q) for q in qs])
    solved = ik.chunk(targets, seed)
    for target, q in zip(targets, solved):
        actual = ik.pose(q)
        assert np.linalg.norm(actual[:3] - target[:3]) <= 1e-4
        error = Rotation.from_quat(actual[3:]).inv() * Rotation.from_quat(target[3:])
        assert error.magnitude() <= 1e-3
        assert np.all(q >= ik.low) and np.all(q <= ik.high)


def test_unreachable_pose_fails_without_returning_joint_command(config):
    ik = PoseIK(config['mujoco'], config['hardware'])
    with pytest.raises(RuntimeError, match='did not converge'):
        ik.solve([10, 10, 10, 0, 0, 0, 1], config['hardware']['endpoint']['rest_q'])


def test_direct_control_uses_time_aligned_q_and_holds_missing_actions():
    rt = Runtime.__new__(Runtime)
    rt.plans = Plans(20)
    rt.direct_token = None
    rt.direct_target = None
    q = np.repeat(np.arange(50, dtype=float)[:, None], 7, axis=1)
    rt.plans.add(q, 20, 0, 40)
    for step in range(80, 84):
        rt.step = step
        np.testing.assert_array_equal(rt.direct_command(), np.full(7, 10))
    rt.step = 84
    np.testing.assert_array_equal(rt.direct_command(), np.full(7, 11))
    rt.step = 160
    assert rt.direct_command() is None


def test_pure_pi_calibration_never_uses_wm(config, monkeypatch):
    config['wm']['enable'] = False
    config['mujoco']['enabled'] = False
    rt = Runtime(config, mock=True)
    rt.pi_snapshot = lambda: {}
    rt.history.append(0, SimpleNamespace(q=np.zeros(7), dq=np.zeros(7), torque=np.zeros(7)), np.zeros(7), 1.)
    rt.pi_worker = object()
    rt.submit_pi = lambda: None
    rt.log_pi_chunk = lambda: None
    rt.cfg['calibration'].update(warmup_samples=1, minimum_samples=1, pi_seconds=.001)
    rt.await_result = lambda worker: SimpleNamespace(value=np.zeros((50, 7)))
    rt.calibrate()
    assert rt.wm_worker is None
    assert rt.plans.consume <= 50
