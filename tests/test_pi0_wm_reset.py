from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from inference.pi0_wm.config import load_config
from inference.pi0_wm.reset import reset_to_rest
from inference.pi0_wm.runtime import Runtime, StopAndReset


def fake_runtime(monkeypatch, *, follows=True):
    import inference.pi0_wm.reset as module
    clock = [0.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(module.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0] + dt))
    cfg = load_config(Path(__file__).parents[1] / 'inference/configs/pi0_wm.yaml')
    commands = []
    rt = SimpleNamespace(cfg=cfg, command_enabled=True, simulated_arm=False,
                         state=SimpleNamespace(q=np.zeros(7)), held=np.zeros(7))
    def move(q):
        commands.append(q.copy())
        if follows:
            rt.state.q = q.copy()
    rt.arm = SimpleNamespace(move_joints=move, wait_motion_done=lambda timeout: True)
    rt.read_hardware_state = lambda: None
    return rt, commands


def test_reset_interpolates_to_rest_and_checks_feedback(monkeypatch):
    rt, commands = fake_runtime(monkeypatch)
    reset_to_rest(rt)
    np.testing.assert_allclose(rt.state.q, rt.cfg['hardware']['endpoint']['rest_q'])
    assert np.max(np.abs(np.diff(np.vstack([np.zeros(7), *commands]), axis=0))) <= .05
    np.testing.assert_array_equal(rt.held, commands[-1])


def test_dry_run_does_not_move(monkeypatch):
    rt, commands = fake_runtime(monkeypatch)
    rt.command_enabled = False
    reset_to_rest(rt)
    assert not commands


def test_reset_rejects_failure_to_reach_rest(monkeypatch):
    rt, commands = fake_runtime(monkeypatch, follows=False)
    with pytest.raises(RuntimeError, match='self-check failed'):
        reset_to_rest(rt)
    assert commands


def test_failed_reset_send_does_not_update_held(monkeypatch):
    rt, _ = fake_runtime(monkeypatch)
    def fail(q):
        raise RuntimeError('CAN failed')
    rt.arm.move_joints = fail
    with pytest.raises(RuntimeError, match='CAN failed'):
        reset_to_rest(rt)
    np.testing.assert_array_equal(rt.held, np.zeros(7))


def test_i_interrupts_calibration_wait_before_another_hold_command():
    rt = Runtime.__new__(Runtime)
    rt.keys = SimpleNamespace(read_key=lambda timeout: 'i')
    rt.send = lambda q: pytest.fail('inference command after i')
    with pytest.raises(StopAndReset):
        rt.pump_hold()


@pytest.mark.parametrize('wm_enabled', [True, False])
def test_startup_reset_precedes_calibration_and_i_resets_then_exits(monkeypatch, wm_enabled):
    import inference.pi0_wm.runtime as module
    rt = Runtime.__new__(Runtime)
    rt.cfg = load_config(Path(__file__).parents[1] / 'inference/configs/pi0_wm.yaml')
    rt.mock = False
    rt.control_mode = 'q'
    rt.wm_enabled = wm_enabled
    rt.command_enabled = True
    rt.simulated_arm = False
    rt.hz = 100
    rt.step = 0
    rt.held = np.zeros(7)
    rt.state = SimpleNamespace(q=np.zeros(7))
    rt.pi_worker = rt.wm_worker = None
    events = []
    rt.arm = SimpleNamespace(connect=lambda: events.append('connect'),
        set_follower_mode=lambda: None, enable=lambda: events.append('enable'),
        command_joint_positions=lambda q: None, disconnect=lambda: events.append('disconnect'))
    rt.cameras = SimpleNamespace(start=lambda: None, stop=lambda: None)
    rt.visualizer = SimpleNamespace(start=lambda: None, close=lambda: None)
    rt.wm = SimpleNamespace(history_horizon=2, operations={}, infer=lambda _: None)
    rt.pi = SimpleNamespace(infer=lambda _: None)
    rt.acquire = lambda: None
    rt.send = lambda q: None
    def calibrate():
        events.append('calibrate')
        raise StopAndReset()
    rt.calibrate = calibrate
    monkeypatch.setattr(module, 'reset_to_rest', lambda runtime: events.append('reset'))
    monkeypatch.setattr(module, 'Worker', lambda *args: SimpleNamespace(close=lambda: events.append('worker_closed')))
    result = rt._run()
    assert result == {'stopped_by': 'i', 'reset_to_rest': True}
    assert events == (['connect', 'enable', 'reset', 'calibrate']
                      + ['worker_closed'] * (2 if wm_enabled else 1) + ['reset', 'disconnect'])
