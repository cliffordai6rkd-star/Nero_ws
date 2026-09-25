from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from inference.pi0_wm.config import load_config
from inference.pi0_wm.runtime import Runtime
from inference.pi0_wm.core import Execution, Schedule, Result, Request


def runtime(stride):
    rt = Runtime.__new__(Runtime)
    rt.command_stride = stride
    rt.command_dt = stride / 100
    rt.last_command_step = None
    rt.dt = .01
    rt.command_enabled = True
    rt.simulated_arm = False
    rt.held = np.zeros(7)
    rt.state = SimpleNamespace(q=np.zeros(7), dq=np.zeros(7))
    rt.cfg = {'hardware': {'maximum_step_rad': [10]*7, 'q_min': [-100]*7, 'q_max': [100]*7}}
    rt.visualizer = SimpleNamespace(update=lambda *a: None)
    rt.execution = Execution(Schedule(10, 0))
    values = np.repeat(np.arange(20)[None, :, None], 7, axis=2)
    rt.execution.receive(Result(Request(0, 0, (), None, 0), {'q': values, 'tau': values+100}, 0))
    rt.execution.take_over(0)
    return rt


@pytest.mark.parametrize('stride', [1, 2, 4, 5])
def test_command_rate_selects_current_prediction_without_slowing_time(stride):
    rt = runtime(stride)
    sent = []
    rt.arm = SimpleNamespace(command_joint_positions=lambda q: sent.append(q.copy()))
    for step in range(10):
        rt.step = step
        old = rt.held.copy()
        rt.send(rt.execution.command(step))
        if step % stride:
            np.testing.assert_array_equal(rt.held, old)
        else:
            np.testing.assert_array_equal(rt.held, np.full(7, step))
        rt.execution.advance()
    assert [q[0] for q in sent] == list(range(0, 10, stride))
    assert rt.execution.wm_execute_step == 10


def test_mtc_uses_send_interval_and_synchronized_latest_tau(monkeypatch):
    import inference.pi0_wm.runtime as module
    clock = [0.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    rt = runtime(2)
    rt.mtc_active = True
    rt.mtc_sent_at = None
    prepared, sent = [], []
    def prepare(q, tau, state, held, dt):
        prepared.append((q[0], tau[0], dt))
        return SimpleNamespace(q=q, velocity=np.zeros(7), kp=np.ones(7), kd=np.ones(7), feedforward=tau)
    rt.mtc = SimpleNamespace(cfg={'watchdog_timeout_s': .05}, prepare=prepare, commit=lambda ref: None)
    rt.arm = SimpleNamespace(command_joint_impedance=lambda *args: sent.append(args))
    for step in range(6):
        rt.step = step
        clock[0] = step*.01
        rt.send(rt.execution.command(step), rt.execution.torque(step))
    np.testing.assert_allclose(prepared, [(0,100,.02), (2,102,.02), (4,104,.02)])
    assert len(sent) == 3
    rt.step = 5  # Still an unscheduled step, but a wall-clock stall must trip.
    clock[0] = .11
    with pytest.raises(RuntimeError, match='watchdog'):
        rt.send(None)


def test_failed_send_does_not_advance_rate_schedule():
    rt = runtime(2)
    rt.step = 0
    def fail(q):
        raise RuntimeError('CAN failed')
    rt.arm = SimpleNamespace(command_joint_positions=fail)
    with pytest.raises(RuntimeError):
        rt.send(np.ones(7))
    assert rt.last_command_step is None
    np.testing.assert_array_equal(rt.held, np.zeros(7))


@pytest.mark.parametrize('rate', [0, -1, 30, 101, True])
def test_invalid_command_rate_rejected(tmp_path, rate):
    c = load_config(Path(__file__).parents[1]/'inference/configs/pi0_wm.yaml')
    c['control']['mode'] = 'q'
    c['control']['command_hz'] = rate
    path=tmp_path/'rate.yaml'
    path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match='command_hz'):
        load_config(path)


def test_mtc_watchdog_must_allow_command_interval(tmp_path):
    c = load_config(Path(__file__).parents[1]/'inference/configs/pi0_wm.yaml')
    c['control']['mode'] = 'mtc'
    c['wm']['enable'] = True
    c['control']['command_hz'] = 10
    c['control']['mtc']['watchdog_timeout_s'] = .05
    path=tmp_path/'rate.yaml'
    path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match='watchdog_timeout_s'):
        load_config(path)
