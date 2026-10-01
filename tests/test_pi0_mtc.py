from pathlib import Path
from types import SimpleNamespace
import copy

import numpy as np
import pytest
import yaml

from inference.pi0_wm.config import load_config
from inference.pi0_wm.core import Execution, Schedule, Request, Result
from inference.pi0_wm.mtc import MtcController, validate_mtc
from inference.pi0_wm.runtime import Runtime


@pytest.fixture
def config():
    c = load_config(Path(__file__).parents[1] / 'inference/configs/pi0_wm.yaml')
    validate_mtc(c['control']['mtc'])
    return c


def controller(config):
    c = MtcController(config['control']['mtc'], config['hardware'], lambda q: np.ones(7))
    c.reset(np.zeros(7))
    # Most tests exercise steady-state behavior; startup has dedicated tests.
    c.feedforward = np.ones(7)
    c.transition_time_s = c.cfg['startup_blend_s']
    return c


def test_synchronized_q_tau_selection_and_expiry():
    ex = Execution(Schedule(2, 1), selected_sample=1)
    q = np.arange(2*8*7).reshape(2, 8, 7)
    tau = q + 1000
    ex.receive(Result(Request(0, 10, (), None, 0), {'q': q, 'tau': tau}, .01))
    assert ex.take_over(12)
    np.testing.assert_array_equal(ex.command(12), q[1, 2])
    np.testing.assert_array_equal(ex.torque(12), tau[1, 2])
    assert ex.command(18) is None and ex.torque(18) is None
    ex.current.value.pop('tau')
    with pytest.raises(ValueError, match='requires WM tau'):
        ex.torque(12)


def test_gravity_is_not_double_counted_and_prediction_weight_is_applied(config):
    c = controller(config)
    c.cfg['tau_scale'] = [1.] * 7
    state = SimpleNamespace(q=np.zeros(7), dq=np.zeros(7))
    out = c.prepare(np.zeros(7), np.full(7, 1.1), state, np.zeros(7), .01)
    np.testing.assert_allclose(out.feedforward, 1.1)
    c.cfg['tau_scale'] = [0.] * 7
    out = c.prepare(np.zeros(7), np.full(7, 10), state, np.zeros(7), .01)
    np.testing.assert_allclose(out.feedforward, 1.)


def test_limits_total_torque_and_reference_transition(config):
    c = controller(config)
    c.cfg['kp'] = [500.] * 7
    state = SimpleNamespace(q=np.zeros(7), dq=np.full(7, -10.))
    previous = np.zeros(7)
    old_v = np.zeros(7)
    old_ff = c.feedforward.copy()
    for _ in range(10):
        out = c.prepare(np.full(7, .3), np.full(7, 100.), state, previous, .01)
        total = out.kp * (out.q-state.q) + out.kd * (out.velocity-state.dq) + out.feedforward
        assert np.all(np.abs(total) <= np.asarray(c.cfg['total_torque_limit_nm']) + 1e-9)
        assert np.all(np.abs(out.feedforward-old_ff) <= .2 + 1e-9)
        assert np.all(np.abs(out.velocity-old_v) <= .02 + 1e-9)
        c.commit(out)
        previous, old_v, old_ff = out.q, out.velocity, out.feedforward


def test_expired_reference_returns_to_gravity_and_watchdog_fails(config):
    c = controller(config)
    c.feedforward = np.full(7, 2.)
    state = SimpleNamespace(q=np.zeros(7), dq=np.zeros(7))
    out = c.prepare(None, None, state, np.zeros(7), .01)
    np.testing.assert_allclose(out.feedforward, 1.8)
    np.testing.assert_array_equal(out.velocity, np.zeros(7))
    with pytest.raises(RuntimeError, match='watchdog'):
        c.prepare(None, None, state, np.zeros(7), .1)
    with pytest.raises(ValueError, match='predicted tau'):
        c.prepare(np.zeros(7), np.full(7, np.nan), state, np.zeros(7), .01)


def test_pure_pi0_mit_tracks_q_without_wm_tau(config):
    c = MtcController(config['control']['mtc'], config['hardware'], lambda q: np.ones(7),
                      require_tau=False)
    c.reset(np.zeros(7))
    c.feedforward = np.ones(7)
    c.transition_time_s = c.cfg['startup_blend_s']
    state = SimpleNamespace(q=np.zeros(7), dq=np.zeros(7))
    out = c.prepare(np.full(7, .1), None, state, np.zeros(7), .01)
    assert np.all(out.q > 0)
    np.testing.assert_allclose(out.feedforward, np.ones(7))
    np.testing.assert_array_equal(out.kp, np.asarray(config['control']['mtc']['kp']))
    np.testing.assert_array_equal(out.kd, np.asarray(config['control']['mtc']['kd']))


def test_config_requires_wm_and_respects_per_joint_torque_limit(config, tmp_path):
    config['control']['mode'] = 'mtc'
    config['wm']['enable'] = False
    path = tmp_path/'mtc.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='requires wm.enable'):
        load_config(path)
    c = copy.deepcopy(config['control']['mtc'])
    c['total_torque_limit_nm'][-1] = 9
    with pytest.raises(ValueError, match='Nero joint limits'):
        validate_mtc(c)


def test_send_commits_only_after_success_and_restores_mode(config):
    rt = Runtime.__new__(Runtime)
    rt.mtc = controller(config)
    rt.control_mode = 'mtc'
    rt.mtc_active = True
    rt.mtc_sent_at = None
    rt.dt = .01
    rt.command_enabled = True
    rt.simulated_arm = False
    rt.held = np.zeros(7)
    rt.state = SimpleNamespace(q=np.zeros(7), dq=np.zeros(7), torque=np.zeros(7))
    def fail(*args):
        raise RuntimeError('CAN failed')
    moves = []
    rt.arm = SimpleNamespace(command_joint_impedance=fail, move_joints=lambda q: moves.append(q.copy()))
    rt.visualizer = SimpleNamespace(update=lambda *a: None)
    rt.step = 0
    rt.execution = None
    old_ff = rt.mtc.feedforward.copy()
    with pytest.raises(RuntimeError, match='CAN failed'):
        rt.send(np.full(7,.01), np.full(7,2.))
    np.testing.assert_array_equal(rt.held, np.zeros(7))
    np.testing.assert_array_equal(rt.mtc.feedforward, old_ff)
    rt.leave_mtc()
    assert not rt.mtc_active and len(moves) == 1


def test_mtc_dry_run_does_not_send(config):
    rt = Runtime.__new__(Runtime)
    rt.mtc = controller(config)
    rt.control_mode = 'mtc'
    rt.mtc_active = False
    rt.mtc_sent_at = None
    rt.command_enabled = False
    rt.simulated_arm = False
    rt.held = np.zeros(7)
    rt.state = SimpleNamespace(q=np.zeros(7), dq=np.zeros(7), torque=np.zeros(7))
    rt.dt = .01
    rt.step = 0
    rt.execution = None
    rt.visualizer = SimpleNamespace(update=lambda *a: None)
    rt.arm = SimpleNamespace()  # No hardware command method may be used.
    rt.enter_mtc()
    rt.leave_mtc()
    np.testing.assert_array_equal(rt.held, np.zeros(7))


def test_fixed_target_never_overshoots_and_converges(config):
    c = controller(config)
    held = np.zeros(7)
    target = np.array([.02, -.02, .001, -.001, .3, -.3, 0.])
    for _ in range(500):
        previous = held.copy()
        out = c.prepare(target, np.ones(7),
                        SimpleNamespace(q=held, dq=c.velocity), held, .02)
        assert np.all(out.q >= np.minimum(previous, target) - 1e-12)
        assert np.all(out.q <= np.maximum(previous, target) + 1e-12)
        c.commit(out)
        held = out.q
    np.testing.assert_allclose(held, target, atol=1e-9)


def test_target_reversal_and_hold_do_not_continue_old_motion(config):
    c = controller(config)
    held = np.zeros(7)
    state = SimpleNamespace(q=held, dq=np.zeros(7))
    for _ in range(10):
        out = c.prepare(np.full(7, .3), np.ones(7), state, held, .02)
        c.commit(out)
        held = out.q
    target = held - .001
    out = c.prepare(target, np.ones(7), state, held, .02)
    assert np.all(out.q <= held) and np.all(out.q >= target)
    c.commit(out)
    stopped = c.prepare(None, None, state, out.q, .02)
    np.testing.assert_array_equal(stopped.q, out.q)
    np.testing.assert_array_equal(stopped.velocity, np.zeros(7))


def test_startup_blends_from_measured_torque_and_commits_only_on_success(config):
    c = controller(config)
    c.reset(np.zeros(7), np.full(7, -2.))
    state = SimpleNamespace(q=np.zeros(7), dq=np.zeros(7))
    first = c.prepare(None, None, state, state.q, .02)
    np.testing.assert_allclose(first.feedforward, -1.94)
    np.testing.assert_array_equal(c.feedforward, np.full(7, -2.))
    assert c.transition_time_s == 0
    for _ in range(50):
        out = c.prepare(None, None, state, state.q, .02)
        assert np.all(np.abs(out.feedforward - c.feedforward) <= .4 + 1e-12)
        c.commit(out)
    np.testing.assert_allclose(c.feedforward, 1.)
    with pytest.raises(ValueError, match='initial torque'):
        c.reset(state.q, np.full(7, np.nan))


def test_enter_mtc_anchors_to_feedback_before_switching(config):
    rt = Runtime.__new__(Runtime)
    rt.control_mode = 'mtc'
    rt.mtc = controller(config)
    rt.command_enabled = True
    rt.simulated_arm = False
    rt.held = np.full(7, .1)  # Position mode has not reached its last target.
    rt.state = SimpleNamespace(q=np.zeros(7), torque=np.full(7, -.5))
    events = []
    rt.arm = SimpleNamespace(configure_joint_impedance_mode=lambda: events.append('switch'))
    rt.send = lambda target: events.append((rt.held.copy(), rt.mtc.feedforward.copy()))
    rt.enter_mtc()
    assert events[0] == 'switch'
    np.testing.assert_array_equal(events[1][0], rt.state.q)
    np.testing.assert_array_equal(events[1][1], rt.state.torque)
    events.clear()
    rt.state.torque[:] = np.nan
    with pytest.raises(ValueError, match='initial torque'):
        rt.enter_mtc()
    assert events == []


@pytest.mark.parametrize('duration', [0, -1, float('nan'), True])
def test_invalid_startup_blend_rejected(config, duration):
    config['control']['mtc']['startup_blend_s'] = duration
    with pytest.raises(ValueError, match='startup_blend_s'):
        validate_mtc(config['control']['mtc'])


@pytest.mark.parametrize('stop_key', [True, False])
@pytest.mark.parametrize('mode', ['mtc', 'tau'])
def test_mtc_lifecycle_restores_position_before_reset_or_error_exit(config, monkeypatch, stop_key, mode):
    import inference.pi0_wm.runtime as module
    import nero_collection.inverse_dynamics as dynamics
    config['control']['mode'] = mode
    config['control']['command_hz'] = 100 if mode == 'tau' else 50
    config['wm']['enable'] = True
    config['mujoco']['enabled'] = False
    monkeypatch.setattr(dynamics, 'PinocchioJointTorqueResidualEstimator',
                        lambda cfg: SimpleNamespace(gravity_torque=lambda q: np.zeros(7)))
    rt = Runtime(config, mock=True)
    rt.cameras = SimpleNamespace(start=lambda: None, stop=lambda: None, poll=lambda: [])
    events = []
    monkeypatch.setattr(module, 'reset_to_rest', lambda rt: events.append('reset'))
    monkeypatch.setattr(module, 'Worker', lambda *args: SimpleNamespace(close=lambda: None))
    rt.arm.configure_joint_impedance_mode = lambda: events.append('mit_mode')
    rt.arm.command_joint_impedance = lambda *args: events.append('mit_command')
    rt.arm.move_joints = lambda q: events.append('position_mode')
    def interrupt():
        if stop_key:
            raise module.StopAndReset()
        raise RuntimeError('inference fault')
    rt.calibrate = interrupt
    if stop_key:
        assert rt._run()['stopped_by'] == 'i'
        assert events == ['reset', 'mit_mode', 'mit_command', 'position_mode', 'reset']
    else:
        with pytest.raises(RuntimeError, match='inference fault'):
            rt._run()
        assert events == ['reset', 'mit_mode', 'mit_command', 'position_mode']
