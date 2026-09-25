from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from inference.pi0_wm.config import load_config
from inference.pi0_wm.mtc import MtcController, validate_mtc
from inference.pi0_wm.runtime import Runtime


@pytest.fixture
def controller():
    raw = yaml.safe_load((Path(__file__).parents[1] / 'inference/configs/pi0_wm.yaml').read_text())
    cfg = raw['control']['mtc']
    cfg.update(kp=4., kd=.5, tau_scale=.5, maximum_tracking_error_rad=.25)
    validate_mtc(cfg)
    c = MtcController(cfg, raw['hardware'], lambda q: np.ones(7), torque_only=True)
    c.reset(np.zeros(7), np.ones(7))
    c.transition_time_s = cfg['startup_blend_s']
    return c


def test_host_pd_with_zero_motor_gains_and_synchronized_tau(controller):
    c = controller
    q = np.zeros(7)
    state = SimpleNamespace(q=np.full(7, -.02), dq=np.full(7, .1))
    out = c.prepare(q, np.full(7, 1.2), state, q, .01)
    # .08 position + -.05 damping + 1 gravity + .1 learned residual.
    np.testing.assert_allclose(out.feedforward, 1.13)
    np.testing.assert_array_equal(out.kp, np.zeros(7))
    np.testing.assert_array_equal(out.kd, np.zeros(7))


def test_hold_still_corrects_position_and_velocity(controller):
    state = SimpleNamespace(q=np.full(7, .02), dq=np.full(7, .1))
    out = controller.prepare(None, None, state, np.zeros(7), .01)
    np.testing.assert_allclose(out.feedforward, .87)
    np.testing.assert_array_equal(out.q, np.zeros(7))


def test_limits_apply_to_total_torque_including_pd(controller):
    c = controller
    c.cfg['kp'] = [500.] * 7
    c.cfg['total_torque_limit_nm'] = [2.] * 7
    state = SimpleNamespace(q=np.full(7, -.2), dq=np.zeros(7))
    for _ in range(30):
        out = c.prepare(None, None, state, np.zeros(7), .01)
        assert np.all(np.abs(out.feedforward) <= 2.)
        assert np.all(np.abs(out.feedforward - c.feedforward) <= .2 + 1e-12)
        c.commit(out)
    np.testing.assert_allclose(out.feedforward, 2.)
    state.q *= -1
    reverse = c.prepare(None, None, state, np.zeros(7), .01)
    np.testing.assert_allclose(reverse.feedforward, 1.8)


def test_tracking_error_and_nonfinite_feedback_rejected(controller):
    state = SimpleNamespace(q=np.full(7, .3), dq=np.zeros(7))
    with pytest.raises(RuntimeError, match='tracking error'):
        controller.prepare(None, None, state, np.zeros(7), .01)
    state.q[:] = 0
    state.dq[:] = np.nan
    with pytest.raises(ValueError, match='measured dq'):
        controller.prepare(None, None, state, np.zeros(7), .01)


def test_tau_startup_blends_complete_torque(controller):
    c = controller
    c.reset(np.zeros(7), np.full(7, -.5))
    state = SimpleNamespace(q=np.full(7, -.02), dq=np.full(7, .1))
    out = c.prepare(None, None, state, np.zeros(7), .01)
    np.testing.assert_allclose(out.feedforward, -.5 + .01 * (1.03 + .5))
    assert c.transition_time_s == 0


@pytest.mark.parametrize('dry_run', [False, True])
def test_runtime_sends_only_tau_and_keeps_internal_reference(controller, dry_run):
    rt = Runtime.__new__(Runtime)
    rt.control_mode = 'tau'
    rt.mtc = controller
    rt.mtc_active = True
    rt.mtc_sent_at = None
    rt.command_enabled = not dry_run
    rt.simulated_arm = False
    rt.held = np.full(7, .1)
    rt.state = SimpleNamespace(q=rt.held.copy(), dq=np.zeros(7))
    rt.dt = .01
    rt.step = 0
    rt.execution = None
    rt.visualizer = SimpleNamespace(update=lambda *a: None)
    sent = []
    rt.arm = SimpleNamespace(command_joint_impedance=lambda *args: sent.append(args))
    old_time = rt.mtc.transition_time_s
    rt.send(None)
    np.testing.assert_array_equal(rt.held, np.full(7, .1))
    if dry_run:
        assert sent == [] and rt.mtc.transition_time_s == old_time
    else:
        assert len(sent) == 1
        for field in sent[0][:4]:
            np.testing.assert_array_equal(field, np.zeros(7))
        np.testing.assert_allclose(sent[0][4], 1.)


@pytest.mark.parametrize('enabled,rate,error', [(True,100,None), (False,100,'requires wm.enable'),
                                               (True,50,'command_hz=100')])
def test_tau_config(tmp_path, enabled, rate, error):
    source = Path(__file__).parents[1] / 'inference/configs/pi0_wm.yaml'
    cfg = yaml.safe_load(source.read_text())
    cfg['control'].update(mode='tau', command_hz=rate)
    cfg['wm']['enable'] = enabled
    path = tmp_path / 'tau.yaml'
    path.write_text(yaml.safe_dump(cfg))
    if error:
        with pytest.raises(ValueError, match=error):
            load_config(path)
    else:
        assert load_config(path)['control']['mode'] == 'tau'
