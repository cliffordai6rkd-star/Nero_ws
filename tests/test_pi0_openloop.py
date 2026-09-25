from pathlib import Path
import numpy as np
import pytest
import yaml

from inference.pi0_wm.core import Execution, Schedule, Request, Result
from inference.pi0_wm.config import load_config


def prediction(anchor, offset=0, horizon=6):
    q = np.repeat((np.arange(horizon)+offset)[None, :, None], 7, axis=2).astype(float)
    return Result(Request(offset, anchor, (), None, 0), {'q': q, 'tau': q+100}, .5)


def test_openloop_replays_from_zero_then_holds_without_prefetch_or_tail():
    ex = Execution(Schedule(3, 2), open_loop=True)
    assert ex.should_request()
    ex.requested = True
    assert ex.command(40) is None and not ex.should_request()
    ex.receive(prediction(0))
    assert ex.take_over(50)  # Inference delay does not consume playback frames.
    assert ex.start_step == 50
    for i in range(3):
        assert not ex.should_request()
        np.testing.assert_array_equal(ex.command(50+i), np.full(7, i))
        np.testing.assert_array_equal(ex.torque(50+i), np.full(7, i+100))
        ex.advance()
    assert ex.should_request()
    ex.requested = True
    for step in range(53, 70):
        assert ex.command(step) is None and ex.torque(step) is None
        assert not ex.should_request()
        ex.advance()
    assert ex.overruns == 0 and ex.wm_execute_step == 3
    ex.receive(prediction(53, 10))
    assert ex.take_over(70)
    np.testing.assert_array_equal(ex.command(70), np.full(7,10))


def test_openloop_never_interrupts_current_segment():
    ex = Execution(Schedule(3, 0), open_loop=True)
    ex.receive(prediction(0))
    ex.take_over(5)
    ex.receive(prediction(6, 20))
    assert not ex.take_over(6)
    for _ in range(3):
        ex.advance()
    assert ex.take_over(8)
    assert ex.command(8)[0] == 20


def test_openloop_rejects_segment_longer_than_prediction():
    ex = Execution(Schedule(7,0), open_loop=True)
    ex.receive(prediction(0))
    assert not ex.take_over(10)
    assert ex.rejected == 1


def test_mode_config_validation_and_legacy_default(tmp_path):
    cfg = load_config(Path(__file__).parents[1]/'inference/configs/pi0_wm.yaml')
    cfg['wm'].pop('inference_mode')
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    assert load_config(path)['wm']['inference_mode'] == 'prefetch'
    cfg['wm']['inference_mode']='invalid'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='inference_mode'):
        load_config(path)
