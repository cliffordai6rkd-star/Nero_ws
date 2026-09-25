from types import SimpleNamespace

import numpy as np
import pytest

from inference.pi0_wm.runtime import Runtime
from inference.pi0_wm.core import Execution, Schedule, Result, Request


def test_pause_resume_end_and_next_trial(monkeypatch):
    import inference.pi0_wm.runtime as module
    rt = Runtime.__new__(Runtime)
    rt.cfg = {'control': {'inference_runs': 2}}
    rt.interactive_session = True
    sequence = iter(['s', 'd', 's', 'i', 's', 'i'])
    rt.keys = SimpleNamespace(read_key=lambda _: next(sequence))
    events = []
    rt._prepare_trial = lambda: events.append(('fresh', rt.round_steps_done))
    def execute(limit):
        rt.round_steps_done += 1
        rt.check_keys()
    rt._execute_round_steps = execute
    rt._hold_trial = lambda: events.append('hold')
    rt._discard_trial_results = lambda: events.append('drain')
    rt.wait_cycle = rt.acquire = lambda: None
    rt.send = lambda target: None
    monkeypatch.setattr(module, 'reset_to_rest', lambda r: events.append('reset'))
    result = rt._session_loop(100)
    assert result['completed_runs'] == 2
    assert [x['execution_steps'] for x in result['trials']] == [2, 1]
    assert [x for x in events if isinstance(x, tuple)] == [('fresh', 0), ('fresh', 1), ('fresh', 0)]
    assert events.count('reset') == 2  # d does not park or consume a trial.


def test_automatic_step_limit_finishes_and_resets(monkeypatch):
    import inference.pi0_wm.runtime as module
    rt = Runtime.__new__(Runtime)
    rt.cfg = {'control': {'inference_runs': 1}}
    rt.keys = SimpleNamespace(read_key=lambda _: 's')
    rt._prepare_trial = lambda: None
    rt._execute_round_steps = lambda limit: setattr(rt, 'round_steps_done', limit)
    rt._hold_trial = lambda: None
    resets = []
    monkeypatch.setattr(module, 'reset_to_rest', lambda r: resets.append(True))
    result = rt._session_loop(17)
    assert result['trials'][0] == {'trial': 1, 'stopped_by': 'maximum_steps', 'execution_steps': 17}
    assert resets == [True]


@pytest.mark.parametrize('open_loop', [False, True])
def test_contact_phase_matches_executing_sample_and_index(open_loop):
    ex = Execution(Schedule(3, 0), selected_sample=1, open_loop=open_loop)
    phase = np.array([[0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 1, 2]])
    ex.receive(Result(Request(1, 10, (), None, 0),
                      {'q': np.zeros((2,6,7)), 'contact_phase': phase}, 0))
    assert ex.take_over(12)
    assert ex.contact_phase(12) == (0 if open_loop else 2)
    ex.advance()
    assert ex.contact_phase(13) == (1 if open_loop else 0)
    ex.advance(); ex.advance()
    assert ex.contact_phase(16) is None
    ex.current.value.pop('contact_phase')
    assert ex.contact_phase(12) is None


def test_wm_adapter_preserves_contact_phase():
    import torch
    from inference.pi0_wm.wm import WMAdapter
    w = WMAdapter.__new__(WMAdapter)
    w.history_horizon = w.action_horizon = 2
    w.future_horizon, w.num_samples = 3, 2
    w.device = torch.device('cpu')
    w.inputs, w.outputs, w.normalize_keys = ['q'], ['q', 'tau'], []
    w.steps, w.solver = 1, 'heun'
    w.contract = {'contact': {'classes': ['free', 'precontact_or_transition', 'contact']}}
    phase = torch.tensor([[[[0], [1], [2]], [[2], [1], [0]]]], dtype=torch.float32)
    w.model = SimpleNamespace(sample=lambda *a, **kw: {
        'q_pred': torch.zeros(1,2,3,7), 'tau_pred': torch.zeros(1,2,3,7), 'contact_state_pred': phase})
    result = w.infer(({'q': np.zeros((2,7))}, np.zeros((2,7))))
    np.testing.assert_array_equal(result['contact_phase'], [[0,1,2], [2,1,0]])
    w.contract = {}  # Unknown class ordering must not invent labels.
    assert 'contact_phase' not in w.infer(({'q': np.zeros((2,7))}, np.zeros((2,7))))


def test_contact_overlay_white_top_right():
    from nero_collection.cameras import _draw_contact_phase
    draws = []
    cv = SimpleNamespace(FONT_HERSHEY_SIMPLEX=0, LINE_AA=16,
        getTextSize=lambda *a: ((70,15), 2),
        putText=lambda *a: draws.append(a))
    frame = np.zeros((192,256,3), np.uint8)
    for phase in [0,1,2,-1]:
        _draw_contact_phase(frame, phase, cv)
    assert [x[1] for x in draws] == ['free motion', 'alignment', 'contact']
    assert all(x[2] == (178,23) and x[5] == (255,255,255) for x in draws)
