from pathlib import Path
import sys
import threading
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

from inference.pi0_wm.core import Execution, Plans, MissingActions, Request, Result, Schedule, Worker, pi_trigger
from inference.pi0_wm.wm import CausalChain, History, WMAdapter, checkpoint_preprocessing
from inference.pi0_wm.config import load_config


def chunk(start=0, length=50):
    return np.repeat(np.arange(start, start + length, dtype=np.float32)[:, None], 7, axis=1)


def result(anchor=0, horizon=12, ident=0):
    return Result(Request(ident, anchor, (0,), None, time.perf_counter()),
                  {'q': chunk(100, horizon)[None], 'tau': np.zeros((1, horizon, 7))}, .01)


def test_native_actions_cross_committed_boundary_with_offset_and_substeps():
    plans = Plans(50)
    plans.add(chunk(), 0, 0, 0)
    with pytest.raises(MissingActions):
        plans.window(4 * 40, 20, 1)
    plans.add(chunk(50), 50, 1, 100)
    for substep in range(4):
        actions, versions = plans.window(4 * 40 + substep, 20, 1)
        np.testing.assert_array_equal(actions[:, 0], np.arange(41, 61))
        assert versions == (0, 1)
    assert plans.phase(199) == (0, 49, 3)
    assert plans.phase(200) == (1, 0, 0)
    assert plans.plans[0].actions.shape == (50, 7)
    with pytest.raises(ValueError, match='overlap'):
        plans.add(chunk(), 80, 2, 200)


def test_full_response_retained_and_short_response_rejected():
    plans = Plans(30)
    assert plans.add(chunk(), 0, 0, 0).actions.shape[0] == 50
    with pytest.raises(ValueError, match='exceeds'):
        Plans(51).add(chunk(), 0, 0, 0)


def test_calibration_schedules_are_external_steps_and_fail_infeasible():
    s = Schedule.calibrate(20, .051, .02, 100, 80)
    assert (s.prefetch_steps, s.trigger_step) == (8, 12)
    assert pi_trigger(50, 20, 1, .20, .08, 25) == 22
    with pytest.raises(ValueError, match='single WM'):
        Schedule.calibrate(4, .051, .02, 100, 80)
    with pytest.raises(ValueError, match='horizon'):
        Schedule.calibrate(20, .051, .02, 100, 25)
    with pytest.raises(ValueError, match='pi0 latency'):
        pi_trigger(20, 20, 1, .2, .08, 25)


def test_prefetch_does_not_reset_count_and_takeover_accounts_for_wait():
    execution = Execution(Schedule(4, 2))
    execution.receive(result())
    assert execution.take_over(0)
    for step in range(2):
        np.testing.assert_equal(execution.command(step)[0], 100 + step)
        execution.advance()
    assert execution.should_request()
    execution.requested = True
    execution.advance()  # current count advances while next worker runs
    execution.receive(result(anchor=2, ident=1))
    assert not execution.take_over(3)
    assert execution.wm_execute_step == 3
    execution.advance()
    assert execution.take_over(4)
    assert execution.wm_execute_step == 0
    assert execution.wm_loop_id == 1
    assert execution.command(4)[0] == 102  # d=2, includes waiting after ready at 3


def test_late_prediction_consumes_valid_tail_then_holds_and_recovers():
    execution = Execution(Schedule(4, 2))
    execution.receive(result(horizon=8))
    execution.take_over(0)
    execution.requested = True
    for step in range(10):
        q = execution.command(step)
        assert (q is None) == (step >= 8)
        execution.advance()
    assert execution.overruns == 1
    execution.receive(result(anchor=2, horizon=8, ident=1))
    assert not execution.take_over(10)  # d+E > horizon
    assert execution.rejected == 1 and execution.should_request()
    execution.receive(result(anchor=10, horizon=8, ident=2))
    assert execution.take_over(12)
    assert execution.command(12)[0] == 102


def test_takeover_horizon_equality_and_complete_sample_selection():
    execution = Execution(Schedule(4, 2), selected_sample=1)
    r = result(horizon=6)
    r.value['q'] = np.concatenate([r.value['q'], r.value['q'] + 1000])
    execution.receive(r)
    assert execution.take_over(2)  # d+E == H is valid
    assert execution.command(2)[0] == 1102


def test_worker_has_one_inflight_including_ready_result():
    release = threading.Event()
    worker = Worker('test', lambda value: (release.wait(1), value)[1])
    try:
        worker.submit(Request(1, 10, (0,), 'first', time.perf_counter()))
        with pytest.raises(RuntimeError, match='in-flight'):
            worker.submit(Request(2, 11, (), 'stale', time.perf_counter()))
        release.set()
        deadline = time.monotonic() + 2
        answer = None
        while answer is None and time.monotonic() < deadline:
            answer = worker.poll()
            time.sleep(.001)
        assert answer.value == 'first'
        assert not worker.busy
    finally:
        release.set()
        worker.close()


def test_real_history_delta_q_uses_held_bounded_command_and_never_predictions():
    history = History(3, 100, {})
    state = SimpleNamespace(q=np.ones(7), dq=np.zeros(7), torque=np.full(7, 2))
    for step in range(3):
        history.append(step, state, np.full(7, 1.02), 1 + step * .01)
    batch = history.snapshot()
    np.testing.assert_allclose(batch['delta_q'], .02, atol=1e-7)
    np.testing.assert_equal(batch['tau'], 2)
    batch['q'][:] = 100
    np.testing.assert_equal(history.snapshot()['q'], 1)


def test_config_independent_of_dp_and_explicit_modes():
    config = load_config(Path(__file__).parents[1] / 'inference/configs/pi0_wm.yaml')
    assert config['control']['hz'] == 100
    assert config['wm']['num_samples'] == 1
    assert 'dp_checkpoint' not in config
    assert config['pi0']['consume_steps'] == 50
    assert 'preprocessing' not in config['wm']


def test_checkpoint_preprocessing_has_no_offline_metadata_dependency():
    data = {'dq_source': 'hardware'}
    filters = {
        key: {'enabled': False, 'operations': [], 'dataset_preprocessed_operations': []}
        for key in ('q', 'dq', 'delta_q', 'tau')
    }
    assert checkpoint_preprocessing(data, filters) == ('hardware', {})


def test_wm_adapter_initializes_without_offline_preprocessing_paths(monkeypatch, tmp_path):
    import torch

    class FakeModel:
        inputs = ('q',)
        predicted_state_streams = ('q',)

        def __init__(self, config):
            self.config = config

        def validate_checkpoint(self, payload):
            return {
                'external_history_horizon': 1, 'external_future_horizon': 1,
                'external_action_horizon': 1, 'input_state_streams': ['q'],
                'predicted_continuous_streams': ['q'], 'joint_dim': 7,
                'action': {'dimension': 7, 'start_offset': 0,
                           'type': 'absolute_ee_pose',
                           'representation': 'xyz_quaternion',
                           'quaternion_order': 'xyzw',
                           'quaternion_sign': 'canonical_w_nonnegative',
                           'coordinate_frame': 'link7',
                           'absolute_or_relative': 'absolute',
                           'inference_delay_s': 0.0},
                'external_state_rate_hz': 100, 'action_rate_hz': 25,
                'flow': {'steps': 1, 'solver': 'euler'},
            }

        def load_state_dict(self, state, strict=True):
            return None

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeNormalizer:
        def __init__(self, stats, eps):
            pass

    modules = {
        'model': types.ModuleType('model'),
        'model.pinn_model': types.ModuleType('model.pinn_model'),
        'model.pinn_model.contact_world_model': types.ModuleType('model.pinn_model.contact_world_model'),
        'train': types.ModuleType('train'),
        'train.nomalizer': types.ModuleType('train.nomalizer'),
        'data_process': types.ModuleType('data_process'),
        'data_process.causal_data_filter': types.ModuleType('data_process.causal_data_filter'),
    }
    modules['model.pinn_model.contact_world_model'].ContactWorldModel = FakeModel
    modules['train.nomalizer'].Normalizer = FakeNormalizer
    modules['data_process.causal_data_filter'].normalize_dataloader_filters = lambda data: {}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    cfg = {
        'dataloader': {'dq_source': 'hardware', 'normalize_mode': 'gaussian',
                       'normalize_lowdim_keys': [], 'filters': {}},
        'train': {'ema': {'enabled': False}},
    }
    payload = {'config': cfg, 'model': {}, 'normalizer': {'stats': {}},
               'dataloader_filters': {}, 'sample_rate_hz': 100}
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: payload)
    from inference.pi0_wm.wm import WMAdapter

    adapter = WMAdapter({'pinn_root': str(tmp_path), 'checkpoint': 'missing.pt',
                         'use_ema': False, 'device': 'cpu', 'num_samples': 1,
                         'flow_steps': 1, 'solver': 'euler'}, 100, 25)
    assert adapter.dq_source == 'hardware'
    assert adapter.operations == {}


def test_checkpoint_preprocessing_restores_only_pending_causal_operations():
    data = {'dq_source': 'backward_difference'}
    filters = {
        'q': {'enabled': True,
              'operations': [{'type': 'lowpass', 'cutoff_hz': 15.0, 'order': 2}],
              'dataset_preprocessed_operations':
                  [{'type': 'lowpass', 'cutoff_hz': 15.0, 'order': 2}]},
        'dq': {'enabled': True,
               'operations': [{'type': 'lowpass', 'cutoff_hz': 15.0}],
               'dataset_preprocessed_operations': []},
    }
    source, operations = checkpoint_preprocessing(data, filters)
    assert source == 'backward_difference'
    assert operations == {'dq': [{'type': 'lowpass', 'cutoff_hz': 15.0}]}


def test_unknown_checkpoint_dq_source_fails_closed():
    with pytest.raises(ValueError, match='dq source'):
        checkpoint_preprocessing({}, {})
    with pytest.raises(ValueError, match='unsupported checkpoint dq source'):
        checkpoint_preprocessing({'dq_source': 'unknown'}, {})


def test_position_limits_record_only_successful_sent_command():
    from inference.pi0_wm.runtime import Runtime
    runtime = Runtime.__new__(Runtime)
    runtime.cfg = {'hardware': {'q_min': [-1] * 7, 'q_max': [1] * 7, 'maximum_step_rad': [.02] * 7}}
    runtime.held = np.zeros(7)
    runtime.state = SimpleNamespace(q=np.zeros(7))
    runtime.step, runtime.execution = 0, None
    runtime.visualizer = SimpleNamespace(update=lambda *args: None)
    commands = []
    runtime.arm = SimpleNamespace(command_joint_positions=lambda q: commands.append(q.copy()))
    runtime.command_enabled, runtime.simulated_arm = True, False
    runtime.send(np.ones(7))
    np.testing.assert_allclose(runtime.held, .02)
    np.testing.assert_array_equal(runtime.held, commands[-1])
    def fail(q):
        raise RuntimeError('CAN send failed')
    runtime.arm.command_joint_positions = fail
    with pytest.raises(RuntimeError, match='CAN send'):
        runtime.send(np.ones(7))
    np.testing.assert_allclose(runtime.held, .02)
    runtime.command_enabled = False
    runtime.send(np.ones(7))  # dry-run must not call the failing transport
    np.testing.assert_allclose(runtime.held, .02)


def pinn_imports():
    root = Path(__file__).resolve().parents[2] / 'PINN'
    if not root.exists():
        pytest.skip('optional sibling PINN checkout unavailable')
    sys.path.insert(0, str(root))
    package = sys.modules.get('model')
    if package is not None and hasattr(package, '__path__'):
        package.__path__ = [str(root / 'model'), *package.__path__]
    return root


def test_streaming_filters_match_training_episode_operations():
    pinn_imports()
    from data_process.causal_data_filter import filter_episode_values
    rng = np.random.default_rng(12)
    values = rng.normal(size=(100, 7)).astype(np.float32)
    times = np.cumsum(rng.uniform(.007, .014, 100))
    ops = [{'type': 'median', 'window': 3}, {'type': 'moving_average', 'window': 4},
           {'type': 'lowpass', 'cutoff_hz': 15, 'order': 2}]
    chain = CausalChain(ops)
    actual = np.stack([chain.apply(v, .01 if i == 0 else times[i] - times[i-1]) for i, v in enumerate(values)])
    expected = filter_episode_values(times, values, ops)
    np.testing.assert_allclose(actual, expected, atol=2e-7)


def test_actual_current_model_stride_keeps_all_action_tokens_and_expands_once():
    pinn_imports()
    import torch
    from model.pinn_model.contact_world_model import ContactWorldModel
    from train.nomalizer import Normalizer
    config = {'dataloader': {'state_history_horizon': 12, 'prediction_horizon': 8,
                             'action_condition_horizon': 5, 'action_start_offset': 1,
                             'high_fps': 100, 'expert_fps': 25},
              'train': {'downsample': 2},
              'model': {'inputs': ['q', 'dq', 'delta_q', 'tau'], 'outputs': ['q', 'tau'],
                        'hidden_dim': 16, 'state_layers': 1, 'action_layers': 1,
                        'flow_layers': 1, 'flow_attention_heads': 2, 'dropout': 0}}
    model = ContactWorldModel(config).eval()
    adapter = WMAdapter.__new__(WMAdapter)
    adapter.model, adapter.device = model, torch.device('cpu')
    adapter.num_samples, adapter.steps, adapter.solver = 2, 1, 'euler'
    adapter.history_horizon, adapter.action_horizon, adapter.future_horizon = 12, 5, 8
    adapter.inputs, adapter.outputs = model.inputs, model.predicted_state_streams
    adapter.normalizer, adapter.mode, adapter.normalize_keys = Normalizer({}), 'gaussian', []
    history = {key: np.zeros((12, 7), np.float32) for key in model.inputs}
    seen = []
    original = model._action_inputs
    def capture(batch):
        seen.append(batch['action'].detach().cpu().numpy().copy())
        return original(batch)
    model._action_inputs = capture
    output = adapter.infer((history, chunk(length=5)))
    assert output['q'].shape == (2, 8, 7)
    assert seen and all(x.shape == (1, 5, 7) for x in seen)
    for x in seen:
        np.testing.assert_array_equal(x[0, :, 0], np.arange(5))
    np.testing.assert_array_equal(output['q'][:, ::2], output['q'][:, 1::2])
