"""Current PINN public sampler and checkpoint-defined preprocessing, independent of DP."""
from __future__ import annotations

from collections import deque
import logging
from pathlib import Path
import sys
import time

import numpy as np

log = logging.getLogger(__name__)


class CausalChain:
    """Streaming equivalent of PINN causal_data_filter.filter_episode_values."""
    def __init__(self, operations):
        self.stages = []
        for op in operations:
            if op['type'] == 'lowpass':
                for _ in range(int(op.get('order', 1))):
                    self.stages.append((op, None))
            elif op['type'] in ('median', 'moving_average'):
                self.stages.append((op, deque(maxlen=int(op['window']))))
            else:
                raise ValueError(f"unsupported causal operation: {op}")

    def apply(self, value, dt):
        value = np.asarray(value, dtype=np.float64).copy()
        for index, (op, state) in enumerate(self.stages):
            if op['type'] == 'lowpass':
                alpha = 1 - np.exp(-2 * np.pi * op['cutoff_hz'] * dt)
                value = value if state is None else alpha * value + (1 - alpha) * state
                self.stages[index] = (op, value.copy())
            else:
                if not state:
                    state.extend([value.copy() for _ in range(state.maxlen)])
                else:
                    state.append(value.copy())
                value = (np.median(state, axis=0) if op['type'] == 'median' else np.mean(state, axis=0))
        return value.astype(np.float32)


def checkpoint_preprocessing(normalized_filters):
    """Restore only causal operations that the checkpoint expects at runtime.

    Operations already applied while creating the training dataset are not
    repeated.  No offline episode metadata is consulted.
    """
    operations = {}
    for key, spec in normalized_filters.items():
        if not spec.get('enabled'):
            continue
        prefix = len(spec['dataset_preprocessed_operations'])
        pending = list(spec['operations'][prefix:])
        if pending:
            operations[key] = pending
    log.info('checkpoint preprocessing operations=%s', operations)
    return operations


# Kept as a narrow compatibility name for callers that used the old helper;
# it no longer reads any recorded episode metadata.
recorded_preprocessing = checkpoint_preprocessing


class History:
    def __init__(self, horizon, hz, operations):
        self.rows = deque(maxlen=horizon)
        self.horizon = horizon
        self.hz = hz
        self.filters = {key: CausalChain(operations.get(key, [])) for key in ('q', 'dq', 'delta_q', 'tau')}
        self.last_time = None
        self.anchor = -1

    def append(self, step, state, held_command, now):
        if step <= self.anchor:
            raise ValueError('history step must increase')
        dt = 1 / self.hz if self.last_time is None else now - self.last_time
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('nonpositive history sample interval')
        q = np.asarray(state.q, dtype=np.float32)
        # Deployment always uses the measured hardware velocity.
        dq = np.asarray(state.dq, dtype=np.float32)
        raw = {'q': q, 'dq': dq, 'tau': state.torque,
               'delta_q': np.asarray(held_command) - q}
        if any(np.shape(v) != (7,) or not np.isfinite(v).all() for v in raw.values()):
            raise ValueError('invalid measured state/held command; refusing WM history')
        if any(self.filters[k].stages for k in raw):
            row = {k: self.filters[k].apply(v, dt) for k, v in raw.items()}
        else:
            row = {k: np.asarray(v, dtype=np.float32).copy() for k, v in raw.items()}
        self.rows.append(row)
        self.last_time, self.anchor = now, step

    @property
    def ready(self):
        return len(self.rows) == self.horizon

    def snapshot(self):
        if not self.ready:
            raise RuntimeError('real history not filled')
        return {key: np.stack([row[key] for row in self.rows]) for key in self.filters}


class WMAdapter:
    def __init__(self, config, control_hz, action_hz):
        import torch
        root = Path(config['pinn_root']).resolve()
        sys.path.insert(0, str(root))
        # Accommodate an already imported workspace `model` namespace.
        package = sys.modules.get('model')
        if package is not None and hasattr(package, '__path__'):
            package.__path__ = [str(root / 'model'), *package.__path__]
        from model.pinn_model.contact_world_model import ContactWorldModel
        from train.nomalizer import Normalizer
        from data_process.causal_data_filter import normalize_dataloader_filters

        payload = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
        self.cfg = payload.get('config', payload.get('cfg'))
        deterministic = payload.get('model_version') == 'deterministic_wm_v1'
        if deterministic:
            from model.pinn_model.deterministic_world_model import DeterministicRobotStateWorldModel
            self.model = DeterministicRobotStateWorldModel(self.cfg)
        else:
            self.model = ContactWorldModel(self.cfg)
        self.contract = self.model.validate_checkpoint(payload)
        if deterministic:
            # Expose the same deployment metadata after validating either v1
            # envelope. The deterministic architecture has no flow contract.
            data = self.cfg.get('dataloader') or {}
            action = self.cfg.get('action_contract') or {}
            self.contract = {
                **self.model.checkpoint_contract(),
                'external_action_horizon': self.model.external_action_condition_horizon,
                'input_state_streams': self.model.inputs,
                'predicted_continuous_streams': self.model.outputs,
                'contact': {
                    'classes': (['free', 'precontact_or_transition', 'contact']
                                if self.model.contact_state_count == 3
                                and (self.cfg.get('contact_gate') or {}).get('label_mode', 'three_phase') == 'three_phase'
                                else []),
                },
                'action': {
                    'dimension': self.model.action_dim,
                    'start_offset': self.model.action_start_offset,
                    'type': action.get('type', 'absolute_ee_pose'),
                    'representation': action.get('representation', 'xyz_quaternion'),
                    'quaternion_order': action.get('quaternion_order', 'xyzw'),
                    'quaternion_sign': action.get('quaternion_sign', 'canonical_w_nonnegative'),
                    'coordinate_frame': action.get('coordinate_frame', 'link7'),
                    'absolute_or_relative': action.get('absolute_or_relative', 'absolute'),
                    'inference_delay_s': float(data.get('inference_delay_s', 0.0)),
                },
            }
        # PINN trainer saves the EMA deployment model under `model` and raw
        # optimizer weights under `model_raw`. Never silently substitute weights.
        ema_trained = (self.cfg.get('train', {}).get('ema') or {}).get('enabled', False)
        if not config['use_ema'] and ema_trained and payload.get('model_raw') is None:
            raise ValueError('raw weights requested but EMA checkpoint has no model_raw')
        weights_key = 'model' if config['use_ema'] or not ema_trained else 'model_raw'
        if config['use_ema'] and ema_trained and payload.get('ema') is None:
            raise ValueError('EMA requested but checkpoint has no EMA state')
        self.model.load_state_dict(payload[weights_key], strict=True)
        self.device = torch.device(config['device'])
        self.model.to(self.device).eval()
        self.num_samples = int(config['num_samples'])
        self.steps = config.get('flow_steps')
        self.solver = config.get('solver')
        self.history_horizon = self.contract['external_history_horizon']
        self.future_horizon = self.contract['external_future_horizon']
        self.action_horizon = self.contract['external_action_horizon']
        self.offset = self.contract['action']['start_offset']
        self.inputs = self.contract['input_state_streams']
        self.outputs = self.contract['predicted_continuous_streams']
        if not set(self.inputs) <= {'q', 'dq', 'delta_q', 'tau'} or 'q' not in self.outputs:
            raise ValueError(f'unsupported model modalities: {self.inputs} -> {self.outputs}')
        if self.contract['joint_dim'] != 7 or self.contract['action']['dimension'] != 7:
            raise ValueError('Nero requires joint_dim=7 and EE pose action_dim=7')
        if control_hz != self.contract['external_state_rate_hz'] or action_hz != self.contract['action_rate_hz']:
            raise ValueError('configured control/action rates do not match checkpoint')
        expected = {'type': 'absolute_ee_pose', 'representation': 'xyz_quaternion',
                    'quaternion_order': 'xyzw', 'quaternion_sign': 'canonical_w_nonnegative',
                    'coordinate_frame': 'link7', 'absolute_or_relative': 'absolute', 'inference_delay_s': 0.0}
        for name, value in expected.items():
            if self.contract['action'][name] != value:
                raise ValueError(f'unsupported WM action contract {name}={self.contract["action"][name]}')
        if self.offset < 0:
            raise ValueError('negative action_start_offset unsupported')
        n = payload['normalizer']
        self.normalizer = Normalizer(n['stats'], eps=float(n.get('eps', 1e-6)))
        data = self.cfg['dataloader']
        self.mode = data.get('normalize_mode')
        self.normalize_keys = data.get('normalize_lowdim_keys', [])
        if n.get('normalize_mode', self.mode) != self.mode:
            raise ValueError('normalizer mode disagrees with checkpoint data config')
        if self.mode not in ('gaussian', 'limit', 'quantile'):
            raise ValueError(f'unsupported normalization {self.mode}')
        for key in self.normalize_keys:
            if key not in n['stats']:
                raise ValueError(f'missing normalizer statistics for {key}')
        filters = normalize_dataloader_filters(data)
        if payload.get('dataloader_filters') is not None and payload['dataloader_filters'] != filters:
            raise ValueError('saved dataloader_filters disagree with checkpoint config')
        if payload.get('sample_rate_hz') is not None and payload['sample_rate_hz'] != control_hz:
            raise ValueError('saved sample_rate_hz disagrees with external control rate')
        if n.get('normalize_lowdim_keys', self.normalize_keys) != self.normalize_keys:
            raise ValueError('normalizer keys disagree with checkpoint data config')
        self.operations = checkpoint_preprocessing(filters)
        acceleration = dict(config.get('acceleration') or {})
        self.acceleration = {
            'enabled': bool(acceleration.get('enabled', False)),
            'cache_condition_kv': bool(acceleration.get('cache_condition_kv', False)),
            'compile': bool(acceleration.get('compile', False)),
            'compile_mode': str(acceleration.get('compile_mode', 'reduce-overhead')),
        }
        self._compiled_integrate = None
        self._compile_warmup_seconds = None
        self._compile_warmup_reported = False
        self._compile_counter_snapshot = None
        self._compile_unique_graphs = None
        self._last_timing = None
        if self.acceleration['enabled'] and self.acceleration['compile']:
            self._setup_compiled_integrator(torch)
        log.info('WM acceleration enabled=%s cache_condition_kv=%s compile=%s mode=%s',
                 self.acceleration['enabled'], self.acceleration['cache_condition_kv'],
                 self.acceleration['compile'], self.acceleration['compile_mode'])
        log.info('WM restored %s weights=%s device=%s samples=%s flow_steps=%s solver=%s contract=%s',
                 config['checkpoint'], weights_key, self.device, self.num_samples,
                 self.steps or self.contract.get('flow', {}).get('steps'),
                 self.solver or self.contract.get('flow', {}).get('solver'), self.contract)

    def _setup_compiled_integrator(self, torch):
        """Compile the exact integration path used by ``model.sample``.

        Flow steps and solver are captured from this deployment instance. A
        different configured step count gets its own adapter and compilation,
        while the model implementation remains generic for Euler and Heun.
        """
        if not hasattr(torch, 'compile'):
            log.warning('WM acceleration compile unavailable: torch.compile is not present; using eager path')
            return
        if not hasattr(self.model, 'integrate_flow'):
            log.warning('WM acceleration compile unavailable: model has no flow integration path; using eager path')
            return
        mode = self.acceleration['compile_mode']
        try:
            def integrate(source, encoded, *, steps=None, solver=None):
                return self.model.integrate_flow(
                    source, encoded, steps=self.steps if self.steps is not None else steps,
                    solver=self.solver if self.solver is not None else solver,
                )
            self._compiled_integrate = torch.compile(
                integrate, mode=mode, dynamic=False, fullgraph=False,
            )
            log.info('WM acceleration compile enabled backend=inductor(default) mode=%s steps=%s solver=%s; lazy compile will run in WM worker warmup',
                     mode, self.steps or self.model.flow_inference_steps,
                     self.solver or self.model.flow_solver)
        except Exception as exc:
            self._compiled_integrate = None
            log.warning('WM acceleration compile setup failed (%s: %s); using eager path', type(exc).__name__, exc)

    @staticmethod
    def _compile_diagnostics():
        try:
            from torch._dynamo.utils import counters
            graph_breaks = sum(
                values.get('graph_break', 0) + values.get('graph_breaks', 0)
                for values in counters.values()
            )
            unique_graphs = sum(values.get('unique_graphs', 0) for values in counters.values())
            return graph_breaks, unique_graphs
        except Exception:
            return None

    def infer(self, payload, source_noise=None):
        import torch
        started_total = time.perf_counter()
        history, action = payload
        if np.shape(action) != (self.action_horizon, 7):
            raise ValueError('incorrect native action window shape')
        if any(np.shape(history[key]) != (self.history_horizon, 7) for key in self.inputs):
            raise ValueError('incorrect external history shape')
        batch = {key: torch.as_tensor(history[key], device=self.device).float()[None] for key in self.inputs}
        batch['action'] = torch.as_tensor(action, device=self.device).float()[None]
        for key in batch:
            if key in self.normalize_keys:
                batch[key] = getattr(self.normalizer, f'{self.mode}_normalize')(key, batch[key])
        if source_noise is not None:
            source_noise = torch.as_tensor(
                source_noise, device=self.device, dtype=batch[self.inputs[0]].dtype,
            )
        # Public API owns state stride and expansion. Action is never sliced.
        compiled_integrate = getattr(self, '_compiled_integrate', None)
        acceleration = getattr(self, 'acceleration', {})
        compile_started = time.perf_counter() if compiled_integrate is not None and not getattr(self, '_compile_warmup_reported', False) else None
        gpu_start = gpu_end = None
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_start.record()
        sample_kwargs = {
            'num_samples': self.num_samples,
            'steps': self.steps,
            'solver': self.solver,
            'source_noise': source_noise,
        }
        if hasattr(self.model, 'flow_velocity'):
            sample_kwargs.update(
                cache_condition_kv=(
                    acceleration.get('enabled', False) and acceleration.get('cache_condition_kv', False)
                ),
                integration_fn=compiled_integrate,
            )
        try:
            with torch.inference_mode():
                result = self.model.sample(batch, **sample_kwargs)
        except Exception as exc:
            if compiled_integrate is None:
                raise
            log.warning('WM compiled inference failed (%s: %s); disabling compile and retrying eager path',
                        type(exc).__name__, exc)
            self._compiled_integrate = None
            sample_kwargs.pop('integration_fn', None)
            with torch.inference_mode():
                result = self.model.sample(batch, **sample_kwargs)
        if gpu_end is not None:
            gpu_end.record()
            gpu_end.synchronize()
        if compile_started is not None:
            self._compile_warmup_seconds = time.perf_counter() - compile_started
            self._compile_warmup_reported = True
            try:
                from torch._dynamo.utils import counters
                self._compile_counter_snapshot = {
                    group: dict(values) for group, values in counters.items()
                }
                graph_breaks, unique_graphs = self._compile_diagnostics()
                self._compile_unique_graphs = unique_graphs
                log.info('WM compile diagnostics: graph_breaks=%s unique_graphs=%s', graph_breaks, unique_graphs)
            except Exception as exc:
                log.debug('WM compile diagnostics unavailable: %s', exc)
            log.info('WM compile+first worker warmup completed in %.6fs; excluded from stable inference timings',
                     self._compile_warmup_seconds)
        elif compiled_integrate is not None and self._compile_unique_graphs is not None:
            diagnostics = self._compile_diagnostics()
            if diagnostics is not None and diagnostics[1] > self._compile_unique_graphs:
                log.warning('WM compile produced additional graphs after warmup: unique_graphs=%s (initial=%s); check input shapes/dtypes',
                            diagnostics[1], self._compile_unique_graphs)
        physical = {}
        for key in self.outputs:
            value = result[f'{key}_pred'][0]
            if key in self.normalize_keys:
                value = getattr(self.normalizer, f'{self.mode}_denormalize')(key, value)
            physical[key] = value.float().cpu().numpy()
            if physical[key].shape != (self.num_samples, self.future_horizon, 7) or not np.isfinite(physical[key]).all():
                raise ValueError(f'invalid external prediction {key}: {physical[key].shape}')
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        phase = result.get('contact_state_pred')
        classes = (self.contract.get('contact') or {}).get('classes')
        if phase is not None and classes == ['free', 'precontact_or_transition', 'contact']:
            phase = phase[0].squeeze(-1).cpu().numpy()
            if phase.shape != (self.num_samples, self.future_horizon) or not np.isin(phase, [0, 1, 2]).all():
                raise ValueError('invalid WM contact phase prediction')
            physical['contact_phase'] = phase.astype(np.int8)
        self._last_timing = {
            'gpu_seconds': None if gpu_start is None else gpu_start.elapsed_time(gpu_end) / 1000.0,
            'adapter_seconds': time.perf_counter() - started_total,
        }
        return physical


class MockWM:
    history_horizon = 12
    future_horizon = 80
    action_horizon = 20
    offset = 1
    operations = {}

    def __init__(self, samples=1, latency=0.015):
        self.num_samples, self.latency = samples, latency

    def infer(self, payload):
        import time
        history, action = payload
        if action.shape != (self.action_horizon, 7):
            raise ValueError('mock WM needs a full native action window')
        time.sleep(self.latency)
        q = np.broadcast_to(history['q'][-1], (self.num_samples, self.future_horizon, 7)).copy()
        return {'q': q, 'tau': np.zeros_like(q),
                'contact_phase': np.zeros(q.shape[:2], dtype=np.int8)}
