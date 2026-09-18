"""Current PINN public sampler and checkpoint-defined preprocessing, independent of DP."""
from __future__ import annotations

from collections import deque
import logging
from pathlib import Path
import sys

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


def _declared_dq_source(data_config):
    """Return the explicitly declared source of the checkpoint's dq stream.

    The deployment cannot infer this from a feature name or from the current
    robot adapter.  Training must record either hardware motor velocity or a
    causal backward difference in the checkpoint dataloader configuration.
    """
    declarations = []
    for key in ('dq_source', 'velocity_source', 'dq_derivation', 'derivative_method'):
        if key in data_config:
            declarations.append((key, data_config[key]))
    for parent in ('state_sources', 'sources', 'features'):
        value = data_config.get(parent)
        if isinstance(value, dict):
            for key in ('dq', 'velocity', 'observation.velocity'):
                if key in value:
                    declarations.append((f'{parent}.{key}', value[key]))
    if not declarations:
        raise ValueError('checkpoint dataloader does not declare dq source')

    aliases = {
        'hardware': 'hardware',
        'hardware_motor_velocity': 'hardware',
        'motor_velocity': 'hardware',
        'measured_motor_velocity': 'hardware',
        'official_motor_velocity': 'hardware',
        'sign_corrected_official_motor_velocity_unfiltered': 'hardware',
        'backward_difference': 'backward_difference',
        'backward-difference': 'backward_difference',
        'q_backward_difference': 'backward_difference',
    }
    resolved = []
    for key, value in declarations:
        if not isinstance(value, str):
            raise ValueError(f'checkpoint dq source {key} must be a string')
        source = aliases.get(value.strip().lower())
        if source is None:
            raise ValueError(f'unsupported checkpoint dq source {value!r}')
        resolved.append(source)
    if len(set(resolved)) != 1:
        raise ValueError(f'checkpoint dq source declarations disagree: {declarations}')
    return resolved[0]


def checkpoint_preprocessing(data_config, normalized_filters):
    """Restore only causal operations that the checkpoint expects at runtime.

    Operations already applied while creating the training dataset are not
    repeated.  No offline episode metadata is consulted.
    """
    dq_source = _declared_dq_source(data_config)
    operations = {}
    for key, spec in normalized_filters.items():
        if not spec.get('enabled'):
            continue
        prefix = len(spec['dataset_preprocessed_operations'])
        pending = list(spec['operations'][prefix:])
        if pending:
            operations[key] = pending
    log.info('checkpoint preprocessing dq_source=%s operations=%s', dq_source, operations)
    return dq_source, operations


# Kept as a narrow compatibility name for callers that used the old helper;
# it no longer reads any recorded episode metadata.
recorded_preprocessing = checkpoint_preprocessing


class History:
    def __init__(self, horizon, hz, operations, dq_source='hardware'):
        if dq_source not in ('hardware', 'backward_difference'):
            raise ValueError(f'unsupported dq source {dq_source!r}')
        self.rows = deque(maxlen=horizon)
        self.horizon = horizon
        self.hz = hz
        self.filters = {key: CausalChain(operations.get(key, [])) for key in ('q', 'dq', 'delta_q', 'tau')}
        self.dq_source = dq_source
        self.previous_q = None
        self.last_time = None
        self.anchor = -1

    def append(self, step, state, held_command, now):
        if step <= self.anchor:
            raise ValueError('history step must increase')
        dt = 1 / self.hz if self.last_time is None else now - self.last_time
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('nonpositive history sample interval')
        q = np.asarray(state.q, dtype=np.float32)
        dq = np.asarray(state.dq, dtype=np.float32)
        if self.dq_source == 'backward_difference':
            dq = np.zeros_like(q) if self.previous_q is None else (q - self.previous_q) / dt
        raw = {'q': q, 'dq': dq, 'tau': state.torque,
               'delta_q': np.asarray(held_command) - q}
        if any(np.shape(v) != (7,) or not np.isfinite(v).all() for v in raw.values()):
            raise ValueError('invalid measured state/held command; refusing WM history')
        if any(self.filters[k].stages for k in raw):
            row = {k: self.filters[k].apply(v, dt) for k, v in raw.items()}
        else:
            row = {k: np.asarray(v, dtype=np.float32).copy() for k, v in raw.items()}
        self.rows.append(row)
        self.previous_q, self.last_time, self.anchor = q.copy(), now, step

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
        self.model = ContactWorldModel(self.cfg)
        self.contract = self.model.validate_checkpoint(payload)
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
        self.dq_source, self.operations = checkpoint_preprocessing(data, filters)
        log.info('WM restored %s weights=%s device=%s samples=%s flow_steps=%s solver=%s contract=%s',
                 config['checkpoint'], weights_key, self.device, self.num_samples,
                 self.steps or self.contract['flow']['steps'], self.solver or self.contract['flow']['solver'], self.contract)

    def infer(self, payload):
        import torch
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
        # Public API owns state stride and expansion. Action is never sliced.
        with torch.inference_mode():
            result = self.model.sample(batch, num_samples=self.num_samples, steps=self.steps, solver=self.solver)
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
        return physical


class MockWM:
    history_horizon = 12
    future_horizon = 80
    action_horizon = 20
    offset = 1
    operations = {}
    dq_source = 'hardware'

    def __init__(self, samples=1, latency=0.015):
        self.num_samples, self.latency = samples, latency

    def infer(self, payload):
        import time
        history, action = payload
        if action.shape != (self.action_horizon, 7):
            raise ValueError('mock WM needs a full native action window')
        time.sleep(self.latency)
        q = np.broadcast_to(history['q'][-1], (self.num_samples, self.future_horizon, 7)).copy()
        return {'q': q, 'tau': np.zeros_like(q)}
