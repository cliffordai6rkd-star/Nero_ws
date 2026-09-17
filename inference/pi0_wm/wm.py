"""Current PINN public sampler and recorded-data preprocessing, independent of DP."""
from __future__ import annotations

from collections import deque
import json
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


def recorded_preprocessing(config, data_config, normalized_filters):
    """Use export metadata plus ONLY pending training operations.

    A checkpoint's source_already_filtered flag describes the dataset, not
    raw hardware. The actual exported features, including deliberately skipped
    training operations, are authoritative. Unknown raw processing fails closed.
    """
    import h5py
    timeline_path = Path(config['timeline'])
    timeline = json.loads(timeline_path.read_text())
    if timeline['mode'] != 'raw_lowdim_action_hold':
        raise ValueError('only raw_lowdim_action_hold dataset conversion is supported')
    if (float(timeline['nominal_lerobot_fps']) != float(data_config['high_fps']) or
            float(timeline['action_fps']) != float(data_config['expert_fps'])):
        raise ValueError('export timeline rates disagree with checkpoint')
    columns = {'q': 'observation.joint', 'dq': 'observation.velocity',
               'tau': 'observation.torque', 'delta_q': 'observation.delta_q'}
    paths = {'q': 'teleop/q_follower', 'dq': 'teleop/dq_follower',
             'tau': 'teleop/tau_follower', 'q_cmd': 'teleop/q_cmd'}
    with h5py.File(config['source_h5'], 'r') as source:
        attrs = {key: dict(source[path].attrs) for key, path in paths.items()}
    for key, meta in attrs.items():
        if meta.get('lowpass', False) or int(meta.get('median_window', 1)) != 1:
            raise ValueError(f'raw H5 {key} has preprocessing not implemented by this deployment profile')
    method = attrs['dq'].get('derivative_method')
    if method == 'sign_corrected_official_motor_velocity_unfiltered':
        from nero_collection.coordinates import NERO_V120_MOTOR_VELOCITY_TO_JOINT_SIGN
        recorded_sign = json.loads(attrs['dq'].get('coordinate_sign_correction_json', 'null'))
        if recorded_sign != list(NERO_V120_MOTOR_VELOCITY_TO_JOINT_SIGN):
            raise ValueError('recorded dq coordinate signs differ from the hardware adapter')
        dq_source = 'hardware'  # PyAgxArm already applies the same coordinate sign.
    elif method == 'backward_difference':
        dq_source = 'backward_difference'
    else:
        raise ValueError(f'unsupported recorded dq derivation {method!r}; require exact causal source contract')
    if attrs['q_cmd'].get('command_semantics') != 'causal_zoh_at_state_sample':
        raise ValueError('recorded q_cmd must mean last successfully issued held command')
    if attrs['tau'].get('processing_method') != 'nearest_motor_sample_unfiltered':
        raise ValueError('tau must be measured motor torque, not external/gravity-relative torque')
    operations = {}
    for key, column in columns.items():
        ops = []
        spec = timeline.get('feature_filters', {}).get(column, {})
        if spec.get('enabled'):
            if spec.get('contract') != 'causal_variable_dt_one_pole_cascade_v1' or not spec.get('causal'):
                raise ValueError(f'unsupported export filter: {column}: {spec}')
            ops.append({'type': 'lowpass', 'cutoff_hz': spec['cutoff_hz'], 'order': spec['order']})
        spec = normalized_filters.get(key, {})
        if spec.get('enabled'):
            prefix = len(spec['dataset_preprocessed_operations'])
            ops.extend(spec['operations'][prefix:])
        operations[key] = ops
    log.info('preprocessing evidence H5=%s timeline=%s dq=%s actual chains=%s',
             config['source_h5'], timeline_path, dq_source, operations)
    return dq_source, operations


class History:
    def __init__(self, horizon, hz, operations, dq_source='hardware'):
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
        self.rows.append({k: self.filters[k].apply(v, dt) for k, v in raw.items()})
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
        self.dq_source, self.operations = recorded_preprocessing(config['preprocessing'], data, filters)
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
