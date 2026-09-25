"""Small independent YAML parser; no legacy inference configuration imports."""
from pathlib import Path
import math
import yaml


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    sections = {'pi0', 'wm', 'control', 'calibration', 'hardware', 'cameras', 'mujoco'}
    if set(config) != sections:
        raise ValueError(f'pi0_wm configuration needs exactly {sorted(sections)}')
    def resolve(value):
        return str((path.parent / value).resolve()) if value else value
    wm = config['wm']
    wm.setdefault('enable', True)
    wm.setdefault('inference_mode', 'prefetch')
    if wm['inference_mode'] not in ('prefetch', 'openloop'):
        raise ValueError('wm.inference_mode must be prefetch or openloop')
    if not isinstance(wm['enable'], bool):
        raise ValueError('wm.enable must be true or false')
    if not wm['enable']:
        # Inactive WM fields need neither a checkpoint nor model dependencies.
        wm.update(num_samples=1, selected_sample=0, flow_steps=None, solver=None)
    for key in ('checkpoint', 'pinn_root'):
        if wm['enable']:
            wm[key] = resolve(wm[key])
    config['mujoco']['mujoco_model_path'] = resolve(config['mujoco']['mujoco_model_path'])
    control, pi, cal = config['control'], config['pi0'], config['calibration']
    control.setdefault('mode', 'q')
    control.setdefault('inference_runs', 1)
    control.setdefault('wait_for_start', False)
    if not isinstance(control['wait_for_start'], bool):
        raise ValueError('control.wait_for_start must be boolean')
    if not control['wait_for_start'] and control['inference_runs'] != 1:
        raise ValueError('multiple inference_runs requires wait_for_start=true')
    if control['mode'] not in ('q', 'mtc', 'tau'):
        raise ValueError('control.mode must be q, mtc or tau')
    if control['mode'] in ('mtc', 'tau'):
        if not wm['enable']:
            raise ValueError(f"control.mode={control['mode']} requires wm.enable=true for synchronized q/tau")
        from inference.pi0_wm.mtc import validate_mtc
        validate_mtc(control.setdefault('mtc', {}))
        control['mtc']['urdf_path'] = resolve(control['mtc']['urdf_path'])
    def positive(value, name, integer=False):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite')
        if integer and not isinstance(value, int):
            raise ValueError(f'{name} must be an integer')
    for name, value in [('control.hz', control['hz']), ('pi0.action_hz', pi['action_hz']),
                        ('control.inference_runs', control['inference_runs']),
                        ('execute_steps', control['execute_steps']), ('consume_steps', pi['consume_steps']),
                        ('num_samples', wm['num_samples']), ('minimum_samples', cal['minimum_samples']),
                        ('warmup_samples', cal['warmup_samples'])]:
        positive(value, name, integer=True)
    if control['hz'] != 100 or pi['action_hz'] != 25:
        raise ValueError('this deployment uses control=100 Hz and action=25 Hz')
    control.setdefault('command_hz', control['hz'])
    positive(control['command_hz'], 'control.command_hz')
    ratio = control['hz'] / control['command_hz']
    if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-9):
        raise ValueError('control.command_hz must divide control.hz exactly and cannot exceed it')
    if control['mode'] == 'tau' and control['command_hz'] != control['hz']:
        raise ValueError('control.mode=tau requires command_hz=100 for host torque feedback')
    if control['mode'] in ('mtc', 'tau') and control['mtc']['watchdog_timeout_s'] <= 1 / control['command_hz']:
        raise ValueError('MTC watchdog_timeout_s must exceed the command interval with jitter margin')
    for name in ('wm_seconds', 'pi_seconds', 'request_timeout_s'):
        positive(cal[name], name)
    for name in ('maximum_state_age_s', 'maximum_camera_age_s'):
        positive(control[name], name)
    positive(control['maximum_steps'], 'maximum_steps', integer=True)
    positive(pi['quaternion_tolerance'], 'quaternion_tolerance')
    for name in ('render_fps', 'prediction_visualization_hz', 'point_size'):
        positive(config['mujoco'][name], name)
    for name in ('wm_margin_s', 'pi_margin_s'):
        if not math.isfinite(cal[name]) or cal[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not isinstance(wm['selected_sample'], int) or isinstance(wm['selected_sample'], bool) or not 0 <= wm['selected_sample'] < wm['num_samples']:
        raise ValueError('selected_sample must identify one complete sampled trajectory')
    if wm['flow_steps'] is not None:
        positive(wm['flow_steps'], 'flow_steps', integer=True)
    if wm['solver'] not in (None, 'euler', 'heun'):
        raise ValueError('solver must be null/euler/heun')
    interface = pi['interface']
    pi.setdefault('anchor_camera', 'wrist' if 'wrist' in interface['images'] else next(iter(interface['images'])))
    if pi['anchor_camera'] not in interface['images']:
        raise ValueError('pi0.anchor_camera must name a configured image view')
    required = {'training_config', 'state_key', 'prompt_key', 'images', 'state_semantic',
                'action_semantic', 'coordinate_frame', 'representation', 'quaternion_order',
                'image_format', 'output_space', 'action_hz'}
    if set(interface) != required:
        raise ValueError(f'pi0.interface must contain {sorted(required)}')
    paths = [interface['state_key'], interface['prompt_key'], *interface['images'].values()]
    if any(not isinstance(p, str) or not all(p.split('/')) for p in paths):
        raise ValueError('pi0 observation paths must have nonempty string components')
    if len(set(paths)) != len(paths) or any(a.startswith(b + '/') for a in paths for b in paths if a != b):
        raise ValueError('pi0 observation paths must not overlap')
    if (interface['state_semantic'] != 'observation.ee_pose' or
            interface['action_semantic'] != 'action.ee_pose' or
            interface['coordinate_frame'] != 'link7' or interface['representation'] != 'xyz_quaternion' or
            interface['quaternion_order'] != 'xyzw' or interface['image_format'] != 'uint8_hwc_rgb' or
            interface['output_space'] != 'physical_absolute' or interface['action_hz'] != pi['action_hz']):
        raise ValueError('unsupported pi0 physical EE pose interface')
    config['cameras'] = [c for c in config['cameras'] if c.get('enabled', True)]
    names = [c['name'] for c in config['cameras']]
    if len(names) != len(set(names)) or set(names) != set(interface['images']):
        raise ValueError('camera sources must exactly match pi0 configured views')
    if config['hardware']['backend'] not in ('mock', 'pyagx'):
        raise ValueError('hardware.backend must be mock or pyagx')
    import numpy as np
    hw = config['hardware']
    low, high, limit = [np.asarray(hw[k], dtype=float) for k in ('q_min', 'q_max', 'maximum_step_rad')]
    if any(x.shape != (7,) or not np.isfinite(x).all() for x in (low, high, limit)) or np.any(low >= high) or np.any(limit <= 0):
        raise ValueError('hardware joint bounds/step limits must be finite seven-vectors')
    rest = np.asarray(hw['endpoint'].get('rest_q', []), dtype=float)
    if rest.shape != (7,) or not np.isfinite(rest).all() or np.any(rest < low) or np.any(rest > high):
        raise ValueError('hardware.endpoint.rest_q must be a finite seven-vector within joint bounds')
    return config
