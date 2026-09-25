"""Single-arm rest reset using the collection reset defaults and move_j path."""
import logging
import math
import time

import numpy as np

from nero_collection.config import CommandConfig

log = logging.getLogger(__name__)


def reset_to_rest(runtime):
    if not (runtime.command_enabled or runtime.simulated_arm):
        log.info('dry-run: skipping rest reset')
        return
    hw = runtime.cfg['hardware']
    rest = np.asarray(hw['endpoint']['rest_q'], dtype=float)
    low, high = np.asarray(hw['q_min']), np.asarray(hw['q_max'])
    if rest.shape != (7,) or not np.isfinite(rest).all() or np.any(rest < low) or np.any(rest > high):
        raise ValueError('rest_q must be a finite seven-vector within joint bounds')
    settings = CommandConfig()
    target = rest.copy()
    deadline = None
    log.info('resetting follower to rest_q=%s', rest.tolist())
    while True:
        runtime.read_hardware_state()
        start = np.asarray(runtime.state.q, dtype=float)
        delta = float(np.max(np.abs(target - start)))
        duration = max(settings.reset_min_duration_s, delta / settings.reset_joint_speed_rad_s)
        steps = max(1, math.ceil(duration * settings.reset_interpolation_rate_hz),
                    math.ceil(delta / settings.reset_max_step_rad))
        began = time.monotonic()
        for index in range(1, steps + 1):
            runtime.read_hardware_state()
            command = start + (target - start) * (index / steps)
            runtime.arm.move_joints(command)
            runtime.held = command.copy()
            time.sleep(max(0, began + index / settings.reset_interpolation_rate_hz - time.monotonic()))
        if not runtime.arm.wait_motion_done(settings.reset_timeout_s):
            raise RuntimeError('rest reset motion timed out')
        if deadline is None:
            deadline = time.monotonic() + settings.reset_timeout_s
        time.sleep(settings.reset_wait_s)
        samples = []
        for index in range(settings.reset_test_sample_time):
            runtime.read_hardware_state()
            samples.append(runtime.state.q.copy())
            if index + 1 < settings.reset_test_sample_time:
                time.sleep(1 / settings.idle_rate_hz)
        error = rest - np.mean(samples, axis=0)
        maximum = float(np.max(np.abs(error)))
        if maximum <= settings.reset_error_limit_rad:
            log.info('rest reset passed: max joint error %.6f rad', maximum)
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f'rest reset self-check failed: max joint error {maximum:.6f} rad')
        target = np.clip(target + np.clip(error, -settings.joint_step_limit_rad,
                                          settings.joint_step_limit_rad), low, high)
