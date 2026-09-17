"""Three permanent workers/loops: pi0, WM, and 100 Hz measured-state q control."""
from __future__ import annotations

import logging
import math
import time
import numpy as np

from nero_collection.config import ArmEndpointConfig, CameraConfig
from nero_collection.cameras import CameraManager, CameraVisualizer
from inference.pi0_wm.core import Plans, Worker, Request, Schedule, Execution, MissingActions, pi_trigger
from inference.pi0_wm.pi import PiClient, observation
from inference.pi0_wm.wm import WMAdapter, MockWM, History
from inference.pi0_wm.visualization import Visualizer

log = logging.getLogger(__name__)


class Runtime:
    def __init__(self, config, *, enable_commands=False, mock=False, mock_wm=False):
        self.cfg = config
        self.mock = mock
        if mock and enable_commands:
            raise ValueError('--mock cannot be combined with --enable-commands')
        self.command_enabled = enable_commands
        self.hz = config['control']['hz']
        self.dt = 1 / self.hz
        self.step = 0
        self.request_id = 0
        self.deadline = time.monotonic()
        self.control_overruns = 0
        self.frames = {}
        self.held = None
        self.state = None
        self.pi_worker = self.wm_worker = None
        self.plans = Plans(config['pi0']['consume_steps'], int(self.hz / config['pi0']['action_hz']))
        self.execution = Execution(Schedule(config['control']['execute_steps'], 0), config['wm']['selected_sample'])
        self.wm = MockWM(config['wm']['num_samples']) if mock_wm or mock else WMAdapter(config['wm'], self.hz, config['pi0']['action_hz'])
        if mock_wm and enable_commands:
            raise ValueError('mock WM is never allowed to command real hardware')
        self.history = History(self.wm.history_horizon, self.hz, self.wm.operations, self.wm.dq_source)
        self.simulated_arm = mock or config['hardware']['backend'] == 'mock'
        if enable_commands and self.simulated_arm:
            raise ValueError('--enable-commands requires hardware.backend=pyagx')
        if self.simulated_arm:
            from nero_collection.arms.mock import MockArm
            arm_type = MockArm
        else:
            from nero_collection.arms.pyagx import PyAgxArmAdapter
            arm_type = PyAgxArmAdapter
        self.arm = arm_type(ArmEndpointConfig(**config['hardware']['endpoint']))
        camera_configs = tuple(CameraConfig(**{**c, **({'backend': 'mock', 'visualize': False} if mock else {})}) for c in config['cameras'])
        self.cameras = CameraManager.from_config(camera_configs, CameraVisualizer.from_config(camera_configs))
        self.visualizer = Visualizer(config['mujoco'])
        self.pi = PiClient(config['pi0'])
        self.requested_plans = set()
        self.pi_handoff = 0
        self.last_plan = None

    def next_request(self, anchor, versions, payload, started):
        request = Request(self.request_id, anchor, versions, payload, started)
        self.request_id += 1
        return request

    def wait_cycle(self):
        self.deadline += self.dt
        delay = self.deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self.control_overruns += 1
            # No burst of synthetic catch-up commands/history samples.
            self.deadline = time.monotonic()
        self.step += 1

    def acquire(self):
        self.state = self.arm.read_state()
        now = time.time()
        if not self.simulated_arm:
            stamps = self.state.q_component_timestamp_us
            if len(stamps) == 0 or np.any(stamps <= 0):
                raise RuntimeError('missing hardware joint feedback timestamps')
            age = max(now - float(np.min(stamps)) / 1e6,
                      now - self.state.timestamp_us / 1e6)
            motors = self.state.motor_timestamp_us
            if len(motors) != 7 or np.any(motors <= 0):
                raise RuntimeError('missing measured velocity/torque feedback timestamps')
            age = max(age, now - float(np.min(motors)) / 1e6)
            if age > self.cfg['control']['maximum_state_age_s']:
                raise RuntimeError(f'stale hardware state: {age:.3f}s')
        if self.held is None:
            self.held = np.asarray(self.state.q).copy()
        self.history.append(self.step, self.state, self.held, time.monotonic())
        for frame in self.cameras.poll():
            self.frames[frame.camera_name] = frame

    def send(self, target):
        hw = self.cfg['hardware']
        if target is None:
            target = self.held
        target = np.asarray(target, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError('invalid q command')
        limit = np.asarray(hw['maximum_step_rad'])
        target = np.clip(target, np.maximum(self.held - limit, hw['q_min']),
                         np.minimum(self.held + limit, hw['q_max']))
        if self.command_enabled or self.simulated_arm:
            self.arm.command_joint_positions(target)
            # Update ONLY after a successful hardware send, including clipping.
            self.held = target.copy()
        # Physical dry-run does not pretend an unsent predicted q was applied.
        self.visualizer.update(self.step, self.state.q, self.execution)

    def pi_snapshot(self):
        now = time.time()
        names = self.cfg['pi0']['interface']['images']
        if any(name not in self.frames or now - self.frames[name].timestamp_us / 1e6 >
               self.cfg['control']['maximum_camera_age_s'] for name in names):
            raise MissingActions('waiting for a fresh complete pi0 camera snapshot')
        return observation(self.cfg['pi0'], self.state, self.frames)

    def submit_pi(self):
        started = time.perf_counter()
        obs = self.pi_snapshot()
        current = self.plans.active(self.step)
        versions = () if current is None else (current.loop_id,)
        request = self.next_request(self.step, versions, obs, started)
        self.pi_worker.submit(request)
        return request

    def submit_wm(self):
        started = time.perf_counter()
        action, versions = self.plans.window(self.step, self.wm.action_horizon, self.wm.offset)
        request = self.next_request(self.step, versions, (self.history.snapshot(), action), started)
        self.wm_worker.submit(request)
        self.execution.requested = True
        log.debug('WM submit req=%s anchor=%s plans=%s execute_step=%s',
                  request.request_id, request.anchor, versions, self.execution.wm_execute_step)

    def pump_hold(self):
        self.send(None)
        self.wait_cycle()
        self.acquire()

    def await_result(self, worker):
        deadline = time.monotonic() + self.cfg['calibration']['request_timeout_s']
        while True:
            result = worker.poll()
            if result is not None:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{worker.thread.name} startup calibration timed out')
            self.pump_hold()

    def calibrate(self):
        cal = self.cfg['calibration']
        deadline = time.monotonic() + cal['request_timeout_s']
        while True:
            try:
                self.pi_snapshot()
                if self.history.ready:
                    break
            except MissingActions:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('startup requires full real history and all configured cameras')
            self.pump_hold()
        action = None
        for name, worker in [('pi0', self.pi_worker), ('WM', self.wm_worker)]:
            timings = []
            count = 0
            measurement_started = None
            seconds = cal['pi_seconds' if name == 'pi0' else 'wm_seconds']
            while measurement_started is None or len(timings) < cal['minimum_samples'] or time.monotonic() - measurement_started < seconds:
                started = time.perf_counter()
                if name == 'pi0':
                    self.submit_pi()
                else:
                    request = self.next_request(self.step, (), (self.history.snapshot(), action[:self.wm.action_horizon]), started)
                    worker.submit(request)
                result = self.await_result(worker)
                elapsed = time.perf_counter() - started  # includes prepare, worker transfer and poll availability
                if name == 'pi0':
                    action = result.value
                    if len(action) < max(self.plans.consume, self.wm.action_horizon + self.wm.offset):
                        raise ValueError('pi0 chunk shorter than configured consumption / WM condition window')
                else:
                    self.execution.latest = result
                    self.visualizer.update(self.step, self.state.q, self.execution)
                count += 1
                if count == cal['warmup_samples']:
                    measurement_started = time.monotonic()
                elif count > cal['warmup_samples']:
                    timings.append(elapsed)
            maximum = max(timings)
            log.info('%s calibration: samples=%s duration=%.3fs peak=%.6fs', name, len(timings),
                     time.monotonic() - measurement_started, maximum)
            if name == 'pi0':
                self.pi_trigger_step = pi_trigger(self.plans.consume, self.wm.action_horizon, self.wm.offset,
                                                  maximum, cal['pi_margin_s'], self.cfg['pi0']['action_hz'])
                log.info('pi0 fixed action trigger=%s margin=%.3fs consume=%s window=%s offset=%s',
                         self.pi_trigger_step, cal['pi_margin_s'], self.plans.consume, self.wm.action_horizon, self.wm.offset)
            else:
                schedule = Schedule.calibrate(self.cfg['control']['execute_steps'], maximum,
                                              cal['wm_margin_s'], self.hz, self.wm.future_horizon)
                self.execution = Execution(schedule, self.cfg['wm']['selected_sample'])
                log.info('WM fixed schedule: peak=%.6fs margin=%.6fs prefetch_steps=%s trigger_step=%s execute_steps=%s horizon=%s',
                         maximum, cal['wm_margin_s'], schedule.prefetch_steps, schedule.trigger_step,
                         schedule.execute_steps, self.wm.future_horizon)

    def pi_update(self):
        result = self.pi_worker.poll()
        if result is not None:
            # A missing-plan recovery gets an explicit new 25 Hz boundary;
            # already committed cross-chunk boundaries never move.
            start = max(self.pi_handoff, math.ceil(self.step / self.plans.ratio))
            old = self.last_plan
            new = self.plans.add(result.value, start, result.request.request_id, result.request.anchor)
            self.last_plan = new
            if old is not None:
                jump = new.actions[0] - old.actions[old.length - 1]
                log.info('pi0 boundary old=%s new=%s token=%s jump_xyz_xyzw=%s gap_tokens=%s',
                         old.loop_id, new.loop_id, start, jump.tolist(), start - old.end_token)
            log.info('pi0 pending loop=%s request=%s snapshot=%s activation_token=%s observed_age_steps=%s',
                     new.loop_id, new.request_id, new.snapshot_anchor, start, self.step - new.snapshot_anchor)
        self.plans.prune(self.step)
        active = self.plans.active(self.step)
        if self.pi_worker.busy:
            return
        if active is None:
            if self.plans.plans:  # returned recovery plan activates at next token boundary
                return
            self.pi_handoff = math.ceil(self.step / self.plans.ratio)
        else:
            phase = self.step // self.plans.ratio - active.start_token
            if phase < self.pi_trigger_step or active.loop_id in self.requested_plans:
                return
            self.pi_handoff = active.end_token
        try:
            self.submit_pi()
            if active is not None:
                self.requested_plans = {active.loop_id}
        except MissingActions:
            pass  # latest snapshot only, never enqueue incomplete/old images

    def run(self, maximum_steps=None):
        limit = self.cfg['control']['maximum_steps'] if maximum_steps is None else maximum_steps
        if not self.mock and not self.cfg['pi0']['interface']['training_config']:
            raise ValueError('set pi0.interface.training_config to the actual verified Nero LoRA config')
        try:
            self.arm.connect()
            self.cameras.start()
            self.visualizer.start()
            self.acquire()
            q = self.state.q
            if np.any(q < self.cfg['hardware']['q_min']) or np.any(q > self.cfg['hardware']['q_max']):
                raise ValueError('initial measured q outside configured hardware bounds')
            if self.command_enabled:
                self.arm.set_follower_mode()
                self.arm.enable()
                self.send(q)
            log.info('position-only runtime: command_enabled=%s simulated_arm=%s', self.command_enabled, self.simulated_arm)
            self.pi_worker = Worker('pi0', self._mock_pi if self.mock else self.pi.infer)
            self.wm_worker = Worker('WM', self.wm.infer)
            self.deadline = time.monotonic()
            self.calibrate()
            start = self.step
            while self.step - start < limit:
                self.pi_update()
                result = self.wm_worker.poll()
                if result is not None:
                    self.execution.receive(result)
                rejected = self.execution.rejected
                if self.execution.take_over(self.step):
                    r = self.execution.current.request
                    log.info('WM takeover loop=%s request=%s anchor=%s d=%s plans=%s',
                             self.execution.wm_loop_id, r.request_id, r.anchor, self.step - r.anchor, r.plan_versions)
                if self.execution.rejected != rejected:
                    log.warning('discard expired WM result at step=%s: d+execute_steps exceeds horizon', self.step)
                if not self.wm_worker.busy and self.execution.should_request():
                    try:
                        self.submit_wm()
                    except MissingActions:
                        pass
                overruns = self.execution.overruns
                target = self.execution.command(self.step)
                if self.execution.overruns != overruns:
                    log.warning('WM overrun loop=%s step=%s; consume valid tail then hold', self.execution.wm_loop_id, self.step)
                self.send(target)
                self.execution.advance()
                if self.step % 100 == 0:
                    log.info('step=%s pi_loop/action/substep=%s wm_loop=%s execute_step=%s holds=%s',
                             self.step, self.plans.phase(self.step), self.execution.wm_loop_id,
                             self.execution.wm_execute_step, target is None)
                self.wait_cycle()
                self.acquire()
            return {'wm_loops': self.execution.wm_loop_id + 1, 'pi_loops': self.plans.next_id,
                    'wm_overruns': self.execution.overruns, 'expired_results': self.execution.rejected,
                    'control_overruns': self.control_overruns}
        finally:
            # Preserve an enabled position hold; never initiate an exit/reset move.
            if self.held is not None and self.command_enabled:
                try:
                    self.arm.command_joint_positions(self.held)
                except Exception:
                    log.exception('final position hold failed')
            for worker in (self.pi_worker, self.wm_worker):
                if worker is not None:
                    worker.close()
            self.visualizer.close()
            self.cameras.stop()
            self.arm.disconnect()

    def _mock_pi(self, obs):
        time.sleep(0.04)
        value = obs
        for key in self.cfg['pi0']['interface']['state_key'].split('/'):
            value = value[key]
        return np.tile(value, (50, 1))
