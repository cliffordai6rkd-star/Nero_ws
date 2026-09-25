"""Three permanent workers/loops: pi0, WM, and 100 Hz measured-state q control."""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from types import SimpleNamespace
import numpy as np

from nero_collection.config import ArmEndpointConfig, CameraConfig
from nero_collection.cameras import CameraManager, CameraVisualizer
from nero_collection.keyboard import TerminalKeys
from inference.pi0_wm.reset import reset_to_rest
from inference.pi0_wm.diagnostics import ChunkDiagnostics
from inference.pi0_wm.core import Plans, Worker, Request, Schedule, Execution, MissingActions, aligned_pi_schedule
from inference.pi0_wm.pi import PiClient, observation
from inference.pi0_wm.wm import WMAdapter, MockWM, History
from inference.pi0_wm.visualization import Visualizer

log = logging.getLogger(__name__)


class StopAndReset(Exception):
    """Terminal request to leave inference and park the arm."""


class PauseInference(Exception):
    """Pause the current trial without parking."""


class Runtime:
    def __init__(self, config, *, enable_commands=False, mock=False, mock_wm=False):
        self.cfg = config
        self.mock = mock
        if mock and enable_commands:
            raise ValueError('--mock cannot be combined with --enable-commands')
        self.command_enabled = enable_commands
        self.control_mode = config['control'].get('mode', 'q')
        self.mtc = None
        self.mtc_active = False
        self.mtc_sent_at = None
        self.hz = config['control']['hz']
        self.dt = 1 / self.hz
        self.command_dt = 1 / config['control'].get('command_hz', self.hz)
        self.command_stride = round(self.command_dt * self.hz)
        self.last_command_step = None
        self.step = 0
        self.request_id = 0
        self.deadline = time.monotonic()
        self.control_overruns = 0
        self.frames = {}
        self.pi_clock_samples = deque(maxlen=max(100, math.ceil(config['control']['maximum_camera_age_s'] * self.hz) + 4))
        self.held = None
        self.state = None
        self.pi_worker = self.wm_worker = None
        self.plans = Plans(config['pi0']['consume_steps'], int(self.hz / config['pi0']['action_hz']))
        self.wm_enabled = config['wm'].get('enable', True)
        self.wm_open_loop = config['wm'].get('inference_mode', 'prefetch') == 'openloop'
        self.wm_request_started = None
        self.execution = Execution(Schedule(config['control']['execute_steps'], 0), config['wm']['selected_sample'])
        if mock_wm and not self.wm_enabled:
            raise ValueError('--mock-wm requires wm.enable=true')
        if mock_wm and enable_commands:
            raise ValueError('mock WM is never allowed to command real hardware')
        self.ik = None
        if self.wm_enabled:
            self.wm = MockWM(config['wm']['num_samples']) if mock_wm or mock else WMAdapter(config['wm'], self.hz, config['pi0']['action_hz'])
        else:
            from inference.pi0_wm.ik import PoseIK
            self.ik = PoseIK(config['mujoco'], config['hardware'])
            self.wm = SimpleNamespace(history_horizon=1, operations={}, action_horizon=0, offset=0)
        self.history = History(self.wm.history_horizon, self.hz, self.wm.operations)
        if self.wm_enabled and self.wm_open_loop and config['control']['execute_steps'] > self.wm.future_horizon:
            raise ValueError('open-loop execute_steps exceeds WM prediction horizon')
        if self.control_mode in ('mtc', 'tau'):
            if not self.wm_enabled:
                raise ValueError('MTC requires WM q/tau predictions')
            if 'tau' not in getattr(self.wm, 'outputs', ('q', 'tau')):
                raise ValueError('MTC requires a WM checkpoint that predicts tau')
            from pathlib import Path
            from nero_collection.config import InverseDynamicsConfig
            from nero_collection.inverse_dynamics import PinocchioJointTorqueResidualEstimator
            from inference.pi0_wm.mtc import MtcController
            dynamics = PinocchioJointTorqueResidualEstimator(InverseDynamicsConfig(
                urdf_path=Path(config['control']['mtc']['urdf_path'])))
            self.mtc = MtcController(config['control']['mtc'], config['hardware'], dynamics.gravity_torque,
                                     torque_only=self.control_mode == 'tau')
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
        self.pi_requested_tail = None
        self.pi_handoff = 0
        self.last_plan = None
        self.keys = None
        self.direct_target = None
        self.direct_token = None

    def check_keys(self):
        key = self.keys.read_key(0) if self.keys is not None else None
        if key in ('i', 'I'):
            raise StopAndReset()
        if key in ('d', 'D') and getattr(self, 'interactive_session', False):
            raise PauseInference()

    def set_contact_phase(self, phase):
        preview = getattr(self.cameras, 'visualizer', None)
        if preview is not None and hasattr(preview, 'set_contact_phase'):
            preview.set_contact_phase(phase)

    def next_request(self, anchor, versions, payload, started):
        request = Request(self.request_id, anchor, versions, payload, started)
        self.request_id += 1
        return request

    def log_pi_chunk(self):
        # Snapshot the latest WM input q before the asynchronous logger consumes it.
        self.diagnostics.emit({'q': self.history.rows[-1]['q'].copy()})

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

    def read_hardware_state(self):
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
        q = np.asarray(self.state.q)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise RuntimeError('invalid measured joint positions')
        if np.any(q < self.cfg['hardware']['q_min']) or np.any(q > self.cfg['hardware']['q_max']):
            raise RuntimeError('measured joint positions outside hardware bounds')

    def acquire(self):
        # Fetch queued frames before the new state sample so their timestamps
        # can be bracketed even for cameras that stamp synchronously on poll.
        for frame in self.cameras.poll():
            self.frames[frame.camera_name] = frame
        self.read_hardware_state()
        # Keep the actual wall-clock/step mapping and causal state snapshots.
        # Cameras use the same host wall clock; never infer their age solely
        # from the number of nominal 100 Hz ticks.
        self.pi_clock_samples.append((time.time(), self.step,
                                      SimpleNamespace(ee_pose=np.asarray(self.state.ee_pose).copy())))
        if self.held is None:
            self.held = np.asarray(self.state.q).copy()
        self.history.append(self.step, self.state, self.held, time.monotonic())

    def enter_mtc(self):
        if getattr(self, 'control_mode', 'q') not in ('mtc', 'tau'):
            return
        # Anchor MIT to actual feedback, rather than the last position-mode
        # target which may still have tracking error after reset.
        self.held = np.asarray(self.state.q).copy()
        self.mtc.reset(self.state.q, self.state.torque)
        if self.command_enabled or self.simulated_arm:
            # Mark before switching so even a partial mode-switch failure
            # attempts to restore position mode during cleanup.
            self.mtc_active = True
            self.arm.configure_joint_impedance_mode()
        else:
            self.mtc_active = True
        self.mtc_sent_at = None
        self.last_command_step = None
        self.send(None)

    def leave_mtc(self):
        if not getattr(self, 'mtc_active', False):
            return
        if self.command_enabled or self.simulated_arm:
            # move_joints explicitly restores SDK automatic mode selection,
            # then selects position mode at the last successful position target.
            self.arm.move_joints(self.held)
        self.mtc_active = False
        self.mtc_sent_at = None

    def send(self, target, tau=None):
        # Advance state/history/prediction time at full rate. Never queue the
        # targets skipped while decimating the physical command stream.
        now = time.monotonic()
        if getattr(self, 'mtc_active', False) and self.mtc_sent_at is not None:
            if now - self.mtc_sent_at > self.mtc.cfg['watchdog_timeout_s']:
                raise RuntimeError('MTC command interval exceeded watchdog')
        previous_step = getattr(self, 'last_command_step', None)
        if (getattr(self, 'command_stride', 1) > 1 and previous_step is not None
                and self.step - previous_step < self.command_stride):
            self.visualizer.update(self.step, self.state.q, self.execution)
            return
        if getattr(self, 'mtc_active', False):
            dt = getattr(self, 'command_dt', self.dt) if self.mtc_sent_at is None else now - self.mtc_sent_at
            reference = self.mtc.prepare(target, tau, self.state, self.held, dt)
            if self.command_enabled or self.simulated_arm:
                torque_only = getattr(self, 'control_mode', 'q') == 'tau'
                self.arm.command_joint_impedance(np.zeros(7) if torque_only else reference.q,
                                                 np.zeros(7) if torque_only else reference.velocity, reference.kp,
                                                 reference.kd, reference.feedforward)
                self.held = reference.q.copy()
                self.mtc.commit(reference)
            self.mtc_sent_at = now
            self.last_command_step = self.step
            self.visualizer.update(self.step, self.state.q, self.execution)
            return
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
        self.last_command_step = self.step
        self.visualizer.update(self.step, self.state.q, self.execution)

    def pi_snapshot(self):
        now = time.time()
        names = self.cfg['pi0']['interface']['images']
        if any(name not in self.frames or not 0 <= now - self.frames[name].timestamp_us / 1e6 <=
               self.cfg['control']['maximum_camera_age_s'] for name in names):
            raise MissingActions('waiting for a fresh complete pi0 camera snapshot')
        camera = self.cfg['pi0'].get('anchor_camera', 'wrist' if 'wrist' in names else next(iter(names)))
        frame_time = self.frames[camera].timestamp_us / 1e6
        samples = self.pi_clock_samples
        if not samples or frame_time < samples[0][0] or frame_time > samples[-1][0]:
            raise MissingActions('waiting for state history bracketing the pi0 camera timestamp')
        times = np.array([s[0] for s in samples])
        steps = np.array([s[1] for s in samples])
        self.pi_snapshot_anchor = float(np.interp(frame_time, times, steps))
        self.pi_snapshot_age_s = now - frame_time
        state_index = int(np.searchsorted(times, frame_time, side='right')) - 1
        return observation(self.cfg['pi0'], samples[state_index][2], self.frames)

    def submit_pi(self):
        started = time.perf_counter()
        obs = self.pi_snapshot()
        if not self.wm_enabled:
            obs = (obs, self.state.q.copy())
        current = self.plans.active(self.step)
        versions = () if current is None else (current.loop_id,)
        request = self.next_request(self.pi_snapshot_anchor, versions, obs, started)
        self.pi_worker.submit(request)
        return request

    def infer_pi(self, payload):
        if self.wm_enabled:
            return self._mock_pi(payload) if self.mock else self.pi.infer(payload)
        obs, seed = payload
        poses = (np.tile(self.ik.pose(seed), (50, 1)) if self.mock else self.pi.infer(obs))
        # Solve off the control thread. Successive poses use the previous solution
        # as seed; no WM prediction or feedback correction enters the chunk.
        return self.ik.chunk(poses, seed)

    def direct_command(self):
        plan = self.plans.active(self.step)
        if plan is None:
            self.direct_token = None
            return None
        token = self.step // self.plans.ratio
        key = (plan.loop_id, token)
        if key != self.direct_token:
            self.direct_target = plan.action_at(token, self.plans.ratio).copy()
            self.direct_token = key
        return self.direct_target

    def submit_wm(self):
        started = time.perf_counter()
        action, versions = self.plans.window(self.step, self.wm.action_horizon, self.wm.offset)
        request = self.next_request(self.step, versions, (self.history.snapshot(), action), started)
        self.wm_worker.submit(request)
        self.wm_request_started = started
        self.execution.requested = True
        log.debug('WM submit req=%s anchor=%s plans=%s execute_step=%s',
                  request.request_id, request.anchor, versions, self.execution.wm_execute_step)

    def pump_hold(self):
        self.check_keys()
        self.send(None)
        self.wait_cycle()
        self.acquire()

    def await_result(self, worker):
        deadline = time.monotonic() + self.cfg['calibration']['request_timeout_s']
        while True:
            self.check_keys()
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
        chunk_lengths = []
        workers = [('pi0', self.pi_worker)]
        if self.wm_enabled:
            workers.append(('WM', self.wm_worker))
        for name, worker in workers:
            timings = []
            count = 0
            measurement_started = None
            seconds = cal['pi_seconds' if name == 'pi0' else 'wm_seconds']
            alignment_deadline = time.monotonic() + cal['request_timeout_s']
            while measurement_started is None or len(timings) < cal['minimum_samples'] or time.monotonic() - measurement_started < seconds:
                started = time.perf_counter()
                if name == 'pi0':
                    try:
                        self.submit_pi()
                    except MissingActions:
                        if time.monotonic() > alignment_deadline:
                            raise TimeoutError('pi0 calibration timed out waiting for aligned cameras')
                        self.pump_hold()
                        continue
                    camera_age = getattr(self, 'pi_snapshot_age_s', 0.0)
                else:
                    request = self.next_request(self.step, (), (self.history.snapshot(), action[:self.wm.action_horizon]), started)
                    worker.submit(request)
                result = self.await_result(worker)
                alignment_deadline = time.monotonic() + cal['request_timeout_s']
                elapsed = time.perf_counter() - started  # includes prepare, worker transfer and poll availability
                if name == 'pi0':
                    elapsed += camera_age  # Plan validity starts at the image, not submission.
                if name == 'pi0':
                    action = result.value
                    self.log_pi_chunk()
                    chunk_lengths.append(len(action))
                    if len(action) < self.wm.action_horizon + self.wm.offset:
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
                self.plans.consume, self.pi_lookahead = aligned_pi_schedule(
                    self.cfg['pi0']['consume_steps'], min(chunk_lengths), self.wm.action_horizon,
                    self.wm.offset, maximum, cal['pi_margin_s'], self.cfg['pi0']['action_hz'])
                log.info('pi0 aligned schedule: consume=%s requested_max=%s lookahead=%s',
                         self.plans.consume, self.cfg['pi0']['consume_steps'], self.pi_lookahead)
            else:
                if self.wm_open_loop:
                    schedule = Schedule(self.cfg['control']['execute_steps'], 0)
                else:
                    schedule = Schedule.calibrate(self.cfg['control']['execute_steps'], maximum,
                                                  cal['wm_margin_s'], self.hz, self.wm.future_horizon)
                self.execution = Execution(schedule, self.cfg['wm']['selected_sample'], open_loop=self.wm_open_loop)
                log.info('WM execution mode=%s', 'openloop' if self.wm_open_loop else 'prefetch')
                if self.wm_open_loop:
                    play_seconds = schedule.execute_steps / self.hz
                    log.info('WM openloop timing: playback=%.3fs measured_peak_wait=%.3fs estimated_playback_fraction=%.1f%%',
                             play_seconds, maximum, 100 * play_seconds / (play_seconds + maximum))
                log.info('WM fixed schedule: peak=%.6fs margin=%.6fs prefetch_steps=%s trigger_step=%s execute_steps=%s horizon=%s',
                         maximum, cal['wm_margin_s'], schedule.prefetch_steps, schedule.trigger_step,
                         schedule.execute_steps, self.wm.future_horizon)

    def pi_update(self):
        result = self.pi_worker.poll()
        if result is not None:
            # A missing-plan recovery gets an explicit new 25 Hz boundary;
            # already committed cross-chunk boundaries never move.
            start = max(self.pi_handoff, math.ceil(self.step / self.plans.ratio))
            self.log_pi_chunk()
            try:
                new = self.plans.add(result.value, start, result.request.request_id, result.request.anchor)
            except MissingActions:
                log.warning('discard expired pi0 request=%s', result.request.request_id)
                self.pi_requested_tail = None
            else:
                self.last_plan = new
                log.info('pi0 aligned plan loop=%s request=%s snapshot=%s start=%s end=%s skipped=%s',
                         new.loop_id, new.request_id, new.snapshot_anchor, new.start_token, new.end_token,
                         (new.start_token * self.plans.ratio - new.snapshot_anchor) // self.plans.ratio)
        self.plans.prune(self.step)
        if self.pi_worker.busy:
            return
        # Schedule against the last committed plan, even before it activates.
        # Waiting for active() would strand short, latency-cropped plans.
        tail = self.plans.plans[-1] if self.plans.plans else None
        if tail is not None:
            if (self.step // self.plans.ratio < tail.end_token - self.pi_lookahead
                    or self.pi_requested_tail == tail.loop_id):
                return
            self.pi_handoff = tail.end_token
        else:
            self.pi_handoff = math.ceil(self.step / self.plans.ratio)
        try:
            self.submit_pi()
            self.pi_requested_tail = None if tail is None else tail.loop_id
        except MissingActions:
            pass

    def run(self, maximum_steps=None):
        with ChunkDiagnostics() as diagnostics, TerminalKeys() as keys:
            self.diagnostics = diagnostics
            self.keys = keys
            self.interactive_session = self.cfg['control'].get('wait_for_start', False)
            if self.interactive_session and not keys.is_tty:
                raise RuntimeError('interactive inference requires a terminal; use control.wait_for_start=false for unattended runs')
            try:
                return self._run(maximum_steps)
            finally:
                self.keys = None

    def _run(self, maximum_steps=None):
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
            reset_to_rest(self)
            self.history = History(self.wm.history_horizon, self.hz, self.wm.operations)
            self.step += 1
            self.acquire()
            self.last_command_step = None
            log.info('state/control rate=%s Hz command rate=%s Hz', self.hz,
                     self.cfg['control'].get('command_hz', self.hz))
            log.info('keys: s=start/resume, d=pause, i=end trial and reset; Ctrl+C=exit' if getattr(self, 'interactive_session', False)
                     else 'press i to stop inference, reset to rest_q, and exit')
            log.info('control mode=%s command_enabled=%s simulated_arm=%s', self.control_mode, self.command_enabled, self.simulated_arm)
            self.pi_worker = Worker('pi0', self.infer_pi)
            if self.wm_enabled:
                self.wm_worker = Worker('WM', self.wm.infer)
            log.info('inference mode: %s', 'pi0 + WM' if self.wm_enabled else 'pi0 EE pose -> IK -> open-loop q')
            self.deadline = time.monotonic()
            if getattr(self, 'interactive_session', False):
                return self._session_loop(limit)
            self.enter_mtc()
            self.calibrate()
            self.round_steps_done = 0
            return self._execute_round_steps(limit)
        except StopAndReset:
            log.info('i pressed: stopping inference and resetting to rest_q')
            self.leave_mtc()
            for worker in (self.pi_worker, self.wm_worker):
                if worker is not None:
                    worker.close()
            self.pi_worker = self.wm_worker = None
            reset_to_rest(self)
            return {'stopped_by': 'i', 'reset_to_rest': self.command_enabled or self.simulated_arm}
        finally:
            self.set_contact_phase(None)
            # Hold the last successful target, including an explicit rest reset.
            try:
                self.leave_mtc()
            except Exception:
                log.exception('failed to restore position mode from MTC')
            if self.held is not None and self.command_enabled and not getattr(self, 'mtc_active', False):
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

    def _hold_trial(self):
        self.set_contact_phase(None)
        self.read_hardware_state()
        self.held = self.state.q.copy()
        self.leave_mtc()
        self.last_command_step = None
        self.send(None)

    def _discard_trial_results(self):
        for worker in (self.pi_worker, self.wm_worker):
            if worker is not None:
                worker.poll()  # Drain old results, including errors; never install them.

    def _prepare_trial(self):
        # Reuse permanent workers. Starting another WM job while an old one
        # still runs would race the same model/CUDA context.
        deadline = time.monotonic() + self.cfg['calibration']['request_timeout_s']
        while any(w is not None and w.busy for w in (self.pi_worker, self.wm_worker)):
            self.check_keys()
            self._discard_trial_results()
            if time.monotonic() > deadline:
                raise TimeoutError('previous trial inference did not finish before restart')
            self.pump_hold()
        self.plans = Plans(self.cfg['pi0']['consume_steps'], int(self.hz / self.cfg['pi0']['action_hz']))
        self.pi_requested_tail = self.last_plan = None
        self.pi_handoff = math.ceil(self.step / self.plans.ratio)
        self.direct_target = self.direct_token = None
        self.wm_request_started = None
        self.execution = Execution(Schedule(self.cfg['control']['execute_steps'], 0),
                                   self.cfg['wm']['selected_sample'], open_loop=self.wm_open_loop)
        self.history = History(self.wm.history_horizon, self.hz, self.wm.operations)
        self.pi_clock_samples.clear()
        self.deadline = time.monotonic()
        self.enter_mtc()
        cached = getattr(self, 'session_calibration', None)
        if cached is None:
            self.calibrate()
            self.session_calibration = (self.plans.consume, self.pi_lookahead, self.execution.schedule)
        else:
            self.plans.consume, self.pi_lookahead, schedule = cached
            self.execution = Execution(schedule, self.cfg['wm']['selected_sample'], open_loop=self.wm_open_loop)
            deadline = time.monotonic() + self.cfg['calibration']['request_timeout_s']
            while True:
                self.pump_hold()
                try:
                    self.pi_snapshot()
                    if self.history.ready:
                        break
                except MissingActions:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('trial restart requires fresh history and camera frames')

    def _session_loop(self, limit):
        runs = self.cfg['control']['inference_runs']
        completed, active = [], False
        self.round_steps_done = 0
        log.info('ready: press s to start trial 1/%s', runs)
        while len(completed) < runs:
            key = self.keys.read_key(0)
            key = key.lower() if key else None
            finish = None
            if key == 'i':
                self._hold_trial()
                reset_to_rest(self)
                self.deadline = time.monotonic()
                if active:
                    finish = 'i'
            elif key == 's':
                if not active:
                    self.round_steps_done = 0
                    active = True
                log.info('starting/resuming trial %s/%s at execution step %s', len(completed)+1, runs, self.round_steps_done)
                try:
                    self._prepare_trial()
                    self._execute_round_steps(limit)
                    finish = 'maximum_steps'
                except PauseInference:
                    self._hold_trial()
                    log.info('trial paused: s=resume with fresh predictions, i=end and reset')
                except StopAndReset:
                    finish = 'i'
                if finish is not None:
                    self._hold_trial()
                    reset_to_rest(self)
                    self.deadline = time.monotonic()
            if finish is not None:
                completed.append({'trial': len(completed)+1, 'stopped_by': finish,
                                  'execution_steps': self.round_steps_done})
                active = False
                log.info('trial %s/%s finished and reset', len(completed), runs)
                if len(completed) == runs:
                    break
                log.info('press s to start next trial')
            self._discard_trial_results()
            self.wait_cycle()
            self.acquire()
            self.send(None)
        return {'completed_runs': len(completed), 'trials': completed}

    def _execute_round_steps(self, limit):
        while self.round_steps_done < limit:
            self.check_keys()
            self.pi_update()
            if not self.wm_enabled:
                target = self.direct_command()
            else:
                result = self.wm_worker.poll()
                if result is not None:
                    self.wm_request_started = None
                    self.execution.receive(result)
                if (self.wm_open_loop and self.wm_worker.busy and self.wm_request_started is not None
                        and time.perf_counter() - self.wm_request_started > self.cfg['calibration']['request_timeout_s']):
                    raise TimeoutError('open-loop WM inference timed out while holding position')
                rejected = self.execution.rejected
                command_due = (self.last_command_step is None or
                               self.step - self.last_command_step >= self.command_stride)
                if (not self.wm_open_loop or command_due) and self.execution.take_over(self.step):
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
            tau = self.execution.torque(self.step) if getattr(self, 'mtc_active', False) else None
            self.set_contact_phase(self.execution.contact_phase(self.step) if self.wm_enabled else None)
            self.send(target, tau)
            self.execution.advance()
            self.round_steps_done += 1
            if self.step % 100 == 0:
                if self.wm_enabled:
                    log.info('step=%s pi_loop/action/substep=%s wm_loop=%s execute_step=%s holds=%s',
                             self.step, self.plans.phase(self.step), self.execution.wm_loop_id,
                             self.execution.wm_execute_step, target is None)
                else:
                    log.info('step=%s pi_loop/action/substep=%s direct_ik holds=%s',
                             self.step, self.plans.phase(self.step), target is None)
            self.wait_cycle()
            self.acquire()
        return {'wm_loops': self.execution.wm_loop_id + 1, 'pi_loops': self.plans.next_id,
                'wm_overruns': self.execution.overruns, 'expired_results': self.execution.rejected,
                'control_overruns': self.control_overruns}

    def _mock_pi(self, obs):
        time.sleep(0.04)
        value = obs[self.cfg['pi0']['interface']['state_key']]
        return np.tile(value, (50, 1))
