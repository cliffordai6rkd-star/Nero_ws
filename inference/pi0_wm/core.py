"""Step-indexed plans and fixed scheduling; no hardware or model imports."""
from __future__ import annotations

from dataclasses import dataclass
import math
from queue import Queue, Empty
from threading import Thread
import time
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class Schedule:
    execute_steps: int
    prefetch_steps: int

    @property
    def trigger_step(self):
        return self.execute_steps - self.prefetch_steps

    @classmethod
    def calibrate(cls, execute_steps, latency_max, margin, hz, horizon):
        prefetch = math.ceil(math.nextafter((latency_max + margin) * hz, -math.inf))
        if prefetch > execute_steps:
            raise ValueError(f"single WM worker cannot sustain execute_steps={execute_steps}: "
                             f"prefetch_steps={prefetch}; reduce inference cost or explicitly change execute_steps")
        if prefetch + execute_steps > horizon:
            raise ValueError(f"prefetch + execute_steps={prefetch + execute_steps} exceeds external horizon={horizon}")
        return cls(execute_steps, prefetch)


@dataclass(frozen=True)
class Plan:
    loop_id: int
    request_id: int
    start_token: int
    actions: np.ndarray  # retain the FULL response, consume only length tokens
    length: int
    snapshot_anchor: float

    @property
    def end_token(self):
        return self.start_token + self.length

    def action_at(self, token, ratio):
        # Each source action retains its time relative to the observation.
        index = math.floor((token * ratio - self.snapshot_anchor) / ratio)
        if index < 0 or index >= len(self.actions):
            raise MissingActions(f'action token {token} outside request {self.request_id} validity')
        return self.actions[index]


class MissingActions(RuntimeError):
    pass


class Plans:
    def __init__(self, consume, ratio=4):
        self.consume = consume
        self.ratio = ratio
        self.plans: list[Plan] = []
        self.next_id = 0

    def add(self, actions, start_token, request_id, anchor):
        actions = np.asarray(actions, dtype=np.float32).copy()
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError("pi0 actions must be finite physical EE poses [T,7]")
        skipped = math.floor((start_token * self.ratio - anchor) / self.ratio)
        if skipped < 0:
            raise ValueError('plan cannot activate before its observation')
        available = len(actions) - skipped
        if available <= 0:
            raise MissingActions('returned pi0 chunk has expired before activation')
        if self.plans and start_token < self.plans[-1].end_token:
            raise ValueError("a committed plan boundary may not move or overlap")
        plan = Plan(self.next_id, request_id, start_token, actions, min(self.consume, available), anchor)
        self.next_id += 1
        self.plans.append(plan)
        return plan

    def active(self, step):
        token = step // self.ratio
        return next((p for p in self.plans if p.start_token <= token < p.end_token), None)

    def phase(self, step):
        plan = self.active(step)
        return (None if plan is None else plan.loop_id,
                None if plan is None else step // self.ratio - plan.start_token,
                step % self.ratio)

    def window(self, step, horizon, offset):
        start = step // self.ratio + offset
        tokens, versions = [], []
        for token in range(start, start + horizon):
            plan = next((p for p in self.plans if p.start_token <= token < p.end_token), None)
            if plan is None:
                raise MissingActions(f"no native action token {token} for anchor {step}")
            tokens.append(plan.action_at(token, self.ratio))
            if not versions or versions[-1] != plan.loop_id:
                versions.append(plan.loop_id)
        return np.stack(tokens), tuple(versions)

    def prune(self, step):
        # Retain all still-valid committed plans, including pending successors.
        self.plans[:] = [p for p in self.plans if p.end_token > step // self.ratio]


def aligned_pi_schedule(requested_consume, chunk_length, horizon, offset, latency_max, margin, action_hz):
    """Reserve the expired prefix and enough coverage for serial inference.

    A pending plan may need its successor before it becomes active. Never wait
    for activation to schedule that successor. One extra token covers phase.
    """
    lead = math.ceil(math.nextafter((latency_max + margin) * action_hz, -math.inf))
    lookahead = horizon + offset + lead
    consume = min(requested_consume, chunk_length - lookahead - 1)
    if consume <= lead:
        raise ValueError('pi0 chunk too short for time-aligned continuous inference: '
                         f'available consumption={consume}, latency reserve={lead}; '
                         'reduce latency/WM action horizon or use a longer trained pi0 chunk')
    return consume, lookahead


@dataclass(frozen=True)
class Request:
    request_id: int
    anchor: float  # WM uses integer state ticks; pi0 uses interpolated camera time.
    plan_versions: tuple[int, ...]
    payload: Any
    started: float


@dataclass(frozen=True)
class Result:
    request: Request
    value: Any
    elapsed: float
    error: BaseException | None = None


class Worker:
    """One permanent thread, one in-flight job including an unconsumed result."""
    def __init__(self, name: str, infer: Callable):
        self.infer = infer
        self.jobs = Queue(maxsize=1)
        self.results = Queue(maxsize=1)
        self.busy = False
        self.thread = Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def submit(self, request):
        if self.busy:
            raise RuntimeError("worker already has an in-flight request")
        self.busy = True
        self.jobs.put_nowait(request)

    def _run(self):
        while True:
            request = self.jobs.get()
            if request is None:
                return
            try:
                value = self.infer(request.payload)
                result = Result(request, value, time.perf_counter() - request.started)
            except BaseException as exc:
                result = Result(request, None, time.perf_counter() - request.started, exc)
            self.results.put(result)

    def poll(self):
        try:
            result = self.results.get_nowait()
        except Empty:
            return None
        self.busy = False
        if result.error is not None:
            raise RuntimeError(
                f"{self.thread.name} request {result.request.request_id} failed: "
                f"{type(result.error).__name__}: {result.error}"
            ) from result.error
        return result

    def close(self):
        # A remote infer may block. Never join it indefinitely on shutdown.
        try:
            self.jobs.put_nowait(None)
        except Exception:
            pass
        self.thread.join(timeout=0.2)


class Execution:
    def __init__(self, schedule, selected_sample=0, *, open_loop=False):
        self.schedule = schedule
        self.open_loop = open_loop
        self.start_step = None
        self.selected_sample = selected_sample
        self.current: Result | None = None
        self.pending: Result | None = None
        self.latest: Result | None = None
        self.wm_loop_id = -1
        self.wm_execute_step = 0
        self.requested = False
        self.overruns = 0
        self.rejected = 0
        self._late = False

    def receive(self, result):
        self.pending = self.latest = result

    def take_over(self, step):
        if self.pending is None or (self.current is not None and self.wm_execute_step < self.schedule.execute_steps):
            return False
        candidate = self.pending
        self.pending = None
        d = 0 if self.open_loop else step - candidate.request.anchor
        horizon = candidate.value['q'].shape[1]
        if d < 0 or d + self.schedule.execute_steps > horizon:
            self.rejected += 1
            self.requested = False
            return False
        self.current = candidate
        self.start_step = step
        self.wm_loop_id += 1
        self.wm_execute_step = 0  # ONLY at actual takeover
        self.requested = False
        self._late = False
        return True

    def should_request(self):
        trigger = self.schedule.execute_steps if self.open_loop else self.schedule.trigger_step
        return not self.requested and (self.current is None or self.wm_execute_step >= trigger)

    def command(self, step):
        if self.current is None:
            return None
        if self.open_loop and self.wm_execute_step >= self.schedule.execute_steps:
            return None
        if self.wm_execute_step >= self.schedule.execute_steps and not self._late:
            self.overruns += 1
            self._late = True
        index = self.wm_execute_step if self.open_loop else step - self.current.request.anchor
        q = self.current.value['q']
        if index < 0 or index >= q.shape[1]:
            return None  # caller holds the last successfully issued bounded q
        return q[self.selected_sample, index].copy()

    def advance(self):
        if self.current is not None and (not self.open_loop or self.wm_execute_step < self.schedule.execute_steps):
            self.wm_execute_step += 1

    def torque(self, step):
        """Use exactly the same result, sample and delay index as command()."""
        if self.current is None:
            return None
        if self.open_loop and self.wm_execute_step >= self.schedule.execute_steps:
            return None
        index = self.wm_execute_step if self.open_loop else step - self.current.request.anchor
        q = self.current.value['q']
        if index < 0 or index >= q.shape[1]:
            return None
        tau = self.current.value.get('tau')
        if tau is None or tau.shape != q.shape:
            raise ValueError('MTC requires WM tau predictions with the same shape as q')
        return tau[self.selected_sample, index].copy()

    def contact_phase(self, step):
        if self.current is None or (self.open_loop and self.wm_execute_step >= self.schedule.execute_steps):
            return None
        phase = self.current.value.get('contact_phase')
        index = self.wm_execute_step if self.open_loop else step - self.current.request.anchor
        if phase is None or not 0 <= index < phase.shape[1]:
            return None
        return int(phase[self.selected_sample, index])
