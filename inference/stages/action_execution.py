"""Timestamp-driven execution of a high-level action plan."""

from __future__ import annotations

import numpy as np


class ActionPlanExecutor:
    """Hold and advance a resolved action plan without knowing its semantics."""

    def __init__(self) -> None:
        self.plan: np.ndarray | None = None
        self.index = 0
        self.next_step_s: float | None = None
        self.step_s: float | None = None
        self.active = False
        self.action: np.ndarray | None = None
        self.action_chunk: np.ndarray | None = None
        self.held_action: np.ndarray | None = None

    def reset(self) -> None:
        self.plan = None
        self.index = 0
        self.next_step_s = None
        self.step_s = None
        self.active = False
        self.action = None
        self.action_chunk = None
        self.held_action = None

    def set_idle(self, action: np.ndarray, *, future_horizon: int) -> None:
        value = np.asarray(action, dtype=np.float64).reshape(-1)
        if value.ndim != 1 or value.size != 7 or not np.all(np.isfinite(value)):
            raise ValueError("idle action must be a finite seven-vector")
        horizon = max(1, int(future_horizon))
        self.plan = np.repeat(value[None], horizon, axis=0)
        self.index = 0
        self.next_step_s = None
        self.step_s = None
        self.active = False
        self.action = value.copy()
        self.action_chunk = self.plan.copy()
        self.held_action = value.copy()

    def install(
        self,
        plan: np.ndarray,
        *,
        held_action: np.ndarray,
        timestamp_s: float,
        scheduled: bool,
        step_s: float | None,
    ) -> None:
        values = np.asarray(plan, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != 7:
            raise ValueError(f"action plan must have shape [H,7], got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("action plan must contain only finite values")
        held = np.asarray(held_action, dtype=np.float64).reshape(-1)
        if held.shape != (7,) or not np.all(np.isfinite(held)):
            raise ValueError("held action must be a finite seven-vector")
        self.plan = values.copy()
        self.index = 0
        self.action = self.plan[0].copy()
        self.action_chunk = self.plan.copy()
        self.held_action = held.copy()
        if scheduled:
            interval = float(step_s) if step_s is not None else None
            if interval is None or not np.isfinite(interval) or interval <= 0.0:
                raise ValueError("scheduled action plan requires a positive step_s")
            self.active = True
            self.step_s = interval
            self.next_step_s = float(timestamp_s) + interval
        else:
            self.active = False
            self.next_step_s = None
            self.step_s = None

    def advance(self, timestamp_s: float) -> bool:
        """Advance to the timestamp's waypoint; return true when plan completes."""
        if (
            not self.active
            or self.plan is None
            or self.next_step_s is None
            or self.step_s is None
        ):
            return False
        timestamp_s = float(timestamp_s)
        while timestamp_s + 1.0e-9 >= self.next_step_s:
            next_index = self.index + 1
            if next_index >= len(self.plan):
                self.active = False
                self.next_step_s = None
                self.step_s = None
                self.index = len(self.plan) - 1
                self.action = self.plan[-1].copy()
                self.action_chunk = self.plan[-1:].copy()
                return True
            self.index = next_index
            self.action = self.plan[next_index].copy()
            self.action_chunk = self.plan[next_index:].copy()
            self.next_step_s += self.step_s
        return False


__all__ = ["ActionPlanExecutor"]
