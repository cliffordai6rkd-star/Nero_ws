import bisect
import math
from typing import Optional, Sequence

import torch
import torch.nn as nn


class ContactDetector(nn.Module):
    """Detect contact from raw, physical-unit wrench history."""

    def __init__(
        self,
        threshold: float,
        force_dims: Sequence[int] = (0, 1, 2),
        history_reducer: str = "max",
    ) -> None:
        super().__init__()
        if threshold < 0:
            raise ValueError("contact threshold must be non-negative")
        if not force_dims:
            raise ValueError("force_dims cannot be empty")
        if history_reducer not in {"last", "max", "mean"}:
            raise ValueError(
                f"Unsupported history_reducer={history_reducer!r}; "
                "expected 'last', 'max', or 'mean'."
            )
        self.threshold = float(threshold)
        self.force_dims = tuple(int(dim) for dim in force_dims)
        self.history_reducer = history_reducer

    def forward(self, raw_wrench: torch.Tensor) -> torch.Tensor:
        if raw_wrench.ndim != 4:
            raise ValueError(
                "raw_wrench must have shape [B, To, K, D], "
                f"got {tuple(raw_wrench.shape)}"
            )
        if max(self.force_dims) >= raw_wrench.shape[-1]:
            raise ValueError(
                f"force_dims={self.force_dims} exceed wrench dim {raw_wrench.shape[-1]}"
            )

        force = raw_wrench[..., list(self.force_dims)]
        magnitude = torch.linalg.vector_norm(force, dim=-1)
        if self.history_reducer == "last":
            reduced = magnitude[..., -1]
        elif self.history_reducer == "max":
            reduced = magnitude.amax(dim=-1)
        else:
            reduced = magnitude.mean(dim=-1)
        return reduced > self.threshold


class MaskProbabilityScheduler(nn.Module):
    """Map optimizer update steps to contact-image masking probability."""

    def __init__(
        self,
        schedule_type: str = "cosine",
        start_step: int = 0,
        end_step: int = 50_000,
        start_probability: float = 1.0,
        end_probability: float = 0.0,
        exponential_rate: float = 5.0,
        piecewise_steps: Optional[Sequence[int]] = None,
        piecewise_probabilities: Optional[Sequence[float]] = None,
        piecewise_interpolation: str = "linear",
    ) -> None:
        super().__init__()
        supported = {"constant", "linear", "cosine", "exponential", "piecewise"}
        if schedule_type not in supported:
            raise ValueError(
                f"Unsupported schedule_type={schedule_type!r}; expected one of {sorted(supported)}"
            )
        if end_step < start_step:
            raise ValueError("end_step must be greater than or equal to start_step")
        for value in (start_probability, end_probability):
            if not 0.0 <= value <= 1.0:
                raise ValueError("mask probabilities must be in [0, 1]")
        if exponential_rate <= 0:
            raise ValueError("exponential_rate must be positive")
        if piecewise_interpolation not in {"linear", "hold"}:
            raise ValueError("piecewise_interpolation must be 'linear' or 'hold'")

        self.schedule_type = schedule_type
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.start_probability = float(start_probability)
        self.end_probability = float(end_probability)
        self.exponential_rate = float(exponential_rate)
        self.piecewise_interpolation = piecewise_interpolation

        self.piecewise_steps = tuple(int(step) for step in (piecewise_steps or ()))
        self.piecewise_probabilities = tuple(
            float(value) for value in (piecewise_probabilities or ())
        )
        if self.schedule_type == "piecewise":
            if len(self.piecewise_steps) < 1:
                raise ValueError("piecewise schedule requires at least one step")
            if len(self.piecewise_steps) != len(self.piecewise_probabilities):
                raise ValueError("piecewise steps and probabilities must have equal length")
            if tuple(sorted(self.piecewise_steps)) != self.piecewise_steps:
                raise ValueError("piecewise steps must be sorted")
            if any(not 0.0 <= value <= 1.0 for value in self.piecewise_probabilities):
                raise ValueError("piecewise probabilities must be in [0, 1]")

    def probability(self, optimizer_step: int) -> float:
        step = int(optimizer_step)
        if self.schedule_type == "constant":
            return self.start_probability
        if self.schedule_type == "piecewise":
            return self._piecewise_probability(step)
        if step <= self.start_step:
            return self.start_probability
        if step >= self.end_step:
            return self.end_probability

        duration = max(1, self.end_step - self.start_step)
        progress = (step - self.start_step) / duration
        if self.schedule_type == "linear":
            weight = 1.0 - progress
        elif self.schedule_type == "cosine":
            weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            end_value = math.exp(-self.exponential_rate)
            weight = (math.exp(-self.exponential_rate * progress) - end_value) / (
                1.0 - end_value
            )
        return self.end_probability + (
            self.start_probability - self.end_probability
        ) * weight

    def _piecewise_probability(self, step: int) -> float:
        insertion = bisect.bisect_right(self.piecewise_steps, step)
        if insertion == 0:
            return self.piecewise_probabilities[0]
        if insertion >= len(self.piecewise_steps):
            return self.piecewise_probabilities[-1]
        left = insertion - 1
        if self.piecewise_interpolation == "hold":
            return self.piecewise_probabilities[left]
        left_step = self.piecewise_steps[left]
        right_step = self.piecewise_steps[insertion]
        progress = (step - left_step) / max(1, right_step - left_step)
        left_value = self.piecewise_probabilities[left]
        right_value = self.piecewise_probabilities[insertion]
        return left_value + (right_value - left_value) * progress

    def forward(self, optimizer_step: int) -> float:
        return self.probability(optimizer_step)


class ContactAwareImageMasker(nn.Module):
    """Sample whole-image token masks for contact observations."""

    def __init__(self, scope: str = "full_context") -> None:
        super().__init__()
        if scope not in {"current_observation", "full_context"}:
            raise ValueError(
                f"Unsupported mask scope={scope!r}; "
                "expected 'current_observation' or 'full_context'."
            )
        self.scope = scope

    def forward(
        self,
        contact_mask: torch.Tensor,
        probability: float,
        enabled: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if contact_mask.ndim != 2:
            raise ValueError(
                f"contact_mask must have shape [B, To], got {tuple(contact_mask.shape)}"
            )
        result = torch.zeros_like(contact_mask, dtype=torch.bool)
        probability = float(probability)
        if not enabled or probability <= 0.0:
            return result

        latest_contact = contact_mask[:, -1]
        if probability >= 1.0:
            selected = latest_contact
        else:
            random_values = torch.rand(
                latest_contact.shape,
                device=contact_mask.device,
                generator=generator,
            )
            selected = latest_contact & (random_values < probability)

        if self.scope == "full_context":
            return selected[:, None].expand_as(contact_mask).clone()
        result[:, -1] = selected
        return result
