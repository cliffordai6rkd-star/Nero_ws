"""Compatibility exports for legacy DP imports.

The LeRobot checkpoint interface now lives in :mod:`inference.policies.lerobotdp`.
The historical ``inference.policies.dp`` package remains import-compatible for
existing callers and tests.
"""

from inference.policies.dp.adapter import (
    DiffusionPolicyAdapter,
    predict_diffusion_action,
)
from inference.policies.dp.policy import DPPolicy, DiffusionPolicy
from inference.policies.lerobotdp import (
    LeRobotDP,
    LeRobotDiffusionPolicy,
    is_lerobot_checkpoint,
)

__all__ = [
    "DiffusionPolicy",
    "DPPolicy",
    "DiffusionPolicyAdapter",
    "predict_diffusion_action",
    "LeRobotDiffusionPolicy",
    "LeRobotDP",
    "is_lerobot_checkpoint",
]
