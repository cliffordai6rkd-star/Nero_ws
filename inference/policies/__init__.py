"""High-level model adapters."""

from inference.policies.base import CallablePolicy
from inference.policies.dp import (
    DPPolicy,
    DiffusionPolicy,
    DiffusionPolicyAdapter,
    predict_diffusion_action,
)
from inference.policies.lerobotdp import (
    LeRobotDP,
    LeRobotDiffusionPolicy,
    is_lerobot_checkpoint,
)
from inference.policies.tavla import (
    TAVLA,
    TAVLAAdapter,
    TAVLAInferencePolicy,
    TAVLAPolicy,
    TAVLAObservationBuilder,
)
from inference.factory import POLICY_REGISTRY

POLICY_REGISTRY.register("callable", CallablePolicy)
POLICY_REGISTRY.register("diffusion_policy", DiffusionPolicyAdapter)
POLICY_REGISTRY.register("dp", DiffusionPolicy)
POLICY_REGISTRY.register("lerobotdp", LeRobotDiffusionPolicy)
POLICY_REGISTRY.register("lerobot_diffusion_policy", LeRobotDiffusionPolicy)
POLICY_REGISTRY.register("tavla", TAVLA)

__all__ = [
    "CallablePolicy",
    "DiffusionPolicyAdapter",
    "DiffusionPolicy",
    "DPPolicy",
    "predict_diffusion_action",
    "LeRobotDiffusionPolicy",
    "LeRobotDP",
    "is_lerobot_checkpoint",
    "TAVLA",
    "TAVLAAdapter",
    "TAVLAInferencePolicy",
    "TAVLAPolicy",
    "TAVLAObservationBuilder",
]
