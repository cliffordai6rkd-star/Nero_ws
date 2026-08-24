"""High-level model adapters."""

from inference.policies.base import CallablePolicy
from inference.policies.dp import DPPolicy, DiffusionPolicy, DiffusionPolicyAdapter
from inference.factory import POLICY_REGISTRY

POLICY_REGISTRY.register("callable", CallablePolicy)
POLICY_REGISTRY.register("diffusion_policy", DiffusionPolicyAdapter)
POLICY_REGISTRY.register("dp", DiffusionPolicy)

__all__ = [
    "CallablePolicy",
    "DiffusionPolicyAdapter",
    "DiffusionPolicy",
    "DPPolicy",
]
