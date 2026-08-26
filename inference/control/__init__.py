"""Robot control adapters."""

from inference.control.base import ArmRobotController, CallableRobotController
from inference.control.nero import NeroPipelineOutputController
from inference.control.resolver import CallableActionResolver, DirectActionResolver
from inference.control.safety import BasicSafetyGuard
from inference.factory import CONTROLLER_REGISTRY

CONTROLLER_REGISTRY.register("callable", CallableRobotController)
CONTROLLER_REGISTRY.register("arm", ArmRobotController)
CONTROLLER_REGISTRY.register("legacy_pipeline", NeroPipelineOutputController)

__all__ = [
    "ArmRobotController",
    "CallableRobotController",
    "NeroPipelineOutputController",
    "CallableActionResolver",
    "DirectActionResolver",
    "BasicSafetyGuard",
]
