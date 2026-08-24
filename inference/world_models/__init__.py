"""World-model stage implementations."""

from inference.world_models.base import CallableWorldModel, NullWorldModel
from inference.factory import WORLD_MODEL_REGISTRY

WORLD_MODEL_REGISTRY.register("none", NullWorldModel)
WORLD_MODEL_REGISTRY.register("callable", CallableWorldModel)

__all__ = ["CallableWorldModel", "NullWorldModel"]
