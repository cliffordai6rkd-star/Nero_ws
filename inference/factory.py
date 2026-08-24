"""Small registries for policy/WM/controller implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, TypeVar


T = TypeVar("T")


class ComponentRegistry:
    def __init__(self, name: str) -> None:
        self.name = str(name)
        self._items: dict[str, type[Any] | Callable[..., Any]] = {}

    def register(self, key: str, component: T | None = None):
        normalized = str(key).strip().lower()
        if not normalized:
            raise ValueError(f"{self.name} registry key must be non-empty")

        def bind(value):
            if normalized in self._items:
                if self._items[normalized] is value:
                    return value
                raise ValueError(
                    f"{self.name} registry already contains {normalized!r}"
                )
            self._items[normalized] = value
            return value

        return bind(component) if component is not None else bind

    def get(self, key: str):
        normalized = str(key).strip().lower()
        try:
            return self._items[normalized]
        except KeyError as exc:
            raise KeyError(
                f"unknown {self.name} component {key!r}; "
                f"available={sorted(self._items)}"
            ) from exc

    def create(self, key: str, *args, **kwargs):
        return self.get(key)(*args, **kwargs)

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


def load_symbol(reference: str) -> Any:
    """Load ``package.module:Symbol`` for project-local extensions."""
    module_name, separator, symbol_name = str(reference).partition(":")
    if not separator or not module_name or not symbol_name:
        raise ValueError("component reference must be module.path:Symbol")
    return getattr(import_module(module_name), symbol_name)


POLICY_REGISTRY = ComponentRegistry("policy")
WORLD_MODEL_REGISTRY = ComponentRegistry("world_model")
CONTROLLER_REGISTRY = ComponentRegistry("controller")


__all__ = [
    "ComponentRegistry",
    "CONTROLLER_REGISTRY",
    "POLICY_REGISTRY",
    "WORLD_MODEL_REGISTRY",
    "load_symbol",
]
