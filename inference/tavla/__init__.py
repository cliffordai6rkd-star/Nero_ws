"""TA-VLA remote inference client for ICRA2027."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_CLIENT_EXPORTS = {
    "EffortHistoryBuffer",
    "TavlaObservationBuilder",
    "TavlaRemotePolicy",
}

__all__ = [
    "EffortHistoryBuffer",
    "TavlaObservationBuilder",
    "TavlaRemotePolicy",
    "clip_joint_target",
]


def __getattr__(name: str) -> Any:
    """Load client pieces lazily so ``python -m ...tavla_client`` stays clean."""
    if name in _CLIENT_EXPORTS:
        value = getattr(import_module(".tavla_client", __name__), name)
    elif name == "clip_joint_target":
        value = getattr(import_module(".nero_runtime", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
