from __future__ import annotations

"""Utilities for lazily accessing the service container."""

import importlib
from types import SimpleNamespace
from typing import Any


def get_container(default: Any | None = None) -> Any:
    """Return the global ``ServiceContainer`` instance if available.

    Parameters
    ----------
    default:
        Object returned when the container cannot be imported.  By default a
        simple ``SimpleNamespace`` is provided.
    """
    if default is None:
        default = SimpleNamespace()

    base_pkg = __package__.rsplit(".", 1)[0]
    try:
        module = importlib.import_module(f"{base_pkg}.service_container")
        return getattr(module, "container")
    except Exception:  # pragma: no cover - container may not exist
        return default


__all__ = ["get_container"]
