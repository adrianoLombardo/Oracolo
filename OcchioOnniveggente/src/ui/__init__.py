"""Helper exports for the :mod:`src.ui` package.

The heavy ``core`` module is intentionally **not** imported here in order to
avoid circular import issues when ``python -m src.ui.core`` is executed.  Only
lightweight utilities are re-exported and the main UI should be imported
explicitly via ``import src.ui.core``.
"""

from typing import TYPE_CHECKING, Any

__all__ = ["RealtimeWSClient", "highlight_terms"]

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from .realtime_ws import RealtimeWSClient
    from .utils import highlight_terms


def __getattr__(name: str) -> Any:
    """Lazily import optional helpers."""
    if name == "RealtimeWSClient":
        from .realtime_ws import RealtimeWSClient as _RealtimeWSClient

        return _RealtimeWSClient
    if name == "highlight_terms":
        from .utils import highlight_terms as _highlight_terms

        return _highlight_terms
    raise AttributeError(f"module {__name__} has no attribute {name}")
