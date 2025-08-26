"""Backward compatibility wrappers for chat utilities.

This module now re-exports :class:`ChatState` and ``summarize_history`` from
the shared :mod:`conversation` module so that previous imports continue to
work.
"""

from __future__ import annotations

from .conversation import ChatState, summarize_history
from .service_container import container


def get_chat() -> ChatState:
    """Return the shared :class:`ChatState` from the service container."""

    if container.ui_state.conversation is None:
        container.ui_state.conversation = ChatState()
    return container.ui_state.conversation


__all__ = ["ChatState", "summarize_history", "get_chat"]

