"""Backward compatibility wrappers for chat utilities.

This module now re-exports :class:`ChatState`, :class:`ConversationManager`
and ``summarize_history`` from the shared :mod:`conversation` module so that
previous imports continue to work while new code can rely on the consolidated
conversation utilities.
"""

from __future__ import annotations

from .conversation import ChatState, ConversationManager, summarize_history

__all__ = ["ChatState", "ConversationManager", "summarize_history"]

