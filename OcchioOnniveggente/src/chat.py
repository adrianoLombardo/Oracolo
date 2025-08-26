"""Backward compatibility wrappers for chat utilities.

This module now re-exports :class:`ChatState` and ``summarize_history`` from
the shared :mod:`conversation` module so that previous imports continue to
work.
"""

from __future__ import annotations

from .conversation import ChatState, summarize_history

__all__ = ["ChatState", "summarize_history"]

