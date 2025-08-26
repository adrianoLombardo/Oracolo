"""Backward compatibility wrappers for chat utilities.

This module now re-exports :class:`ChatState`, :class:`ConversationManager`
and ``summarize_history`` from the shared :mod:`conversation` module so that
previous imports continue to work while new code can rely on the consolidated
conversation utilities.
"""

from __future__ import annotations
from .conversation import ChatState, summarize_history
from .event_bus import event_bus


def publish_recording_started() -> None:
    """Notify subscribers that audio recording has begun."""

    event_bus.publish("recording_started")


def publish_transcript(text: str) -> None:
    """Publish a transcript for downstream consumers."""

    event_bus.publish("transcript_ready", text)


def publish_response(text: str) -> None:
    """Publish a ready assistant response."""

    event_bus.publish("response_ready", text)


__all__ = [
    "ChatState",
    "summarize_history",
    "publish_recording_started",
    "publish_transcript",
    "publish_response",
]
