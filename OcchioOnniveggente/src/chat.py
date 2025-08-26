"""HTTP-based chat helpers.

This module exposes small helper functions used by the rest of the
application to interact with the backend API.  The previous implementation
relied on an in-process event bus; the helpers now forward the information to
the web API so that multiple clients can share the same backend service.
"""

from __future__ import annotations

import os
import requests

from .conversation import ChatState, ConversationManager, summarize_history


API_URL = os.getenv("ORACOLO_API_URL", "http://localhost:5000")


def publish_recording_started() -> None:
    """Notify the backend that audio recording has begun."""

    try:
        requests.post(
            f"{API_URL}/voice",
            json={"event": "recording_started"},
            timeout=10,
        )
    except Exception:
        pass


def publish_transcript(text: str) -> None:
    """Send a transcript to the backend service."""

    if not text:
        return
    try:
        requests.post(
            f"{API_URL}/voice",
            json={"transcript": text},
            timeout=10,
        )
    except Exception:
        pass


def publish_response(text: str) -> None:
    """Forward an assistant response to the backend service."""

    if not text:
        return
    try:
        requests.post(
            f"{API_URL}/chat",
            json={"message": text},
            timeout=10,
        )
    except Exception:
        pass


__all__ = [
    "ChatState",
    "ConversationManager",
    "summarize_history",
    "publish_recording_started",
    "publish_transcript",
    "publish_response",
]

