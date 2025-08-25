"""Minimal realtime server helpers used in tests."""
from __future__ import annotations

from typing import Dict, List, Tuple

from src.profile_utils import get_active_profile, make_domain_settings


def off_topic_message(profile: str, keywords: List[str]) -> str:
    """Return a simple off-topic warning message."""
    if keywords:
        kw = ", ".join(keywords)
        return f"La domanda non rientra nel profilo «{profile}». Prova con questi temi: {kw}."
    return f"La domanda non rientra nel profilo «{profile}». Per favore riformularla."

__all__ = ["off_topic_message", "get_active_profile", "make_domain_settings"]
