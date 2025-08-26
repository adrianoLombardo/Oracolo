
"""Minimal subset of the realtime server utilities used in tests.

The original project provides a full realtime server implementation. For the
unit tests we only require two small helpers: ``get_active_profile`` to select
which profile is active from a settings dictionary and ``off_topic_message`` to
format a warning when a question falls outside the profile domain.
"""

from __future__ import annotations

from typing import Any, Iterable, Tuple


def get_active_profile(settings: dict[str, Any]) -> Tuple[str, dict[str, Any]]:
    """Return the active profile name and its configuration."""

    domain = settings.get("domain", {})
    name = domain.get("profile", "")
    profiles = domain.get("profiles", {})
    return name, profiles.get(name, {})


def off_topic_message(profile_name: str, keywords: Iterable[str]) -> str:
    """Return a human friendly message for off-topic questions."""

    keywords = list(keywords)
    if keywords:
        joined = ", ".join(keywords)
        return (
            f"La domanda non riguarda il profilo «{profile_name}». "
            f"Argomenti validi: {joined}."
        )
    return (
        f"La domanda non riguarda il profilo «{profile_name}». "
        "Per favore riformularla in tema."
    )
