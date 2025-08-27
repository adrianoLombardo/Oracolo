"""Utilities for managing conversation language preference."""
from __future__ import annotations


def update_language(current: str | None, detected: str | None, text: str) -> str:
    """Return the effective session language.

    - ``current`` is the previously selected language (``"it"`` or ``"en"``).
    - ``detected`` is the language guessed from the latest user input.
    - ``text`` is the raw user text which may contain an explicit
      language request.
    The function preserves ``current`` unless ``text`` explicitly asks for a
    change. If no language has yet been chosen, ``detected`` is used. The
    default fallback is Italian (``"it"``).
    """

    t = (text or "").lower()
    if "inglese" in t or "english" in t:
        return "en"
    if "italiano" in t or "italian" in t:
        return "it"
    if current in ("it", "en"):
        return current
    if detected in ("it", "en"):
        return detected
    return "it"
