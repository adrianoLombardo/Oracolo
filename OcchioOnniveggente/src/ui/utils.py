from __future__ import annotations

"""Miscellaneous utility helpers for the UI layer."""

import re


def highlight_terms(text: str, query: str) -> str:
    """Highlight terms from ``query`` inside ``text``."""
    tokens = set(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", query.lower()))
    for t in sorted(tokens, key=len, reverse=True):
        pattern = re.compile(re.escape(t), re.IGNORECASE)
        text = pattern.sub(lambda m: f"[{m.group(0)}]", text)
    return text


__all__ = ["highlight_terms"]
