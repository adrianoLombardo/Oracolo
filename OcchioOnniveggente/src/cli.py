"""Simple console helpers used by the main application."""

from __future__ import annotations

import sys


def _ensure_utf8_stdout() -> None:
    """Force UTF-8 output on ``sys.stdout`` if supported."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def say(msg: str) -> None:
    """Print a message intended for the user conversation."""
    print(msg, flush=True)


def oracle_greeting(lang: str) -> str:
    """Return a greeting string based on ``lang``."""
    if (lang or "").lower().startswith("en"):
        return "Hello, I am the Oracle. Ask your question."
    return "Ciao, sono l'Oracolo. Fai pure la tua domanda."
