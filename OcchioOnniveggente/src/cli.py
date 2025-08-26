"""Simple console helpers used by the main application."""

from __future__ import annotations

import sys
import time
from typing import Iterator
from threading import Event


def _ensure_utf8_stdout() -> None:
    """Force UTF-8 output on ``sys.stdout`` if supported."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def say(msg: str) -> None:
    """Print a message intended for the user conversation."""
    print(msg, flush=True)


def stream_say(
    tokens: Iterator[str], *, stop_event: Event | None = None, timeout: float | None = None
) -> str:
    """Print ``tokens`` as they are produced.

    The function consumes the ``tokens`` iterator, printing each chunk as soon
    as it becomes available.  It returns the accumulated text.  Streaming can
    be interrupted either by setting ``stop_event`` or by pressing
    ``Ctrl+C``/``KeyboardInterrupt``; a ``timeout`` in seconds can also be
    provided to stop after a fixed duration.
    """

    text = ""
    start = time.monotonic()
    try:
        for chunk in tokens:
            if stop_event is not None and stop_event.is_set():
                break
            if timeout is not None and (time.monotonic() - start) > timeout:
                break
            text += chunk
            print(chunk, end="", flush=True)
    except KeyboardInterrupt:  # pragma: no cover - interactive usage
        pass
    finally:
        print()
    return text


def oracle_greeting(lang: str) -> str:
    """Return a greeting string based on ``lang``."""
    if (lang or "").lower().startswith("en"):
        return "Hello, I am the Oracle. Ask your question."
    return "Ciao, sono l'Oracolo. Fai pure la tua domanda."
