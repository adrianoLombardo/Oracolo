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



def oracle_greeting(lang: str, tone: str = "informal") -> str:
    """Return a greeting string based on ``lang`` and ``tone``."""
    english = (lang or "").lower().startswith("en")
    if english:
        if tone == "formal":
            return "Good day, I am the Oracle. Please state your question."
        return "Hello, I am the Oracle. Ask your question."
    if tone == "formal":
        return "Salve, sono l'Oracolo. Ponga pure la sua domanda."
    return "Ciao, sono l'Oracolo. Fai pure la tua domanda."


def default_response(kind: str, lang: str, tone: str = "informal") -> str:
    """Return a default response message for ``kind``."""
    english = (lang or "").lower().startswith("en")
    kind = kind.lower()
    if kind == "profanity":
        if english:
            return (
                "🚫 Please avoid offensive language."
                if tone == "formal"
                else "🚫 Hey, let's keep it clean!"
            )
        return (
            "🚫 Per favore evita linguaggio offensivo."
            if tone == "formal"
            else "🚫 Ehi, niente parolacce!"
        )
    if kind == "filtered":
        return "⚠️ Filtered text: " if english else "⚠️ Testo filtrato: "
    if english:
        return "I did not understand." if tone == "formal" else "I didn't understand."
    return "Non ho capito." if tone == "formal" else "Non ti ho capito."

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

