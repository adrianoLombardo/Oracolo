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
