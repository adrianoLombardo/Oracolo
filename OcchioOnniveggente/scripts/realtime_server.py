"""Utility e placeholder per il server realtime."""

from __future__ import annotations


def off_topic_message(profile: str, keywords: list[str]) -> str:
    """Genera un messaggio per input fuori contesto."""

    if keywords:
        kw = ", ".join(keywords)
        return f"La domanda non riguarda il profilo «{profile}». Prova con parole chiave come {kw}."
    return (
        f"La domanda non riguarda il profilo «{profile}». "
        "Per favore riformularla in tema."
    )


__all__ = ["off_topic_message"]


def get_active_profile(settings: dict) -> tuple[str, dict]:
    """Restituisce il profilo attivo dalle impostazioni."""

    dom = settings.get("domain", {})
    name = dom.get("profile", "")
    profiles = dom.get("profiles", {})
    return name, profiles.get(name, {})


__all__.append("get_active_profile")

