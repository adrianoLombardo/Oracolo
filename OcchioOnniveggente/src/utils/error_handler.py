"""Utility per la gestione centralizzata degli errori.

Questa funzione classifica gli errori comuni dell'applicazione e restituisce
un messaggio comprensibile per l'utente insieme a suggerimenti d'azione.
Gli errori vengono inoltre registrati nel log con timestamp e contesto.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def handle_error(exc: Exception, context: str | None = None) -> str:
    """Gestisce ``exc`` classificandolo e restituendo un messaggio utente.

    Parameters
    ----------
    exc:
        Eccezione sollevata.
    context:
        Contesto in cui l'errore è avvenuto (es. "transcribe").

    Returns
    -------
    str
        Messaggio comprensibile per l'utente.
    """

    if isinstance(exc, (ConnectionError, TimeoutError)):
        category = "network"
        message = "Errore di rete. Controlla la connessione."
        level = logging.WARNING
    elif isinstance(exc, ValueError):
        category = "api"
        message = "Errore dell'API. Riprova più tardi."
        level = logging.ERROR
    elif isinstance(exc, OSError):
        category = "audio"
        message = "Errore audio. Verifica i dispositivi audio."
        level = logging.ERROR
    else:
        category = "unknown"
        message = "Si è verificato un errore inatteso."
        level = logging.ERROR

    logger.log(level, "%s error: %s | context: %s", category, exc, context, exc_info=level >= logging.ERROR)
    return message
