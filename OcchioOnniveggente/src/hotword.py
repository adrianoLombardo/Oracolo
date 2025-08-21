# src/hotword.py
from __future__ import annotations
"""
Utilities for wake word detection.

- listen_for_wakeword(...)  -> prova a usare Picovoice Porcupine; se non disponibile, ritorna False.
- matches_hotword_text(...) -> match robusto (accenti/maiuscole/punteggiatura) su testo STT.
- strip_hotword_prefix(...) -> per "ciao oracolo, domanda...", rimuove l’hotword e restituisce la domanda.
"""

from typing import Iterable, Optional, Tuple
import os
import re
import unicodedata


# ----------------------------- TEXT MATCH HELPERS ----------------------------- #

def _strip_accents(s: str) -> str:
    nf = unicodedata.normalize("NFD", s or "")
    return "".join(ch for ch in nf if not unicodedata.combining(ch))

def _normalize_lang(s: str) -> str:
    return _strip_accents(s).lower()

def _phrase_to_regex_tokens(phrase: str) -> re.Pattern:
    """
    Esempio: "ciao oracolo" -> regex tollerante a punteggiatura/spazi:
    r"\\bciao[\\W_]*oracolo\\b"
    """
    phrase = _normalize_lang(phrase)
    tokens = re.split(r"\s+", phrase.strip())
    parts = [re.escape(t) for t in tokens if t]
    if not parts:
        parts = [re.escape(phrase.strip())]
    pattern = r"\b" + r"[\W_]*".join(parts) + r"\b"
    return re.compile(pattern, flags=re.IGNORECASE)

def matches_hotword_text(text: str, phrases: Iterable[str]) -> bool:
    """True se QUALSIASI hot-phrase è presente nel testo (robusto)."""
    norm = _normalize_lang(text or "")
    for p in phrases or []:
        if _phrase_to_regex_tokens(p).search(norm):
            return True
    return False

def strip_hotword_prefix(text: str, phrases: Iterable[str]) -> Tuple[bool, str]:
    """
    Se il testo inizia con una hotword (tollerante), rimuove la porzione iniziale e
    restituisce (matched, resto). Utile per "ciao oracolo, domanda...".
    """
    norm = _normalize_lang(text or "")
    for p in phrases or []:
        rx = _phrase_to_regex_tokens(p)
        m = rx.search(norm)
        if m and m.start() == 0:  # match all'inizio
            cut_norm_len = m.end()
            remainder = text[cut_norm_len:].lstrip(" ,;:.-–—\n\t")
            return True, remainder
    return False, text or ""


# ----------------------------- PORCUPINE ENGINE ------------------------------ #

def _porcupine_create_with_optional_access_key(keywords, sensitivities=None):
    """
    Gestisce differenze tra versioni di pvporcupine:
    - Alcune richiedono access_key, altre no.
    Prova prima senza, poi con access_key da env (PV_ACCESS_KEY o PICOVOICE_ACCESS_KEY).
    """
    import pvporcupine  # type: ignore

    sens = sensitivities or [0.6] * len(keywords)
    try:
        # tentativo "classico"
        return pvporcupine.create(keywords=keywords, sensitivities=sens)
    except TypeError:
        # versione che richiede access_key
        access_key = (
            os.environ.get("PV_ACCESS_KEY")
            or os.environ.get("PICOVOICE_ACCESS_KEY")
            or ""
        )
        if not access_key:
            raise RuntimeError(
                "pvporcupine richiede access_key. Imposta env PV_ACCESS_KEY o PICOVOICE_ACCESS_KEY."
            )
        return pvporcupine.create(access_key=access_key, keywords=keywords, sensitivities=sens)


def listen_for_wakeword(
    wakeword: str,
    device_id: Optional[int] = None,
    sensitivity: float = 0.6,
) -> bool:
    """
    Blocca finché 'wakeword' non viene rilevata tramite Picovoice Porcupine.
    Se Porcupine o pyaudio non sono disponibili, stampa un warning e ritorna False.

    Parameters
    ----------
    wakeword: str
        Nome keyword tra quelle supportate (es. 'porcupine', 'picovoice') oppure
        una keyword custom se il pacchetto è configurato di conseguenza.
    device_id: Optional[int]
        Indice del dispositivo input (None = default di sistema).
    sensitivity: float
        Sensibilità Porcupine per la/e keyword [0..1].

    Returns
    -------
    bool
        True se la wakeword è stata rilevata, False se non è stato possibile usare il motore.
    """
    try:
        import struct
        import pvporcupine  # type: ignore
        import pyaudio      # type: ignore
    except Exception:
        print("Wakeword engine non disponibile (pvporcupine/pyaudio). Fallback al match testuale.", flush=True)
        return False

    porcupine = None
    pa = None
    stream = None
    try:
        porcupine = _porcupine_create_with_optional_access_key(
            keywords=[wakeword], sensitivities=[sensitivity]
        )
        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length,
            input_device_index=device_id,
        )

        while True:
            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            result = porcupine.process(pcm)
            if result >= 0:
                return True

    finally:
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
        except Exception:
            pass
        try:
            if pa is not None:
                pa.terminate()
        except Exception:
            pass
        try:
            if porcupine is not None:
                porcupine.delete()
        except Exception:
            pass
