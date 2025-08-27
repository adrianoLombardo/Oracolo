from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol


class SpeechToText(Protocol):
    """Interface for speech-to-text providers."""

    def transcribe(self, audio_path: Path, lang: str = "it") -> str:
        ...


class TextToSpeech(Protocol):
    """Interface for text-to-speech providers."""

    def synthesize(self, text: str, lang: str = "it") -> bytes:
        ...


class LocalSpeechToText:
    """Simple local STT implementation based on :mod:`local_audio`."""

    def transcribe(self, audio_path: Path, lang: str = "it") -> str:
        from .local_audio import stt_local

        return stt_local(audio_path, lang=lang)


class LocalTextToSpeech:
    """Simple local TTS implementation based on :mod:`local_audio`."""

    def synthesize(self, text: str, lang: str = "it") -> bytes:
        from .local_audio import tts_local

        tmp = Path(tempfile.gettempdir()) / "tts_tmp.wav"
        tts_local(text, tmp, lang=lang)
        return tmp.read_bytes()


# Plugin registration -------------------------------------------------------

def create_local_stt(container: "ServiceContainer") -> SpeechToText:
    """Factory for the built-in local STT backend."""
    return LocalSpeechToText()


def create_local_tts(container: "ServiceContainer") -> TextToSpeech:
    """Factory for the built-in local TTS backend."""
    return LocalTextToSpeech()


try:  # pragma: no cover - registry may not be available during docs build
    from ..plugins import register_stt, register_tts

    register_stt("local", create_local_stt)
    register_tts("local", create_local_tts)
except Exception:  # pragma: no cover - defensive
    pass
