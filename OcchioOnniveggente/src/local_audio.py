from __future__ import annotations

"""Local TTS/STT helpers and simple chunked streaming utilities."""

from pathlib import Path
from typing import Generator


def stream_file(path: Path, chunk_size: int = 4096) -> Generator[bytes, None, None]:
    """Yield audio chunks from ``path`` for streaming playback."""

    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            yield chunk


def tts_local(text: str, out_path: Path, lang: str = "it") -> None:
    """Synthesize ``text`` locally if possible.

    The function tries a couple of lightweight libraries (``gTTS`` and
    ``pyttsx3``) and falls back to an empty file if neither is available.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:  # gTTS first because it is tiny and widely available
        from gtts import gTTS  # type: ignore

        gTTS(text=text, lang=lang).save(out_path.as_posix())
        return
    except Exception:
        pass

    try:  # fall back to pyttsx3 (offline, cross-platform)
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        engine.save_to_file(text, out_path.as_posix())
        engine.runAndWait()
        return
    except Exception:
        pass

    # Fallback: create an empty placeholder so callers don't explode
    out_path.write_bytes(b"")


def stt_local(audio_path: Path, lang: str = "it") -> str:
    """Attempt a local transcription of ``audio_path``.

    The function uses `speech_recognition` with the PocketSphinx backend when
    available.  It is intentionally simple and meant only as a latency saving
    fallback.
    """

    try:
        import speech_recognition as sr  # type: ignore

        r = sr.Recognizer()
        with sr.AudioFile(str(audio_path)) as source:
            audio = r.record(source)
        try:
            return r.recognize_sphinx(audio, language=lang)
        except Exception:
            return ""
    except Exception:
        return ""
