from __future__ import annotations

"""Local TTS/STT helpers and simple chunked streaming utilities."""

from pathlib import Path

from typing import Generator, Literal
import logging
import hashlib
import shutil
import tempfile
import asyncio

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import sounddevice as sd

from .cache import get_tts_cache, set_tts_cache

try:  # pragma: no cover - fallback for isolated test environment
    from .service_container import container  # type: ignore
except Exception:  # pragma: no cover
    from types import SimpleNamespace

    container = SimpleNamespace(
        load_tts_model=lambda: ("gtts", lambda text, lang: None),
        load_stt_model=lambda: ("", None),
    )


# Cache per i modelli Whisper caricati localmente. La chiave identifica
# la combinazione ``(device, compute_type)`` utilizzata per costruire il
# modello e consente di riutilizzare la stessa istanza tra più chiamate
# senza ricaricarlo in VRAM.
_WHISPER_CACHE: dict[tuple[str, str, bool], object] = {}


def _get_whisper(
    model_name: str, device: str, compute_type: str, use_onnx: bool = False
) -> object:
    """Restituisce un'istanza ``WhisperModel`` riutilizzabile.

    Se la combinazione ``device``/``compute_type`` è già stata utilizzata,
    viene ritornato il modello memorizzato; altrimenti il modello viene
    costruito e salvato nel cache.
    """

    if use_onnx and device == "cpu":
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - runtime dep
            raise RuntimeError("onnxruntime è richiesto per modelli ONNX") from exc
        key = (model_name, device, True)
        model = _WHISPER_CACHE.get(key)
        if model is None:
            model = ort.InferenceSession(model_name)
            _WHISPER_CACHE[key] = model
        return model

    from faster_whisper import WhisperModel  # type: ignore

    key = (model_name, device, False)
    model = _WHISPER_CACHE.get(key)
    if model is None:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        _WHISPER_CACHE[key] = model
    return model





def stream_file(path: Path, chunk_size: int = 4096) -> Generator[bytes, None, None]:
    """Yield audio chunks from ``path`` for streaming playback."""

    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            yield chunk


def tts_local(
    text: str,
    out_path: Path,
    *,
    lang: str = "it",
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> None:
    """Synthesize ``text`` using a cached local backend when available."""


    cached = get_tts_cache(text, lang)
    if cached is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, out_path)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:  # gTTS first because it is tiny and widely available
        from gtts import gTTS  # type: ignore

        gTTS(text=text, lang=lang).save(out_path.as_posix())
        set_tts_cache(text, lang, out_path)
        return
    except Exception:
        pass

    try:  # fall back to pyttsx3 (offline, cross-platform)
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        engine.save_to_file(text, out_path.as_posix())
        engine.runAndWait()
        set_tts_cache(text, lang, out_path)
        return
    except Exception:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    backend, engine = container.load_tts_model()
    try:
        if backend == "gtts":
            engine(text=text, lang=lang).save(out_path.as_posix())
            return
        if backend == "pyttsx3":
            engine.save_to_file(text, out_path.as_posix())
            engine.runAndWait()
            return

    except Exception:
        logger.warning("Errore sintesi vocale locale", exc_info=True)

    # Fallback: create an empty placeholder so callers don't explode
    out_path.write_bytes(b"")


def _play(data: np.ndarray, sr: int) -> None:
    """Play ``data`` at sample rate ``sr`` using ``sounddevice"."""
    sd.play(data, sr)
    sd.wait()


def tts_speak(text: str, *, lang: str = "it") -> None:
    """Synthesize ``text`` and play it synchronously."""
    tmp = Path(tempfile.gettempdir()) / "tts_tmp.wav"
    try:
        tts_local(text, tmp, lang=lang)
        data, sr = sf.read(tmp.as_posix(), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        _play(data, sr)
    except Exception:  # pragma: no cover - best effort playback
        logger.warning("tts_speak failed", exc_info=True)


async def async_tts_speak(text: str, *, lang: str = "it") -> None:
    """Asynchronous version of :func:`tts_speak`."""
    tmp = Path(tempfile.gettempdir()) / "tts_tmp.wav"
    try:
        tts_local(text, tmp, lang=lang)
        data, sr = sf.read(tmp.as_posix(), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        await asyncio.to_thread(_play, data, sr)
    except Exception:  # pragma: no cover - best effort playback
        logger.warning("async_tts_speak failed", exc_info=True)


def stt_local(audio_path: Path, lang: str = "it") -> str:
    """Placeholder local transcription returning an empty string."""
    try:
        backend, model = container.load_stt_model()
    except Exception:
        return ""
    if backend == "faster_whisper":
        try:
            segments, _ = model.transcribe(
                audio_path.as_posix(), language=lang, task="transcribe"
            )
            return "".join(seg.text for seg in segments).strip()
        except Exception:
            return ""
    return ""
