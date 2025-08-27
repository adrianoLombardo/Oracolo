from __future__ import annotations

"""Minimal local TTS/STT utilities used in the test-suite.

The real project contains a much richer implementation with streaming and
advanced caching.  For the purposes of the exercises we provide a small subset
that offers deterministic behaviour without external dependencies.
"""

from pathlib import Path
from typing import Generator, Literal
import asyncio
import logging
import shutil
import tempfile
import threading

import numpy as np
import soundfile as sf
import sounddevice as sd

from .storage.cache import get_tts_cache, set_tts_cache
from types import SimpleNamespace

from .utils.container import get_container


_container_stub = SimpleNamespace(
    load_tts_model=lambda: ("gtts", lambda text, lang: (np.zeros(1), 16000)),
    load_stt_model=lambda: ("", None),
)

container = get_container(_container_stub)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Whisper model cache
# ---------------------------------------------------------------------------

_WHISPER_CACHE: dict[tuple[str, str, bool], object] = {}
_WHISPER_LOCK = threading.Lock()

def _get_whisper(
    model_name: str, device: str, compute_type: str, use_onnx: bool = False
) -> object:
    """Return a cached instance of ``WhisperModel`` or an ONNX session."""
    key = (model_name, device, use_onnx)
    with _WHISPER_LOCK:
        model = _WHISPER_CACHE.get(key)
        if model is None:
            if use_onnx and device == "cpu":
                import onnxruntime as ort  # type: ignore
                model = ort.InferenceSession(model_name)
            else:
                from faster_whisper import WhisperModel  # type: ignore
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

# ---------------------------------------------------------------------------
# TTS helpers
# ---------------------------------------------------------------------------

def tts_local(
    text: str,
    out_path: Path,
    *,
    lang: str = "it",
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> None:
    """Synthesize ``text`` to ``out_path`` using a best-effort strategy."""
    cached = get_tts_cache(text, lang)
    if cached is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, out_path)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:  # small, widely available online engine
        from gtts import gTTS  # type: ignore
        gTTS(text=text, lang=lang).save(out_path.as_posix())
        set_tts_cache(text, lang, out_path)
        return
    except Exception:
        pass

    try:  # offline fallback
        import pyttsx3  # type: ignore
        engine = pyttsx3.init()
        engine.save_to_file(text, out_path.as_posix())
        engine.runAndWait()
        set_tts_cache(text, lang, out_path)
        return
    except Exception:
        pass

    # Use container-provided backend as a last resort
    try:
        backend, synth = container.load_tts_model()
        data, sr = synth(text, lang=lang)
        sf.write(out_path, data, sr)
        set_tts_cache(text, lang, out_path)
    except Exception:
        out_path.write_bytes(b"")


def _play(data: np.ndarray, sr: int) -> None:
    """Play ``data`` at ``sr`` using ``sounddevice``."""
    sd.play(data, sr)
    sd.wait()


def tts_speak(text: str, *, lang: str = "it") -> None:
    """Synthesize ``text`` and play it synchronously."""
    tmp = Path(tempfile.gettempdir()) / "tts_tmp.wav"
    tts_local(text, tmp, lang=lang)
    data, sr = sf.read(tmp.as_posix(), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    _play(data, sr)


async def async_tts_speak(text: str, *, lang: str = "it") -> None:
    """Asynchronous variant of :func:`tts_speak`."""
    tmp = Path(tempfile.gettempdir()) / "tts_tmp.wav"
    tts_local(text, tmp, lang=lang)
    data, sr = sf.read(tmp.as_posix(), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    await asyncio.to_thread(_play, data, sr)

# ---------------------------------------------------------------------------
# STT helpers
# ---------------------------------------------------------------------------

def stt_local(audio_path: Path, lang: str = "it") -> str:
    """Very small placeholder STT function returning a transcription or ''."""
    try:
        backend, model = container.load_stt_model()
    except Exception:
        return ""
    if backend == "faster_whisper" and model is not None:
        try:
            segments, _ = model.transcribe(
                audio_path.as_posix(), language=lang, task="transcribe"
            )
            return "".join(seg.text for seg in segments).strip()
        except Exception:
            return ""
    return ""


def stt_local_faster(
    audio_path: Path,
    lang: str = "it",
    *,
    device: str = "cpu",
    compute_type: str = "int8",
    model_path: str = "base",
    use_onnx: bool = False,
) -> str:
    """Transcribe ``audio_path`` using a cached ``faster-whisper`` model."""
    try:
        model = _get_whisper(model_path, device, compute_type, use_onnx)
    except Exception:
        return ""
    try:
        segments, _ = model.transcribe(str(audio_path), language=lang)
        return "".join(seg.text for seg in segments).strip()
    except Exception:
        logger.error("Errore trascrizione locale con faster-whisper", exc_info=True)
        return ""

__all__ = [
    "_get_whisper",
    "_WHISPER_CACHE",
    "tts_local",
    "tts_speak",
    "async_tts_speak",
    "stt_local",
    "stt_local_faster",
]
