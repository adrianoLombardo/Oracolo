from __future__ import annotations

"""Local TTS/STT helpers and simple chunked streaming utilities."""

from pathlib import Path

from typing import Generator, Literal
import logging
import hashlib
import shutil


logger = logging.getLogger(__name__)

import numpy as np

from .audio import AudioPreprocessor, load_audio_as_float
from .config import Settings
from .utils.device import resolve_device
from .cache import get_tts_cache, set_tts_cache, get_stt_cache, set_stt_cache
from .service_container import container


# Cache per i modelli Whisper caricati localmente. La chiave identifica
# la combinazione ``(device, compute_type)`` utilizzata per costruire il
# modello e consente di riutilizzare la stessa istanza tra più chiamate
# senza ricaricarlo in VRAM.
_WHISPER_CACHE: dict[tuple[str, str], "WhisperModel"] = {}


def _get_whisper(model_name: str, device: str, compute_type: str) -> "WhisperModel":
    """Restituisce un'istanza ``WhisperModel`` riutilizzabile.

    Se la combinazione ``device``/``compute_type`` è già stata utilizzata,
    viene ritornato il modello memorizzato; altrimenti il modello viene
    costruito e salvato nel cache.
    """

    from faster_whisper import WhisperModel  # type: ignore

    key = (device, compute_type)
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
    """Synthesize ``text`` locally if possible.

    The function tries a couple of lightweight libraries (``gTTS`` and
    ``pyttsx3``) and falls back to an empty file if neither is available.
    """

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

    # Fallback: create an empty placeholder so callers don't explode
    out_path.write_bytes(b"")



def stt_local(
    audio_path: Path,
    *,
    lang: str = "it",
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> str:
    """Attempt a local transcription of ``audio_path`` using a cleaned signal."""

    try:
        import torch  # type: ignore


        actual_device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
    except Exception:
        actual_device = "cpu" if device == "auto" else device

    compute_type = "int8_float16" if actual_device == "cuda" else "int8"

    try:
        model = _get_whisper("base", actual_device, compute_type)
        segments, _ = model.transcribe(
            audio_path.as_posix(), language=lang, task="transcribe"
        )

        device = resolve_device("auto")
        compute_type = "int8_float16" if device == "cuda" else "int8"
        model = WhisperModel("base", device=device, compute_type=compute_type)
        segments, _ = model.transcribe(audio_path.as_posix(), language=lang, task="transcribe")

        return "".join(seg.text for seg in segments).strip()
    except Exception:
        pass

    try:
        import speech_recognition as sr  # type: ignore

        setts = Settings.model_validate_yaml(Path("settings.yaml"))
        sr_cfg = setts.audio.sample_rate
        pre = AudioPreprocessor(
            sr_cfg,
            denoise=getattr(setts.audio, "denoise", False),
            echo_cancel=getattr(setts.audio, "echo_cancel", False),
        )
        y, sr_in = load_audio_as_float(audio_path, sr_cfg)
        y = pre.process(y)
        pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()
        audio = sr.AudioData(pcm, sr_in, 2)
        r = sr.Recognizer()
        try:
            return r.recognize_sphinx(audio, language=lang)
        except Exception:
            return ""
    except Exception:
        return ""


def stt_local_faster(
    audio_path: Path,
    lang: str = "it",
    *,
    device: str = "cpu",
    compute_type: str = "int8",
) -> str:
    """Transcribe ``audio_path`` using ``faster-whisper`` if available.

    ``device`` can be ``"cpu"`` or ``"cuda"``; ``compute_type`` controls the
    precision used by the model.  On failure or missing dependencies an empty
    string is returned.
    """

    setts = container.settings
    pre = AudioPreprocessor(
        setts.audio.sample_rate,
        denoise=getattr(setts.audio, "denoise", False),
        echo_cancel=getattr(setts.audio, "echo_cancel", False),
    )
    y, _ = load_audio_as_float(audio_path, setts.audio.sample_rate)
    y = pre.process(y)
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()
    audio_hash = hashlib.sha256(pcm).hexdigest()

    cached = get_stt_cache(audio_hash)
    if cached is not None:
        return cached

    try:
        model = _get_whisper("base", device, compute_type)
    except Exception:
        logger.warning("faster-whisper non disponibile, fallback a stt_local")
        return stt_local(audio_path, lang)

    try:
        segments, _ = model.transcribe(str(audio_path), language=lang)
        text = "".join(seg.text for seg in segments).strip()
        set_stt_cache(audio_hash, text)
        return text
    except Exception:
        logger.error("Errore trascrizione locale con faster-whisper", exc_info=True)
        return ""
