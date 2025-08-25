from __future__ import annotations

"""Local TTS/STT helpers and simple chunked streaming utilities."""

from pathlib import Path

from typing import Generator, Literal
import logging

logger = logging.getLogger(__name__)

import numpy as np

from .audio import AudioPreprocessor, load_audio_as_float
from .config import Settings

from . import metrics

from .service_container import container
from .utils.device import resolve_device


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
    """Synthesize ``text`` using a cached local backend when available."""

    metrics.record_system_metrics()
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




def stt_local(audio_path: Path, lang: str = "it") -> str:



def stt_local(audio_path: Path, lang: str = "it") -> str:
    """Attempt a local transcription of ``audio_path`` using cached models."""

    backend, model = container.load_stt_model()

    if backend == "faster_whisper":
        try:
            segments, _ = model.transcribe(
                audio_path.as_posix(), language=lang, task="transcribe"
            )
            return "".join(seg.text for seg in segments).strip()
        except Exception:
            return ""

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


    metrics.record_system_metrics()

    compute_type = "int8_float16" if actual_device == "cuda" else "int8"


    try:
        model = _get_whisper("base", actual_device, compute_type)
        segments, _ = model.transcribe(
            audio_path.as_posix(), language=lang, task="transcribe"
        )


            device = metrics.resolve_device("auto")
        except Exception:
            device = "cpu"

        device = resolve_device("auto")

        compute_type = "int8_float16" if device == "cuda" else "int8"
        model = WhisperModel("base", device=device, compute_type=compute_type)
        segments, _ = model.transcribe(audio_path.as_posix(), language=lang, task="transcribe")

        return "".join(seg.text for seg in segments).strip()
    except Exception:
        pass


    if backend == "speech_recognition":
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
            return model.recognize_sphinx(audio, language=lang)
        except Exception:
            return ""

    return ""


def stt_local_faster(
    audio_path: Path,
    lang: str = "it",
    *,
    device: str = "cpu",
    compute_type: str = "int8",
) -> str:
    """Transcribe ``audio_path`` using a cached ``faster-whisper`` model."""


    backend, model = container.load_stt_model()
    if backend != "faster_whisper":

    ``device`` can be ``"cpu"`` or ``"cuda"``; ``compute_type`` controls the
    precision used by the model.  On failure or missing dependencies an empty
    string is returned.
    """

    device = metrics.resolve_device(device)
    metrics.record_system_metrics()
    try:
        model = _get_whisper("base", device, compute_type)
    except Exception:

        logger.warning("faster-whisper non disponibile, fallback a stt_local")
        return stt_local(audio_path, lang)

    try:
        segments, _ = model.transcribe(str(audio_path), language=lang)
        return "".join(seg.text for seg in segments).strip()
    except Exception:
        logger.error("Errore trascrizione locale con faster-whisper", exc_info=True)
        return ""
