from __future__ import annotations

"""Local TTS/STT helpers and simple chunked streaming utilities."""

from pathlib import Path
from typing import Generator
import logging


logger = logging.getLogger(__name__)

import numpy as np

from .audio import AudioPreprocessor, load_audio_as_float
from .config import Settings


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

    The function prefers the `faster-whisper` package, automatically
    selecting GPU or CPU and using quantized models when available.  If the
    library is missing it falls back to ``speech_recognition`` with the
    PocketSphinx backend.  On failure an empty string is returned.
    """

    """Attempt a local transcription of ``audio_path`` using a cleaned signal."""


    try:
        from faster_whisper import WhisperModel  # type: ignore

        try:
            import torch  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
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

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        logger.warning("faster-whisper non disponibile, fallback a stt_local")
        return stt_local(audio_path, lang)

    try:
        model = WhisperModel("base", device=device, compute_type=compute_type)
        segments, _ = model.transcribe(str(audio_path), language=lang)
        return "".join(seg.text for seg in segments).strip()
    except Exception:
        logger.error("Errore trascrizione locale con faster-whisper", exc_info=True)
        return ""
