from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import soundfile as sf

try:  # pragma: no cover - optional dependency
    import webrtc_audio_processing as wap  # type: ignore
except Exception:  # pragma: no cover
    wap = None
try:  # pragma: no cover - optional dependency
    import rnnoise  # type: ignore
except Exception:  # pragma: no cover
    rnnoise = None


class AudioPreprocessor:
    """Optional noise suppression / echo cancellation / AGC processor."""

    def __init__(
        self,
        sr: int,
        *,
        denoise: bool = False,
        echo_cancel: bool = False,
        agc: bool = True,
    ) -> None:
        self.sr = sr
        self._proc: Any | None = None
        self._kind = "none"
        if wap is not None:
            self._proc = wap.AudioProcessor(
                enable_ns=denoise,
                enable_agc=agc,
                enable_aec=echo_cancel,
            )
            self._proc.set_stream_format(wap.StreamFormat(sr, 1))
            self._kind = "webrtc"
        elif rnnoise is not None and denoise:
            self._proc = rnnoise.RNNoise()
            self._kind = "rnnoise"

    def process(self, y: np.ndarray) -> np.ndarray:
        """Process ``y`` returning a cleaned signal."""
        if self._proc is None:
            return y
        if self._kind == "webrtc":
            arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            out = self._proc.process_stream(arr)
            return out.reshape(-1)
        if self._kind == "rnnoise":
            frame = 480
            padded = np.pad(y, (0, (-len(y)) % frame), mode="constant")
            out = np.empty_like(padded)
            for i in range(0, len(padded), frame):
                out[i : i + frame] = self._proc.process_frame(padded[i : i + frame])
            return out[: len(y)]
        return y


def load_audio_as_float(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """Load an audio file and return mono float32 samples and sample rate."""
    if path.suffix.lower() != ".mp3":
        y, sr = sf.read(path.as_posix(), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y, sr

    try:  # MP3 fallback
        from pydub import AudioSegment
    except Exception:
        raise RuntimeError(
            "File MP3 ma pydub/ffmpeg non disponibili. Usa WAV oppure installa ffmpeg."
        )
    audio = AudioSegment.from_mp3(path.as_posix())
    audio = audio.set_channels(1).set_frame_rate(target_sr)
    y = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return y, target_sr


def apply_agc(y: np.ndarray, target_rms: float = 0.1, max_gain: float = 10.0) -> np.ndarray:
    """Apply a simple automatic gain control to ``y``.

    The signal is amplified towards ``target_rms`` but never more than
    ``max_gain`` and clipped at [-1, 1]."""

    if y.size == 0:
        return y
    rms = float(np.sqrt(np.mean(y**2))) + 1e-8
    gain = min(max_gain, target_rms / rms)
    y = y * gain
    return np.clip(y, -1.0, 1.0)


def apply_limiter(y: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Limit the peak amplitude of ``y`` to ``threshold``."""

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= threshold or peak == 0.0:
        return y
    return y * (threshold / peak)
