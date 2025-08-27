"""Audio utilities and recording helpers.

This module exposes convenient re-exports so other parts of the project can
import symbols directly from ``src.audio``.
"""

from .processing import AudioPreprocessor, apply_agc, apply_limiter, load_audio_as_float
from .recording import play_and_pulse, record_until_silence, record_wav
from . import local_audio

__all__ = [
    "AudioPreprocessor",
    "apply_agc",
    "apply_limiter",
    "load_audio_as_float",
    "record_until_silence",
    "play_and_pulse",
    "record_wav",
    "local_audio",
]

