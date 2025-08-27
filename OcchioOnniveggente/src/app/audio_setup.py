from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sounddevice as sd

from src.audio import AudioPreprocessor
from src.audio_device import pick_device, debug_print_devices
from src.lights import SacnLight, WledLight
from src.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class AudioSetup:
    audio_conf: Any
    audio_sr: int
    input_wav: Path
    output_wav: Path
    preproc: AudioPreprocessor
    in_dev: Any
    out_dev: Any
    recording_conf: dict[str, Any]
    vad_conf: dict[str, Any]
    lighting_conf: dict[str, Any]
    light: Any
    is_tts_playing: threading.Event


def audio_setup(settings: Settings, debug: bool, args: Any) -> AudioSetup:
    """Configure audio devices and lighting."""
    audio_conf = settings.audio
    audio_sr = audio_conf.sample_rate
    input_wav = Path(audio_conf.input_wav)
    output_wav = Path(audio_conf.output_wav)
    preproc = AudioPreprocessor(
        audio_sr,
        denoise=getattr(audio_conf, "denoise", False),
        echo_cancel=getattr(audio_conf, "echo_cancel", False),
    )
    in_spec = audio_conf.input_device
    out_spec = audio_conf.output_device
    in_dev = pick_device(in_spec, "input")
    out_dev = pick_device(out_spec, "output")
    sd.default.device = (in_dev, out_dev)
    if debug:
        debug_print_devices()

    recording_conf = settings.recording.model_dump()
    vad_conf = settings.vad.model_dump()
    lighting_conf = settings.lighting.model_dump()

    is_tts_playing = threading.Event()

    light_mode = settings.lighting.mode
    if light_mode == "sacn":
        light = SacnLight(lighting_conf)
    elif light_mode == "wled":
        light = WledLight(lighting_conf)
    else:
        logger.warning("⚠️ lighting.mode non valido, uso WLED di default")
        light = WledLight(lighting_conf)

    return AudioSetup(
        audio_conf=audio_conf,
        audio_sr=audio_sr,
        input_wav=input_wav,
        output_wav=output_wav,
        preproc=preproc,
        in_dev=in_dev,
        out_dev=out_dev,
        recording_conf=recording_conf,
        vad_conf=vad_conf,
        lighting_conf=lighting_conf,
        light=light,
        is_tts_playing=is_tts_playing,
    )
