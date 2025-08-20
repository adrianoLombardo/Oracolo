from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field


class OpenAIConfig(BaseModel):
    stt_model: str = "gpt-4o-mini-transcribe"
    llm_model: str = "gpt-5-mini"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"


class AudioConfig(BaseModel):
    sample_rate: int = 24_000
    ask_seconds: int = 10
    input_wav: str = "data/temp/input.wav"
    output_wav: str = "data/temp/answer.wav"
    input_device: Optional[str | int] = None
    output_device: Optional[str | int] = None


class RecordingConfig(BaseModel):
    mode: Literal["vad", "timed"] = "vad"
    timed_seconds: int = 10
    fallback_to_timed: bool = False
    min_speech_level: float = 0.01


class VadConfig(BaseModel):
    frame_ms: int = 30
    start_ms: int = 150
    end_ms: int = 800
    max_ms: int = 15_000
    preroll_ms: int = 300
    noise_window_ms: int = 800
    start_mult: float = 1.8
    end_mult: float = 1.3
    base_start: float = 0.006
    base_end: float = 0.0035


class FilterConfig(BaseModel):
    mode: Literal["block", "mask"] = "block"


class SacnConfig(BaseModel):
    universe: int = 1
    destination_ip: str
    rgb_channels: List[int]
    idle_level: int = 10
    peak_level: int = 255


class WledConfig(BaseModel):
    host: str


class LightingConfig(BaseModel):
    mode: Literal["sacn", "wled"] = "sacn"
    sacn: SacnConfig = SacnConfig(destination_ip="192.168.1.50", rgb_channels=[1, 2, 3])
    wled: WledConfig = WledConfig(host="192.168.1.77")


class PaletteItem(BaseModel):
    rgb: Tuple[int, int, int]
    style: str


class Settings(BaseModel):
    debug: bool = False
    openai: OpenAIConfig = OpenAIConfig()
    audio: AudioConfig = AudioConfig()
    recording: RecordingConfig = RecordingConfig()
    vad: VadConfig = VadConfig()
    filter: FilterConfig = FilterConfig()
    lighting: LightingConfig = LightingConfig()
    palette_keywords: Dict[str, PaletteItem] = Field(default_factory=dict)
    oracle_system: str = ""

    @classmethod
    def model_validate_yaml(cls, path: Path) -> "Settings":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)
