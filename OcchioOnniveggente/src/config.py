from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field
class WakeConfig(BaseModel):
    enabled: bool = True
    single_turn: bool = True
    idle_timeout: float = 60.0
    it_phrases: List[str] = Field(default_factory=lambda: ["ciao oracolo", "ehi oracolo", "salve oracolo", "ciao, oracolo"])
    en_phrases: List[str] = Field(default_factory=lambda: ["hello oracle", "hey oracle", "hi oracle", "hello, oracle"])



class OpenAIConfig(BaseModel):
    api_key: str = ""
    stt_model: str = "gpt-4o-mini-transcribe"
    llm_model: str = "gpt-5-mini"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"
    embed_model: str = "text-embedding-3-small"


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
    wakeword: Optional[str] = None
    openai: OpenAIConfig = OpenAIConfig()
    audio: AudioConfig = AudioConfig()
    recording: RecordingConfig = RecordingConfig()
    vad: VadConfig = VadConfig()
    filter: FilterConfig = FilterConfig()
    lighting: LightingConfig = LightingConfig()
    palette_keywords: Dict[str, PaletteItem] = Field(default_factory=dict)
    oracle_system: str = ""
    docstore_path: str = "data/docstore"
    retrieval_top_k: int = 3
    wake: Optional[WakeConfig] = WakeConfig()
    
    @classmethod
    def model_validate_yaml(cls, path: Path) -> "Settings":
        """Load settings from a YAML file.

        Returns the default configuration if the file does not exist and raises
        a ``ValueError`` when the YAML content is invalid. This makes the
        application fail fast on misconfigured YAML files instead of silently
        falling back to defaults.
        """

        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return cls()

        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}") from exc

        return cls.model_validate(data)
