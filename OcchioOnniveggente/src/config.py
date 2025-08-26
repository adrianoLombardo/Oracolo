from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import os
import yaml
from functools import lru_cache
from pydantic import BaseModel, Field
class WakeConfig(BaseModel):
    enabled: bool = True
    single_turn: bool = False
    idle_timeout: float = 60.0
    it_phrases: List[str] = Field(default_factory=lambda: ["ciao oracolo", "ehi oracolo", "salve oracolo", "ciao, oracolo"])
    en_phrases: List[str] = Field(default_factory=lambda: ["hello oracle", "hey oracle", "hi oracle", "hello, oracle"])



class OpenAIConfig(BaseModel):
    api_key: str = ""
    stt_model: str = "gpt-4o-mini-transcribe"
    llm_model: str = "gpt-5-mini"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"
    embed_model: str = "text-embedding-3-large"
    max_workers: int = 4


class ComputeModuleConfig(BaseModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["fp32", "fp16", "bf16", "int4"] = "fp16"
    batch_interval: int = 100  # ms between batch flushes
    max_batch_size: int = 4     # maximum prompts per batch


class ComputeConfig(BaseModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"

    use_onnx: bool = False

    device_concurrency: int = 1

    stt: ComputeModuleConfig = ComputeModuleConfig()
    llm: ComputeModuleConfig = ComputeModuleConfig()
    tts: ComputeModuleConfig = ComputeModuleConfig()


class AudioConfig(BaseModel):
    sample_rate: int = 24_000
    ask_seconds: int = 10
    input_wav: str = "data/temp/input.wav"
    output_wav: str = "data/temp/answer.wav"
    input_device: Optional[str | int] = None
    output_device: Optional[str | int] = None
    barge_rms_threshold: float = 0.25
    denoise: bool = False
    echo_cancel: bool = False


class RecordingConfig(BaseModel):
    mode: Literal["vad", "timed"] = "vad"
    timed_seconds: int = 10
    fallback_to_timed: bool = False
    min_speech_level: float = 0.02
    hold_off_after_tts_ms: int = 500
    use_webrtcvad: bool = False


class VadConfig(BaseModel):
    frame_ms: int = 30
    start_ms: int = 200
    end_ms: int = 800
    max_ms: int = 15_000
    preroll_ms: int = 300
    noise_window_ms: int = 1_000
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


class DomainProfileConfig(BaseModel):
    keywords: List[str] = Field(default_factory=list)
    weights: Dict[str, float] | None = None
    accept_threshold: float | None = None
    clarify_margin: float | None = None


class DomainConfig(BaseModel):
    enabled: bool = True
    profile: str = "museo"
    topic: str = ""
    keywords: List[str] = Field(default_factory=list)
    kw_min_overlap: float = 0.04
    emb_min_sim: float = 0.22
    rag_min_hits: int = 1
    weights: Dict[str, float] = Field(
        default_factory=lambda: {"kw": 0.6, "emb": 0.2, "rag": 0.2}
    )
    accept_threshold: float = 0.65
    clarify_margin: float = 0.10
    profiles: Dict[str, DomainProfileConfig] = Field(default_factory=dict)


class RetrievalConfig(BaseModel):
    hybrid: bool = True
    rerank: bool = True
    max_chunks: int = 6
    chunk_strategy: Literal["semantic", "fixed_overlap"] = "semantic"


class ChatConfig(BaseModel):
    inactivity_timeout_s: int = 60
    remember_turns: int = 8
    pinned: List[str] = Field(default_factory=list)
    summary_model: str = "gpt-4o-mini"
    summary_max_tokens: int = 256
    tone: Literal["formal", "informal"] = "informal"


class RealtimeAudioConfig(BaseModel):
    chunk_ms: int = 20
    start_level: int = 300
    end_sil_ms: int = 700
    max_utt_ms: int = 15_000


class RealtimeConfig(BaseModel):
    barge_in_threshold: float = 0.55
    ducking_db: float = -12.0
    cpu_workers: int = 4
    gpu_workers: int = 1


class Settings(BaseModel):
    debug: bool = False
    stt_backend: Literal["openai", "whisper"] = "openai"
    wakeword: Optional[str] = None
    openai: OpenAIConfig = OpenAIConfig()
    compute: ComputeConfig = ComputeConfig()
    audio: AudioConfig = AudioConfig()
    recording: RecordingConfig = RecordingConfig()
    vad: VadConfig = VadConfig()
    filter: FilterConfig = FilterConfig()
    lighting: LightingConfig = LightingConfig()
    palette_keywords: Dict[str, PaletteItem] = Field(default_factory=dict)
    oracle_system: str = ""  # stile oracolare
    oracle_policy: str = ""  # prompt fattuale/guardrail
    answer_mode: Literal["detailed", "concise"] = "detailed"
    docstore_path: str = "data/docstore"
    retrieval_top_k: int = 3
    wake: Optional[WakeConfig] = WakeConfig()
    domain: DomainConfig = DomainConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    chat: ChatConfig = ChatConfig()
    realtime: RealtimeConfig = RealtimeConfig()

    realtime_audio: RealtimeAudioConfig = RealtimeAudioConfig()

    cache_dir: str = "data/cache"
    cache_ttl: int = 3600

    
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


@lru_cache()
def _load_api_key_from_files() -> str:
    """Load the OpenAI API key from settings files, if present."""
    for name in ("settings.local.yaml", "settings.yaml"):
        p = Path(name)
        if not p.exists():
            continue
        try:
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        key = cfg.get("openai", {}).get("api_key")
        if key:
            return key
    return ""


def get_openai_api_key(settings: Any | None = None) -> str:
    """Return the OpenAI API key.

    The key is resolved from, in order of priority:
    1. Environment variable ``OPENAI_API_KEY``;
    2. the provided ``settings`` object or mapping;
    3. ``settings.local.yaml`` or ``settings.yaml`` on disk.

    Settings files are read only once and cached.
    """

    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    if settings is not None:
        attr_key = getattr(getattr(settings, "openai", None), "api_key", None)
        if not attr_key and isinstance(settings, dict):
            attr_key = settings.get("openai", {}).get("api_key")
        if attr_key:
            return attr_key

    return _load_api_key_from_files()
