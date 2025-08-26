from __future__ import annotations

"""Simple plugin registry for STT, TTS and LLM providers.

Plugins can be registered via :func:`register_stt`, :func:`register_tts` and
:func:`register_llm`.  Third party packages may also expose entry points using
``oracolo.stt``, ``oracolo.tts`` or ``oracolo.llm`` groups.  Each entry point is
expected to provide a factory callable accepting a
:class:`~OcchioOnniveggente.src.service_container.ServiceContainer` instance and
returning the concrete backend implementation.
"""

from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Callable, Dict

if TYPE_CHECKING:  # pragma: no cover - hints only
    from .audio import SpeechToText, TextToSpeech
    from .openai_async import LLMClient
    from .service_container import ServiceContainer

# Factory type aliases -----------------------------------------------------
STTFactory = Callable[["ServiceContainer"], "SpeechToText"]
TTSFactory = Callable[["ServiceContainer"], "TextToSpeech"]
LLMFactory = Callable[["ServiceContainer"], "LLMClient"]

# Registries ---------------------------------------------------------------
_stt_registry: Dict[str, STTFactory] = {}
_tts_registry: Dict[str, TTSFactory] = {}
_llm_registry: Dict[str, LLMFactory] = {}

_ep_loaded = False


def register_stt(name: str, factory: STTFactory) -> None:
    """Register a speech-to-text provider factory under ``name``."""
    _stt_registry[name] = factory


def register_tts(name: str, factory: TTSFactory) -> None:
    """Register a text-to-speech provider factory under ``name``."""
    _tts_registry[name] = factory


def register_llm(name: str, factory: LLMFactory) -> None:
    """Register a language model provider factory under ``name``."""
    _llm_registry[name] = factory


# Entry point loading ------------------------------------------------------

def _load_entry_points() -> None:
    global _ep_loaded
    if _ep_loaded:
        return
    _ep_loaded = True
    eps = entry_points()
    for ep in eps.select(group="oracolo.stt"):
        register_stt(ep.name, ep.load())
    for ep in eps.select(group="oracolo.tts"):
        register_tts(ep.name, ep.load())
    for ep in eps.select(group="oracolo.llm"):
        register_llm(ep.name, ep.load())


# Factory helpers ----------------------------------------------------------

def create_stt(name: str, container: "ServiceContainer"):
    _load_entry_points()
    factory = _stt_registry.get(name)
    if factory is None:
        raise ValueError(f"Unsupported STT provider: {name}")
    return factory(container)


def create_tts(name: str, container: "ServiceContainer"):
    _load_entry_points()
    factory = _tts_registry.get(name)
    if factory is None:
        raise ValueError(f"Unsupported TTS provider: {name}")
    return factory(container)


def create_llm(name: str, container: "ServiceContainer"):
    _load_entry_points()
    factory = _llm_registry.get(name)
    if factory is None:
        raise ValueError(f"Unsupported LLM provider: {name}")
    return factory(container)


__all__ = [
    "register_stt",
    "register_tts",
    "register_llm",
    "create_stt",
    "create_tts",
    "create_llm",
]
