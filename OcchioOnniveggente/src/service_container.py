"""Simple dependency injection container for core services.

The container centralizes creation of shared components such as the OpenAI
client or placeholders for the datastore and audio modules. This makes it
possible to swap implementations during testing or at runtime without touching
consumer code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import atexit

from concurrent.futures import ProcessPoolExecutor

import openai
try:
    import torch
except Exception:  # pragma: no cover
    class _CudaStub:
        @staticmethod
        def is_available() -> bool:
            return False

    class _TorchStub:
        cuda = _CudaStub()

    torch = _TorchStub()  # type: ignore

from openai import AsyncOpenAI
from .config import Settings, get_openai_api_key
from .ui_state import UIState
from . import openai_async


@dataclass
class ServiceContainer:
    """Container holding application-wide services."""

    settings: Settings = field(default_factory=Settings)
    ui_state: UIState = field(default_factory=UIState)
    datastore: Any | None = None
    audio_module: Any | None = None

    _executor: ProcessPoolExecutor | None = field(default=None, init=False)
    _openai_client: AsyncOpenAI | None = field(default=None, init=False)

    _stt_model: Any | None = field(default=None, init=False)
    _tts_model: Any | None = field(default=None, init=False)
    _llm_models: dict[tuple[str, str], tuple[Any, Any]] = field(
        default_factory=dict, init=False
    )


    def openai_client(self) -> AsyncOpenAI:
        """Return a lazily initialized OpenAI client."""

        if self._openai_client is None:
            api_key = get_openai_api_key(self.settings)
            self._openai_client = AsyncOpenAI(api_key=api_key)
        return self._openai_client


    def executor(self) -> ProcessPoolExecutor:
        """Return a lazily initialized process pool executor."""

        if self._executor is None:
            workers = 1 if torch.cuda.is_available() else self.settings.openai.max_workers
            self._executor = ProcessPoolExecutor(max_workers=workers)
        return self._executor

    # ------------------------------------------------------------------
    # Local models
    def load_stt_model(self) -> tuple[str | None, Any | None]:
        """Return a lazily loaded STT model.

        The function tries to load ``faster-whisper`` first and falls back to
        ``speech_recognition``.  The returned tuple contains the backend name and
        the model object (or ``None`` if unavailable).
        """

        if self._stt_model is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore

                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "int8_float16" if device == "cuda" else "int8"
                model = WhisperModel("base", device=device, compute_type=compute_type)
                self._stt_model = ("faster_whisper", model)
            except Exception:
                try:
                    import speech_recognition as sr  # type: ignore

                    self._stt_model = ("speech_recognition", sr.Recognizer())
                except Exception:
                    self._stt_model = (None, None)
        return self._stt_model

    def load_tts_model(self) -> tuple[str | None, Any | None]:
        """Return a lazily loaded TTS model or engine.

        The returned tuple contains the backend name and the model/engine
        instance when available.
        """

        if self._tts_model is None:
            try:
                from gtts import gTTS  # type: ignore

                self._tts_model = ("gtts", gTTS)
            except Exception:
                try:
                    import pyttsx3  # type: ignore

                    engine = pyttsx3.init()
                    self._tts_model = ("pyttsx3", engine)
                except Exception:
                    self._tts_model = (None, None)
        return self._tts_model

    def load_llm(self, model_path: str, device: str) -> tuple[Any, Any]:
        """Return a lazily loaded local LLM (tokenizer, model) pair."""

        key = (model_path, device)
        if key not in self._llm_models:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path)
                model.to(device)
                model.eval()
                self._llm_models[key] = (tokenizer, model)
            except Exception as exc:  # pragma: no cover - import errors handled at runtime
                raise RuntimeError(
                    "transformers and torch are required for the local LLM backend"
                ) from exc
        return self._llm_models[key]

    def shutdown(self) -> None:
        """Shutdown managed resources."""

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


    def close(self) -> None:
        """Shutdown all services including async helpers."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._openai_client = None
        self._stt_model = None
        self._tts_model = None
        for tokenizer, model in self._llm_models.values():
            del tokenizer
            del model
        self._llm_models.clear()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - defensive
                pass

        openai_async.shutdown()


# Default container used by the application
container = ServiceContainer()
atexit.register(container.close)
