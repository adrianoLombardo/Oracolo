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

    _openai_client: openai.OpenAI | None = field(default=None, init=False)
    _executor: ProcessPoolExecutor | None = field(default=None, init=False)

    _openai_client: AsyncOpenAI | None = field(default=None, init=False)


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

    def shutdown(self) -> None:
        """Shutdown managed resources."""

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


    def close(self) -> None:
        """Shutdown all services and free cached models."""

        openai_async.shutdown()

        # Svuota il cache dei modelli Whisper per liberare la VRAM
        try:
            from .local_audio import _WHISPER_CACHE

            _WHISPER_CACHE.clear()
        except Exception:  # pragma: no cover - se il modulo non è caricato
            pass


# Default container used by the application
container = ServiceContainer()
atexit.register(container.close)
