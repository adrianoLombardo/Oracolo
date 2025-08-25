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
    _openai_client: AsyncOpenAI | None = field(default=None, init=False)

    def openai_client(self) -> AsyncOpenAI:
        """Return a lazily initialized OpenAI client."""

        if self._openai_client is None:
            api_key = get_openai_api_key(self.settings)
            self._openai_client = AsyncOpenAI(api_key=api_key)
        return self._openai_client

    def close(self) -> None:
        """Shutdown all services including async helpers."""

        openai_async.shutdown()


# Default container used by the application
container = ServiceContainer()
atexit.register(container.close)
