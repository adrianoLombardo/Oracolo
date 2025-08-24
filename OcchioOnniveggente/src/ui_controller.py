from __future__ import annotations

from typing import Any

from src.conversation import ConversationManager
from src.ui_state import UIState


class UIController:
    """Simple controller that mediates access to :class:`UIState`.

    Components interacting with the UI can use this controller instead of
    manipulating global variables directly.  Only a subset of behaviour is
    required for the tests, so the class intentionally keeps a very small
    surface area.
    """

    def __init__(self, state: UIState | None = None) -> None:
        self.state = state or UIState()

    # ------------------------------------------------------------------
    # configuration
    @property
    def settings(self) -> dict[str, Any]:
        return self.state.settings

    def update_settings(self, new_settings: dict[str, Any]) -> None:
        self.state.settings = new_settings

    # ------------------------------------------------------------------
    # conversation
    @property
    def conversation(self) -> ConversationManager | None:
        return self.state.conversation

    def set_conversation(self, conv: ConversationManager) -> None:
        self.state.conversation = conv

    # ------------------------------------------------------------------
    # audio reference
    @property
    def audio(self) -> Any | None:
        return self.state.audio

    def set_audio(self, audio: Any | None) -> None:
        self.state.audio = audio
