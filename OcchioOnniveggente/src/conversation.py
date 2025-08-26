from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .chat import ChatState
from .dialogue import DialogueManager, DialogState


@dataclass
class ConversationManager:
    """Unified wrapper around :class:`ChatState` and :class:`DialogueManager`.

    It exposes helper methods to push user/assistant messages while keeping
    track of dialogue state and processing turns.
    """

    idle_timeout: float = 60.0
    chat: ChatState = field(default_factory=ChatState)
    dlg: DialogueManager = field(init=False)
    is_processing: bool = False
    turn_id: int = 0

    def __post_init__(self) -> None:  # pragma: no cover - simple delegation
        self.dlg = DialogueManager(self.idle_timeout)

    # ------------------------- dialogue proxies -------------------------
    @property
    def state(self) -> DialogState:
        return self.dlg.state

    @state.setter
    def state(self, value: DialogState) -> None:
        self.dlg.state = value

    def refresh_deadline(self) -> None:
        self.dlg.refresh_deadline()

    @property
    def active_deadline(self) -> float:
        """Proxy access to the underlying dialogue deadline."""
        return self.dlg.active_deadline

    @active_deadline.setter
    def active_deadline(self, value: float) -> None:
        self.dlg.active_deadline = value

    def transition(self, new_state: DialogState) -> None:
        self.dlg.transition(new_state)

    def timed_out(self, now: float) -> bool:
        return self.dlg.timed_out(now)

    # --------------------------- turn helpers ---------------------------
    def start_processing(self) -> int:
        """Mark that a user turn is being processed and return turn id."""
        self.is_processing = True
        self.turn_id += 1
        return self.turn_id

    def end_processing(self) -> None:
        """Clear processing flag after finishing the turn."""
        self.is_processing = False

    # --------------------------- chat helpers ---------------------------
    def push_user(self, text: str) -> None:
        self.chat.push_user(text)

    def push_assistant(self, text: str) -> None:
        self.chat.push_assistant(text)


__all__ = ["ConversationManager", "DialogState"]
