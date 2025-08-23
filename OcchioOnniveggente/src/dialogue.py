from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto


class DialogState(Enum):
    """Possible states for the dialogue state machine."""

    SLEEP = auto()
    AWAKE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    INTERRUPTED = auto()


@dataclass
class DialogueManager:
    """Maintain conversation state and timeouts."""

    idle_timeout: float
    state: DialogState = DialogState.SLEEP
    active_deadline: float = 0.0
    is_processing: bool = False
    turn_id: int = 0

    def refresh_deadline(self) -> None:
        """Refresh the activity deadline."""
        self.active_deadline = time.time() + self.idle_timeout

    def transition(self, new_state: DialogState) -> None:
        """Transition to a new state and refresh deadline when appropriate."""
        self.state = new_state
        if new_state in (
            DialogState.AWAKE,
            DialogState.LISTENING,
            DialogState.SPEAKING,
            DialogState.INTERRUPTED,
        ):
            self.refresh_deadline()

    def start_processing(self) -> int:
        """Mark that a user turn is being processed and return turn id."""
        self.is_processing = True
        self.turn_id += 1
        return self.turn_id

    def end_processing(self) -> None:
        """Clear processing flag after finishing the turn."""
        self.is_processing = False

    def timed_out(self, now: float) -> bool:
        """Return True if current session expired due to inactivity."""
        return self.state is not DialogState.SLEEP and now > self.active_deadline
