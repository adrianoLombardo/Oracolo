from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .conversation import ConversationManager


@dataclass
class UIState:
    """Container for shared UI state.

    The UI previously relied on a couple of module level variables to keep
    track of the current configuration, the conversation history and any
    audio handle in use.  Keeping this information in a dedicated object
    makes it easier to pass the state around explicitly and avoids reliance
    on globals.
    """

    settings: dict[str, Any] = field(default_factory=dict)
    conversation: ConversationManager | None = None
    audio: Any | None = None


def apply_to_chat(state: UIState, text: str) -> None:
    """Append ``text`` to the current conversation history.

    A new :class:`ConversationManager` is created if one isn't present yet.
    Empty strings are ignored.
    """

    if not text:
        return
    if state.conversation is None:
        state.conversation = ConversationManager()
    state.conversation.push_user(text)
