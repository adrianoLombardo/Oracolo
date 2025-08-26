import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.ui_state import UIState, apply_to_chat


def test_apply_to_chat_creates_conversation():
    state = UIState()
    apply_to_chat(state, "ciao")
    assert state.conversation is not None
    assert state.conversation.chat.history[0]["content"] == "ciao"
