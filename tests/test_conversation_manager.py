import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.conversation import ConversationManager


def test_conversation_manager_tracks_and_summarizes():
    cm = ConversationManager(max_history=4)
    for i in range(6):
        cm.push_user(f"u{i}")
        cm.push_assistant(f"a{i}")
    msgs = cm.messages
    assert cm.chat.summary
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["content"] == "a5"
    assert len(msgs) <= cm.max_history + 1
