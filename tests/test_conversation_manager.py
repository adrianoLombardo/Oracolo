import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.conversation import ConversationManager
from OcchioOnniveggente.DataBase.conversation_store import ConversationStore


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


def test_load_session_restores_state(tmp_path):
    db = tmp_path / "conv.sqlite"
    cm1 = ConversationManager(max_history=4, store=ConversationStore(db))
    cm1.push_user("u1")
    cm1.push_assistant("a1")
    sid = cm1.chat.session_id

    cm2 = ConversationManager(max_history=4, store=ConversationStore(db))
    cm2.load_session(sid)
    assert cm2.chat.history[0]["content"] == "u1"
    assert cm2.chat.history[1]["content"] == "a1"
