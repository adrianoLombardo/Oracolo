import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.chat import ChatState


def test_summary_and_pinned_shortlist():
    st = ChatState(max_turns=1, pinned_limit=2)
    st.push_user("ciao")
    st.push_assistant("salve")
    st.push_user("come va?")
    st.push_assistant("bene")
    assert st.summary
    assert len(st.history) == 2 and st.history[-1]["content"] == "bene"
    st.pin_message("uno")
    st.pin_message("due")
    st.pin_message("tre")
    assert st.pinned_shortlist() == ["due", "tre"]
