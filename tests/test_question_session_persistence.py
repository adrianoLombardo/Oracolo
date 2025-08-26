import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.retrieval import Question
from OcchioOnniveggente.src.question_session import QuestionSession


def test_rotation_after_reload(tmp_path: Path) -> None:
    questions = {
        "a": [Question(domanda="qa", type="a")],
        "b": [Question(domanda="qb", type="b")],
    }
    state = tmp_path / "session.json"

    session = QuestionSession(questions, state_path=state)
    first = session.next_question()
    assert first.type == "a"
    session.record_answer("ans", "rep")
    session.save(state)

    restored = QuestionSession(questions, state_path=state)
    assert restored.answers == ["ans"]
    assert restored.replies == ["rep"]

    second = restored.next_question()
    assert second.type == "b"
    third = restored.next_question()
    assert third.type == "a"
