import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.question_session import QuestionSession
from OcchioOnniveggente.src.retrieval import Question


def test_session_serves_all_questions_before_repeat():
    questions = {
        "demo": [
            Question(domanda="q1", type="demo"),
            Question(domanda="q2", type="demo"),
            Question(domanda="q3", type="demo"),
        ]
    }
    session = QuestionSession(questions)
    seen = set()
    total = len(questions["demo"])
    for _ in range(total):
        q = session.next_question("demo")
        assert q.domanda not in seen
        seen.add(q.domanda)
    assert len(seen) == total

    # Next question should come from new cycle
    q = session.next_question("demo")
    assert q.domanda in seen
