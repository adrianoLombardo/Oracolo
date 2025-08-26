import sys
from pathlib import Path
import random

sys.path.append(str(Path(__file__).resolve().parents[1]))



from OcchioOnniveggente.src.oracle import QuestionSession as OracleQuestionSession, get_questions
from OcchioOnniveggente.src.question_session import QuestionSession
from OcchioOnniveggente.src.oracle import get_questions



def test_round_robin_rotation():
    session = OracleQuestionSession(rng=random.Random(0))
    categories = session._categories.copy()
    # consume one full cycle
    seen = [session.next_question().type for _ in range(len(categories))]
    assert seen == categories
    # next call should restart from beginning
    assert session.next_question().type == categories[0]


def test_weighted_selection():
    categories = list(get_questions().keys())
    target = categories[0]
    session = OracleQuestionSession(weights={target: 1.0}, rng=random.Random(0))
    chosen = {session.next_question().type for _ in range(10)}
    assert chosen == {target}

from OcchioOnniveggente.src.retrieval import Question


def test_session_serves_all_questions_before_repeat():
    questions = {
        "demo": [
            Question(domanda="q1", type="demo"),
            Question(domanda="q2", type="demo"),
            Question(domanda="q3", type="demo"),
        ]
    }
    session = QuestionSession(questions, rng=random.Random(0))
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


def test_next_question_filters_by_tags():
    questions = {
        "demo": [
            Question(domanda="q1", type="demo", tag=["a", "b"]),
            Question(domanda="q2", type="demo", tag=["b"]),
        ]
    }
    session = QuestionSession(questions)
    q = session.next_question("demo", tags={"a"})
    assert q.domanda == "q1"
    assert session.next_question("demo", tags={"z"}) is None


def test_next_question_recalculates_category_for_tags():
    questions = {
        "a": [Question(domanda="qa", type="a", tag=["x"])],
        "b": [Question(domanda="qb", type="b", tag=["y"])],
    }
    session = QuestionSession(questions)
    q = session.next_question(tags={"y"})
    assert q.type == "b"
    assert "y" in (q.tag or [])

    # No category contains requested tag
    assert session.next_question(tags={"not-there"}) is None

