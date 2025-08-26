import sys
from pathlib import Path
import random

sys.path.append(str(Path(__file__).resolve().parents[1]))


from OcchioOnniveggente.src.oracle import QuestionSession as OracleQuestionSession, get_questions


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

