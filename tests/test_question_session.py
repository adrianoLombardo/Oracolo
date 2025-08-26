import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


from OcchioOnniveggente.src.oracle import QuestionSession, get_questions


def test_round_robin_rotation():
    session = QuestionSession()
    categories = session._categories.copy()
    # consume one full cycle
    seen = [session.next_question().type for _ in range(len(categories))]
    assert seen == categories
    # next call should restart from beginning
    assert session.next_question().type == categories[0]


def test_weighted_selection():
    categories = list(get_questions().keys())
    target = categories[0]
    session = QuestionSession(weights={target: 1.0})
    chosen = {session.next_question().type for _ in range(10)}
    assert chosen == {target}

from OcchioOnniveggente.src.question_session import QuestionSession
from OcchioOnniveggente.src.retrieval import Question
import random


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


def test_add_question_and_remove_category():
    session = QuestionSession({})
    q = Question(domanda="nuova", type="extra")
    session.add_question("Extra", q)
    assert "extra" in session._categories
    assert session.questions["extra"] == [q]
    assert session._used["extra"] == set()

    session.remove_question("extra", 0)
    assert "extra" not in session.questions
    assert "extra" not in session._categories
    assert "extra" not in session._used


def test_remove_question_updates_used():
    q1 = Question(domanda="q1", type="demo")
    q2 = Question(domanda="q2", type="demo")
    q3 = Question(domanda="q3", type="demo")
    session = QuestionSession({"demo": [q1, q2, q3]})
    session._used["demo"] = {0, 2}

    session.remove_question("demo", 1)
    assert [q.domanda for q in session.questions["demo"]] == ["q1", "q3"]
    assert session._used["demo"] == {0, 1}


def test_reset_category_allows_repeats():
    q1 = Question(domanda="q1", type="demo")
    q2 = Question(domanda="q2", type="demo")
    session = QuestionSession({"demo": [q1, q2]})

    random.seed(0)
    first = session.next_question("demo")
    random.seed(0)
    second = session.next_question("demo")
    assert first != second

    session.reset_category("demo")
    random.seed(0)
    third = session.next_question("demo")
    assert third == first

