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
