import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.retrieval import Context
from OcchioOnniveggente.src.oracle import QuestionSession, get_questions


def test_next_question_cycles_without_repetition():
    session = QuestionSession()
    category = "poetica"
    ctx = Context.GENERIC
    qs = get_questions(ctx)[category]
    seen = set()
    for _ in range(len(qs)):
        q = session.next_question(category, context=ctx)
        assert q is not None
        assert q.domanda not in seen
        seen.add(q.domanda)
    # After exhausting, a new cycle should restart
    q = session.next_question(category, context=ctx)
    assert q is not None
    assert q.domanda in seen
