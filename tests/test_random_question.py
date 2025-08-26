import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.oracle import (
    QUESTIONS_BY_TYPE,
    random_question,
    _USED_QUESTIONS,
)


def test_random_question_no_repeat_until_exhaustion():
    category = "poetica"
    # reset session tracking
    _USED_QUESTIONS.clear()

    total = len(QUESTIONS_BY_TYPE[category])
    seen = set()
    for _ in range(total):
        q = random_question(category)
        assert q.domanda not in seen
        seen.add(q.domanda)

    assert len(seen) == total
    assert len(_USED_QUESTIONS[category]) == total

    # After exhausting all questions, the next call should reset the set
    q = random_question(category)
    assert q.domanda in seen
    assert len(_USED_QUESTIONS[category]) == 1
