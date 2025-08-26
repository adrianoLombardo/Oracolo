import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.oracle import (
    get_questions,
    random_question,
    _USED_QUESTIONS,
)


def test_random_question_no_repeat_until_exhaustion():
    category = "poetica"
    _USED_QUESTIONS.clear()

    total = len(get_questions()[category])
    seen = set()
    for _ in range(total):
        q = random_question(category)
        assert q is not None
        assert q.domanda not in seen
        seen.add(q.domanda)

    assert len(seen) == total
    assert len(_USED_QUESTIONS[category]) == total

    # After exhausting all questions, the next call should reset the set
    q = random_question(category)
    assert q is not None and q.domanda in seen
    assert len(_USED_QUESTIONS[category]) == 1

    # Drawing the remaining questions again should not repeat within the new cycle.
    seen_second_cycle = {q.domanda}
    for _ in range(total - 1):
        q = random_question(category)
        assert q is not None
        assert q.domanda not in seen_second_cycle
        seen_second_cycle.add(q.domanda)

    assert len(_USED_QUESTIONS[category]) == total

