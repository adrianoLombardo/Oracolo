import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.retrieval import load_questions


def test_question_metadata_fields():
    qs = load_questions()
    all_q = [q for qq in qs.values() for q in qq]
    assert all_q, "No questions loaded"
    q = all_q[0]
    assert isinstance(q.id, str)
    assert q["id"] == q.id

