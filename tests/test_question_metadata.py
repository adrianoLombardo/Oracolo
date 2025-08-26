import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.retrieval import load_questions, Context


def test_question_metadata_fields():
    qs = load_questions()[Context.GENERIC]
    tagged = [q for qq in qs.values() for q in qq if q.tag and "CryptoMadonne" in q.tag]
    assert tagged, "No question tagged CryptoMadonne"
    q = tagged[0]
    assert q.opera == "CryptoMadonne"
    assert q["opera"] == "CryptoMadonne"

