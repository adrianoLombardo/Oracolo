import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "OcchioOnniveggente"))
from src.domain import validate_question


def test_accept_keyword_without_embeddings_or_rag():
    dom = SimpleNamespace(
        enabled=True,
        keywords=["museo"],
        accept_threshold=0.75,
        fallback_accept_threshold=0.4,
    )
    settings = SimpleNamespace(domain=dom)
    ok, ctx, clarify, reason, sugg = validate_question("museo", settings=settings)
    assert ok
    thr = float(reason.split("thr=")[1].split()[0])
    assert thr <= 0.4


def test_short_question_with_keyword():
    dom = SimpleNamespace(
        enabled=True,
        keywords=["arte"],
        accept_threshold=0.75,
        fallback_accept_threshold=0.4,
    )
    settings = SimpleNamespace(domain=dom)
    ok, ctx, clarify, reason, sugg = validate_question("mi parli di arte?", settings=settings)
    assert ok
