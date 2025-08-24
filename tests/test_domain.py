import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "OcchioOnniveggente"))
from src.domain import validate_question, _get_field


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


def test_get_field_works_with_dict_and_obj():
    ns = SimpleNamespace(foo=1)
    d = {"foo": 2}
    assert _get_field(ns, "foo") == 1
    assert _get_field(d, "foo") == 2
    assert _get_field(d, "missing", 5) == 5


def test_validate_with_dict_settings():
    dom = {
        "enabled": True,
        "keywords": ["museo"],
        "accept_threshold": 0.75,
        "fallback_accept_threshold": 0.4,
    }
    settings = {"domain": dom}
    ok, ctx, clarify, reason, sugg = validate_question("museo", settings=settings)
    assert ok
