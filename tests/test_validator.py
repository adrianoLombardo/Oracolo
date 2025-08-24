import json
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "OcchioOnniveggente"))
from src.domain import validate_question


def test_adaptive_threshold(tmp_path):
    dom = SimpleNamespace(enabled=True, keywords=["science"], accept_threshold=0.5)
    settings = SimpleNamespace(domain=dom)
    history = [
        {"role": "user", "content": "football"},
        {"role": "user", "content": "music"},
        {"role": "user", "content": "weather"},
    ]
    ok, ctx, clarify, reason, sugg = validate_question(
        "random", settings=settings, history=history
    )
    assert "thr=" in reason
    thr = float(reason.split("thr=")[1].split()[0])
    assert thr > 0.5


def test_topic_switch_suggestion(tmp_path):
    index = tmp_path / "index.json"
    data = {
        "documents": [
            {"id": "1", "text": "Cats are animals", "topic": "animals"},
            {"id": "2", "text": "car goes fast", "topic": "vehicles"},
        ]
    }
    index.write_text(json.dumps(data), encoding="utf-8")

    dom = SimpleNamespace(
        enabled=True,
        keywords=["cat"],
        accept_threshold=0.1,
        clarify_margin=0.15,
        topic="animals",
    )
    settings = SimpleNamespace(domain=dom)
    ok, ctx, clarify, reason, sugg = validate_question(
        "car", settings=settings, docstore_path=index, top_k=1
    )
    assert clarify
    assert sugg == "vehicles"


def test_disabled_returns_full_tuple():
    dom = SimpleNamespace(enabled=False, keywords=["science"])
    settings = SimpleNamespace(domain=dom)
    ok, ctx, clarify, reason, sugg = validate_question("hi", settings=settings)
    assert ok
    assert reason == "disabled"
    assert sugg is None


def test_off_topic_rejected():
    dom = SimpleNamespace(enabled=True, keywords=["arte", "collezione"], accept_threshold=0.5)
    settings = SimpleNamespace(domain=dom)
    q = "Parli della squadra di calcio del Milan"
    ok, ctx, clarify, reason, sugg = validate_question(q, settings=settings)
    assert not ok
    assert not clarify
    assert ctx == []
    assert "rag_hits=0" in reason


def test_rag_hits_allow_without_keywords(tmp_path):
    index = tmp_path / "index.json"
    data = {
        "documents": [
            {"id": "1", "text": "Cats are animals", "topic": "animals"},
        ]
    }
    index.write_text(json.dumps(data), encoding="utf-8")

    dom = SimpleNamespace(
        enabled=True,
        keywords=["football"],
        accept_threshold=0.1,
        weights={"kw": 0.0, "emb": 0.0, "rag": 1.0},
    )
    settings = SimpleNamespace(domain=dom)
    ok, ctx, clarify, reason, sugg = validate_question(
        "Cats", settings=settings, docstore_path=index, top_k=1
    )
    assert ok
    assert ctx
