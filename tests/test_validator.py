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
