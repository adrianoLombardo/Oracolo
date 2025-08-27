import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.scripts.realtime_server import get_active_profile
from src.retrieval import retrieve


def test_custom_profile_loaded_and_retrieval_filters(tmp_path: Path) -> None:
    settings = {
        "domain": {
            "profile": "them",
            "profiles": {
                "museo": {"topic": "museo"},
                "them": {"topic": "them"},
            },
        }
    }
    name, prof = get_active_profile(settings)
    assert name == "them"
    assert prof.get("topic") == "them"

    docs = [
        {"id": "a", "text": "common cat", "topic": "museo"},
        {"id": "b", "text": "common dog", "topic": "them"},
    ]
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"documents": docs}), encoding="utf-8")

    results = retrieve("common", index, top_k=5, topic=prof.get("topic"))
    assert results and all(d.get("topic") == "them" for d in results)
    ids = {d["id"] for d in results}
    assert ids == {"b"}
