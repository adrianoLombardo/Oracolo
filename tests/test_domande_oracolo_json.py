import json
from collections import Counter
from pathlib import Path


def test_domande_oracolo_structure_and_counts():
    path = (
        Path(__file__).resolve().parents[1]
        / "OcchioOnniveggente"
        / "data"
        / "domande_oracolo.json"
    )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)

    counts = Counter()
    for entry in data:
        assert "domanda" in entry
        assert "type" in entry
        assert "id" in entry
        assert isinstance(entry["domanda"], str)
        assert isinstance(entry["type"], str)
        assert isinstance(entry["id"], int)
        if "follow_up" in entry:
            assert isinstance(entry["follow_up"], str)
        counts[entry["type"]] += 1

    assert counts == {
        "poetica": 50,
        "didattica": 50,
        "evocativa": 50,
        "orientamento": 50,
    }
