import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.oracle import append_log


def test_append_log_writes_metadata(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    append_log(
        "q",
        "a",
        log,
        lang="it",
        topic="neuro",
        sources=[{"id": "doc1", "score": 0.9}],
    )
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["lang"] == "it"
    assert entry["topic"] == "neuro"
    assert entry["question"] == "q"
    assert entry["answer"] == "a"
    assert entry["summary"] == "a"
    assert entry["sources"] == [{"id": "doc1", "score": 0.9}]
