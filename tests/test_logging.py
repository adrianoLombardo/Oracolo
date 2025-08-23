import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.oracle import append_log


def test_append_log_writes_metadata(tmp_path: Path):
    log = tmp_path / "log.csv"
    append_log(
        "q",
        "a",
        log,
        lang="it",
        topic="neuro",
        sources=[{"id": "doc1", "score": 0.9}],
    )
    data = log.read_text(encoding="utf-8").strip().splitlines()
    assert data[0] == '"timestamp","lang","topic","question","answer","sources"'
    assert "doc1:0.90" in data[1]
