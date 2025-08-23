import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.oracle import append_log, reset_last_answer_hash


def test_append_log_writes_metadata(tmp_path: Path):
    log = tmp_path / "log.csv"
    reset_last_answer_hash()
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


def test_append_log_deduplicates(tmp_path: Path):
    log = tmp_path / "log.csv"
    reset_last_answer_hash()
    append_log("q1", "same", log)
    append_log("q1", "same", log)
    data = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(data) == 2  # header + single entry


def test_append_log_scrubs_pii(tmp_path: Path):
    log = tmp_path / "log.csv"
    reset_last_answer_hash()
    append_log(
        "contact me at foo@example.com or 123-456-7890",
        "ans",
        log,
    )
    line = log.read_text(encoding="utf-8").strip().splitlines()[1]
    assert "[email]" in line and "[phone]" in line
