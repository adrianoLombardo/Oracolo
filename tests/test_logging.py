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
        session_id="sess-1",
    )
    data = log.read_text(encoding="utf-8").strip().splitlines()
    assert (
        data[0]
        == '"timestamp","session_id","lang","topic","question","answer","sources"'
    )
    assert "doc1:0.90" in data[1]


def test_append_log_includes_session_id(tmp_path: Path):
    log = tmp_path / "log.csv"
    sid = "sess-123"
    append_log("q1", "a1", log, session_id=sid)
    append_log("q2", "a2", log, session_id=sid)
    lines = log.read_text(encoding="utf-8").strip().splitlines()[1:]
    assert all(f'"{sid}"' in line for line in lines)
