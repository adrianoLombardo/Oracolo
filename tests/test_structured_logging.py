import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.logging_utils import setup_logging


def test_setup_logging_produces_json(tmp_path: Path):
    prev_factory = logging.getLogRecordFactory()
    log_path = tmp_path / "log.json"
    listener = setup_logging(log_path, session_id="sess-1")
    try:
        logging.getLogger("test").info("hello")
    finally:
        listener.stop()
        logging.setLogRecordFactory(prev_factory)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["message"] == "hello"
    assert data["session_id"] == "sess-1"
    assert data["level"] == "INFO"
