import logging
import sys
from pathlib import Path

import sentry_sdk

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "OcchioOnniveggente"))
from OcchioOnniveggente.src.logging_utils import setup_logging


def test_setup_logging_with_sentry(tmp_path: Path):
    prev_factory = logging.getLogRecordFactory()
    log_path = tmp_path / "log.json"
    listener = setup_logging(
        log_path, session_id="sess-1", sentry_dsn="https://abc@example.com/1"
    )
    try:
        assert sentry_sdk.Hub.current.client is not None
        dsn = sentry_sdk.Hub.current.client.dsn  # type: ignore[attr-defined]
        assert dsn is not None and "example.com" in str(dsn)
    finally:
        listener.stop()
        logging.setLogRecordFactory(prev_factory)
