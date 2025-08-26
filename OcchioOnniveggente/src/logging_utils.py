"""
Utility per la configurazione del logging strutturato.

Il modulo imposta un logging asincrono basato su ``QueueListener`` e
``RotatingFileHandler``. I messaggi vengono serializzati in JSON e
arricchiti con un identificativo di sessione, così da poterli analizzare con
strumenti esterni senza confonderli con l'output destinato all'utente.
"""

import json
import logging
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from pathlib import Path
from queue import Queue
from typing import Optional

try:
    import sentry_sdk
except Exception:  # pragma: no cover - optional dependency
    sentry_sdk = None  # type: ignore[assignment]


class JsonFormatter(logging.Formatter):
    """Formatter che serializza i record in JSON."""


    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        session_id = getattr(record, "session_id", None)
        if session_id is not None:
            log_record["session_id"] = session_id
        return json.dumps(log_record, ensure_ascii=False)


def setup_logging(
    log_path: Path,
    level: int = logging.INFO,
    *,
    console: bool = True,
    session_id: Optional[str] = None,
    sentry_dsn: Optional[str] = None,
) -> QueueListener:
    """Configure logging with a queue and rotating file handler.

    Parameters
    ----------
    log_path: Path
        File path where logs will be written. The directory is created if
        missing.
    level: int
        Minimum logging level for the root logger (default: INFO).

    Returns
    -------
    QueueListener
        The listener object managing the background logging thread. The caller
        is responsible for stopping it with ``listener.stop()`` on shutdown.

    Notes
    -----
    If ``sentry_dsn`` is provided and ``sentry_sdk`` is installed, events of
    level ERROR and above will also be sent to Sentry for centralized
    monitoring.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Shared queue between main thread and logging thread
    queue: Queue = Queue()

    # Root logger sends records to the queue
    queue_handler = QueueHandler(queue)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(queue_handler)

    if sentry_dsn and sentry_sdk is not None:
        sentry_sdk.init(dsn=sentry_dsn)

    # Ensure each record has a session identifier for context
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.session_id = session_id or "-"
        return record

    logging.setLogRecordFactory(record_factory)

    json_fmt = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    file_handler = RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(json_fmt)

    handlers = [file_handler]

    if console:
        console_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s [%(session_id)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_fmt)
        handlers.append(console_handler)

    listener = QueueListener(queue, *handlers)
    listener.start()
    return listener


def get_backend_logger(level: int | None = None) -> logging.Logger:
    """Return the logger dedicated to UI/backend operations.

    Parameters
    ----------
    level: int | None, optional
        If provided, the logger's level is set accordingly.

    Notes
    -----
    The logger propagates to the root logger configured by
    :func:`setup_logging`, so backend messages are still processed by the
    same handlers while remaining isolated from other loggers when
    desired.
    """

    logger = logging.getLogger("backend")
    if level is not None:
        logger.setLevel(level)
    return logger
