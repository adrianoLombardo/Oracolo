import logging
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from pathlib import Path
from queue import Queue


def setup_logging(
    log_path: Path, level: int = logging.INFO, *, console: bool = True
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
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Shared queue between main thread and logging thread
    queue: Queue = Queue()

    # Root logger sends records to the queue
    queue_handler = QueueHandler(queue)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(queue_handler)

    # Rotating file handler (and optional console handler) run in listener thread
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    handlers = [file_handler]
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        handlers.append(console_handler)

    listener = QueueListener(queue, *handlers)
    listener.start()
    return listener
