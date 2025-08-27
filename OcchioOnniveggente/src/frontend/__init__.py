from __future__ import annotations

"""Reusable front-end components for CLI and web interfaces."""

from typing import Protocol


class Frontend(Protocol):
    """Minimal interface for a front-end component."""

    def run(self) -> None:
        """Start the user interface."""
        ...


class CLIFrontend:
    """Very small wrapper around console helpers."""

    def run(self) -> None:  # pragma: no cover - simple passthrough
        from src.cli import _ensure_utf8_stdout, oracle_greeting

        _ensure_utf8_stdout()
        print(oracle_greeting("it"))


class WebFrontend:
    """Wrapper launching the Flask web application."""

    def __init__(self) -> None:
        from .web.webapp import create_app

        self.app = create_app()

    def run(self) -> None:  # pragma: no cover - starts server
        self.app.run()


__all__ = ["Frontend", "CLIFrontend", "WebFrontend"]
