import logging
from pathlib import Path

from OcchioOnniveggente.src.oracle import transcribe


class NetworkClient:
    def transcribe(self, *args, **kwargs):
        raise ConnectionError("network down")


def test_transcribe_network_error(caplog):
    with caplog.at_level(logging.WARNING):
        msg = transcribe(Path("a.wav"), NetworkClient(), "model")
    assert "controlla la connessione" in msg.lower()
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert "context: transcribe" in caplog.text


class APIClient:
    def transcribe(self, *args, **kwargs):
        raise ValueError("bad request")


def test_transcribe_api_error(caplog):
    with caplog.at_level(logging.ERROR):
        msg = transcribe(Path("a.wav"), APIClient(), "model")
    assert "errore dell'api" in msg.lower()
    assert any(r.levelno == logging.ERROR for r in caplog.records)
    assert "context: transcribe" in caplog.text


class AudioClient:
    def transcribe(self, *args, **kwargs):
        raise OSError("audio failure")


def test_transcribe_audio_error(caplog):
    with caplog.at_level(logging.ERROR):
        msg = transcribe(Path("a.wav"), AudioClient(), "model")
    assert "errore audio" in msg.lower()
    assert any(r.levelno == logging.ERROR for r in caplog.records)
    assert "context: transcribe" in caplog.text
