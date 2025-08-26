import sys
import types
import importlib
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.audio import LocalSpeechToText, LocalTextToSpeech


def _import_local_audio():
    sys.modules.setdefault("sounddevice", types.SimpleNamespace())
    return importlib.import_module("OcchioOnniveggente.src.local_audio")


def test_stt_local_transcribes(monkeypatch, tmp_path):
    la = _import_local_audio()
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"0")

    class DummyModel:
        def transcribe(self, path, language="it", task="transcribe"):
            return ([types.SimpleNamespace(text="ciao")], None)

    monkeypatch.setattr(
        la.container, "load_stt_model", lambda: ("faster_whisper", DummyModel())
    )
    assert la.stt_local(audio_path, lang="it") == "ciao"


def test_tts_local_falls_back(monkeypatch, tmp_path):
    la = _import_local_audio()
    out = tmp_path / "out.wav"
    monkeypatch.setattr(la, "get_tts_cache", lambda text, lang: None)
    monkeypatch.setattr(la, "set_tts_cache", lambda text, lang, path: None)

    fake_gtts = types.SimpleNamespace(
        gTTS=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no gtts"))
    )
    fake_pyttsx3 = types.SimpleNamespace(
        init=lambda: (_ for _ in ()).throw(RuntimeError("no pyttsx3"))
    )
    sys.modules["gtts"] = fake_gtts
    sys.modules["pyttsx3"] = fake_pyttsx3

    monkeypatch.setattr(
        la.container,
        "load_tts_model",
        lambda: ("backend", lambda text, lang: (np.ones(1, dtype=np.float32), 16000)),
    )
    la.tts_local("ciao", out)
    assert out.exists() and out.read_bytes() != b""


def test_local_interfaces_wrap(monkeypatch, tmp_path):
    la = _import_local_audio()
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"0")
    monkeypatch.setattr(la, "stt_local", lambda p, lang="it": "ok")
    stt = LocalSpeechToText()
    assert stt.transcribe(audio_path) == "ok"

    monkeypatch.setattr(la, "tts_local", lambda text, path, lang="it": path.write_bytes(b"d"))
    tts = LocalTextToSpeech()
    data = tts.synthesize("hi")
    assert data == b"d"
