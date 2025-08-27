import pytest

from OcchioOnniveggente.src.hardware import audio


def test_record_wav_requires_sounddevice(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "sd", None)
    with pytest.raises(RuntimeError):
        audio.record_wav(tmp_path / "out.wav", seconds=1, sr=16000)
