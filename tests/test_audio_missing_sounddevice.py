import pytest

from OcchioOnniveggente.src.audio import recording


def test_record_wav_requires_sounddevice(tmp_path, monkeypatch):
    monkeypatch.setattr(recording, "sd", None)
    with pytest.raises(RuntimeError):
        recording.record_wav(tmp_path / "out.wav", seconds=1, sr=16000)
