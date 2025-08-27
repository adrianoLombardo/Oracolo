import asyncio
import time
import numpy as np
import soundfile as sf
import pytest
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "OcchioOnniveggente"))
sys.modules.setdefault("sounddevice", types.SimpleNamespace(play=lambda *a, **k: None, wait=lambda: None))

from OcchioOnniveggente.src.hardware import local_audio
=======from OcchioOnniveggente.src.audio import local_audio



def test_async_tts_speak_non_blocking(monkeypatch):
    def fake_tts_local(text, out_path, lang="it"):
        data = np.zeros(1000, dtype=np.float32)
        sf.write(out_path, data, 16000)

    called = False

    def fake_play(data, sr):
        nonlocal called
        called = True
        time.sleep(0.5)

    monkeypatch.setattr(local_audio, "tts_local", fake_tts_local)
    monkeypatch.setattr(local_audio, "_play", fake_play)

    async def runner():
        start = time.perf_counter()
        task = asyncio.create_task(local_audio.async_tts_speak("ciao"))
        await asyncio.sleep(0.1)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5
        assert not task.done()
        await task
        assert called

    asyncio.run(runner())
