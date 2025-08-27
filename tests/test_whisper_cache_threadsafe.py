import sys
import types
import time
import threading
import importlib
from pathlib import Path


def test_get_whisper_thread_safety(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    dummy_sc = types.ModuleType("service_container")
    dummy_sc.container = types.SimpleNamespace()
    monkeypatch.setitem(
        sys.modules, "OcchioOnniveggente.src.service_container", dummy_sc
    )
    monkeypatch.setitem(sys.modules, "sounddevice", types.ModuleType("sounddevice"))
    monkeypatch.setitem(sys.modules, "soundfile", types.ModuleType("soundfile"))


    local_audio = importlib.import_module("OcchioOnniveggente.src.hardware.local_audio")

    local_audio = importlib.import_module("OcchioOnniveggente.src.audio.local_audio")

    local_audio._WHISPER_CACHE.clear()

    created = []

    class DummyModel:
        def __init__(self, model_name, device, compute_type):
            created.append((model_name, device, compute_type))
            time.sleep(0.1)

    dummy_module = types.ModuleType("faster_whisper")
    dummy_module.WhisperModel = DummyModel
    monkeypatch.setitem(sys.modules, "faster_whisper", dummy_module)

    results = []

    def call():
        results.append(local_audio._get_whisper("base", "cpu", "int8"))

    threads = [threading.Thread(target=call) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(created) == 1
    assert len({id(r) for r in results}) == 1

