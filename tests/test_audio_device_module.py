import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Stub sounddevice before importing module under test
_devices = [
    {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
]

def _fake_query(idx=None):
    return _devices if idx is None else _devices[idx]

sd_stub = types.SimpleNamespace()
sd_stub.default = types.SimpleNamespace(device=(0, 1))
sd_stub.query_devices = _fake_query
sys.modules["sounddevice"] = sd_stub

from OcchioOnniveggente.src.audio_device import pick_device, debug_print_devices


def test_pick_device():
    assert pick_device(None, "input") is None
    assert pick_device("speaker", "output") == 1


def test_debug_print_devices(capsys):
    debug_print_devices()
    out = capsys.readouterr().out
    assert "Mic" in out
