import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "OcchioOnniveggente"))

try:
    from src import validators
except OSError:
    pytest.skip("PortAudio library not available", allow_module_level=True)


def test_validate_device_config_invalid_index(monkeypatch):
    monkeypatch.setattr(validators.sd, "query_devices", lambda: [{"name": "mic"}])
    with pytest.raises(ValueError):
        validators.validate_device_config({"input_device": 5, "output_device": 0})


def test_validate_device_config_valid(monkeypatch):
    devices = [{"name": "mic"}, {"name": "spk"}]
    monkeypatch.setattr(validators.sd, "query_devices", lambda: devices)
    validators.validate_device_config({"input_device": "mic", "output_device": "spk"})
