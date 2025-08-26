import sys
from pathlib import Path

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.config import Settings


def test_model_validate_yaml_invalid_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "settings.yaml"
    bad.write_text("debug: [::]", encoding="utf-8")
    with pytest.raises(ValueError):
        Settings.model_validate_yaml(bad)


def test_model_validate_yaml_valid_yaml(tmp_path: Path) -> None:
    good = tmp_path / "settings.yaml"
    good.write_text("debug: true", encoding="utf-8")
    settings = Settings.model_validate_yaml(good)
    assert settings.debug is True


def test_round_trip_defaults(tmp_path: Path) -> None:
    original = Settings()
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(original.model_dump()), encoding="utf-8")
    loaded = Settings.model_validate_yaml(path)
    assert loaded == original
    assert loaded.domain.profile == "museo"
    assert loaded.domain.topic == ""


def test_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "settings.yaml"
    base.write_text("debug: false", encoding="utf-8")
    override = tmp_path / "settings.test.yaml"
    override.write_text("debug: true", encoding="utf-8")
    monkeypatch.setenv("ORACOLO_ENV", "test")
    settings = Settings.model_validate_yaml(base)
    assert settings.debug is True


def test_device_concurrency_default() -> None:
    settings = Settings()
    assert settings.compute.device_concurrency == 1
