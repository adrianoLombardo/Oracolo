import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock

# ``main`` imports ``sounddevice`` which requires PortAudio; mock it to avoid ImportError.
sys.modules.setdefault("sounddevice", MagicMock())

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "OcchioOnniveggente"))
from OcchioOnniveggente.src.profile_utils import get_active_profile, make_domain_settings


def test_get_active_profile_selects_profile():
    settings = {
        "domain": {
            "profile": "museo",
            "profiles": {
                "museo": {"topic": "museo", "keywords": ["arte"]},
                "science": {"topic": "science", "keywords": ["fisica"]},
            },
        }
    }
    name, prof = get_active_profile(settings)
    assert name == "museo" and prof.get("topic") == "museo"
    name, prof = get_active_profile(settings, forced_name="science")
    assert name == "science" and prof.get("topic") == "science"


def test_make_domain_settings_generates_domain_dict():
    base = {}
    prof_name = "science"
    prof = {"topic": "physics", "keywords": ["fisica", "chimica"], "weights": {"a": 1}}
    new_set = make_domain_settings(base, prof_name, prof)
    domain = new_set["domain"]
    assert domain["enabled"] is True
    assert domain["profile"] == prof_name
    assert domain["topic"] == "physics"
    assert domain["keywords"] == ["fisica", "chimica"]
    assert domain["weights"] == {"a": 1}


def test_make_domain_settings_accepts_namespace():
    base = SimpleNamespace(domain=SimpleNamespace())
    prof = {"topic": "history", "keywords": ["rome"]}
    make_domain_settings(base, "hist-profile", prof)
    dom = base.domain
    assert dom.enabled is True
    assert dom.profile == "hist-profile"
    assert dom.topic == "history"
    assert dom.keywords == ["rome"]
