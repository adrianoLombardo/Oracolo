import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.profile_utils import get_active_profile, make_domain_settings
from OcchioOnniveggente.src.config import Settings


def test_get_active_profile_dict():
    settings = {
        "domain": {
            "profile": "them",
            "profiles": {"museo": {"topic": "museo"}, "them": {"topic": "them"}},
        }
    }
    name, prof = get_active_profile(settings)
    assert name == "them"
    assert prof.get("topic") == "them"


def test_make_domain_settings_dict():
    base = {}
    prof = {"keywords": ["art"], "weights": {"a": 1.0}, "accept_threshold": 0.5}
    result = make_domain_settings(base, "museo", prof)
    assert result["domain"]["profile"] == "museo"
    assert result["domain"]["keywords"] == ["art"]


def test_get_active_profile_settings_object():
    settings = Settings(domain={"profile": "museo", "profiles": {"museo": {"keywords": []}}})
    name, prof = get_active_profile(settings)
    assert name == "museo"
    assert getattr(prof, "keywords", []) == []
