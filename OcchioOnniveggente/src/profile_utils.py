"""Helpers for profile selection and domain configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple


def get_active_profile(SETTINGS: Any, forced_name: str | None = None) -> Tuple[str, Dict]:
    """Return the active profile name and configuration.

    ``SETTINGS`` can be a dictionary or an object with ``domain`` attribute.
    """
    if isinstance(SETTINGS, dict):
        dom = SETTINGS.get("domain", {}) or {}
        prof_name = forced_name or dom.get("profile", "museo")
        profiles = dom.get("profiles", {}) or {}
        prof = profiles.get(prof_name, {})
    else:
        dom = getattr(SETTINGS, "domain", None)
        prof_name = forced_name or (getattr(dom, "profile", "museo") if dom else "museo")
        profiles = getattr(dom, "profiles", {}) if dom else {}
        prof = profiles.get(prof_name, {})
    return prof_name, prof


def make_domain_settings(base_settings: Any, prof_name: str, prof: Dict) -> Any:
    """Return ``base_settings`` with domain info replaced by ``prof``.

    ``base_settings`` may be a ``dict`` or a ``Settings`` object. The returned
    value is of the same type.
    """
    if isinstance(base_settings, dict):
        new_s = dict(base_settings)
        dom = dict(new_s.get("domain", {}))
        dom.update({
            "enabled": True,
            "profile": prof_name,
            "keywords": prof.get("keywords", []),
        })
        if prof.get("weights"):
            dom["weights"] = prof["weights"]
        if prof.get("accept_threshold") is not None:
            dom["accept_threshold"] = prof["accept_threshold"]
        if prof.get("clarify_margin") is not None:
            dom["clarify_margin"] = prof["clarify_margin"]
        new_s["domain"] = dom
        return new_s
    else:
        try:
            base_settings.domain.enabled = True
            base_settings.domain.profile = prof_name
            base_settings.domain.keywords = prof.get("keywords", [])
            if prof.get("weights"):
                base_settings.domain.weights = prof["weights"]
            if prof.get("accept_threshold") is not None:
                base_settings.domain.accept_threshold = prof["accept_threshold"]
            if prof.get("clarify_margin") is not None:
                base_settings.domain.clarify_margin = prof["clarify_margin"]
        except Exception:
            pass
        return base_settings


def save_profile(profile: Dict, path: str | Path = "data/profile.json") -> Path:
    """Persist ``profile`` to ``path`` as JSON.

    Parameters
    ----------
    profile:
        Mapping with the profile information to store.
    path:
        Destination file.  Missing directories are created automatically.

    Returns
    -------
    Path
        The location where the profile has been written.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
