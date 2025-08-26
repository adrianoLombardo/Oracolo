import os

_THEMES = {
    "dark": {
        "bg": "#0f0f0f",
        "fg": "#00ffe1",
        "mid": "#161616",
        "button_bg": "#1e1e1e",
    },
    "text": {
        "bg": "#000000",
        "fg": "#ffffff",
        "mid": "#000000",
        "button_bg": "#000000",
        "text_only": True,
    },
}


def get_theme(name: str | None = None) -> dict:
    """Return theme dictionary by ``name`` or ``ORACOLO_THEME`` env variable."""
    name = name or os.getenv("ORACOLO_THEME", "dark")
    return _THEMES.get(name, _THEMES["dark"])
