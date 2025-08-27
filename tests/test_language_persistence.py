import sys
from pathlib import Path

# Ensure repository root is on the path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.conversation import update_language


def test_language_persists_across_turns():
    lang = None
    # First turn detected english
    lang = update_language(lang, "en", "hello oracle")
    assert lang == "en"

    # Second turn user speaks italian but doesn't request change
    lang = update_language(lang, "it", "ciao oracolo")
    assert lang == "en"

    # Third turn user explicitly asks to switch to Italian
    lang = update_language(lang, "it", "parli in italiano?")
    assert lang == "it"
