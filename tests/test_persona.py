from OcchioOnniveggente.src.config import Settings

def test_persona_defaults():
    s = Settings()
    assert s.persona.current == "standard"
    assert "standard" in s.persona.profiles
