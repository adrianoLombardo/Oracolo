import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.cli import say, oracle_greeting, _ensure_utf8_stdout


def test_say_prints(capsys):
    say("hello")
    captured = capsys.readouterr()
    data = json.loads(captured.out.strip())
    assert data == {"type": "chat", "role": "assistant", "text": "hello"}


def test_oracle_greeting_languages():
    assert oracle_greeting("en") == "Hello, I am the Oracle. Ask your question."
    assert oracle_greeting("it").startswith("Ciao")


def test_ensure_utf8_stdout_does_not_fail():
    _ensure_utf8_stdout()
