import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "OcchioOnniveggente"))
from scripts.realtime_server import off_topic_message


def test_off_topic_message_with_keywords():
    msg = off_topic_message("museo", ["arte", "storia", "scienza"])
    assert "profilo «museo»" in msg
    assert "arte, storia, scienza" in msg


def test_off_topic_message_without_keywords():
    msg = off_topic_message("museo", [])
    assert "profilo «museo»" in msg
    assert "riformularla" in msg.lower()
