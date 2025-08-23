import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.oracle import oracle_answer


class DummyClient:
    def __init__(self):
        self.last_input = None
        class Responses:
            def __init__(self, outer):
                self.outer = outer
            def create(self, model, instructions, input):
                self.outer.last_input = input
                return SimpleNamespace(output_text="ok")
        self.responses = Responses(self)


def test_oracle_answer_includes_sources():
    c = DummyClient()
    ctx = [{"text": "alpha", "id": "a1"}]
    oracle_answer("q", "it", c, "gpt", "", context=ctx)
    sys_msgs = [m for m in c.last_input if m["role"] == "system"]
    src_msg = [m for m in sys_msgs if m["content"].startswith("Fonti:")][0]
    assert "[1] alpha" in src_msg["content"]
