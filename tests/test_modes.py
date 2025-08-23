import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.oracle import oracle_answer


class DummyClient:
    def __init__(self):
        self.last_instructions = ""
        class Responses:
            def __init__(self, outer):
                self.outer = outer
            def create(self, model, instructions, input):
                self.outer.last_instructions = instructions
                return SimpleNamespace(output_text="ok")
        self.responses = Responses(self)


def test_oracle_answer_mode_clauses():
    c = DummyClient()
    oracle_answer("q", "it", c, "gpt", "", policy_prompt="", mode="concise")
    assert "Stile conciso" in c.last_instructions
    oracle_answer("q", "it", c, "gpt", "", policy_prompt="", mode="detailed")
    assert "Struttura: 1)" in c.last_instructions
