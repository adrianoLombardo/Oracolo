import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.oracle import oracle_answer
from OcchioOnniveggente.src.oracle import core as oracle_core


class CountingClient:
    def __init__(self):
        self.calls = 0

        class Responses:
            def __init__(self, outer):
                self.outer = outer

            def create(self, model, instructions, input):
                self.outer.calls += 1
                return SimpleNamespace(output_text="ok")

        self.responses = Responses(self)


def _patch_cache(monkeypatch):
    store = {}
    monkeypatch.setattr(oracle_core, "cache_get_json", lambda k: store.get(k))
    monkeypatch.setattr(oracle_core, "cache_set_json", lambda k, v: store.__setitem__(k, v))


def test_cache_differs_by_context(monkeypatch):
    _patch_cache(monkeypatch)
    client = CountingClient()
    oracle_answer("q", "it", client, "m", "", context=[{"text": "A"}])
    oracle_answer("q", "it", client, "m", "", context=[{"text": "A"}])
    assert client.calls == 1
    oracle_answer("q", "it", client, "m", "", context=[{"text": "B"}])
    assert client.calls == 2


def test_cache_differs_by_mode(monkeypatch):
    _patch_cache(monkeypatch)
    client = CountingClient()
    oracle_answer("q", "it", client, "m", "", mode="concise")
    oracle_answer("q", "it", client, "m", "", mode="concise")
    assert client.calls == 1
    oracle_answer("q", "it", client, "m", "", mode="detailed")
    assert client.calls == 2
