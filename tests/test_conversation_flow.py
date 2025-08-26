import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.oracle import ConversationFlow


def test_default_flow_sequence():
    cf = ConversationFlow()
    phases = [cf.state]
    while not cf.is_finished():
        phases.append(cf.advance())
    assert phases == ConversationFlow.DEFAULT_FLOW


def test_custom_context_flow():
    flows = {
        "mostra": [
            "introduzione",
            "presentazione_opera",
            "domanda_visitatore",
            "follow_up",
            "chiusura",
        ]
    }
    cf = ConversationFlow(context="mostra", flows=flows)
    observed = [cf.state]
    while not cf.is_finished():
        observed.append(cf.advance())
    assert observed == flows["mostra"]
