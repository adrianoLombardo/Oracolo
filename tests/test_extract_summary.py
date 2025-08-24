import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from OcchioOnniveggente.src.oracle import extract_summary


def test_extract_summary_structured():
    text = "1) Sintesi: Questo è il riassunto.\n2) Dettagli: punto A; punto B.\n3) Fonti: [1]"
    assert extract_summary(text) == "Questo è il riassunto."


def test_extract_summary_no_structure():
    text = "Risposta semplice senza struttura."
    assert extract_summary(text) == "Risposta semplice senza struttura."
