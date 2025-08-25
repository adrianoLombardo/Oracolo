import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "OcchioOnniveggente"))
from OcchioOnniveggente.src.docs_api import app
from OcchioOnniveggente.src.metrics import REQUEST_COUNT


def test_metrics_endpoint_records_requests():
    client = TestClient(app)
    REQUEST_COUNT._metrics.clear()
    client.post("/api/docs/options", json={})
    assert REQUEST_COUNT.labels("POST", "/api/docs/options")._value.get() == 1
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"http_requests_total" in resp.content
