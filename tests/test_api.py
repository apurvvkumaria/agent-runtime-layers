"""Integration tests for the FastAPI app via TestClient.

The two LLM-backed endpoints (/ask, /chat) are tested with the `stub_llm` fixture
so no real Claude calls are made; the no-LLM endpoints are tested for real.
"""

_METRIC_FIELDS = [
    "requests_per_second",
    "p99_latency_ms",
    "disk_utilization_pct",
    "replication_lag_ms",
    "active_partitions",
]


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_calc_endpoint_addition(client):
    resp = client.get("/calc", params={"expr": "2+2"})
    assert resp.status_code == 200
    # Integer ops stringify as "4"; compare numerically to match the 4.0 intent.
    assert float(resp.json()["result"]) == 4.0


def test_calc_endpoint_multiplication(client):
    resp = client.get("/calc", params={"expr": "150*223.48"})
    assert resp.status_code == 200
    assert float(resp.json()["result"]) == 33522.0


def test_metrics_endpoint_has_fields(client):
    resp = client.get("/metrics/prod-east-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["cluster"] == "prod-east-1"
    for field in _METRIC_FIELDS:
        assert field in body["metrics"]


def test_ask_endpoint_missing_question_returns_422(client):
    # Missing required field -> FastAPI validation error before the handler runs.
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_chat_endpoint_requires_session_id(client, stub_llm):
    # The chat endpoint is session-aware: a provided session_id is echoed back,
    # and a fresh request without one still gets a generated session_id.
    resp = client.post("/chat", json={"message": "hello", "session_id": "s-test"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "s-test"
    assert "answer" in body

    generated = client.post("/chat", json={"message": "hello"})
    assert generated.status_code == 200
    assert generated.json()["session_id"]  # non-empty, auto-generated
