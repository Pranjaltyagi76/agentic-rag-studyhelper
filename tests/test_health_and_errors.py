"""API surface: health, index, and the structured error envelopes (Phase 1 + 6)."""

import app.config as config_mod


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_chat_validation_error_envelope(client):
    # Missing the required `query` field -> 422 in the {error:{code,message,detail}} shape.
    r = client.post("/chat", json={"session_id": "x"})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == 422
    assert body["error"]["message"] == "Invalid request"
    assert any("query" in (d.get("loc") or []) for d in body["error"]["detail"])


def test_unknown_route_404_envelope(client):
    r = client.get("/no-such-route")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == 404


def test_api_key_gate(client, monkeypatch):
    """When APP_API_KEY is set, every route except /health requires the key."""
    monkeypatch.setattr(config_mod.settings, "APP_API_KEY", "s3cret")

    # /health stays open even with the gate on.
    assert client.get("/health").status_code == 200

    # No key -> 401 envelope.
    r = client.post("/chat", json={"session_id": "s", "query": "hi"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == 401

    # Correct key passes the gate; the request then fails validation (422) instead of
    # 401 — proving the gate let it through without us having to invoke the LLM.
    r2 = client.post("/chat", headers={"X-API-Key": "s3cret"}, json={"session_id": "s"})
    assert r2.status_code == 422
