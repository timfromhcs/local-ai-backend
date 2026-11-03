"""POST /v1/rsi/*: capture verified work as training data, over HTTP."""
import pytest
from fastapi.testclient import TestClient

from localai_gateway.app import create_app
from localai_gateway.config import Settings
from localai_gateway.registry import ModelSpec


class _Scheduler:
    def loaded(self):
        return []

    async def ensure_loaded(self, model_id):
        raise KeyError(model_id)

    async def release(self, model_id):
        return None


@pytest.fixture()
def client(tmp_path):
    settings = Settings(budget_mb=1000, bin_dir=tmp_path, model_dir=tmp_path)
    return TestClient(create_app(settings, {"m": ModelSpec(id="m", kind="llm", files=[], runtime="llama", est_mem_mb=10)}, _Scheduler()))


def _fill(client, session="rsi-test", verified=True):
    assert client.post("/v1/rsi/sessions", json={"session": session}).status_code == 200
    client.post("/v1/rsi/events", json={"session": session, "type": "user", "content": "fix auth.py"})
    client.post("/v1/rsi/events", json={"session": session, "type": "assistant", "content": "running tests"})
    client.post("/v1/rsi/events", json={"session": session, "type": "tool", "name": "bash",
                                        "input": {"cmd": "pytest -q"}, "result": "5 passed", "ok": True})
    client.post("/v1/rsi/events", json={"session": session, "type": "verification",
                                        "ok": verified, "detail": "pytest 5 passed"})
    return session


def test_capture_roundtrip_produces_a_pair(client):
    sid = _fill(client)
    out = client.post("/v1/rsi/finalize", json={"session": sid}).json()
    assert out["pairs"] == 1 and out["verified"] is True
    pair = client.get("/v1/rsi/pairs", params={"session": sid}).json()["pairs"][0]
    assert pair["messages"][0]["content"] == "fix auth.py"
    assert pair["meta"]["verified"] is True


def test_unverified_sessions_yield_nothing(client):
    sid = _fill(client, "unverified", verified=False)
    assert client.post("/v1/rsi/finalize", json={"session": sid}).json()["pairs"] == 0
    assert client.get("/v1/rsi/pairs", params={"session": sid}).json()["pairs"] == []


def test_secrets_never_reach_the_capture_file(client, tmp_path):
    client.post("/v1/rsi/sessions", json={"session": "secret"})
    client.post("/v1/rsi/events", json={"session": "secret", "type": "user",
                                        "content": "token ghp_abcdefghijklmnopqrstuvwxyz0123"})  # secret-scan: allow (fake fixture)
    raw = (tmp_path / "rsi" / "captures" / "secret.jsonl").read_text(encoding="utf8")
    assert "ghp_" not in raw  # secret-scan: allow (fake fixture)


def test_stats_and_error_cases(client):
    _fill(client)
    stats = client.get("/v1/rsi/stats").json()
    assert stats["sessions"] == 1 and stats["pairs"] == 1

    assert client.post("/v1/rsi/events", json={"type": "user", "content": "x"}).status_code == 422
    assert client.post("/v1/rsi/events", json={"session": "s", "type": "bogus"}).json()["error"]["code"] == "capture_error"
    assert client.post("/v1/rsi/finalize", json={"session": "missing"}).json()["error"]["code"] == "capture_error"
    assert client.get("/v1/rsi/pairs", params={"session": "missing"}).json()["error"]["code"] == "capture_error"


def test_events_without_body_is_rejected(client):
    r = client.post("/v1/rsi/events", content=b"not json")
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"
