"""Exposure contract: fail closed on public binds, reject oversized bodies."""
import pytest
from fastapi.testclient import TestClient

from localai_gateway.app import create_app
from localai_gateway.config import Settings
from localai_gateway.exposure import PublicBindRefused, max_body_bytes, resolve_host
from localai_gateway.registry import ModelSpec


class _Scheduler:
    def loaded(self):
        return []

    async def ensure_loaded(self, model_id):
        raise KeyError(model_id)

    async def release(self, model_id):
        return None


def _client(tmp_path):
    settings = Settings(budget_mb=1000, bin_dir=tmp_path, model_dir=tmp_path)
    registry = {"m": ModelSpec(id="m", kind="llm", files=[], runtime="llama", est_mem_mb=10)}
    return TestClient(create_app(settings, registry, _Scheduler()))


# ------------------------------------------------------------------ bind guard
def test_loopback_hosts_are_allowed():
    for host in ("127.0.0.1", "localhost", "::1"):
        assert resolve_host(host, {}) == host


def test_public_bind_is_refused_without_optin():
    with pytest.raises(PublicBindRefused) as exc:
        resolve_host("0.0.0.0", {})
    assert "no authentication" in str(exc.value)
    assert "LOCALAI_ALLOW_PUBLIC=1" in str(exc.value)


def test_public_bind_allowed_with_explicit_optin():
    assert resolve_host("0.0.0.0", {"LOCALAI_ALLOW_PUBLIC": "1"}) == "0.0.0.0"


# ------------------------------------------------------------------ body limit
def test_default_body_limit_is_8mb():
    assert max_body_bytes({}) == 8 * 1024 * 1024


def test_body_limit_is_configurable_and_sane():
    assert max_body_bytes({"LOCALAI_MAX_BODY_MB": "2"}) == 2 * 1024 * 1024
    assert max_body_bytes({"LOCALAI_MAX_BODY_MB": "0"}) == 1024 * 1024, "never allow zero"
    assert max_body_bytes({"LOCALAI_MAX_BODY_MB": "nonsense"}) == 8 * 1024 * 1024


def test_oversized_request_is_rejected_with_413(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAI_MAX_BODY_MB", "1")
    client = _client(tmp_path)
    big = b"{" + b" " * (1024 * 1024 + 10) + b"}"
    r = client.post("/v1/chat/completions", content=big, headers={"content-type": "application/json"})
    assert r.status_code == 413
    body = r.json()["error"]
    assert body["code"] == "payload_too_large"
    assert body["limit_bytes"] == 1024 * 1024


def test_normal_request_still_works(tmp_path):
    client = _client(tmp_path)
    r = client.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert r.status_code == 404
    assert r.headers["x-localai-version"]
