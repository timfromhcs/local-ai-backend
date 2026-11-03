"""Generated client contract: it must match the API and actually work against the app."""
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from localai_gateway.app import create_app
from localai_gateway.client import GatewayClient, GatewayError
from localai_gateway.config import Settings
from localai_gateway.registry import ModelSpec

ROOT = Path(__file__).resolve().parents[1]


class _Scheduler:
    def loaded(self):
        return []

    async def ensure_loaded(self, model_id):
        return type("EP", (), {"base_url": "http://fake"})()

    async def release(self, model_id):
        return None


def _transport(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/healthz":
        return httpx.Response(200, json={"status": "ok"})
    if path == "/v1/models":
        return httpx.Response(200, json={"object": "list", "data": [{"id": "m", "status": "ready"}]})
    if path == "/v1/placement":
        assert request.url.params.get("kinds") == "llm"
        return httpx.Response(200, json={"models": ["m"], "memory_mb": 10, "budget_mb": 100})
    if path == "/v1/autotune":
        assert request.url.params.get("model") == "m"
        return httpx.Response(200, json={"args": ["-ngl 99"], "why": ["test"]})
    if path == "/v1/rsi/stats":
        return httpx.Response(200, json={"sessions": 0, "pairs": 0})
    if path == "/v1/chat/completions":
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]})
    return httpx.Response(404, json={"error": {"code": "model_not_found", "reason": "nope"}})


@pytest.fixture()
def client():
    http = httpx.Client(transport=httpx.MockTransport(_transport))
    return GatewayClient("http://gateway.test", client=http)


def test_client_exposes_every_generated_route():
    names = {n for n in dir(GatewayClient) if not n.startswith("_")}
    expected = {"healthz", "doctor", "models", "chat", "bench", "placement", "autotune",
                "rsi_start", "rsi_event", "rsi_finalize", "rsi_pairs", "rsi_stats"}
    assert expected <= names


def test_get_calls_pass_query_parameters(client):
    assert client.healthz()["status"] == "ok"
    assert client.models()["data"][0]["id"] == "m"
    assert client.placement(budget_mb=100)["models"] == ["m"]
    assert client.autotune("m")["args"] == ["-ngl 99"]
    assert client.rsi_stats()["sessions"] == 0


def test_post_calls_send_the_body(client):
    reply = client.chat({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert reply["choices"][0]["message"]["content"] == "hi"


def test_errors_become_gateway_error(client):
    with pytest.raises(GatewayError) as exc:
        client._request("GET", "/v1/nope")
    assert exc.value.status == 404
    assert exc.value.code == "model_not_found"
    assert "nope" in exc.value.reason


def test_generated_file_is_up_to_date():
    """CI gate: the client must be regenerated when the API snapshot changes."""
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "gen-client.py"), "--check"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_client_roundtrips_against_the_real_app(tmp_path):
    """No mock: the generated client talks to the real ASGI app via FastAPI's TestClient."""
    from fastapi.testclient import TestClient

    settings = Settings(budget_mb=1000, bin_dir=tmp_path, model_dir=tmp_path)
    registry = {"m": ModelSpec(id="m", kind="llm", files=[], runtime="llama", est_mem_mb=10)}
    app = create_app(settings, registry, _Scheduler())
    with GatewayClient("http://testserver", client=TestClient(app)) as gw:
        assert gw.healthz()["status"] == "ok"
        assert gw.doctor()["version"]
        assert gw.models()["object"] == "list"
        assert gw.placement()["models"] == ["m"]
        assert gw.autotune("m")["class"] in {"cpu", "igpu_uma", "dgpu"}
