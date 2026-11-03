"""CORS contract: the deployed demo and local dev servers may call the gateway from a browser.

The gateway must answer the preflight with the concrete origin (never "*"), allow
POST + content-type, and allow Vercel preview deployments.
"""
import os
import socket
import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient

from localai_gateway.app import DEFAULT_CORS_ORIGINS, create_app, cors_origins
from localai_gateway.config import Settings
from localai_gateway.registry import ModelSpec


class _Scheduler:
    def loaded(self):
        return []

    async def ensure_loaded(self, model_id):  # pragma: no cover - not reached
        raise KeyError(model_id)

    async def release(self, model_id):
        return None


@pytest.fixture()
def client(tmp_path):
    settings = Settings(budget_mb=1024, bin_dir=tmp_path, model_dir=tmp_path)
    registry = {"m": ModelSpec(id="m", kind="llm", files=[], runtime="llama", est_mem_mb=1)}
    app = create_app(settings, registry, _Scheduler())
    return TestClient(app)


def test_healthz_is_reachable(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_preflight_from_localhost_dev_origin(client):
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "POST" in r.headers["access-control-allow-methods"]


def test_preflight_from_vercel_preview(client):
    origin = "https://local-ai-backend-git-main-timfromhcs.vercel.app"
    r = client.get("/v1/models", headers={"Origin": origin})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == origin


def test_unknown_origin_gets_no_cors_header(client):
    r = client.get("/v1/models", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_origins_come_from_env(monkeypatch):
    monkeypatch.setenv("LOCALAI_CORS_ORIGINS", "https://a.example, https://b.example/")
    assert cors_origins() == ["https://a.example", "https://b.example"]


def test_default_origins_never_wildcard(monkeypatch):
    monkeypatch.delenv("LOCALAI_CORS_ORIGINS", raising=False)
    origins = cors_origins()
    assert origins == list(DEFAULT_CORS_ORIGINS)
    assert "*" not in origins


def test_cors_headers_on_real_server():
    """TestClient bypasses some middleware paths; prove it on a real socket too."""
    settings = Settings(budget_mb=1024, bin_dir=".", model_dir=".")
    app = create_app(settings, {}, _Scheduler())

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.05).close()
            break
        except OSError:
            time.sleep(0.05)

    import httpx

    r = httpx.get(f"http://127.0.0.1:{port}/v1/models", headers={"Origin": "http://127.0.0.1:3000"}, timeout=5)
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://127.0.0.1:3000"
    server.should_exit = True
