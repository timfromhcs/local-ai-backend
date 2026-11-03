"""External models with an API key: llama-server --api-key answers 401 without the bearer header.

Regression: the duo virtual model failed with `backend HTTP 401 Invalid API Key` because the
gateway never sent the header the Jev server requires.
"""
import httpx
import pytest

from localai_gateway.adapters.llm import BackendError, auth_headers, forward_chat, open_stream
from localai_gateway.config import Settings
from localai_gateway.registry import ModelSpec
from localai_gateway.scheduler import Scheduler


def test_auth_headers_only_when_a_key_exists():
    assert auth_headers(None) == {}
    assert auth_headers("") == {}
    assert auth_headers("local") == {"authorization": "Bearer local"}


async def test_forward_chat_sends_the_bearer_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await forward_chat("http://x", {"model": "m"}, client, api_key="local")
    assert seen["auth"] == "Bearer local"


async def test_forward_chat_without_key_sends_no_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await forward_chat("http://x", {"model": "m"}, client)
    assert seen["auth"] is None


async def test_stream_sends_the_header_and_surfaces_401():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") != "Bearer local":
            return httpx.Response(401, json={"error": "Invalid API Key"})
        return httpx.Response(200, content=b"data: [DONE]\n\n", headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await open_stream("http://x", {"model": "m"}, client, api_key="local")
        assert resp.status_code == 200
        await resp.aclose()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(BackendError) as exc:
            await open_stream("http://x", {"model": "m"}, client)
        assert "401" in str(exc.value)


async def test_external_endpoint_carries_the_configured_key(tmp_path):
    settings = Settings(budget_mb=1000, bin_dir=tmp_path, model_dir=tmp_path)
    registry = {"jev": ModelSpec(id="jev", kind="llm", files=[], runtime="external",
                                 endpoint="http://127.0.0.1:8080/", api_key="local", est_mem_mb=0)}
    sched = Scheduler(settings, registry, supervisor=object())
    ep = await sched.ensure_loaded("jev")
    assert ep.base_url == "http://127.0.0.1:8080"
    assert ep.api_key == "local"


def test_models_yaml_ships_the_key_for_jev():
    from pathlib import Path

    from localai_gateway.config import config_file
    from localai_gateway.registry import load_registry

    root = Path(__file__).resolve().parents[1]
    registry = load_registry(config_file("models.yaml"), root)
    jev = registry.get("jev-1.5b")
    if jev is None:  # pragma: no cover - models.yaml is part of the repo
        pytest.skip("models.yaml without jev")
    assert jev.api_key, "the Jev server is started with --api-key, the registry must know it"
