"""Gateway app + LLM adapter contract tests (written first).

create_app(settings, registry, scheduler) -> FastAPI
Routes: GET /healthz -> {"status":"ok"}; GET /v1/models; POST /v1/chat/completions (stream + non-stream passthrough).
Errors are always {"error": {"code": str, "reason": str, ...}}:
  unknown model id        -> 404 code "model_not_found" and "valid_ids": [...]
  status != ready         -> 503 code "model_unavailable", reason = spec.reason
  DoesNotFit              -> 503 code "does_not_fit"
  BackendStartError / backend connect error -> 503 code "backend_error"
  model of wrong kind     -> 400 code "wrong_kind"
The model's in-use refcount MUST be released (scheduler.release) on every path, incl. client disconnect.
"""
import asyncio
import json
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from localai_gateway.app import create_app
from localai_gateway.config import Settings
from localai_gateway.duo import DuoConfig
from localai_gateway.registry import ModelSpec
from localai_gateway.scheduler import DoesNotFit, Endpoint, ModelUnavailable
from localai_gateway.supervisor import BackendStartError


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app):
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return server, port


# ---- fake llama-server backend -------------------------------------------------------------
BACKEND = {"mode": "ok", "last_body": None, "slow_chunks": False}


def make_backend():
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        BACKEND["last_body"] = body
        if BACKEND["mode"] == "error":
            return JSONResponse({"error": "boom"}, status_code=500)
        if body.get("stream"):
            async def gen():
                for i in range(3):
                    yield f'data: {json.dumps({"choices": [{"delta": {"content": f"t{i}"}}]})}\n\n'.encode()
                    if BACKEND["slow_chunks"]:
                        await asyncio.sleep(0.3)
                yield b"data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return {"choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "timings": {"predicted_per_second": 5.8}}

    return app


# ---- fake scheduler --------------------------------------------------------------------------
class FakeScheduler:
    def __init__(self, backend_port, registry):
        self.url, self.registry = f"http://127.0.0.1:{backend_port}", registry
        self.in_use, self.released, self.error, self.loaded_ids = 0, [], None, []

    async def ensure_loaded(self, model_id):
        spec = self.registry[model_id]  # KeyError for unknown ids, like the real scheduler
        if spec.status != "ready":
            raise ModelUnavailable(spec.reason or spec.status)
        if self.error:
            raise self.error
        self.in_use += 1
        self.loaded_ids = [model_id]
        return Endpoint(base_url=self.url)

    async def release(self, model_id):
        self.in_use -= 1
        self.released.append(model_id)

    def loaded(self):
        return list(self.loaded_ids)


def spec(mid, kind="llm", status="ready", reason=None):
    return ModelSpec(id=mid, kind=kind, files=[Path(mid)], runtime="llama", est_mem_mb=100,
                     status=status, reason=reason)


@pytest.fixture(scope="module")
def backend():
    server, port = _serve(make_backend())
    yield port
    server.should_exit = True


@pytest.fixture
def gw(backend):
    BACKEND.update(mode="ok", last_body=None, slow_chunks=False)
    registry = {
        "m": spec("m"),
        "img": spec("img", kind="image"),
        "lance": spec("lance", kind="video", status="unsupported", reason="tensor name 11 is too long"),
        "duo": ModelSpec(id="duo", kind="llm", files=[], runtime="virtual", est_mem_mb=0),
    }
    sched = FakeScheduler(backend, registry)
    settings = Settings(budget_mb=1000, bin_dir=Path("bin"), model_dir=Path("."))
    server, port = _serve(create_app(settings, registry, sched,
                                     duo_config=DuoConfig(planner="m", executor="m", verifier="m", fast="m")))
    yield sched, f"http://127.0.0.1:{port}"
    server.should_exit = True


async def test_healthz(gw):
    _, url = gw
    async with httpx.AsyncClient() as c:
        assert (await c.get(f"{url}/healthz")).json() == {"status": "ok"}


async def test_models_lists_status(gw):
    sched, url = gw
    sched.loaded_ids = ["m"]
    async with httpx.AsyncClient() as c:
        data = (await c.get(f"{url}/v1/models")).json()["data"]
    by_id = {d["id"]: d for d in data}
    assert set(by_id) == {"m", "img", "lance", "duo"} and by_id["duo"]["runtime"] == "virtual"
    assert by_id["m"]["status"] == "ready" and by_id["m"]["loaded"] is True
    assert by_id["lance"]["status"] == "unsupported" and "too long" in by_id["lance"]["reason"]
    assert by_id["lance"]["loaded"] is False


async def test_chat_nonstream(gw):
    sched, url = gw
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{url}/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "hello"
    assert sched.released == ["m"] and sched.in_use == 0
    assert BACKEND["last_body"]["messages"][0]["content"] == "hi"


async def test_chat_stream_sse_chunks(gw):
    sched, url = gw
    chunks = []
    async with httpx.AsyncClient() as c:
        async with c.stream("POST", f"{url}/v1/chat/completions",
                            json={"model": "m", "stream": True, "messages": []}) as r:
            assert r.headers["content-type"].startswith("text/event-stream")
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    chunks.append(line[6:])
    assert chunks[-1] == "[DONE]" and len(chunks) == 4
    assert json.loads(chunks[0])["choices"][0]["delta"]["content"] == "t0"
    await asyncio.sleep(0.2)
    assert sched.in_use == 0


async def test_unknown_model_404_lists_valid_ids(gw):
    _, url = gw
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{url}/v1/chat/completions", json={"model": "nope", "messages": []})
    err = r.json()["error"]
    assert r.status_code == 404 and err["code"] == "model_not_found"
    assert set(err["valid_ids"]) == {"m", "img", "lance", "duo"}


async def test_missing_model_field_is_422_or_400(gw):
    _, url = gw
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{url}/v1/chat/completions", json={"messages": []})
    assert r.status_code in (400, 422)


async def test_unsupported_model_503_with_reason(gw):
    _, url = gw
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{url}/v1/chat/completions", json={"model": "lance", "messages": []})
    err = r.json()["error"]
    assert r.status_code == 503 and err["code"] == "model_unavailable" and "too long" in err["reason"]


async def test_wrong_kind_400(gw):
    sched, url = gw
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{url}/v1/chat/completions", json={"model": "img", "messages": []})
    assert r.status_code == 400 and r.json()["error"]["code"] == "wrong_kind" and sched.in_use == 0


@pytest.mark.parametrize("exc,code", [(DoesNotFit("busy"), "does_not_fit"),
                                      (BackendStartError("died"), "backend_error")])
async def test_scheduler_errors_map_to_503(gw, exc, code):
    sched, url = gw
    sched.error = exc
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{url}/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 503 and r.json()["error"]["code"] == code
    assert sched.in_use == 0


async def test_backend_http_error_returns_503_json_and_releases(gw):
    sched, url = gw
    BACKEND["mode"] = "error"
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{url}/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 503 and r.json()["error"]["code"] == "backend_error"
    assert sched.in_use == 0


async def test_client_disconnect_releases_model(gw):
    sched, url = gw
    BACKEND["slow_chunks"] = True
    async with httpx.AsyncClient() as c:
        async with c.stream("POST", f"{url}/v1/chat/completions",
                            json={"model": "m", "stream": True, "messages": []}) as r:
            async for _ in r.aiter_lines():
                break  # read one line, then drop the connection
    for _ in range(40):
        if sched.in_use == 0:
            break
        await asyncio.sleep(0.1)
    assert sched.in_use == 0 and sched.released == ["m"]


async def test_backend_unreachable_returns_503(gw):
    sched, url = gw
    sched.url = f"http://127.0.0.1:{_free_port()}"  # nothing listens here
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{url}/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 503 and r.json()["error"]["code"] == "backend_error" and sched.in_use == 0


async def test_duo_virtual_model_runs_plan_execute_verify_and_releases_every_role(gw):
    sched, url = gw
    tools = [{"type": "function", "function": {"name": "bash", "description": "x", "parameters": {"type": "object"}}}]
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{url}/v1/chat/completions",
                         json={"model": "duo", "messages": [{"role": "user", "content": "do it"}], "tools": tools})
    assert r.status_code == 200 and "hello" in r.json()["choices"][0]["message"]["content"]
    assert r.json()["model"] == "duo"
    assert sched.released == ["m"] * 4 and sched.in_use == 0                # Jev mode check-up, planner, executor, verifier


async def test_duo_stream_returns_sse_and_done(gw):
    _, url = gw
    lines = []
    async with httpx.AsyncClient(timeout=30) as c:
        async with c.stream("POST", f"{url}/v1/chat/completions", json={"model": "duo", "stream": True,
                            "messages": [{"role": "user", "content": "hi there " * 100}]}) as r:
            assert r.headers["content-type"].startswith("text/event-stream")
            lines = [ln async for ln in r.aiter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]" and any("hello" in ln for ln in lines)
