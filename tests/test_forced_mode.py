"""Forced mode: a caller can demand the plan/execute/verify loop instead of the fast lane.

Without this, a short tool task is routed to the fast model by design and `duo` never plans - so
the headline behaviour could not be measured at all.
"""
import pytest

from localai_gateway.app import create_app
from localai_gateway.config import Settings
from localai_gateway.duo import Duo, DuoConfig
from localai_gateway.modes import Mode
from localai_gateway.registry import ModelSpec


def completion(content="ok", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"id": "x", "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop", "message": msg}]}


class Recorder:
    def __init__(self):
        self.calls = []

    async def __call__(self, model, body):
        self.calls.append((model, body))
        if "planner" in (body.get("messages") or [{}])[0].get("content", "").lower():
            return completion("1. list files\n2. read package.json")
        return completion("", [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}])


class AlwaysFast:
    async def decide(self, key, body):
        return Mode.FAST


TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]


@pytest.mark.asyncio
async def test_forced_duo_runs_the_planner_even_when_the_controller_says_fast():
    call = Recorder()
    duo = Duo(call, DuoConfig(), modes=AlwaysFast())
    body = {"messages": [{"role": "user", "content": "read the file"}], "tools": TOOLS, "localai_mode": "duo"}
    await duo.chat(body)
    # first call is the planner (big model), so the loop really ran
    assert call.calls[0][0] == DuoConfig().planner


@pytest.mark.asyncio
async def test_without_the_flag_the_controller_decides():
    call = Recorder()
    duo = Duo(call, DuoConfig(), modes=AlwaysFast())
    body = {"messages": [{"role": "user", "content": "read the file"}], "tools": TOOLS}
    await duo.chat(body)
    assert call.calls[0][0] == DuoConfig().executor


@pytest.mark.asyncio
async def test_invalid_mode_raises():
    duo = Duo(Recorder(), DuoConfig())
    with pytest.raises(ValueError):
        await duo.chat({"messages": [{"role": "user", "content": "x"}], "tools": TOOLS, "localai_mode": "turbo"})


@pytest.mark.asyncio
async def test_mode_is_stripped_before_forwarding():
    call = Recorder()
    duo = Duo(call, DuoConfig(), modes=AlwaysFast())
    body = {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS, "localai_mode": "fast"}
    await duo.chat(body)
    assert all("localai_mode" not in sent for _, sent in call.calls), "backend must never see the extension field"


def _client(monkeypatch):
    from fastapi.testclient import TestClient
    import httpx

    def handler(request):
        return httpx.Response(200, json=completion("x"))

    settings = Settings(budget_mb=1000, bin_dir=".", model_dir=".")
    registry = {
        "duo": ModelSpec(id="duo", kind="llm", files=[], runtime="virtual", est_mem_mb=0),
        "llama": ModelSpec(id="llama", kind="llm", files=[], runtime="llama", est_mem_mb=10),
    }
    app = create_app(settings, registry, _DummyScheduler(), http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return TestClient(app)


class _DummyScheduler:
    def loaded(self):
        return []

    async def ensure_loaded(self, model_id):
        return type("EP", (), {"base_url": "http://fake", "api_key": None})()

    async def release(self, model_id):
        return None


def test_bad_mode_is_a_structured_400():
    c = _client(None)
    r = c.post("/v1/chat/completions", json={"model": "duo", "messages": [{"role": "user", "content": "x"}],
                                            "tools": TOOLS, "localai_mode": "turbo"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_mode"


def test_bench_accepts_a_forced_mode_and_rejects_a_bad_one():
    c = _client(None)
    ok = c.post("/v1/bench", json={"model": "duo", "tasks": ["json"], "mode": "fast"})
    assert ok.status_code in (200, 502), ok.text  # mocked backend may score 0, but the mode is accepted
    bad = c.post("/v1/bench", json={"model": "duo", "tasks": ["json"], "mode": "turbo"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "bad_mode"
