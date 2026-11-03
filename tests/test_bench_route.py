"""POST /v1/bench: the same deterministic suite as the CLI, served by the gateway."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from localai_gateway.app import create_app
from localai_gateway.config import Settings
from localai_gateway.registry import ModelSpec

ANSWERS = {
    "planning, implementation and verification": "duo",
    "Migrate three services": "bigout",
    "List the files in": "list_files('C:/work')\nread_file('package.json')",
    "Return only JSON": '{"model":"duo","loaded":true,"ram_mb":9000}',
    "A model decodes 5.9 tokens": "3540",
    "Can a browser run a 27B model": "No. It cannot hold the weights.",
    "one sentence what a KV cache": "It stores attention keys and values so they are not recomputed.",
}


def _transport(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    prompt = payload["messages"][-1]["content"]
    answer = next((v for k, v in ANSWERS.items() if k in prompt), "unknown")
    # the bench must disable thinking by default, otherwise a reasoning model returns empty content
    assert payload.get("chat_template_kwargs", {}).get("enable_thinking") is False, payload
    return httpx.Response(200, json={
        "id": "x", "model": payload["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
    })


class _Scheduler:
    def loaded(self):
        return []

    async def ensure_loaded(self, model_id):
        return type("EP", (), {"base_url": "http://fake", "api_key": None})()

    async def release(self, model_id):
        return None


def _client(registry, transport=_transport):
    settings = Settings(budget_mb=5000, bin_dir=".", model_dir=".")
    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    return TestClient(create_app(settings, registry, _Scheduler(), http_client=client))


@pytest.fixture()
def registry():
    return {
        "good": ModelSpec(id="good", kind="llm", files=[], runtime="llama", est_mem_mb=1000),
        "gone": ModelSpec(id="gone", kind="llm", files=[], runtime="llama", est_mem_mb=1000,
                          status="unavailable", reason="missing file"),
        "img": ModelSpec(id="img", kind="image", files=[], runtime="sd", est_mem_mb=1000),
    }


def test_bench_scores_every_task(registry):
    body = _client(registry).post("/v1/bench", json={"model": "good"}).json()
    assert body["model"] == "good"
    assert body["suite"] == "fast"
    assert len(body["rows"]) == 7
    assert body["score"] == 1.0, body
    assert set(body["by_category"]) == {"routing", "tools", "format", "reasoning", "honesty"}


def test_bench_rejects_unknown_suite_model_and_kind(registry):
    c = _client(registry)
    assert c.post("/v1/bench", json={"model": "good", "suite": "nope"}).json()["error"]["code"] == "unknown_suite"
    assert c.post("/v1/bench", json={"model": "x"}).json()["error"]["code"] == "model_not_found"
    r = c.post("/v1/bench", json={"model": "gone"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"
    r = c.post("/v1/bench", json={"model": "img"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "wrong_kind"


def test_bench_without_model_field(registry):
    r = _client(registry).post("/v1/bench", json={})
    assert r.status_code == 422 and r.json()["error"]["code"] == "missing_model"


def test_bench_reports_total_failure(registry):
    def broken(request):
        return httpx.Response(500, text="backend exploded")

    r = _client(registry, broken).post("/v1/bench", json={"model": "good"})
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "bench_failed"


def test_bench_can_opt_into_thinking(registry):
    """`thinking: true` must pass through for models where the caller wants a trace."""
    seen = {}

    def spy(request):
        payload = json.loads(request.content)
        seen["kwargs"] = payload.get("chat_template_kwargs")
        return _transport(httpx.Request("POST", request.url, content=json.dumps({**payload, "chat_template_kwargs": {"enable_thinking": False}})))

    body = _client(registry, spy).post("/v1/bench", json={"model": "good", "thinking": True}).json()
    assert seen["kwargs"] is None, "thinking=true must not force enable_thinking=false"
    assert body["score"] >= 0


def test_bench_supports_virtual_models(registry):
    """duo is orchestrated, not loaded - the bench must route through duo.chat."""
    registry = dict(registry)
    registry["duo"] = ModelSpec(id="duo", kind="llm", files=[], runtime="virtual", est_mem_mb=0)
    body = _client(registry).post("/v1/bench", json={"model": "duo", "tasks": ["json"]}).json()
    assert body["model"] == "duo"
    assert len(body["rows"]) == 1 and body["rows"][0]["id"] == "json"


def test_bench_task_filter_rejects_unknown_ids(registry):
    r = _client(registry).post("/v1/bench", json={"model": "good", "tasks": ["nope"]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unknown_task"


def test_bench_task_filter_runs_only_the_requested_tasks(registry):
    body = _client(registry).post("/v1/bench", json={"model": "good", "tasks": ["json", "math"]}).json()
    assert [row["id"] for row in body["rows"]] == ["json", "math"]


def test_bench_partial_failure_still_scores(registry):
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            return httpx.Response(500, text="boom")
        return _transport(request)

    body = _client(registry, flaky).post("/v1/bench", json={"model": "good"}).json()
    assert 0 < body["score"] < 1.0
    assert any(row["error"] for row in body["rows"])
