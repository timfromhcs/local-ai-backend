"""Tool-call benchmark suite: a task is only solved by an actual tool call, not prose about one."""
import httpx
import pytest

from localai_gateway.benchscore import TOOL_TASKS, score_expected_tool_calls

from tests.test_bench_route import _client, _Scheduler  # reuse the mock client factory
from localai_gateway.config import Settings
from localai_gateway.registry import ModelSpec


def message(*names: str) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": "{}"}}
        for i, n in enumerate(names)
    ]}


def test_exact_name_sequence_scores_one():
    assert score_expected_tool_calls(("list_files", "read_file"))(message("list_files", "read_file")) == 1.0


def test_wrong_order_or_missing_call_scores_zero():
    score = score_expected_tool_calls(("list_files", "read_file"))
    assert score(message("read_file", "list_files")) == 0.0
    assert score(message("list_files")) == 0.0
    assert score({"role": "assistant", "content": "I would call list_files"}) == 0.0


def test_tools_suite_is_shaped_correctly():
    assert len(TOOL_TASKS) >= 2
    for task in TOOL_TASKS:
        assert task.tools, f"{task.id} must offer tool schemas"
        assert task.score_message is not None, f"{task.id} must score the message, not the text"
        assert task.category == "tools"


def test_bench_route_passes_tools_and_scores_tool_calls(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        seen["tools"] = [t["function"]["name"] for t in payload.get("tools", [])]
        seen["tool_choice"] = payload.get("tool_choice")
        return httpx.Response(200, json={"choices": [{"index": 0, "finish_reason": "tool_calls",
                                                      "message": message("list_files", "read_file")}]})

    registry = {"m": ModelSpec(id="m", kind="llm", files=[], runtime="llama", est_mem_mb=10)}
    client = _client(registry, handler)
    body = client.post("/v1/bench", json={"model": "m", "suite": "tools", "tasks": ["tool-list-read"]}).json()

    assert seen["tools"] == ["list_files", "read_file"]
    assert seen["tool_choice"] == "auto"
    assert body["score"] == 1.0, body
    assert body["rows"][0]["id"] == "tool-list-read"


def test_bench_route_reports_prose_as_zero_on_tool_tasks():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"index": 0, "finish_reason": "stop",
                                                      "message": {"role": "assistant", "content": "list_files('C:/work')"}}]})

    registry = {"m": ModelSpec(id="m", kind="llm", files=[], runtime="llama", est_mem_mb=10)}
    body = _client(registry, handler).post("/v1/bench", json={"model": "m", "suite": "tools"}).json()
    assert body["score"] == 0.0, "text that looks like a call is not a tool call"
