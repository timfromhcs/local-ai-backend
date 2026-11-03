"""The code suite: a real workdir, real tools, and a score produced by running the tests."""
import json

import httpx
import pytest

from localai_gateway.bench_exec import EXEC_TOOLS, Workdir, run_with_tools, summarize_transcript
from localai_gateway.benchscore import CODE_TASKS, SUITES
from localai_gateway.config import Settings
from localai_gateway.registry import ModelSpec

from tests.test_bench_route import _client

GREEN = {"solution.py": "def add(a, b):\n    return a + b\n",
         "test_solution.py": "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"}


def test_code_suite_is_shaped_correctly():
    assert len(CODE_TASKS) >= 2
    assert "code" in SUITES
    for task in CODE_TASKS:
        assert task.files, f"{task.id} needs fixture files"
        assert task.verify is not None, f"{task.id} must be checked mechanically"
        assert any(t["function"]["name"] == "run_tests" for t in task.tools)
        assert task.category == "code"


def test_read_and_write_stay_inside_the_workdir():
    work = Workdir({"a.txt": "hello"})
    try:
        assert work.read_file("a.txt") == "hello"
        assert work.list_files(".") == "a.txt"
        assert "outside" in work.dispatch("read_file", {"path": "../secrets.txt"})
        assert "outside" in work.dispatch("write_file", {"path": "../../evil.txt", "content": "x"})
        assert "unknown tool" in work.dispatch("rm_rf", {})
    finally:
        work.cleanup()


def test_tests_pass_is_a_real_run(tmp_path):
    work = Workdir({"solution.py": "def add(a, b):\n    return a - b\n",
                    "test_solution.py": "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"})
    try:
        passed, output = work.tests_pass()
        assert passed is False and "failed" in output.lower()
        work.write_file("solution.py", "def add(a, b):\n    return a + b\n")
        passed, output = work.tests_pass()
        assert passed is True, output
        assert work.run_tests().startswith("exit=0")
    finally:
        work.cleanup()


def test_workdir_root_is_already_resolved():
    """CI (windows-latest) caught the bug this guards: a temp path can be a short name (RUNNER~1),
    so a later resolve() of a child did not compare equal to an unresolved root."""
    work = Workdir({"a.txt": "x"})
    try:
        assert work.root == work.root.resolve()
        assert work.list_files(".") == "a.txt"
        assert str(work.root / "a.txt").startswith(str(work.root))
    finally:
        work.cleanup()


def test_write_file_reports_and_truncates_reads():
    work = Workdir({"big.txt": "x" * 7000})
    try:
        assert "written" in work.write_file("new.txt", "abc")
        assert work.root.joinpath("new.txt").read_text(encoding="utf-8") == "abc"
        assert "[truncated]" in work.read_file("big.txt")
    finally:
        work.cleanup()


def completion(content="", tool_calls=None, finish="stop"):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"id": "x", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": finish, "message": msg}]}


def call(name, arguments):
    return {"id": f"call_{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


@pytest.mark.asyncio
async def test_tool_loop_executes_calls_and_returns_the_final_answer():
    work = Workdir(GREEN)
    script = [
        completion(tool_calls=[call("list_files", {"path": "."})], finish="tool_calls"),
        completion(tool_calls=[call("run_tests", {})], finish="tool_calls"),
        completion("DONE"),
    ]
    seen = []

    async def fake_call(body):
        seen.append(body)
        return script.pop(0)

    try:
        resp, messages = await run_with_tools(fake_call, {"messages": [{"role": "user", "content": "go"}]}, work)
        assert resp["choices"][0]["message"]["content"] == "DONE"
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert "test_solution.py" in tool_msgs[0]["content"]
        assert tool_msgs[1]["content"].startswith("exit=0")
        # the tool result is fed back to the model, which is the whole point
        assert any(m["role"] == "tool" for m in seen[1]["messages"])
    finally:
        work.cleanup()


@pytest.mark.asyncio
async def test_tool_loop_stops_on_the_round_budget():
    work = Workdir(GREEN)
    calls = {"n": 0}

    async def always_tools(body):
        calls["n"] += 1
        return completion(tool_calls=[call("list_files", {"path": "."})], finish="tool_calls")

    try:
        _, messages = await run_with_tools(always_tools, {"messages": [{"role": "user", "content": "go"}]},
                                           work, max_rounds=3)
        assert calls["n"] == 3, "an endless tool caller must be cut off"
        assert summarize_transcript(messages).count("-> list_files") == 3
    finally:
        work.cleanup()


def _client_with(handler):
    registry = {"m": ModelSpec(id="m", kind="llm", files=[], runtime="llama", est_mem_mb=10),
                "duo": ModelSpec(id="duo", kind="llm", files=[], runtime="virtual", est_mem_mb=0)}
    return _client(registry, handler)


def test_code_suite_refuses_to_run_generated_code_by_default(monkeypatch):
    monkeypatch.delenv("LOCALAI_BENCH_EXEC", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no model call may happen when execution is disabled")

    body = _client_with(handler).post("/v1/bench", json={"model": "m", "suite": "code"}).json()
    assert body["error"]["code"] == "bench_exec_disabled"


def test_code_suite_scores_a_green_patch(monkeypatch):
    monkeypatch.setenv("LOCALAI_BENCH_EXEC", "1")
    script = [
        completion(tool_calls=[call("read_file", {"path": "solution.py"})], finish="tool_calls"),
        completion(tool_calls=[call("write_file", {"path": "solution.py",
                                                   "content": "def add(a, b):\n    return a + b\n"})],
                   finish="tool_calls"),
        completion("DONE"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=script.pop(0))

    body = _client_with(handler).post("/v1/bench", json={"model": "m", "suite": "code",
                                                       "tasks": ["code-fix-bug"]}).json()
    assert body["score"] == 1.0, body
    assert "-> write_file" in body["rows"][0]["answer"]
    assert "[green]" in body["rows"][0]["answer"]


def test_code_suite_scores_a_broken_patch_as_zero(monkeypatch):
    monkeypatch.setenv("LOCALAI_BENCH_EXEC", "1")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion("I think it is fine now. DONE"))

    body = _client_with(handler).post("/v1/bench", json={"model": "m", "suite": "code",
                                                       "tasks": ["code-fix-bug"]}).json()
    assert body["score"] == 0.0, "talking about a fix is not a fix"
    assert "[red]" in body["rows"][0]["answer"]


def test_exec_tools_have_unique_names_and_object_schemas():
    names = [t["function"]["name"] for t in EXEC_TOOLS]
    assert len(names) == len(set(names))
    for tool in EXEC_TOOLS:
        assert tool["function"]["parameters"]["type"] == "object"


def test_settings_defaults_do_not_enable_execution():
    # the opt-in is an environment variable, never a default
    assert Settings(budget_mb=100, bin_dir=".", model_dir=".").model_dump().get("bench_exec") is None

@pytest.mark.asyncio
async def test_repair_round_turns_a_lazy_answer_into_a_green_patch():
    """The client, not the model, closes the loop: a failing test is fed back as a user message."""
    work = Workdir({"solution.py": "def add(a, b):\n    return a - b\n",
                    "test_solution.py": "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"})
    script = [
        completion("it is already fine, DONE"),
        completion(tool_calls=[call("write_file", {"path": "solution.py",
                                                   "content": "def add(a, b):\n    return a + b\n"})]),
        completion("DONE"),
    ]
    seen = []

    async def fake_call(body):
        seen.append(body)
        return script.pop(0)

    try:
        resp, messages = await run_with_tools(fake_call, {"messages": [{"role": "user", "content": "fix it"}]},
                                              work, check=work.tests_pass, repairs=2)
        passed, output = work.tests_pass()
        assert passed, output
        assert resp["choices"][0]["message"]["content"] == "DONE"
        assert "the tests still fail" in seen[1]["messages"][-1]["content"]
        assert any(m["role"] == "tool" for m in seen[2]["messages"])
    finally:
        work.cleanup()


@pytest.mark.asyncio
async def test_repair_round_is_bounded_and_scores_zero():
    work = Workdir({"solution.py": "def add(a, b):\n    return a - b\n",
                    "test_solution.py": "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"})
    calls = {"n": 0}

    async def always_done(body):
        calls["n"] += 1
        return completion("DONE")

    try:
        await run_with_tools(always_done, {"messages": [{"role": "user", "content": "fix it"}]},
                             work, max_rounds=3, check=work.tests_pass, repairs=1)
        assert calls["n"] == 2, "one attempt plus exactly one repair round"
        assert work.tests_pass()[0] is False
    finally:
        work.cleanup()
