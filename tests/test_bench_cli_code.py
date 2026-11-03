"""The CLI is the client of the loop, so it must execute the tools and let the tests score."""
import json

import pytest

from bench import run as benchrun
from localai_gateway.benchscore import CODE_TASKS


def completion(content="", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"id": "x", "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop", "message": msg}]}


def call(name, arguments):
    return {"id": f"call_{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


class FakeClient:
    def __init__(self, script, timeout=None):
        self.script = script
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        self.sent.append(json)

        class R:
            def raise_for_status(self):
                return None

            def json(self_inner):
                return self.script.pop(0)

        return R()


def test_cli_code_task_runs_the_tools_and_scores_green(monkeypatch):
    script = [
        completion(tool_calls=[call("write_file", {"path": "solution.py",
                                                   "content": "def add(a, b):\n    return a + b\n"})]),
        completion("DONE"),
    ]
    fake = FakeClient(script)
    monkeypatch.setattr(benchrun.httpx, "AsyncClient", lambda timeout=None: fake)

    task = next(t for t in CODE_TASKS if t.id == "code-fix-bug")
    answer, seconds, score = benchrun.run_code_task("http://x", "m", task)

    assert score == 1.0, answer
    assert "[green]" in answer and "-> write_file" in answer
    assert seconds >= 0
    assert len(fake.sent) == 2


def test_cli_code_task_scores_red_when_tests_still_fail(monkeypatch):
    # one answer plus the two mechanical repair rounds (the tests keep failing)
    fake = FakeClient([completion("all good, DONE"), completion("still fine"), completion("DONE")])
    monkeypatch.setattr(benchrun.httpx, "AsyncClient", lambda timeout=None: fake)

    task = next(t for t in CODE_TASKS if t.id == "code-fix-bug")
    answer, _, score = benchrun.run_code_task("http://x", "m", task)
    assert score == 0.0 and "[red]" in answer


def test_cli_suites_include_code():
    assert "code" in benchrun.SUITES


@pytest.mark.parametrize("suite", ["fast", "tools", "hard", "code"])
def test_every_cli_suite_is_non_empty(suite):
    assert benchrun.SUITES[suite]
