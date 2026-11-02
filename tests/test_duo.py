"""Duo orchestrator contract: ONE virtual model that plans with the big model, executes with the small one,
verifies with the big one, and loops until verified (bounded). Pure logic: backends are injected as `call`.

Duo(call, cfg=DuoConfig(), clock=time.monotonic)
  call(model_id: str, body: dict) -> awaitable chat.completion dict (always non-stream)
  await duo.chat(body) -> chat.completion dict   (the response the client sees)
Tool-call ids returned to the client are prefixed "x_" (executor) or "v_" (verifier) so the next request
(the tool results) can be routed back to the role that asked for them; prefixes are stripped before forwarding upstream.
"""
import copy

import pytest

from localai_gateway.duo import Duo, DuoConfig, to_sse_chunks

TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run", "parameters": {"type": "object", "properties": {}}}}]
CFG = DuoConfig(planner="P", executor="E", verifier="V", fast="F", max_verify_rounds=2)


def text(content):
    return {"id": "c", "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": content}}]}


def tool(name="bash", args='{"command":"pytest -q"}', cid="call1"):
    return {"id": "c", "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "tool_calls",
            "message": {"role": "assistant", "content": None,
                        "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": args}}]}}]}


class Fake:
    def __init__(self, **script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    async def __call__(self, model, body):
        self.calls.append((model, copy.deepcopy(body)))
        item = self.script[model].pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def models(self):
        return [m for m, _ in self.calls]


def req(*extra, prompt="fix the bug", tools=True):
    msgs = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": prompt}, *extra]
    body = {"model": "duo", "messages": msgs}
    if tools:
        body["tools"] = TOOLS
    return body


def asst_call(cid):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "type": "function", "function": {"name": "bash", "arguments": "{}"}}]}


def tool_msg(cid, content="3 passed"):
    return {"role": "tool", "tool_call_id": cid, "content": content}


async def test_first_request_plans_then_executes_with_plan_injected():
    f = Fake(P=[text("1. read file\n2. patch\n3. run tests")], E=[tool()])
    out = await Duo(f, CFG).chat(req())
    assert f.models() == ["P", "E"]
    assert "tools" not in f.calls[0][1]                                   # planner must not call tools
    sys_msg = f.calls[1][1]["messages"][0]["content"]
    assert "run tests" in sys_msg and "You are a coding agent." in sys_msg
    assert f.calls[1][1]["tools"] == TOOLS
    assert out["choices"][0]["message"]["tool_calls"][0]["id"] == "x_call1"


async def test_tool_result_goes_to_executor_without_replanning():
    f = Fake(P=[text("1. do it")], E=[tool(), text("done")], V=[text("VERIFIED: ok")])
    d = Duo(f, CFG)
    await d.chat(req())
    await d.chat(req(asst_call("x_call1"), tool_msg("x_call1")))
    assert f.models() == ["P", "E", "E", "V"]                             # exactly one plan call
    fwd = f.calls[2][1]["messages"]
    assert [m for m in fwd if m["role"] == "tool"][0]["tool_call_id"] == "call1"   # prefix stripped upstream
    assert fwd[-2]["tool_calls"][0]["id"] == "call1"


async def test_final_answer_is_verified_then_returned():
    f = Fake(P=[text("plan")], E=[text("I fixed it")], V=[text("VERIFIED: tests pass")])
    out = await Duo(f, CFG).chat(req())
    assert f.models() == ["P", "E", "V"]
    msg = out["choices"][0]["message"]
    assert "I fixed it" in msg["content"] and "tool_calls" not in msg


async def test_verifier_can_run_tools_and_next_turn_returns_to_verifier():
    f = Fake(P=[text("plan")], E=[text("done")], V=[tool(cid="t9"), text("VERIFIED: green")])
    d = Duo(f, CFG)
    out1 = await d.chat(req())
    assert out1["choices"][0]["message"]["tool_calls"][0]["id"] == "v_t9"
    out2 = await d.chat(req(asst_call("v_t9"), tool_msg("v_t9", "5 passed")))
    assert f.models() == ["P", "E", "V", "V"]                             # result of a verifier call returns to V
    assert "done" in out2["choices"][0]["message"]["content"]


async def test_fix_feedback_loops_to_executor_and_is_bounded():
    f = Fake(P=[text("plan")], E=[text("v1"), text("v2"), text("v3")],
             V=[text("FIX: test_a fails"), text("FIX: still failing")])
    out = await Duo(f, CFG).chat(req())                                   # max_verify_rounds=2
    assert f.models() == ["P", "E", "V", "E", "V", "E"]
    assert "test_a fails" in str(f.calls[3][1]["messages"])               # feedback reached the executor
    assert "v3" in out["choices"][0]["message"]["content"]                # budget spent: executor answer returned


async def test_no_tools_short_request_goes_to_fast_model():
    f = Fake(F=[text("Title")])
    out = await Duo(f, CFG).chat(req(prompt="make a title", tools=False))
    assert f.models() == ["F"] and out["choices"][0]["message"]["content"] == "Title"


async def test_no_tools_long_request_goes_to_executor_only():
    f = Fake(E=[text("long answer")])
    await Duo(f, CFG).chat(req(prompt="explain " * 200, tools=False))
    assert f.models() == ["E"]


async def test_each_new_task_is_planned_separately():
    f = Fake(P=[text("plan A"), text("plan B")], E=[tool(), tool()])
    d = Duo(f, CFG)
    await d.chat(req(prompt="task A"))
    await d.chat(req(prompt="task B"))
    assert f.models() == ["P", "E", "P", "E"]


async def test_planner_failure_falls_back_to_executor_without_plan():
    f = Fake(P=[RuntimeError("27B busy")], E=[tool()])
    out = await Duo(f, CFG).chat(req())
    assert f.models() == ["P", "E"] and out["choices"][0]["message"]["tool_calls"]


async def test_verifier_failure_returns_executor_answer():
    f = Fake(P=[text("plan")], E=[text("answer")], V=[RuntimeError("27B oom")])
    out = await Duo(f, CFG).chat(req())
    assert out["choices"][0]["message"]["content"].startswith("answer")


async def test_plan_state_expires():
    now = [0.0]
    f = Fake(P=[text("p1"), text("p2")], E=[tool(), tool()])
    d = Duo(f, DuoConfig(planner="P", executor="E", verifier="V", fast="F", state_ttl_s=10), clock=lambda: now[0])
    await d.chat(req())
    now[0] = 100.0
    await d.chat(req())
    assert f.models().count("P") == 2


def test_to_sse_chunks_roundtrip():
    chunks = list(to_sse_chunks(text("hello"), "duo"))
    assert chunks[-1] == b"data: [DONE]\n\n" and all(c.startswith(b"data: ") for c in chunks)
    assert b"hello" in b"".join(chunks)


def test_to_sse_chunks_tool_calls():
    chunks = b"".join(to_sse_chunks(tool(cid="x_1"), "duo"))
    assert b"tool_calls" in chunks and b"x_1" in chunks and b'"finish_reason": "tool_calls"' in chunks
