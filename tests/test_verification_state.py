"""Every duo answer says how it was checked - or admits that it was not.

The previous cycle made a red test run authoritative. This one removes the opposite lie: the fast
lane, the big-output path and an unreachable verifier used to return an answer that looked exactly
as trustworthy as a verified one.
"""
import pytest

from localai_gateway.duo import VERIFIER_SYSTEM, Duo, DuoConfig


def completion(content="", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"id": "x", "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop", "message": msg}]}


class Script:
    def __init__(self, answers):
        self.calls = []
        self.answers = list(answers)

    async def __call__(self, model, body):
        self.calls.append((model, body))
        if "answers" in dir(self) and self.answers:
            return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        raise AssertionError("unexpected call")


def verification(resp: dict) -> dict:
    return resp["localai_verification"]


TOOLS = [{"type": "function", "function": {"name": "run_tests", "parameters": {"type": "object"}}}]


@pytest.mark.asyncio
async def test_fast_lane_admits_it_verified_nothing():
    call = Script([completion("ok")])
    duo = Duo(call, DuoConfig())
    resp = await duo.chat({"messages": [{"role": "user", "content": "hi"}], "localai_mode": "fast"})
    v = verification(resp)
    assert v["state"] == "none"
    assert "fast lane" in v["detail"]


@pytest.mark.asyncio
async def test_bigout_admits_it_verified_nothing():
    call = Script([completion("brief"), completion("the answer")])
    duo = Duo(call, DuoConfig())
    resp = await duo.chat({"messages": [{"role": "user", "content": "write it"}], "localai_mode": "bigout"})
    assert verification(resp)["state"] == "none"
    assert "big-output" in verification(resp)["detail"]


@pytest.mark.asyncio
async def test_llm_verdict_is_labelled_as_an_opinion_not_as_a_check():
    call = Script([completion("plan: do it"), completion("it is done"), completion("VERIFIED: looks right")])
    duo = Duo(call, DuoConfig())
    resp = await duo.chat({"messages": [{"role": "user", "content": "do it"}],
                           "tools": TOOLS, "localai_mode": "duo"})
    assert verification(resp)["state"] == "llm"


@pytest.mark.asyncio
async def test_mechanical_pass_is_labelled_as_mechanical():
    call = Script([completion("plan"), completion("fixed")])
    duo = Duo(call, DuoConfig())
    msgs = [{"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "x_1", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "x_1", "content": "exit=0\n2 passed"}]
    resp = await duo.chat({"messages": msgs, "tools": TOOLS, "localai_mode": "duo"})
    assert verification(resp)["state"] == "mechanical"


@pytest.mark.asyncio
async def test_a_tool_call_is_pending_not_verified():
    call = Script([completion("plan"), completion(tool_calls=[
        {"id": "1", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}}])])
    duo = Duo(call, DuoConfig())
    resp = await duo.chat({"messages": [{"role": "user", "content": "fix it"}],
                           "tools": TOOLS, "localai_mode": "duo"})
    assert verification(resp)["state"] == "pending"


@pytest.mark.asyncio
async def test_an_unreachable_verifier_is_reported_as_such():
    class Broken:
        def __init__(self):
            self.n = 0

        async def __call__(self, model, body):
            self.n += 1
            if self.n == 1:
                return completion("plan")
            if self.n == 2:
                return completion("done")
            raise RuntimeError("verifier down")

    duo = Duo(Broken(), DuoConfig())
    resp = await duo.chat({"messages": [{"role": "user", "content": "do it"}],
                           "tools": TOOLS, "localai_mode": "duo"})
    assert verification(resp)["state"] == "none"
    assert "unreachable" in verification(resp)["detail"]


def test_verification_states_are_documented_in_one_place():
    from localai_gateway import duo

    assert duo.VERIFICATION_STATES == {"mechanical", "llm", "none", "pending", "failed"}

def test_the_field_reaches_the_http_response():
    from tests.test_forced_mode import _client

    body = _client(None).post("/v1/chat/completions", json={
        "model": "duo", "messages": [{"role": "user", "content": "hi"}], "localai_mode": "fast"}).json()
    assert body["localai_verification"]["state"] == "none"


def test_the_field_is_documented_in_the_readme():
    import pathlib

    readme = pathlib.Path("README.md").read_text(encoding="utf-8")
    assert "localai_verification" in readme

@pytest.mark.asyncio
async def test_an_unreachable_planner_is_named_in_the_detail():
    """Found live: the 27B planner could not load, duo swallowed the error and looked normal."""

    class NoPlanner:
        def __init__(self):
            self.n = 0

        async def __call__(self, model, body):
            self.n += 1
            if model == DuoConfig().planner:
                raise RuntimeError("failed to load model")
            return completion("done")

    duo = Duo(NoPlanner(), DuoConfig())
    resp = await duo.chat({"messages": [{"role": "user", "content": "do it"}],
                           "tools": TOOLS, "localai_mode": "duo"})
    detail = verification(resp)["detail"]
    assert "planner unavailable" in detail and "failed to load model" in detail
