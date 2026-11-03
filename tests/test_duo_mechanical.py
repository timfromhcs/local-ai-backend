"""A red test run outranks an LLM verdict.

Measured problem (docs/bench-log.md, hard suite): the verifier replied `VERIFIED` to a wrong answer
because it has no ground truth. When the client returns a check-tool result that says "failed", duo
must not be allowed to finish; and when it says "passed", asking a 27B model for an opinion is
wasted time.
"""
import pytest

from localai_gateway.duo import VERIFIER_SYSTEM, Duo, DuoConfig, mechanical_verdict


def completion(content="", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"id": "x", "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop", "message": msg}]}


class Recorder:
    def __init__(self, answers):
        self.calls = []
        self.answers = answers

    async def __call__(self, model, body):
        self.calls.append((model, body))
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def asked_the_verifier(calls) -> bool:
    return any(m == DuoConfig().verifier and (b.get("messages") or [{}])[0].get("content") == VERIFIER_SYSTEM
               for m, b in calls)


def convo(tool_name, tool_output, call_id="x_1"):
    return [
        {"role": "user", "content": "fix the bug until run_tests is green"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": tool_output},
    ]


TOOLS = [{"type": "function", "function": {"name": "run_tests", "parameters": {"type": "object"}}}]


# ---- the parser -----------------------------------------------------------------------------------
def test_exit_zero_is_a_pass_and_exit_one_is_a_fail():
    assert mechanical_verdict(convo("run_tests", "exit=0\n.. 2 passed"))[0] == "pass"
    assert mechanical_verdict(convo("run_tests", "exit=1\nFF 2 failed"))[0] == "fail"


def test_pytest_summary_without_exit_code_still_counts():
    assert mechanical_verdict(convo("run_tests", "2 passed in 0.01s"))[0] == "pass"
    assert mechanical_verdict(convo("pytest", "1 failed, 1 passed"))[0] == "fail"


def test_a_non_check_tool_is_not_a_verdict():
    assert mechanical_verdict(convo("read_file", "exit=1")) is None
    assert mechanical_verdict(convo("list_files", "solution.py")) is None


def test_unrecognisable_output_defers_to_the_llm():
    assert mechanical_verdict(convo("run_tests", "hmm")) is None


def test_no_tool_message_no_verdict():
    assert mechanical_verdict([{"role": "user", "content": "hi"}]) is None


# ---- the behaviour --------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_red_check_sends_the_executor_back_with_the_test_output():
    call = Recorder([completion(tool_calls=[{"id": "x_1", "type": "function",
                                             "function": {"name": "run_tests", "arguments": "{}"}}]),
                     completion("fixed it")])
    duo = Duo(call, DuoConfig())
    body = {"messages": convo("run_tests", "exit=1\nFF 2 failed in 0.01s"), "tools": TOOLS}
    await duo.chat(body)
    executor_bodies = [b for m, b in call.calls if m == DuoConfig().executor]
    assert executor_bodies, "the executor must be asked again"
    # system_extra is prepended to the first user message (keeps the cached system+tools prefix clean)
    feedback = " ".join(str(m.get("content") or "") for b in executor_bodies for m in b["messages"])
    assert "VERIFIER FEEDBACK" in feedback and "2 failed" in feedback
    assert not asked_the_verifier(call.calls), "no LLM may overrule a red test run"


@pytest.mark.asyncio
async def test_green_check_finishes_without_asking_the_verifier():
    call = Recorder([completion("it is fixed")])
    duo = Duo(call, DuoConfig())
    body = {"messages": convo("run_tests", "exit=0\n.. 2 passed in 0.01s"), "tools": TOOLS}
    result = await duo.chat(body)
    assert result["model"] == "duo"
    assert "tests passed (mechanical check)" in result["choices"][0]["message"]["content"]
    assert not asked_the_verifier(call.calls), "asking a 27B for an opinion is waste"
    assert [m for m, _ in call.calls].count(DuoConfig().planner) == 1, "the planner still runs once"


@pytest.mark.asyncio
async def test_rounds_are_bounded_and_the_failure_is_stated():
    call = Recorder([completion("still broken")])
    duo = Duo(call, DuoConfig(max_verify_rounds=2))
    body = {"messages": convo("run_tests", "exit=1\nFF"), "tools": TOOLS}
    result = await duo.chat(body)
    # the retries are bounded and the last executor answer is returned without a false VERIFIED
    assert [m for m, _ in call.calls].count(DuoConfig().executor) == 2
    assert "still broken" in result["choices"][0]["message"]["content"]
    assert not asked_the_verifier(call.calls)


@pytest.mark.asyncio
async def test_check_can_be_disabled_for_legacy_clients():
    call = Recorder([completion("looks good to me")])  # the LLM verifier's answer
    duo = Duo(call, DuoConfig(respect_tool_checks=False))
    body = {"messages": convo("run_tests", "exit=1"), "tools": TOOLS}
    await duo.chat(body)
    assert asked_the_verifier(call.calls), "opt-out keeps the old behaviour"
