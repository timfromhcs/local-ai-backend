"""Mode controller: Jev (1.5B, fast) decides per task which mode is needed, re-checked regularly, escalating on trouble.

Mode.FAST    small model only (latency first)
Mode.DUO     plan (big) -> execute (small) -> verify (big)           [default]
Mode.BIGOUT  "small in, big out": small model digests the input into a brief, the big model produces the output

ModeController(ask_jev, checkup_every=3, hold_turns=2)
  ask_jev(prompt: str) -> awaitable str | None     (None / exception / timeout => heuristic only)
  await decide(key, body) -> Mode     (call once per incoming request; state per task `key`)
  note_verification_failed(key)       (escalation signal from Duo)
"""
import asyncio

from localai_gateway.modes import Mode, ModeController

TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run", "parameters": {}}}]


def body(prompt="fix it", tools=True, tool_msgs=()):
    msgs = [{"role": "user", "content": prompt}]
    for content in tool_msgs:
        msgs += [{"role": "assistant", "content": None, "tool_calls": [{"id": "x_1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": "x_1", "content": content}]
    b = {"messages": msgs}
    if tools:
        b["tools"] = TOOLS
    return b


class Jev:
    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    async def __call__(self, prompt):
        self.prompts.append(prompt)
        a = self.answers.pop(0) if self.answers else "duo"
        if isinstance(a, Exception):
            raise a
        return a


async def test_speed_hint_forces_fast_without_asking_jev():
    jev = Jev("bigout")
    mc = ModeController(jev)
    assert await mc.decide("k", body("bitte schnell: what is 2+2?")) == Mode.FAST
    assert jev.prompts == []


async def test_short_request_without_tools_is_fast():
    assert await ModeController(Jev("duo")).decide("k", body("title please", tools=False)) == Mode.FAST


async def test_jev_answer_selects_mode_and_prompt_contains_the_task():
    jev = Jev("Bigout.")
    mode = await ModeController(jev).decide("k", body("refactor the whole scheduler across 5 files"))
    assert mode == Mode.BIGOUT and "refactor the whole scheduler" in jev.prompts[0]


async def test_unparseable_or_failing_jev_falls_back_to_duo():
    for ans in ("no idea", RuntimeError("down"), None):
        assert await ModeController(Jev(ans)).decide("k", body("do work")) == Mode.DUO


async def test_jev_timeout_falls_back_quickly():
    async def slow(prompt):
        await asyncio.sleep(5)
        return "bigout"

    mc = ModeController(slow, jev_timeout_s=0.1)
    assert await mc.decide("k", body("do work")) == Mode.DUO


async def test_checkup_cadence_asks_jev_at_start_and_every_n_turns_only():
    jev = Jev("duo", "duo", "duo")
    mc = ModeController(jev, checkup_every=3)
    for i in range(7):                                  # 7 requests of the same task, growing history
        await mc.decide("k", body("task", tool_msgs=["ok"] * i))
    assert len(jev.prompts) == 3                        # turn 0, 3, 6


async def test_tool_errors_escalate_fast_to_duo_to_bigout_and_hold():
    mc = ModeController(Jev("fast", "fast", "fast", "fast"), checkup_every=1, hold_turns=2)
    assert await mc.decide("k", body("task")) == Mode.FAST
    err = "Traceback (most recent call last): ... exit code 1"
    assert await mc.decide("k", body("task", tool_msgs=[err])) == Mode.FAST          # one error: tolerated
    assert await mc.decide("k", body("task", tool_msgs=[err, err])) == Mode.DUO      # two errors: escalate despite Jev
    mc.note_verification_failed("k")
    assert await mc.decide("k", body("task", tool_msgs=[err, err, "ok"])) == Mode.BIGOUT
    assert await mc.decide("k", body("task", tool_msgs=[err, err, "ok", "ok"])) == Mode.BIGOUT   # held, no flapping


async def test_downgrade_after_hold_when_stable_and_jev_agrees():
    mc = ModeController(Jev("duo", "duo", "duo", "duo", "duo"), checkup_every=1, hold_turns=1)
    err = "Error: exit code 2"
    await mc.decide("k", body("t"))
    mc.note_verification_failed("k")
    assert await mc.decide("k", body("t", tool_msgs=[err])) == Mode.BIGOUT
    modes = [await mc.decide("k", body("t", tool_msgs=[err] + ["ok"] * n)) for n in (1, 2, 3)]
    assert modes[-1] == Mode.DUO


async def test_tasks_are_independent():
    mc = ModeController(Jev("bigout", "fast"))
    assert await mc.decide("a", body("hard one")) == Mode.BIGOUT
    assert await mc.decide("b", body("easy one")) == Mode.FAST
