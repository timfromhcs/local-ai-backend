"""Mode controller: Jev decides which mode a task needs, re-checked on a fixed cadence; errors escalate deterministically."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable


class Mode(str, Enum):
    FAST = "fast"        # small model only
    DUO = "duo"          # plan (big) -> execute (small) -> verify (big)
    BIGOUT = "bigout"    # small model digests the input, the big model produces the output


_RANK = {Mode.FAST: 0, Mode.DUO: 1, Mode.BIGOUT: 2}
_ERR = re.compile(r"traceback|error|exception|exit code [1-9]|failed|not found|permission denied", re.I)
_HINT = re.compile(r"\b(schnell|fast|quick|quickly|asap)\b", re.I)
_WORD = re.compile(r"(fast|duo|bigout)")
ASK = ("Pick the processing mode for this coding-agent task. Answer with ONE word.\n"
       "fast = trivial or latency-critical (small model alone)\n"
       "duo = normal task (plan, execute, verify)\n"
       "bigout = hard, multi-file or reasoning-heavy (big model must write the output)\n"
       "Task: {task}\nRecent tool results: {recent}\nAnswer:")


def _text(m: dict) -> str:
    c = m.get("content")
    return " ".join(b.get("text", "") for b in c if isinstance(b, dict)) if isinstance(c, list) else (c or "")


@dataclass
class _S:
    mode: Mode = Mode.DUO
    jev: Mode | None = None
    last_ask_turn: int = -10_000
    last_changed_turn: int = -10_000
    fail_turn: int | None = None
    pending_fail: bool = False


class ModeController:
    def __init__(self, ask_jev: Callable[[str], Awaitable[str | None]] | None, checkup_every: int = 3,
                 hold_turns: int = 2, jev_timeout_s: float = 3.0) -> None:
        self.ask_jev, self.checkup_every, self.hold_turns, self.jev_timeout_s = ask_jev, checkup_every, hold_turns, jev_timeout_s
        self._s: dict[str, _S] = {}

    def note_verification_failed(self, key: str) -> None:
        self._s.setdefault(key, _S()).pending_fail = True

    async def decide(self, key: str, body: dict) -> Mode:
        s = self._s.setdefault(key, _S())
        msgs = body["messages"]
        last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
        task = _text(msgs[last_user]) if last_user >= 0 else ""
        results = [_text(m) for m in msgs[last_user + 1:] if m.get("role") == "tool"]
        turn = len(results)
        if _HINT.search(task):
            return Mode.FAST
        if not body.get("tools") and len(task) < 400:
            return Mode.FAST
        # deterministic escalation floor (Jev cannot override trouble)
        errors = sum(1 for r in results[-4:] if _ERR.search(r))
        if s.pending_fail:
            s.pending_fail, s.fail_turn = False, turn
        floor = Mode.FAST
        if errors >= 2:
            floor = Mode.DUO
        if errors >= 4 or (s.fail_turn is not None and turn - s.fail_turn <= self.hold_turns):
            floor = Mode.BIGOUT
        # periodic Jev check-up
        if self.ask_jev is not None and turn - s.last_ask_turn >= self.checkup_every:
            s.last_ask_turn = turn
            try:
                ans = await asyncio.wait_for(self.ask_jev(ASK.format(task=task[:600], recent=" | ".join(r[:120] for r in results[-2:]) or "none")),
                                             self.jev_timeout_s)
                m = _WORD.search((ans or "").lower())
                s.jev = Mode(m.group(1)) if m else None
            except Exception:
                s.jev = None
        desired = s.jev or Mode.DUO
        mode = desired if _RANK[desired] >= _RANK[floor] else floor
        if _RANK[mode] < _RANK[s.mode] and turn - s.last_changed_turn < self.hold_turns:
            mode = s.mode                                   # hold an escalation for a few turns: no flapping
        if mode != s.mode:
            s.mode, s.last_changed_turn = mode, turn
        return mode
