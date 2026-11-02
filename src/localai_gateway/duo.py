"""Duo: one virtual model = plan with the big model, execute with the small one, verify with the big one, loop.

Stateless towards the client (OpenAI chat.completions); per-task state (plan, rounds, feedback) lives here, keyed by
the last human message + tool names. Backends are injected as `call(model_id, body) -> chat.completion dict`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterator

CallFn = Callable[[str, dict], Awaitable[dict]]

PLANNER_SYSTEM = ("You are the planner. Write a numbered plan of at most 8 short steps for the task. "
                  "Each step is one concrete action (file or command). No tools, no extra prose.")
VERIFIER_SYSTEM = ("You are the verifier. Decide whether the executor's work is complete and correct. "
                   "If code changed and is not yet verified, call a tool to run the relevant tests or commands. "
                   "When verified reply exactly 'VERIFIED: <one line>'. If something is wrong reply exactly "
                   "'FIX: <what to fix>'. Keep replies short.")


@dataclass
class DuoConfig:
    planner: str = "bonsai-27b"
    executor: str = "qwen3-4b"
    verifier: str = "bonsai-27b"
    fast: str = "jev-1.5b"
    max_verify_rounds: int = 3
    plan_max_tokens: int = 300
    verify_max_tokens: int = 400
    state_ttl_s: float = 6 * 3600
    fast_max_chars: int = 400


@dataclass
class _State:
    plan: str
    created: float
    rounds: int = 0
    answer: str = ""
    feedback: str = ""


def _text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(b.get("text", "") for b in c if isinstance(b, dict))
    return c or ""


def _strip(cid: str | None) -> str | None:
    return cid[2:] if cid and cid[:2] in ("x_", "v_") else cid


class Duo:
    def __init__(self, call: CallFn, cfg: DuoConfig | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.call, self.cfg, self._clock = call, cfg or DuoConfig(), clock
        self._states: dict[str, _State] = {}

    # ---- public -------------------------------------------------------------------------------
    async def chat(self, body: dict) -> dict:
        msgs, tools = body["messages"], body.get("tools")
        if not tools:
            chars = sum(len(_text(m)) for m in msgs)
            model = self.cfg.fast if chars < self.cfg.fast_max_chars else self.cfg.executor
            return self._label(await self.call(model, self._upstream(body, model)))
        key = self._key(msgs, tools)
        st = self._get(key)
        if st is None:
            st = self._states[key] = _State(plan=await self._plan(msgs, tools), created=self._clock())
        if self._pending_role(msgs) == "v":
            return await self._verify(body, st)
        return await self._execute(body, st)

    # ---- phases -------------------------------------------------------------------------------
    async def _plan(self, msgs: list[dict], tools: list[dict]) -> str:
        names = ", ".join(t["function"]["name"] for t in tools)
        body = {"model": self.cfg.planner, "temperature": 0.2, "max_tokens": self.cfg.plan_max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": PLANNER_SYSTEM},
                             {"role": "user", "content": f"Task:\n{self._task(msgs)}\n\nAvailable tools: {names}"}]}
        try:
            return (_msg(await self.call(self.cfg.planner, body)).get("content") or "").strip()
        except Exception:
            return ""

    async def _execute(self, body: dict, st: _State) -> dict:
        extra = ""
        if st.plan:
            extra += f"PLAN (from the planner, follow it):\n{st.plan}"
        if st.feedback:
            extra += f"\n\nVERIFIER FEEDBACK on your previous answer (fix this first):\n{st.feedback}"
        ebody = self._upstream(body, self.cfg.executor, system_extra=extra.strip())
        resp = await self.call(self.cfg.executor, ebody)
        msg = _msg(resp)
        if msg.get("tool_calls"):
            return self._tag(resp, "x_")
        st.answer = msg.get("content") or ""
        if st.rounds >= self.cfg.max_verify_rounds:
            return self._label(resp)
        st.rounds += 1
        return await self._verify(body, st, resp)

    async def _verify(self, body: dict, st: _State, executor_resp: dict | None = None) -> dict:
        msgs = body["messages"]
        user = (f"Task:\n{self._task(msgs)}\n\nPlan:\n{st.plan or '(none)'}\n\nWork so far:\n{self._activity(msgs)}"
                f"\n\nExecutor's latest answer:\n{st.answer or '(none yet)'}")
        vbody = {"model": self.cfg.verifier, "temperature": 0.1, "max_tokens": self.cfg.verify_max_tokens,
                 "tools": body.get("tools"), "chat_template_kwargs": {"enable_thinking": False},
                 "messages": [{"role": "system", "content": VERIFIER_SYSTEM}, {"role": "user", "content": user}]}
        try:
            resp = await self.call(self.cfg.verifier, vbody)
        except Exception:
            return self._final(st, executor_resp, note="")
        msg = _msg(resp)
        if msg.get("tool_calls"):
            return self._tag(resp, "v_")
        verdict = (msg.get("content") or "").strip()
        if verdict.upper().startswith("FIX"):
            st.feedback = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
            return await self._execute(body, st)
        st.feedback = ""
        return self._final(st, executor_resp, note=verdict)

    # ---- helpers ------------------------------------------------------------------------------
    def _final(self, st: _State, template: dict | None, note: str) -> dict:
        content = st.answer + (f"\n\n[{note}]" if note else "")
        resp = copy.deepcopy(template) if template else {"id": "duo", "object": "chat.completion", "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": ""}}]}
        resp["choices"][0]["message"] = {"role": "assistant", "content": content}
        resp["choices"][0]["finish_reason"] = "stop"
        return self._label(resp)

    def _key(self, msgs: list[dict], tools: list[dict]) -> str:
        last_user = next((_text(m) for m in reversed(msgs) if m.get("role") == "user"), "")
        names = ",".join(sorted(t["function"]["name"] for t in tools))
        return hashlib.sha1(f"{last_user}|{names}".encode()).hexdigest()

    def _get(self, key: str) -> _State | None:
        st = self._states.get(key)
        if st and self._clock() - st.created > self.cfg.state_ttl_s:
            del self._states[key]
            return None
        return st

    @staticmethod
    def _task(msgs: list[dict]) -> str:
        return next((_text(m) for m in reversed(msgs) if m.get("role") == "user"), "")

    @staticmethod
    def _pending_role(msgs: list[dict]) -> str | None:
        for m in reversed(msgs):
            if m.get("role") == "tool":
                continue
            if m.get("role") == "assistant" and m.get("tool_calls"):
                cid = m["tool_calls"][0].get("id", "")
                return "v" if cid.startswith("v_") else "x" if cid.startswith("x_") else None
            return None
        return None

    @staticmethod
    def _activity(msgs: list[dict], cap: int = 1800) -> str:
        start = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=0)
        lines: list[str] = []
        for m in msgs[start + 1:]:
            for tc in m.get("tool_calls") or []:
                lines.append(f"- {tc['function']['name']}({str(tc['function'].get('arguments'))[:120]})")
            if m.get("role") == "tool":
                lines.append(f"  -> {_text(m)[:200]}")
        out = "\n".join(lines) or "(no tool activity)"
        return out[-cap:]

    def _upstream(self, body: dict, model: str, system_extra: str = "") -> dict:
        b = copy.deepcopy(body)
        b.update(model=model, stream=False)
        b.pop("stream_options", None)
        b.setdefault("chat_template_kwargs", {"enable_thinking": False})
        for m in b["messages"]:
            if m.get("role") == "tool":
                m["tool_call_id"] = _strip(m.get("tool_call_id"))
            for tc in m.get("tool_calls") or []:
                tc["id"] = _strip(tc.get("id"))
        if system_extra:
            first = b["messages"][0]
            if first.get("role") == "system":
                first["content"] = f"{_text(first)}\n\n{system_extra}"
            else:
                b["messages"].insert(0, {"role": "system", "content": system_extra})
        return b

    @staticmethod
    def _tag(resp: dict, prefix: str) -> dict:
        r = copy.deepcopy(resp)
        for tc in _msg(r).get("tool_calls") or []:
            tc["id"] = prefix + (_strip(tc.get("id")) or "")
        r["model"] = "duo"
        return r

    @staticmethod
    def _label(resp: dict) -> dict:
        r = copy.deepcopy(resp)
        r["model"] = "duo"
        return r


def _msg(resp: dict) -> dict:
    return resp["choices"][0]["message"]


def to_sse_chunks(resp: dict, model: str) -> Iterator[bytes]:
    """Re-emit a complete chat.completion as OpenAI streaming chunks (single content/tool_calls delta + finish + DONE)."""
    msg, finish = _msg(resp), resp["choices"][0].get("finish_reason") or "stop"
    base = {"id": resp.get("id", "duo"), "object": "chat.completion.chunk", "created": int(time.time()), "model": model}

    def chunk(delta: dict, fin: str | None) -> bytes:
        return b"data: " + json.dumps({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}).encode() + b"\n\n"

    yield chunk({"role": "assistant"}, None)
    if msg.get("tool_calls"):
        calls = [{"index": i, "id": tc["id"], "type": "function",
                  "function": {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments", "")}}
                 for i, tc in enumerate(msg["tool_calls"])]
        yield chunk({"tool_calls": calls}, None)
        finish = "tool_calls"
    elif msg.get("content"):
        yield chunk({"content": msg["content"]}, None)
    yield chunk({}, finish)
    yield b"data: [DONE]\n\n"
