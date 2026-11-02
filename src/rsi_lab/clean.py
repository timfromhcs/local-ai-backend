"""Turn a raw episode into a compact, scrubbed chat-format SFT example (fewer tokens, fewer calls)."""
from __future__ import annotations

import json
from typing import Any

from .scrub import scrub_text

UNTRUSTED_TOOLS = ("WebFetch", "WebSearch", "firecrawl", "browser", "playwright", "fetch", "BrowsingBee")
HEAD, TAIL = 500, 300


def estimate_tokens(s: str) -> int:
    return len(s) // 4 + 1


def _truncate(s: str) -> str:
    if len(s) <= HEAD + TAIL + 40:
        return s
    return s[:HEAD] + f"\n...[{len(s) - HEAD - TAIL} chars omitted]...\n" + s[-TAIL:]


def _untrusted(name: str) -> bool:
    n = name.lower()
    return any(u.lower() in n for u in UNTRUSTED_TOOLS)


def clean_episode(ep: dict, max_tokens: int = 1536, require_verified: bool = True) -> dict | None:
    if require_verified and not ep.get("verified"):
        return None
    calls = [c for t in ep["turns"] for c in t.get("tool_calls", [])]
    if any(_untrusted(c["name"]) for c in calls):
        return None                       # context may carry injected instructions: never train on it
    # collapse retries: drop an errored call when a later call of the same tool succeeded
    ok_names = {c["name"] for c in calls if not c.get("is_error")}
    drop = {c["id"] for c in calls if c.get("is_error") and c["name"] in ok_names}

    messages: list[dict[str, Any]] = [{"role": "user", "content": scrub_text(ep["prompt"])}]
    for t in ep["turns"]:
        kept = [c for c in t.get("tool_calls", []) if c["id"] not in drop]
        if not t.get("text", "").strip() and not kept:
            continue
        msg: dict[str, Any] = {"role": "assistant", "content": scrub_text(t.get("text", "").strip())}
        if kept:
            msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c["name"],
                                               "arguments": scrub_text(json.dumps(c["input"], ensure_ascii=False))}}
                                 for c in kept]
        messages.append(msg)
        for c in kept:
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": scrub_text(_truncate(str(c.get("result", ""))))})
    out = {"session_id": ep["session_id"], "messages": messages}
    if estimate_tokens(json.dumps(out, ensure_ascii=False)) > max_tokens:
        return None
    return out
