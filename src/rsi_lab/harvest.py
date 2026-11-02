"""Parse Claude Code transcript JSONL into episodes (one human request and everything the agent did for it)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_PASS = re.compile(r"\b(\d+)\s+passed\b")
_FAIL = re.compile(r"\b(\d+)\s+(?:failed|error|errors)\b|\bFAILED\b|\bERROR\b")
_TEST_CMD = re.compile(r"\b(pytest|vitest|jest|npm (?:run )?test|cargo test|go test|unittest)\b")


def _blocks(msg: dict) -> list[dict]:
    c = msg.get("content")
    if isinstance(c, str):
        return [{"type": "text", "text": c}]
    return [b for b in c or [] if isinstance(b, dict)]


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def _human_text(blocks: list[dict]) -> str | None:
    texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
    if not texts or any(b.get("type") == "tool_result" for b in blocks):
        return None
    text = "\n".join(texts).strip()
    if not text or text.startswith("<system-reminder>") or "[SYSTEM NOTIFICATION" in text[:200]:
        return None
    return text


def _verified(turns: list[dict]) -> bool:
    """Verified = the LAST test command in the episode reported passes and no failures."""
    last = None
    for t in turns:
        for c in t.get("tool_calls", []):
            if c["name"] in ("Bash", "PowerShell") and _TEST_CMD.search(str(c["input"].get("command", ""))):
                last = c
    if last is None:
        return False
    res = last.get("result", "")
    return bool(_PASS.search(res)) and not _FAIL.search(res)


def parse_transcript(path: Path) -> list[dict]:
    episodes: list[dict] = []
    cur: dict | None = None
    pending: dict[str, dict] = {}
    for raw in Path(path).read_text(encoding="utf8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if row.get("type") not in ("user", "assistant") or row.get("isSidechain"):
            continue
        blocks = _blocks(row.get("message") or {})
        if row["type"] == "user":
            human = _human_text(blocks)
            if human is not None:
                cur = {"session_id": row.get("sessionId", ""), "prompt": human, "turns": []}
                episodes.append(cur)
                continue
            for b in blocks:
                if b.get("type") == "tool_result" and b.get("tool_use_id") in pending:
                    call = pending[b["tool_use_id"]]
                    call["result"] = _result_text(b.get("content"))
                    call["is_error"] = bool(b.get("is_error"))
        elif cur is not None:
            turn: dict[str, Any] = {"role": "assistant", "text": "", "tool_calls": []}
            for b in blocks:
                if b.get("type") == "text":
                    turn["text"] += b.get("text", "")
                elif b.get("type") == "tool_use":
                    call = {"id": b.get("id"), "name": b.get("name"), "input": b.get("input") or {}, "result": ""}
                    turn["tool_calls"].append(call)
                    pending[call["id"]] = call
            cur["turns"].append(turn)
    for ep in episodes:
        ep["verified"] = _verified(ep["turns"])
    return episodes
