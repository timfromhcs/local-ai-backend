"""Capture verified work into training data, without inventing a single step.

Design rules (from the RSI spec):
* append-only JSONL first, derivation second — a crash never loses the raw session;
* every string is scrubbed before it touches disk;
* only a session with an explicit successful verification becomes training data;
* synthetic steps are allowed but are marked, never presented as executed work.

The output is compatible with `rsi_lab.clean.clean_episode`, so the existing
scrub/clean/split/gate chain keeps working unchanged.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .clean import clean_episode, estimate_tokens
from .scrub import scrub_text

EVENT_TYPES = {"user", "assistant", "tool", "verification", "note"}


def _scrub(value: Any) -> Any:
    """Scrub every string in a nested structure; keep the shape."""
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


@dataclass(frozen=True)
class Event:
    ts: float
    type: str
    payload: dict

    def as_dict(self) -> dict:
        return {"ts": self.ts, "type": self.type, **self.payload}


class CaptureError(ValueError):
    pass


class CaptureLog:
    """One append-only file per session under <root>/captures."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        (self.root / "captures").mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- write
    def _path(self, session_id: str) -> Path:
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:80]
        if not safe:
            raise CaptureError("session id must contain letters or digits")
        return self.root / "captures" / f"{safe}.jsonl"

    def append(self, session_id: str, event: dict) -> dict:
        kind = event.get("type")
        if kind not in EVENT_TYPES:
            raise CaptureError(f"unknown event type {kind!r}; allowed: {sorted(EVENT_TYPES)}")
        payload = {k: v for k, v in event.items() if k != "type"}
        if kind in ("user", "assistant", "note") and not str(payload.get("content", "")).strip():
            raise CaptureError(f"{kind} event needs content")
        if kind == "tool" and not str(payload.get("name", "")).strip():
            raise CaptureError("tool event needs a name")
        if kind == "verification" and "ok" not in payload:
            raise CaptureError("verification event needs ok: true/false")
        record = Event(ts=time.time(), type=kind, payload=_scrub(payload)).as_dict()
        with self._path(session_id).open("a", encoding="utf8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    # ----------------------------------------------------------------- read
    def events(self, session_id: str) -> list[dict]:
        path = self._path(session_id)
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def sessions(self) -> list[str]:
        # derived pair files share the directory; they are not sessions
        return sorted(p.stem for p in (self.root / "captures").glob("*.jsonl")
                      if not p.stem.endswith(".pairs"))

    # ------------------------------------------------------------- episode
    def episode(self, session_id: str) -> dict:
        """Fold events into the episode shape that clean_episode expects."""
        events = self.events(session_id)
        if not events:
            raise CaptureError(f"no events for session {session_id!r}")
        prompt = next((e["content"] for e in events if e["type"] == "user"), "")
        turns: list[dict] = []
        current = {"text": "", "tool_calls": []}
        synthetic = 0
        verified = False
        verification_detail = ""

        for e in events:
            if e["type"] == "assistant":
                if current["text"] or current["tool_calls"]:
                    turns.append(current)
                current = {"text": e.get("content", ""), "tool_calls": []}
                synthetic += bool(e.get("synthetic"))
            elif e["type"] == "tool":
                call_id = f"call_{len(turns)}_{len(current['tool_calls'])}"
                current["tool_calls"].append({
                    "id": call_id,
                    "name": e["name"],
                    "input": e.get("input") or {},
                    "result": e.get("result", ""),
                    "is_error": not bool(e.get("ok", True)),
                })
                synthetic += bool(e.get("synthetic"))
            elif e["type"] == "verification":
                verified = bool(e.get("ok"))
                verification_detail = str(e.get("detail", ""))
        if current["text"] or current["tool_calls"]:
            turns.append(current)

        return {
            "session_id": session_id,
            "prompt": prompt,
            "turns": turns,
            "verified": verified,
            "verification_detail": verification_detail,
            "synthetic_steps": synthetic,
            "events": len(events),
        }

    # --------------------------------------------------------------- pairs
    def pairs(self, session_id: str, require_verified: bool = True) -> list[dict]:
        ep = self.episode(session_id)
        cleaned = clean_episode(ep, require_verified=require_verified)
        if cleaned is None:
            return []
        cleaned["meta"] = {
            "session_id": session_id,
            "verified": ep["verified"],
            "verification": ep["verification_detail"][:300],
            "synthetic_steps": ep["synthetic_steps"],
            "turns": len(ep["turns"]),
        }
        return [cleaned]

    def finalize(self, session_id: str) -> dict:
        pairs = self.pairs(session_id)
        out = self.root / "captures" / f"{self._path(session_id).stem}.pairs.jsonl"
        with out.open("w", encoding="utf8") as fh:
            for pair in pairs:
                fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
        ep = self.episode(session_id)
        return {
            "session_id": session_id,
            "verified": ep["verified"],
            "pairs": len(pairs),
            "path": str(out),
            "tokens": sum(estimate_tokens(json.dumps(p, ensure_ascii=False)) for p in pairs),
            "synthetic_steps": ep["synthetic_steps"],
        }

    def stats(self) -> dict:
        sessions = self.sessions()
        verified = 0
        pairs = 0
        events = 0
        synthetic = 0
        for sid in sessions:
            ep = self.episode(sid)
            events += ep["events"]
            verified += bool(ep["verified"])
            pairs += len(self.pairs(sid))
            synthetic += ep["synthetic_steps"]
        return {
            "root": str(self.root),
            "sessions": len(sessions),
            "verified_sessions": verified,
            "pairs": pairs,
            "events": events,
            "synthetic_steps": synthetic,
        }


def dataset_files(root: Path) -> Iterable[Path]:
    return sorted((Path(root) / "captures").glob("*.pairs.jsonl"))
