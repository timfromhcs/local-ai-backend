"""Precompute the OpenCode prompt prefix once, keep the llama-server slot state on SSD, restore it at every start."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

log = logging.getLogger("localai_gateway.warmup")

MIN_TOKENS = 16
MARGIN_BYTES = 64


@dataclass
class WarmResult:
    state: str          # "restored" | "computed" | "failed"
    tokens: int = 0
    detail: str = ""
    seconds: float = 0.0


def _system_text(body: dict) -> str:
    for m in body.get("messages", []):
        if m.get("role") == "system":
            c = m.get("content")
            return c if isinstance(c, str) else "".join(b.get("text", "") for b in c or [] if isinstance(b, dict))
    return ""


def cut_point(pieces: list[bytes], text: bytes, limit_bytes: int) -> int | None:
    """Number of leading tokens that end within `limit_bytes` of `text` (None if the pieces do not rebuild the text)."""
    if b"".join(pieces) != text:
        return None
    total, n = 0, 0
    for p in pieces:
        if total + len(p) > limit_bytes:
            break
        total += len(p)
        n += 1
    return n


_ENV = re.compile(r"<env>.*?</env>\n?", re.S)


def normalize_for_cache(body: dict) -> dict:
    """Move the volatile <env>..</env> block (cwd etc.) from the system prompt to the first user message.

    Chat templates render tool schemas right after the system text, so a volatile block inside the system prompt
    would invalidate the cache for the whole tool section too. Idempotent; never mutates the input."""
    out = json.loads(json.dumps(body))
    msgs = out.get("messages", [])
    sys_i = next((i for i, m in enumerate(msgs) if m.get("role") == "system"), None)
    usr_i = next((i for i, m in enumerate(msgs) if m.get("role") == "user"), None)
    if sys_i is None or usr_i is None:
        return out
    sm, block = msgs[sys_i], None
    if isinstance(sm.get("content"), str):
        m = _ENV.search(sm["content"])
        if m:
            block, sm["content"] = m.group(0), (sm["content"][:m.start()] + sm["content"][m.end():]).rstrip() + "\n"
    elif isinstance(sm.get("content"), list):
        for part in sm["content"]:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                m = _ENV.search(part["text"])
                if m:
                    block, part["text"] = m.group(0), (part["text"][:m.start()] + part["text"][m.end():]).rstrip() + "\n"
                    break
    if block is None:
        return out
    um = msgs[usr_i]
    if isinstance(um.get("content"), str):
        um["content"] = block.rstrip() + "\n" + um["content"]
    elif isinstance(um.get("content"), list):
        um["content"].insert(0, {"type": "text", "text": block.rstrip() + "\n"})
    return out


class Warmer:
    def __init__(self, directory: Path, client: httpx.AsyncClient) -> None:
        self.dir, self.client = Path(directory), client

    @staticmethod
    def prefix_key(body: dict) -> str:
        stable = _ENV.sub("", _system_text(body))
        return hashlib.sha1(json.dumps([body.get("tools") or [], stable], sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def filename(self, model_id: str, key: str) -> str:
        return f"{re.sub(r'[^A-Za-z0-9_.-]', '_', model_id)}-{key[:16]}.bin"

    def known(self, model_id: str, key: str) -> bool:
        return (self.dir / self.filename(model_id, key)).exists()

    async def _slot(self, base_url: str, action: str, model_id: str, key: str) -> bool:
        try:
            r = await self.client.post(f"{base_url.rstrip('/')}/slots/0", params={"action": action},
                                       json={"filename": self.filename(model_id, key)}, timeout=600)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def restore(self, base_url: str, model_id: str, key: str) -> bool:
        return await self._slot(base_url, "restore", model_id, key)

    async def save(self, base_url: str, model_id: str, key: str) -> bool:
        return await self._slot(base_url, "save", model_id, key)

    async def warm(self, base_url: str, model_id: str, prefix_body: dict) -> WarmResult:
        t0, key, base = time.monotonic(), self.prefix_key(prefix_body), base_url.rstrip("/")
        done = lambda state, tokens=0, detail="": WarmResult(state, tokens, detail, time.monotonic() - t0)  # noqa: E731
        if self.known(model_id, key) and await self.restore(base, model_id, key):
            return done("restored")
        try:
            body = normalize_for_cache(prefix_body)
            rendered: list[str] = []
            for probe in ("\u0001A", "\u0001B"):  # two different user turns -> their common prefix is the stable part
                variant = json.loads(json.dumps({k: v for k, v in body.items() if k in ("messages", "tools")}))
                u = next(m for m in reversed(variant["messages"]) if m.get("role") == "user")
                if isinstance(u["content"], str):
                    u["content"] = u["content"] + probe
                else:
                    u["content"] = u["content"] + [{"type": "text", "text": probe}]
                rendered.append((await self.client.post(f"{base}/apply-template", json=variant, timeout=60)).json()["prompt"])
            a_b, b_b = rendered[0].encode(), rendered[1].encode()
            lcp = 0
            while lcp < min(len(a_b), len(b_b)) and a_b[lcp] == b_b[lcp]:
                lcp += 1
            tok = (await self.client.post(f"{base}/tokenize", timeout=120, json={
                "content": rendered[0], "add_special": False, "parse_special": True, "with_pieces": True})).json()["tokens"]
            pieces = [bytes(t["piece"]) if isinstance(t["piece"], list) else t["piece"].encode() for t in tok]
            n = cut_point(pieces, a_b, lcp - MARGIN_BYTES)
            if n is None or n < MIN_TOKENS:
                return done("failed", 0, "no stable cut point")
            r = await self.client.post(f"{base}/completion", timeout=7200, json={
                "prompt": [t["id"] for t in tok][:n], "n_predict": 0, "cache_prompt": True, "id_slot": 0, "temperature": 0})
            if r.status_code != 200:
                return done("failed", 0, f"prefill HTTP {r.status_code}")
            saved = await self.save(base, model_id, key)
            return done("computed", n, "" if saved else "computed but slot save failed (is --slot-save-path set?)")
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            return done("failed", 0, repr(exc))


def make_hook(warmer: Warmer, load_prefix_body: Callable[[], dict | None]) -> Callable[[Any, Any], Awaitable[None]]:
    """Scheduler `on_started` hook: for models flagged warm_prefix, restore (or compute once + save) the OpenCode prefix."""
    async def hook(spec: Any, endpoint: Any) -> None:
        if not getattr(spec, "warm_prefix", False):
            return
        body = load_prefix_body()
        if body is None:
            log.warning("no captured OpenCode prefix; run scripts/capture_opencode_prefix.py")
            return
        res = await warmer.warm(endpoint.base_url, spec.id, body)
        log.info("prefix warm-up %s: %s tokens=%s %.1fs %s", spec.id, res.state, res.tokens, res.seconds, res.detail)
    return hook
