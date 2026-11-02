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

MARKER = b"<env>"
MIN_TOKENS = 16


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


def cut_point(pieces: list[bytes], text: bytes, marker: bytes = MARKER, margin_bytes: int = 64) -> int | None:
    """Number of leading tokens that end at least `margin_bytes` before the volatile marker (token-exact prefix)."""
    if b"".join(pieces) != text:
        return None
    pos = text.find(marker)
    if pos < 0:
        return None
    limit, total, n = pos - margin_bytes, 0, 0
    for p in pieces:
        if total + len(p) > limit:
            break
        total += len(p)
        n += 1
    return n


class Warmer:
    def __init__(self, directory: Path, client: httpx.AsyncClient) -> None:
        self.dir, self.client = Path(directory), client

    @staticmethod
    def prefix_key(body: dict) -> str:
        sysc = _system_text(body)
        cut = sysc.find(MARKER.decode())
        stable = sysc if cut < 0 else sysc[:cut]
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
            tpl = {k: v for k, v in prefix_body.items() if k in ("messages", "tools")}
            rendered = (await self.client.post(f"{base}/apply-template", json=tpl, timeout=60)).json()["prompt"]
            tok = (await self.client.post(f"{base}/tokenize", timeout=120, json={
                "content": rendered, "add_special": False, "parse_special": True, "with_pieces": True})).json()["tokens"]
            pieces = [bytes(t["piece"]) if isinstance(t["piece"], list) else t["piece"].encode() for t in tok]
            n = cut_point(pieces, rendered.encode())
            if n is None or n < MIN_TOKENS:
                return done("failed", 0, "no stable cut point before <env>")
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
