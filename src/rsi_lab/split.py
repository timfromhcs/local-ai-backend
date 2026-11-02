"""Deterministic train / held-out split by episode hash (stable when more episodes arrive); blacklist protects benchmarks."""
from __future__ import annotations

import hashlib


def _bucket(ep: dict) -> int:
    h = hashlib.sha1(f"{ep.get('session_id', '')}|{ep.get('prompt', '')}".encode()).hexdigest()
    return int(h[:8], 16) % 1000


def split_episodes(episodes: list[dict], held_frac: float = 0.2,
                   blacklist_prompts: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    bl = blacklist_prompts or set()
    train: list[dict] = []
    held: list[dict] = []
    for ep in episodes:
        if _bucket(ep) < int(held_frac * 1000):
            held.append(ep)
        elif ep.get("prompt") not in bl:
            train.append(ep)
    return train, held
