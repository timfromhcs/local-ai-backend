"""Exposure rules: the gateway is a localhost tool and must fail closed.

Two failure modes this module prevents:
* starting on a public interface because someone typed `--host 0.0.0.0` (there is no auth),
* accepting an unbounded request body and letting one client exhaust memory.
"""
from __future__ import annotations

import os
from typing import Mapping

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
DEFAULT_MAX_BODY_MB = 8


class PublicBindRefused(RuntimeError):
    """Raised when a non-loopback bind is requested without an explicit opt-in."""


def resolve_host(host: str, env: Mapping[str, str] | None = None) -> str:
    """Allow non-loopback binds only with LOCALAI_ALLOW_PUBLIC=1."""
    env = os.environ if env is None else env
    if host in LOCAL_HOSTS:
        return host
    if env.get("LOCALAI_ALLOW_PUBLIC") == "1":
        return host
    raise PublicBindRefused(
        f"refusing to bind {host}: the gateway has no authentication. "
        "Use 127.0.0.1 (default), put an authenticating proxy in front, or set "
        "LOCALAI_ALLOW_PUBLIC=1 if you really mean it."
    )


def max_body_bytes(env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    raw = env.get("LOCALAI_MAX_BODY_MB")
    try:
        mb = int(raw) if raw else DEFAULT_MAX_BODY_MB
    except ValueError:
        mb = DEFAULT_MAX_BODY_MB
    return max(1, mb) * 1024 * 1024
