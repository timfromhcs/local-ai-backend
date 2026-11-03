"""Tell a released binary whether a newer version exists — without phoning home silently.

`--check-update` is the only command that touches the network, it is explicit, it sends nothing
but a plain GET to the public GitHub API, and it fails soft: an offline machine gets a clear
message, not a traceback.
"""
from __future__ import annotations

import json
import re
from typing import Callable

RELEASES_URL = "https://api.github.com/repos/timfromhcs/local-ai-backend/releases/latest"
_VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")


def parse_version(tag: str) -> tuple[int, int, int] | None:
    match = _VERSION.match(tag.strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def is_newer(candidate: str, current: str) -> bool:
    a, b = parse_version(candidate), parse_version(current)
    if a is None or b is None:
        return False
    return a > b


def check(current: str, fetch: Callable[[str], bytes] | None = None, url: str = RELEASES_URL) -> dict:
    """Return {status: up_to_date|update_available|unavailable, ...}. Never raises for network."""
    fetch = fetch or (lambda target: __import__("urllib.request").request.urlopen(target, timeout=10).read())
    try:
        payload = json.loads(fetch(url).decode("utf8"))
    except Exception as exc:
        return {"status": "unavailable", "current": current, "reason": str(exc)[:160],
                "hint": "offline or rate-limited; this check is optional and never automatic"}
    latest = str(payload.get("tag_name") or "")
    if parse_version(latest) is None:
        return {"status": "unavailable", "current": current, "reason": f"unparseable tag {latest!r}"}
    newer = is_newer(latest, current)
    return {
        "status": "update_available" if newer else "up_to_date",
        "current": current,
        "latest": latest.lstrip("v"),
        "url": payload.get("html_url"),
        "assets": [a.get("name") for a in payload.get("assets", []) if a.get("name")],
    }


def render(report: dict) -> str:
    status = report["status"]
    if status == "update_available":
        return (f"update available: {report['current']} -> {report['latest']}\n"
                f"  {report.get('url') or 'see the releases page'}\n"
                f"  assets: {', '.join(report.get('assets') or []) or '(none listed)'}")
    if status == "up_to_date":
        return f"up to date ({report['current']})"
    return f"could not check for updates: {report.get('reason', 'unknown')}\n  {report.get('hint', '')}".rstrip()


def main() -> int:
    from . import __version__

    print(render(check(__version__)))
    return 0
