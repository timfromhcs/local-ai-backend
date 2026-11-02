"""Secret scrubber. Runs on every string that can reach a training file."""
from __future__ import annotations

import re

_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("GITHUB_TOKEN", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("GITHUB_TOKEN", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("API_KEY", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{20,}\b")),
    ("API_KEY", re.compile(r"\b(?:bd_live|bd_test)_[A-Za-z0-9]{16,}\b")),
    ("API_KEY", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("API_KEY", re.compile(r"\bxox[abpr]-[A-Za-z0-9\-]{20,}\b")),
    ("AWS_KEY", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("BEARER", re.compile(r"(?i)\b(bearer)\s+(?!\[REDACTED)[A-Za-z0-9._\-~+/]{8,}=*")),
    ("ENV_SECRET", re.compile(r"(?im)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|PRIVATE_?KEY)[A-Z0-9_]*)\s*[=:]\s*(?!\[REDACTED)(\"[^\"\n]{4,}\"|'[^'\n]{4,}'|[^\s\"']{4,})")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
]


def scrub_text(text: str) -> str:
    out = text
    for label, rx in _RULES:
        if label == "BEARER":
            out = rx.sub(lambda m: f"{m.group(1)} [REDACTED:{label}]", out)
        elif label == "ENV_SECRET":
            out = rx.sub(lambda m: f"{m.group(1)}=[REDACTED:{label}]", out)
        else:
            out = rx.sub(f"[REDACTED:{label}]", out)
    return out


def contains_secret(text: str) -> bool:
    """True if any rule still matches un-redacted text (after scrub_text this must be False)."""
    return any(rx.search(text) for _, rx in _RULES)
