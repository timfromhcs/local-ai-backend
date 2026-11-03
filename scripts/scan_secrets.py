"""Secret scan with no third-party action: deterministic, fast, and tested.

Scans files tracked by git (optionally a directory) for the credential shapes that have
actually appeared in this project's history: GitHub tokens, bearer headers, private keys,
API keys and env-style assignments.

    python scripts/scan_secrets.py            -> exits 1 on a finding
    python scripts/scan_secrets.py --root web
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("github-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("openai-key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("huggingface", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("aws-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("slack", re.compile(r"\bxox[abpr]-[A-Za-z0-9\-]{20,}\b")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer", re.compile(r"(?i)\bbearer\s+(?!\[REDACTED)[A-Za-z0-9._\-~+/]{24,}")),
    ("bd-token", re.compile(r"\bbd_(?:live|test)_[A-Za-z0-9]{16,}\b")),
]

# placeholders and documentation examples that must not fail the build
ALLOWLIST = re.compile(
    r"(\[REDACTED|REDACTED:|ghp_\*+|sk-\*+|your[-_]?token|example|placeholder|<token>|xxx+|\.\.\.|secret-scan:\s*allow)",
    re.I,
)

SKIP_SUFFIXES = {".gguf", ".onnx", ".safetensors", ".zip", ".png", ".jpg", ".webp", ".ico", ".db",
                 ".safetensors.index.json", ".pt", ".bin"}
SKIP_DIRS = {".git", ".venv", ".venv-train", "node_modules", "bin", "cache", "dist", "build",
             "__pycache__", "rsi", "training", ".superpowers", ".pytest_cache", ".playwright-mcp"}


def tracked_files(root: Path) -> list[Path]:
    try:
        out = subprocess.check_output(["git", "ls-files", "-z"], cwd=root, text=True)
        files = [root / p for p in out.split("\0") if p]
    except Exception:
        files = [p for p in root.rglob("*") if p.is_file()]
    keep = []
    for f in files:
        if not f.is_file():
            continue
        parts = set(f.relative_to(root).parts)
        if parts & SKIP_DIRS:
            continue
        if f.suffix.lower() in SKIP_SUFFIXES or f.name.endswith(".safetensors"):
            continue
        keep.append(f)
    return keep


def scan(root: Path) -> list[dict]:
    findings = []
    for path in tracked_files(root):
        try:
            text = path.read_text(encoding="utf8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if ALLOWLIST.search(line):
                continue
            for name, pattern in PATTERNS:
                if pattern.search(line):
                    findings.append({"file": str(path.relative_to(root)), "line": lineno, "kind": name})
    return findings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Scan tracked files for credential patterns")
    p.add_argument("--root", default=".")
    args = p.parse_args(argv)
    root = Path(args.root).resolve()
    findings = scan(root)
    for f in findings:
        print(f"FAIL {f['file']}:{f['line']} {f['kind']}")
    print(f"secret scan: {'FAIL' if findings else 'PASS'} ({len(findings)} finding(s))")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
