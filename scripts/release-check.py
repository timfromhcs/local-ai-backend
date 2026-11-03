"""One command that proves the release is shippable.

Every gate this repo has is invoked here in a fixed order and summarised as PASS/FAIL/SKIP.
Exit code 1 if anything fails, so a tag is never cut by memory.

    python scripts/release-check.py            # full (needs Playwright for the browser gates)
    python scripts/release-check.py --fast     # skip browser + container checks
    python scripts/release-check.py --json     # machine-readable report
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


@dataclass
class Result:
    name: str
    status: str = "SKIP"
    seconds: float = 0.0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {"PASS", "SKIP"}


@dataclass
class Check:
    name: str
    command: list[str]
    needs_browser: bool = False
    needs_container: bool = False
    detail: str = ""


CHECKS: list[Check] = [
    Check("version consistency", [PY, "-c",
          "import localai_gateway, tomllib, json; "
          "py=tomllib.load(open('pyproject.toml','rb'))['project']['version']; "
          "npm=json.load(open('packages/npm/package.json'))['version']; "
          "assert py==localai_gateway.__version__==npm, (py, localai_gateway.__version__, npm); "
          "print(py)"]),
    Check("unit tests + coverage", [PY, "-m", "pytest", "-m", "not real", "-q",
          "--cov=localai_gateway", "--cov=rsi_lab", "--cov-report=term", "--cov-fail-under=85"]),
    Check("web static gate", ["node", "scripts/check-web.mjs"]),
    Check("secret scan", [PY, "scripts/scan_secrets.py"]),
    Check("sbom", [PY, "scripts/sbom.py", "--out", "dist/sbom.cdx.json"]),
    Check("openapi snapshot", [PY, "scripts/snapshot-openapi.py", "--check"]),
    Check("generated client", [PY, "scripts/gen-client.py", "--check"]),
    Check("self-test", [PY, "-m", "localai_gateway", "--self-test"]),
    Check("accessibility audit", ["node", "scripts/audit-web.mjs"], needs_browser=True),
    Check("pwa offline", ["node", "scripts/verify-pwa-offline.mjs"], needs_browser=True),
    Check("local storage", ["node", "scripts/verify-storage.mjs"], needs_browser=True),
    Check("container smoke", ["docker", "build", "-f", "api/Dockerfile", "-t", "localai-release-check", "."],
          needs_container=True, detail="docker build"),
]


def run(check: Check, timeout: float = 1800.0) -> Result:
    started = time.perf_counter()
    env = {**os.environ, "RELEASE_CHECK_ACTIVE": "1"}  # stops the tests gate recursing into itself
    try:
        proc = subprocess.run(check.command, cwd=ROOT, capture_output=True, text=True, env=env,
                              encoding="utf-8", errors="replace", timeout=timeout)
        elapsed = time.perf_counter() - started
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()
        detail = tail[-1][:120] if tail else ""
        return Result(check.name, "PASS" if proc.returncode == 0 else "FAIL", elapsed, detail)
    except FileNotFoundError:
        return Result(check.name, "SKIP", time.perf_counter() - started, "command not installed")
    except subprocess.TimeoutExpired:
        return Result(check.name, "FAIL", time.perf_counter() - started, f"timeout after {timeout:.0f}s")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run every release gate and summarise")
    p.add_argument("--fast", action="store_true", help="skip browser and container gates")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)

    results: list[Result] = []
    nested = os.environ.get("RELEASE_CHECK_ACTIVE") == "1"
    for check in CHECKS:
        if args.fast and (check.needs_browser or check.needs_container):
            results.append(Result(check.name, "SKIP", 0.0, "fast mode"))
            continue
        if nested and check.name.startswith("unit tests"):
            results.append(Result(check.name, "SKIP", 0.0, "already inside a release check"))
            continue
        print(f"... {check.name}", flush=True)
        results.append(run(check))

    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
    else:
        print()
        width = max(len(r.name) for r in results)
        for r in results:
            print(f"{r.status:4} {r.name:<{width}} {r.seconds:6.1f}s  {r.detail}")
        failed = [r for r in results if not r.ok]
        print(f"\n{len(results) - len(failed)}/{len(results)} gates passed")
        if failed:
            print("release NOT ready: " + ", ".join(r.name for r in failed))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
