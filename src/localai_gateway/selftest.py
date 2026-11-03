"""`--self-test`: prove the API works without opening a port or touching a model.

Runs entirely in-process against a temporary registry, so it is safe on a CI runner, on a
machine with no models, and inside the one-file EXE.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from .app import create_app
from .config import Settings
from .registry import ModelSpec


class _NoScheduler:
    def loaded(self) -> list[str]:
        return []

    async def ensure_loaded(self, model_id: str):
        raise KeyError(model_id)

    async def release(self, model_id: str) -> None:
        return None


CHECKS = (
    ("GET", "/healthz", 200),
    ("GET", "/v1/models", 200),
    ("GET", "/doctor", 200),
    ("GET", "/v1/placement", 200),
    ("GET", "/openapi.json", 200),
    ("POST", "/v1/chat/completions", 404),  # unknown model must be a structured 404
)


def run() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        settings = Settings(budget_mb=2048, bin_dir=root, model_dir=root)
        registry = {
            "selftest-llm": ModelSpec(id="selftest-llm", kind="llm", files=[], runtime="llama",
                                      est_mem_mb=512),
        }
        client = TestClient(create_app(settings, registry, _NoScheduler()))
        results = []
        ok = True
        for method, path, expected in CHECKS:
            kwargs = {"json": {"model": "nope", "messages": []}} if method == "POST" else {}
            try:
                response = client.request(method, path, **kwargs)
                status = response.status_code
                passed = status == expected
                detail = ""
                if not passed:
                    detail = str(response.text)[:120]
                elif path == "/v1/models":
                    data = response.json().get("data", [])
                    detail = f"{len(data)} model(s)"
                elif path == "/openapi.json":
                    detail = f"{len(response.json().get('paths', {}))} path(s)"
            except Exception as exc:  # a crash is a failed check, not a traceback
                status, passed, detail = 0, False, f"{type(exc).__name__}: {exc}"[:120]
            ok = ok and passed
            results.append({"method": method, "path": path, "status": status,
                            "expected": expected, "ok": passed, "detail": detail})
    return {"ok": ok, "checks": results}


def render(report: dict[str, Any]) -> str:
    lines = ["localai-gateway self-test"]
    for check in report["checks"]:
        mark = "ok  " if check["ok"] else "FAIL"
        lines.append(f"  {mark} {check['method']:<4} {check['path']:<26} {check['status']:>3} "
                     f"(expected {check['expected']}) {check['detail']}".rstrip())
    lines.append("self-test: " + ("PASS" if report["ok"] else "FAIL"))
    return "\n".join(lines)


def main() -> int:
    report = run()
    print(render(report))
    return 0 if report["ok"] else 1
