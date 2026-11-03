"""Release gate contract: one command must cover every gate the project claims."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("release_check", ROOT / "scripts" / "release-check.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


release_check = _load()


def test_every_claimed_gate_is_in_the_checklist():
    names = " ".join(c.name for c in release_check.CHECKS).lower()
    for required in ["version", "unit tests", "coverage", "web", "secret", "sbom", "openapi",
                     "client", "self-test", "accessibility", "pwa", "container"]:
        assert required in names, f"release-check is missing a gate: {required}"


def test_no_gate_is_a_no_op():
    for check in release_check.CHECKS:
        assert check.command, f"{check.name} has no command"


def test_fast_mode_skips_browser_and_container_but_never_tests():
    import os

    # RELEASE_CHECK_ACTIVE marks "already inside a release check": the tests gate must skip
    # instead of recursing, and the cheap gates must still run
    env = {**os.environ, "RELEASE_CHECK_ACTIVE": "1"}
    fast = subprocess.run([sys.executable, str(ROOT / "scripts" / "release-check.py"), "--fast", "--json"],
                          cwd=ROOT, capture_output=True, text=True, env=env,
                          encoding="utf-8", errors="replace", timeout=900)
    report = json.loads(fast.stdout[fast.stdout.index("["):])
    by_name = {r["name"]: r["status"] for r in report}
    assert by_name["unit tests + coverage"] == "SKIP"
    assert by_name["accessibility audit"] == "SKIP"
    assert by_name["container smoke"] == "SKIP"
    assert by_name["version consistency"] == "PASS"
    assert by_name["secret scan"] == "PASS"
    assert fast.returncode in (0, 1), "must exit with a verdict, not crash"


def test_failure_produces_a_clear_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(release_check, "CHECKS", [release_check.Check("always fails", [sys.executable, "-c", "raise SystemExit(3)"])])
    assert release_check.main(["--json"]) == 1
    result = release_check.run(release_check.CHECKS[0])
    assert result.status == "FAIL" and result.ok is False
