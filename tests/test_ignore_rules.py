"""Ignore rules must keep runtime data out without swallowing tests.

Regression: `.gitignore` had `rsi/`, which matched `tests/rsi/` too, so three test files were
never committed. CI then ran 265 - 29 tests and coverage silently fell from 88% to 79.5%.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _check_ignore(*paths: str) -> list[str]:
    result = subprocess.run(["git", "check-ignore", *paths], cwd=ROOT, capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_test_directories_are_never_ignored():
    ignored = _check_ignore("tests/rsi/test_capture.py", "tests/rsi/test_train.py", "tests/rsi/test_cycle.py")
    assert ignored == [], f"test files are ignored by .gitignore: {ignored}"


def test_runtime_rsi_data_is_still_ignored():
    ignored = _check_ignore("rsi/captures/x.pairs.jsonl", "rsi/state.json", "rsi/runs/2026/metrics.json")
    assert len(ignored) == 3, ignored


def test_every_test_file_on_disk_is_tracked_or_new():
    """A test that exists but is ignored is worse than no test: CI silently runs less."""
    on_disk = {p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").rglob("test_*.py")}
    ignored = set(_check_ignore(*sorted(on_disk)))
    assert ignored == set(), f"these test files are invisible to git: {sorted(ignored)}"
