"""Capture contract: append-only, scrubbed, verified-only, synthetic marked."""
import json

import pytest

from rsi_lab.capture import CaptureError, CaptureLog, dataset_files


@pytest.fixture()
def log(tmp_path):
    return CaptureLog(tmp_path / "rsi")


def _session(log, sid="s1", verified=True):
    log.append(sid, {"type": "user", "content": "fix the failing test in auth.py"})
    log.append(sid, {"type": "assistant", "content": "I will run the tests first."})
    log.append(sid, {"type": "tool", "name": "bash", "input": {"cmd": "pytest -q"},
                     "result": "1 failed", "ok": False})
    log.append(sid, {"type": "assistant", "content": "The fixture is missing a token."})
    log.append(sid, {"type": "tool", "name": "edit_file", "input": {"path": "auth.py"},
                     "result": "patched", "ok": True})
    log.append(sid, {"type": "tool", "name": "bash", "input": {"cmd": "pytest -q"},
                     "result": "5 passed", "ok": True})
    log.append(sid, {"type": "verification", "ok": verified, "detail": "pytest 5 passed"})
    return sid


def test_events_are_append_only_and_ordered(log):
    sid = _session(log)
    events = log.events(sid)
    assert [e["type"] for e in events][:2] == ["user", "assistant"]
    assert all("ts" in e for e in events)
    assert log.sessions() == [sid]


def test_secrets_are_scrubbed_before_disk(log, tmp_path):
    log.append("s2", {"type": "user", "content": "my token is ghp_abcdefghijklmnopqrstuvwxyz0123"})  # secret-scan: allow (fake fixture)
    raw = (tmp_path / "rsi" / "captures" / "s2.jsonl").read_text(encoding="utf8")
    assert "ghp_" not in raw  # secret-scan: allow (fake fixture)
    assert "[REDACTED:GITHUB_TOKEN]" in raw


def test_nested_tool_input_is_scrubbed_too(log, tmp_path):
    log.append("s3", {"type": "tool", "name": "http", "input": {"headers": {"authorization": "Bearer sk-abcdefghijklmnopqrstuvwx"}}})  # secret-scan: allow (fake fixture)
    raw = (tmp_path / "rsi" / "captures" / "s3.jsonl").read_text(encoding="utf8")
    assert "sk-abc" not in raw  # secret-scan: allow (fake fixture)


def test_episode_folds_turns_and_verification(log):
    sid = _session(log)
    ep = log.episode(sid)
    assert ep["prompt"].startswith("fix the failing test")
    assert ep["verified"] is True
    assert ep["verification_detail"] == "pytest 5 passed"
    assert len(ep["turns"]) >= 2
    calls = [c for t in ep["turns"] for c in t["tool_calls"]]
    assert [c["name"] for c in calls] == ["bash", "edit_file", "bash"]
    assert calls[0]["is_error"] is True and calls[-1]["is_error"] is False


def test_only_verified_sessions_produce_pairs(log):
    good = _session(log, "good", verified=True)
    bad = _session(log, "bad", verified=False)
    assert len(log.pairs(good)) == 1
    assert log.pairs(bad) == []
    assert log.pairs(bad, require_verified=False), "explicit opt-in must still work"


def test_pair_shape_is_chat_compatible(log):
    pair = log.pairs(_session(log))[0]
    assert pair["messages"][0]["role"] == "user"
    assert any(m["role"] == "assistant" for m in pair["messages"])
    assert any(m["role"] == "tool" for m in pair["messages"])
    assert pair["meta"]["verified"] is True
    assert pair["meta"]["synthetic_steps"] == 0


def test_synthetic_steps_are_counted_not_hidden(log):
    log.append("syn", {"type": "user", "content": "explain the scheduler"})
    log.append("syn", {"type": "assistant", "content": "step one", "synthetic": True})
    log.append("syn", {"type": "verification", "ok": True, "detail": "reviewed"})
    ep = log.episode("syn")
    assert ep["synthetic_steps"] == 1
    assert log.pairs("syn")[0]["meta"]["synthetic_steps"] == 1


def test_retries_are_collapsed_by_clean_episode(log):
    """A failed call followed by a successful one of the same tool is dropped."""
    pair = log.pairs(_session(log))[0]
    bash_calls = [m for m in pair["messages"] if m.get("tool_calls")
                  and any(c["function"]["name"] == "bash" for c in m["tool_calls"])]
    assert all("1 failed" not in (m.get("content") or "") for m in pair["messages"])


def test_invalid_events_are_rejected(log):
    with pytest.raises(CaptureError):
        log.append("x", {"type": "nope"})
    with pytest.raises(CaptureError):
        log.append("x", {"type": "user", "content": "   "})
    with pytest.raises(CaptureError):
        log.append("x", {"type": "tool", "result": "no name"})
    with pytest.raises(CaptureError):
        log.append("x", {"type": "verification"})
    with pytest.raises(CaptureError):
        log.append("!!!", {"type": "user", "content": "bad id"})


def test_finalize_writes_pairs_file_and_stats(log, tmp_path):
    sid = _session(log)
    result = log.finalize(sid)
    assert result["pairs"] == 1 and result["verified"] is True
    assert (tmp_path / "rsi" / "captures" / f"{sid}.pairs.jsonl").is_file()
    assert "dataset_files" in globals() and list(dataset_files(tmp_path / "rsi"))

    stats = log.stats()
    assert stats["sessions"] == 1
    assert stats["verified_sessions"] == 1
    assert stats["pairs"] == 1
    assert stats["events"] == 7


def test_empty_session_is_an_error(log):
    with pytest.raises(CaptureError):
        log.episode("does-not-exist")
    assert json.loads(json.dumps(log.stats()))["sessions"] == 0
