"""RSI cycle contract: refuse to train below the threshold, and record the promotion decision."""
import json

import pytest

from rsi_lab.capture import CaptureLog
from rsi_lab.cycle import CycleConfig, plan_cycle, read_state, run_cycle
from rsi_lab.train import build_dataset


def _pairs(tmp_path, count, verified=True):
    log = CaptureLog(tmp_path)
    for i in range(count):
        sid = f"s{i}"
        log.append(sid, {"type": "user", "content": f"task {i}"})
        log.append(sid, {"type": "assistant", "content": "working"})
        log.append(sid, {"type": "tool", "name": "bash", "input": {"cmd": "pytest"}, "result": "ok", "ok": True})
        log.append(sid, {"type": "verification", "ok": verified, "detail": "tests pass"})
        log.finalize(sid)
    return str(tmp_path / "captures" / "*.pairs.jsonl")


def test_plan_refuses_below_threshold(tmp_path):
    pattern = _pairs(tmp_path, 2)
    cfg = CycleConfig(pairs_glob=pattern, state_path=str(tmp_path / "state.json"), min_pairs=5)
    plan = plan_cycle(build_dataset(pattern), cfg)
    assert plan["ready"] is False
    assert plan["missing"] == 3
    assert "only 2 verified pairs" in plan["reason"]


def test_plan_is_ready_at_the_threshold(tmp_path):
    pattern = _pairs(tmp_path, 5)
    cfg = CycleConfig(pairs_glob=pattern, state_path=str(tmp_path / "state.json"), min_pairs=5)
    plan = plan_cycle(build_dataset(pattern), cfg)
    assert plan["ready"] is True and plan["missing"] == 0
    assert plan["plan"]["rows"] == 5


def test_cycle_skips_and_records_when_not_ready(tmp_path):
    pattern = _pairs(tmp_path, 1)
    cfg = CycleConfig(pairs_glob=pattern, state_path=str(tmp_path / "state.json"), min_pairs=10)
    result = run_cycle(cfg)
    assert result["entry"]["outcome"] == "skipped"
    state = read_state(tmp_path / "state.json")
    assert state["runs"][-1]["outcome"] == "skipped"
    assert state["champion"] is None


def test_cycle_promotes_a_better_candidate(tmp_path):
    pattern = _pairs(tmp_path, 3)
    cfg = CycleConfig(pairs_glob=pattern, state_path=str(tmp_path / "state.json"), min_pairs=3)

    trained = {"run": "rsi/runs/fake"}
    bench_scores = {"bench": 0.75, "tool_call": 1.0, "routing": 0.8}
    result = run_cycle(cfg, trainer=lambda ds, c: trained, bench=lambda m: bench_scores)

    assert result["entry"]["outcome"] == "promoted"
    state = read_state(tmp_path / "state.json")
    assert state["champion"] == bench_scores
    assert state["champion_run"] == "rsi/runs/fake"


def test_cycle_rejects_a_regression_and_keeps_the_champion(tmp_path):
    pattern = _pairs(tmp_path, 3)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"version": 1, "champion": {"bench": 0.8, "tool_call": 1.0}, "runs": []}), encoding="utf8")
    cfg = CycleConfig(pairs_glob=pattern, state_path=str(state_path), min_pairs=3)

    result = run_cycle(cfg, trainer=lambda ds, c: {"run": "x"}, bench=lambda m: {"bench": 0.5, "tool_call": 1.0})
    assert result["entry"]["outcome"] == "rejected"
    assert read_state(state_path)["champion"] == {"bench": 0.8, "tool_call": 1.0}


def test_cycle_records_a_missing_trainer_instead_of_pretending(tmp_path):
    pattern = _pairs(tmp_path, 3)
    cfg = CycleConfig(pairs_glob=pattern, state_path=str(tmp_path / "state.json"), min_pairs=3)
    result = run_cycle(cfg)
    assert result["entry"]["outcome"] == "no_trainer"
    assert "train-rsi.ps1" in result["entry"]["reason"]


def test_cycle_without_bench_does_not_touch_the_champion(tmp_path):
    pattern = _pairs(tmp_path, 3)
    state_path = tmp_path / "state.json"
    cfg = CycleConfig(pairs_glob=pattern, state_path=str(state_path), min_pairs=3)
    result = run_cycle(cfg, trainer=lambda ds, c: {"run": "x"})
    assert result["entry"]["outcome"] == "trained_not_benchmarked"
    assert read_state(state_path)["champion"] is None


def test_state_file_is_append_only_history(tmp_path):
    pattern = _pairs(tmp_path, 3)
    cfg = CycleConfig(pairs_glob=pattern, state_path=str(tmp_path / "state.json"), min_pairs=3)
    run_cycle(cfg)
    run_cycle(cfg)
    state = read_state(tmp_path / "state.json")
    assert len(state["runs"]) == 2


def test_cli_reports_the_plan_without_training(tmp_path, capsys):
    pattern = _pairs(tmp_path, 2)
    from rsi_lab.cycle import main

    assert main(["--pairs", pattern, "--state", str(tmp_path / "s.json"), "--min-pairs", "10"]) == 0
    out = capsys.readouterr().out
    assert "only 2 verified pairs" in out
    assert "no training run performed" in out
