"""RSI stage 2 contract: dataset from verified pairs, plan math, promotion gate, honest failure."""
import json

import pytest

from rsi_lab.capture import CaptureLog
from rsi_lab.train import TrainConfig, TrainingUnavailable, build_dataset, plan_run, promote, train


def _write_pairs(tmp_path, verified=True, sessions=("a",)):
    outdir = tmp_path / "captures"
    outdir.mkdir(parents=True, exist_ok=True)
    for sid in sessions:
        log = CaptureLog(tmp_path)
        log.append(sid, {"type": "user", "content": "fix the failing test"})
        log.append(sid, {"type": "assistant", "content": "running pytest"})
        log.append(sid, {"type": "tool", "name": "bash", "input": {"cmd": "pytest -q"},
                         "result": "5 passed", "ok": True})
        log.append(sid, {"type": "verification", "ok": verified, "detail": "pytest 5 passed"})
        log.finalize(sid)
    return str(tmp_path / "captures" / "*.pairs.jsonl")


def test_dataset_only_includes_verified_pairs(tmp_path):
    pattern = _write_pairs(tmp_path, verified=True, sessions=("a", "b"))
    ds = build_dataset(pattern)
    assert len(ds.rows) == 2
    assert ds.tokens > 0
    assert set(ds.sessions) == {"a", "b"}
    assert ds.stats()["avg_tokens"] > 0


def test_dataset_skips_unverified_and_malformed_files(tmp_path):
    pattern = _write_pairs(tmp_path, verified=False)
    assert build_dataset(pattern).rows == []

    broken = tmp_path / "captures" / "broken.pairs.jsonl"
    broken.write_text("{not json}\n", encoding="utf8")
    ds = build_dataset(pattern)
    assert ds.rows == [] and ds.skipped >= 1


def test_plan_math_is_monotonic_and_explainable(tmp_path):
    ds = build_dataset(_write_pairs(tmp_path, sessions=("a", "b", "c")))
    cfg = TrainConfig(epochs=2, micro_batch=1, grad_accum=2, checkpoint_every=1)
    plan = plan_run(ds, cfg)
    assert plan["rows"] == 3
    assert plan["effective_batch"] == 2
    assert plan["steps_per_epoch"] == 2          # ceil(3 / 2)
    assert plan["total_steps"] == 4              # 2 epochs
    assert plan["planned_checkpoints"] == 4
    assert plan["lora"]["rank"] == cfg.rank
    assert "hours" in plan["note"]

    more = plan_run(ds, TrainConfig(epochs=4, micro_batch=1, grad_accum=2, checkpoint_every=1))
    assert more["total_steps"] > plan["total_steps"]


def test_dataset_deduplicates_repeated_pairs(tmp_path):
    pattern = _write_pairs(tmp_path, sessions=("a",))
    # a copied pairs file (e.g. re-exported) must not double-weight the episode
    src = tmp_path / "captures" / "a.pairs.jsonl"
    (tmp_path / "captures" / "a-copy.pairs.jsonl").write_text(src.read_text(encoding="utf8"), encoding="utf8")
    ds = build_dataset(pattern)
    assert len(ds.rows) == 1, "the same episode must not be trained twice"
    assert ds.skipped >= 1


def test_plan_rejects_empty_dataset(tmp_path):
    with pytest.raises(ValueError):
        plan_run(build_dataset(str(tmp_path / "nothing" / "*.pairs.jsonl")), TrainConfig())


def test_promotion_gate_rules():
    assert promote({"bench": 0.70, "tool_call": 1.0}, {"bench": 0.60, "tool_call": 1.0}).promote
    assert not promote({"bench": 0.605, "tool_call": 1.0}, {"bench": 0.60, "tool_call": 1.0}).promote, "below margin"
    assert promote({"bench": 0.61, "tool_call": 1.0}, {"bench": 0.60, "tool_call": 1.0}).promote, "exactly at margin promotes"
    assert not promote({"bench": 0.90, "tool_call": 0.70}, {"bench": 0.60, "tool_call": 1.0}).promote, "never_drop"
    assert promote({"bench": 0.70, "tool_call": 1.0}, None).promote, "no champion -> promote"


def test_training_without_the_stack_fails_with_instructions(tmp_path):
    """The failure must explain how to install the CPU stack, not traceback into torch."""
    ds = build_dataset(_write_pairs(tmp_path))
    try:
        train(ds, TrainConfig(out_dir=str(tmp_path / "runs")))
    except TrainingUnavailable as exc:
        assert ".venv-train" in str(exc)
        assert "torch" in str(exc)
    else:
        pytest.skip("training stack is installed here; the happy path is a multi-hour run")


def test_cli_plan_mode_prints_json(tmp_path, capsys):
    from rsi_lab.train import main

    pattern = _write_pairs(tmp_path)
    assert main(["--pairs", pattern]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"):out.rindex("}") + 1])
    assert payload["dataset"]["rows"] == 1
    assert payload["plan"]["rows"] == 1
