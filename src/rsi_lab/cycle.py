"""The RSI cycle: dataset -> (train) -> bench -> promotion gate -> state file.

Everything that costs hours is injected, so the control flow is testable in milliseconds:

    trainer(dataset, cfg) -> metrics           # real CPU LoRA, or a fake in tests
    bench(model_id) -> {category: score}       # runs the fast suite against a live gateway

The cycle refuses to train below a data threshold. That number is not decoration: six episodes
cannot teach a model anything, and pretending otherwise is how "self-improvement" becomes a lie.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .train import Dataset, TrainConfig, build_dataset, plan_run, promote

STATE_VERSION = 1
DEFAULT_MIN_PAIRS = 20


@dataclass(frozen=True)
class CycleConfig:
    pairs_glob: str = "rsi/captures/*.pairs.jsonl"
    state_path: str = "rsi/state.json"
    min_pairs: int = DEFAULT_MIN_PAIRS
    bench_model: str = "duo"
    never_drop: tuple[str, ...] = ("tool_call",)


def read_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return {"version": STATE_VERSION, "champion": None, "runs": []}


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf8")


def plan_cycle(dataset: Dataset, cfg: CycleConfig | None = None) -> dict[str, Any]:
    """Decide whether a training run is justified, and what it would cost."""
    cfg = cfg or CycleConfig()
    ready = len(dataset.rows) >= cfg.min_pairs
    plan = plan_run(dataset, TrainConfig()) if dataset.rows else None
    return {
        "ready": ready,
        "pairs": len(dataset.rows),
        "min_pairs": cfg.min_pairs,
        "missing": max(0, cfg.min_pairs - len(dataset.rows)),
        "tokens": dataset.tokens,
        "plan": plan,
        "reason": (
            "threshold reached: training is worth the CPU hours"
            if ready
            else f"only {len(dataset.rows)} verified pairs; {cfg.min_pairs - len(dataset.rows)} more needed "
                 "before a run can move the benchmark"
        ),
    }


def run_cycle(
    cfg: CycleConfig | None = None,
    *,
    trainer: Callable[[Dataset, TrainConfig], dict[str, Any]] | None = None,
    bench: Callable[[str], dict[str, float]] | None = None,
    train_cfg: TrainConfig | None = None,
) -> dict[str, Any]:
    """One full pass. Returns the state it wrote; never raises for "not enough data"."""
    cfg = cfg or CycleConfig()
    train_cfg = train_cfg or TrainConfig()
    state_path = Path(cfg.state_path)
    state = read_state(state_path)

    dataset = build_dataset(cfg.pairs_glob)
    plan = plan_cycle(dataset, cfg)
    entry: dict[str, Any] = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pairs": plan["pairs"],
        "tokens": plan["tokens"],
        "ready": plan["ready"],
        "reason": plan["reason"],
    }

    if not plan["ready"]:
        entry["outcome"] = "skipped"
        state["runs"].append(entry)
        write_state(state_path, state)
        return {"state": state, "entry": entry}

    if trainer is None:
        entry["outcome"] = "no_trainer"
        entry["reason"] = "training stack not provided; pass trainer=... or use scripts/train-rsi.ps1"
        state["runs"].append(entry)
        write_state(state_path, state)
        return {"state": state, "entry": entry}

    metrics = trainer(dataset, train_cfg)
    entry["run"] = metrics.get("run")
    candidate = bench(cfg.bench_model) if bench else None
    if candidate is None:
        entry["outcome"] = "trained_not_benchmarked"
        state["runs"].append(entry)
        write_state(state_path, state)
        return {"state": state, "entry": entry}

    decision = promote(candidate, state.get("champion"), "bench", never_drop=list(cfg.never_drop))
    entry["candidate"] = candidate
    entry["promote"] = decision.promote
    entry["decision"] = list(decision.reasons)
    entry["outcome"] = "promoted" if decision.promote else "rejected"
    if decision.promote:
        state["champion"] = candidate
        state["champion_run"] = metrics.get("run")
    state["runs"].append(entry)
    write_state(state_path, state)
    return {"state": state, "entry": entry}


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Run one RSI cycle (plan, optional train, bench, gate)")
    p.add_argument("--pairs", default=CycleConfig.pairs_glob)
    p.add_argument("--state", default=CycleConfig.state_path)
    p.add_argument("--min-pairs", type=int, default=DEFAULT_MIN_PAIRS)
    p.add_argument("--gateway", default="http://127.0.0.1:8090")
    p.add_argument("--model", default="duo")
    p.add_argument("--train", action="store_true", help="actually train when the threshold is met")
    args = p.parse_args(list(argv) if argv is not None else None)

    cfg = CycleConfig(pairs_glob=args.pairs, state_path=args.state, min_pairs=args.min_pairs,
                      bench_model=args.model)
    dataset = build_dataset(cfg.pairs_glob)
    plan = plan_cycle(dataset, cfg)
    print(json.dumps({"plan": plan}, indent=2))
    if not plan["ready"] or not args.train:
        print("\nno training run performed" + (" (threshold not met)" if not plan["ready"] else " (pass --train)"))
        return 0

    from .train import train

    def bench(model: str) -> dict[str, float]:
        from localai_gateway.client import GatewayClient

        with GatewayClient(args.gateway) as gw:
            result = gw.bench({"model": model, "suite": "fast"})
        return {**result["by_category"], "bench": result["score"]}

    result = run_cycle(cfg, trainer=train, bench=bench)
    print(json.dumps(result["entry"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
