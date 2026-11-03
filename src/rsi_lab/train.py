"""RSI stage 2: turn captured pairs into a LoRA training plan, run it, and gate the result.

Two layers, on purpose:
* everything that can be computed without torch (dataset assembly, step math, token budget,
  promotion decision) lives here and is unit-tested;
* `train()` imports torch/peft lazily and raises TrainingUnavailable with a real message when
  the training venv is missing — a slow CPU job must never look like a bug.

Honest scope: this machine has no CUDA. A 1.5B LoRA run is hours of CPU work, so the default
command is `--plan`; `--run` is opt-in and checkpoints every N steps.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Sequence

from .clean import estimate_tokens
from .gate import Decision, decide


class TrainingUnavailable(RuntimeError):
    """torch/peft are not installed in this environment."""


@dataclass(frozen=True)
class TrainConfig:
    model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    rank: int = 8
    alpha: int = 16
    lr: float = 2e-4
    epochs: int = 1
    seq_len: int = 1024
    micro_batch: int = 1
    grad_accum: int = 4
    checkpoint_every: int = 20
    out_dir: str = "rsi/runs"

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.grad_accum


@dataclass
class Dataset:
    rows: list[dict]
    tokens: int = 0
    sessions: list[str] = field(default_factory=list)
    skipped: int = 0

    def stats(self) -> dict[str, Any]:
        return {
            "rows": len(self.rows),
            "tokens": self.tokens,
            "sessions": len(self.sessions),
            "skipped_files": self.skipped,
            "avg_tokens": round(self.tokens / len(self.rows), 1) if self.rows else 0,
        }


def build_dataset(pattern: str) -> Dataset:
    """Read *.pairs.jsonl files produced by rsi_lab.capture and keep only verified pairs."""
    rows: list[dict] = []
    tokens = 0
    sessions: list[str] = []
    skipped = 0
    for path in sorted(glob(pattern)):
        try:
            lines = Path(path).read_text(encoding="utf8").splitlines()
        except OSError:
            skipped += 1
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            messages = pair.get("messages") or []
            if not pair.get("meta", {}).get("verified") or not messages:
                skipped += 1
                continue
            rows.append({"messages": messages, "meta": pair.get("meta", {})})
            tokens += estimate_tokens(json.dumps(messages, ensure_ascii=False))
            sid = pair.get("meta", {}).get("session_id") or pair.get("session_id")
            if sid and sid not in sessions:
                sessions.append(sid)
    return Dataset(rows=rows, tokens=tokens, sessions=sessions, skipped=skipped)


def plan_run(dataset: Dataset, cfg: TrainConfig) -> dict[str, Any]:
    """Everything you can know before spending an hour of CPU."""
    if not dataset.rows:
        raise ValueError("dataset is empty: capture verified work first (rsi-capture-mcp)")
    # one optimizer step per effective batch of sequences
    steps_per_epoch = max(1, -(-len(dataset.rows) // cfg.effective_batch))
    total_steps = steps_per_epoch * cfg.epochs
    checkpoints = max(1, total_steps // max(1, cfg.checkpoint_every))
    return {
        "model": cfg.model,
        "rows": len(dataset.rows),
        "tokens": dataset.tokens,
        "seq_len": cfg.seq_len,
        "effective_batch": cfg.effective_batch,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "planned_checkpoints": checkpoints,
        "lora": {"rank": cfg.rank, "alpha": cfg.alpha, "lr": cfg.lr, "epochs": cfg.epochs},
        "note": "CPU LoRA on a 1.5B model is hours; use --run in the background with checkpoints",
    }


def promote(candidate: dict[str, float], champion: dict[str, float] | None,
            primary: str = "bench", never_drop: Sequence[str] = ("tool_call",)) -> Decision:
    """Thin wrapper so training and benchmarking share one promotion rule."""
    return decide(candidate, champion, primary, margin=0.01, tolerance=0.02, never_drop=list(never_drop))


def _require_training_stack() -> tuple[Any, Any, Any]:
    try:
        import torch  # noqa: F401
        from datasets import Dataset as HFDataset  # noqa: F401
        from peft import LoraConfig, get_peft_model  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise TrainingUnavailable(
            "training stack missing. Install it in the training venv:\n"
            "  python -m venv .venv-train\n"
            "  .venv-train\\Scripts\\pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  .venv-train\\Scripts\\pip install transformers datasets peft accelerate\n"
            f"original error: {exc}"
        ) from exc
    import torch
    from datasets import Dataset as HFDataset, load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    return (torch, (HFDataset, load_dataset), (LoraConfig, get_peft_model), (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments))


def train(dataset: Dataset, cfg: TrainConfig) -> dict[str, Any]:
    """Real LoRA run. Checkpoints every `checkpoint_every` steps, metrics written as JSON."""
    torch, hf, peft, tr = _require_training_stack()
    HFDataset = hf[0]
    LoraConfig, get_peft_model = peft
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments = tr

    out = Path(cfg.out_dir) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(batch):
        text = tokenizer.apply_chat_template(batch["messages"], tokenize=False)
        enc = tokenizer(text, truncation=True, max_length=cfg.seq_len, padding="max_length")
        enc["labels"] = enc["input_ids"].copy()
        return enc

    hf_ds = HFDataset.from_list([{"messages": r["messages"]} for r in dataset.rows]).map(encode, remove_columns=["messages"])
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    args = TrainingArguments(
        output_dir=str(out), num_train_epochs=cfg.epochs, per_device_train_batch_size=cfg.micro_batch,
        gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.lr, logging_steps=5,
        save_steps=cfg.checkpoint_every, save_total_limit=3, report_to=[], use_cpu=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=hf_ds)
    result = trainer.train()
    adapter = out / "adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    metrics = {"run": str(out), "adapter": str(adapter), **asdict(cfg), **result.metrics}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")
    return metrics


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Plan or run LoRA training on captured RSI pairs")
    p.add_argument("--pairs", default="rsi/captures/*.pairs.jsonl")
    p.add_argument("--model", default=TrainConfig.model)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-rows", type=int, default=0, help="cap rows for a smoke run (0 = all)")
    p.add_argument("--run", action="store_true", help="actually train (CPU, hours)")
    args = p.parse_args(list(argv) if argv is not None else None)

    dataset = build_dataset(args.pairs)
    if args.max_rows:
        dataset = type(dataset)(rows=dataset.rows[: args.max_rows], tokens=dataset.tokens,
                                sessions=dataset.sessions, skipped=dataset.skipped)
    cfg = TrainConfig(model=args.model, epochs=args.epochs, rank=args.rank, lr=args.lr,
                      checkpoint_every=args.checkpoint_every, seq_len=args.seq_len)
    plan = plan_run(dataset, cfg)
    print(json.dumps({"dataset": dataset.stats(), "plan": plan}, indent=2))
    if not args.run:
        print("\nplan only. Re-run with --run to train (CPU, hours).")
        return 0
    metrics = train(dataset, cfg)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
