"""Benchmark suite for any OpenAI-compatible endpoint (local gateway by default).

Two suites:
  fast  routing, tool-call validity, strict JSON, one arithmetic task, honesty, concision (6 tasks)
  long  the fast suite plus multi-step coding and refactoring prompts, scored on structure

Scoring is deterministic Python, never an LLM judge, so a run is cheap and repeatable.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

ROUTES = ("fast", "duo", "bigout")


def score_route(answer: str, want: str) -> float:
    return 1.0 if re.search(rf"\b{want}\b", answer, re.I) else 0.0


def score_tool_call(answer: str) -> float:
    """Both a listing and a read call must appear, in a callable shape."""
    return 1.0 if re.search(r"(list_files?|list_dir)\s*\(", answer, re.I) and re.search(r"read_file\s*\(", answer, re.I) else 0.0


def score_json(answer: str) -> float:
    m = re.search(r"\{[\s\S]*\}", answer)
    if not m:
        return 0.0
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return 0.0
    return 1.0 if data.get("model") == "duo" and data.get("loaded") is True else 0.0


def score_number(answer: str, want: float) -> float:
    nums = re.findall(r"-?\d+(?:[.,]\d+)?", answer.replace(".", "").replace(",", ""))
    if not nums:
        return 0.0
    try:
        return 1.0 if abs(float(nums[0]) - want) <= max(1.0, want * 0.01) else 0.0
    except ValueError:
        return 0.0


def score_starts_no(answer: str) -> float:
    return 1.0 if re.match(r"\s*no\b", answer, re.I) else 0.0


def score_one_sentence(answer: str) -> float:
    sentences = [s for s in re.split(r"[.!?]", answer.strip()) if s.strip()]
    return 1.0 if len(sentences) <= 2 and len(answer) < 400 else 0.0


def score_patch_shape(answer: str) -> float:
    """Long suite: a coding answer should contain a file reference and a code or diff block."""
    has_file = bool(re.search(r"[\w./-]+\.(py|ts|tsx|js|json|yaml|yml|md)", answer))
    has_code = "```" in answer or bool(re.search(r"^\s*(diff --git|[-+]{1,3} )", answer, re.M))
    return 1.0 if has_file and has_code else 0.0


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    weight: float
    score: Callable[[str], float]
    category: str = "core"


@dataclass
class Run:
    model: str
    suite: str
    started: str
    results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def score(self) -> float:
        total = sum(r["weight"] for r in self.results) or 1.0
        return round(sum(r["score"] * r["weight"] for r in self.results) / total, 3)

    def by_category(self) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for r in self.results:
            out.setdefault(r["category"], []).append(r["score"])
        return {k: round(sum(v) / len(v), 3) for k, v in out.items()}


FAST: list[Task] = [
    Task("routing", "A task needs planning, implementation and verification. Answer with one word: fast, duo or bigout.", 1.0,
         lambda a: score_route(a, "duo"), "routing"),
    Task("routing-hard", "Migrate three services, keep every test green and prove rollback works. One word: fast, duo or bigout.", 1.0,
         lambda a: score_route(a, "bigout"), "routing"),
    Task("tool-call", "List the files in C:\\work, then read package.json. Reply with the tool calls only.", 1.0,
         lambda a: score_tool_call(a), "tools"),
    Task("json", 'Return only JSON: {"model":"duo","loaded":true,"ram_mb":9000}', 1.0,
         lambda a: score_json(a), "format"),
    Task("math", "A model decodes 5.9 tokens per second. How many tokens in 10 minutes? Number only.", 1.0,
         lambda a: score_number(a, 3540), "reasoning"),
    Task("honesty", "Can a browser run a 27B model at 50 tokens per second? Answer No or Yes, then one short reason.", 1.0,
         lambda a: score_starts_no(a), "honesty"),
    Task("concise", "Explain in exactly one sentence what a KV cache does.", 0.5,
         lambda a: score_one_sentence(a), "format"),
]

LONG: list[Task] = FAST + [
    Task("refactor", "Refactor a blocking request handler into async, name the file and show the diff.", 1.0,
         lambda a: score_patch_shape(a), "coding"),
    Task("test-first", "Write the failing pytest first, then the fix. Show the file and the code.", 1.0,
         lambda a: score_patch_shape(a), "coding"),
    Task("rollback", "Describe a rollback plan for a bad production deploy, with the command that triggers it.", 1.0,
         lambda a: 1.0 if re.search(r"vercel|promote|rollback|git revert", a, re.I) else 0.0, "ops"),
    Task("perf", "The gateway is slow on the first request but fast on the second. Give the cause and the fix.", 1.0,
         lambda a: 1.0 if re.search(r"cache|prefix|prefill|slot", a, re.I) else 0.0, "reasoning"),
]

SUITES = {"fast": FAST, "long": LONG}


def complete(client: httpx.Client, base_url: str, model: str, prompt: str, max_tokens: int = 400) -> tuple[str, float]:
    started = time.perf_counter()
    r = client.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens, "temperature": 0},
    )
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content, time.perf_counter() - started


def run_suite(base_url: str, model: str, suite: str = "fast", timeout: float = 300.0) -> Run:
    run = Run(model=model, suite=suite, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with httpx.Client(timeout=timeout) as client:
        for task in SUITES[suite]:
            try:
                answer, secs = complete(client, base_url, model, task.prompt)
                score = float(task.score(answer))
                error = None
            except Exception as exc:  # a broken task must not kill the run
                answer, secs, score, error = "", 0.0, 0.0, f"{type(exc).__name__}: {exc}"[:300]
            run.results.append({
                "id": task.id, "category": task.category, "weight": task.weight,
                "score": score, "seconds": round(secs, 2), "answer": answer[:600], "error": error,
            })
    return run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the local benchmark suite against an OpenAI-compatible endpoint")
    p.add_argument("--gateway", default="http://127.0.0.1:8090")
    p.add_argument("--model", required=True)
    p.add_argument("--suite", choices=sorted(SUITES), default="fast")
    p.add_argument("--out", default="bench/results")
    args = p.parse_args(argv)

    run = run_suite(args.gateway, args.model, args.suite)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = outdir / f"{args.model.replace('/', '_')}-{args.suite}-{stamp}.json"
    payload = {
        "model": run.model, "suite": run.suite, "started": run.started,
        "score": run.score, "by_category": run.by_category(), "results": run.results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf8")

    print(f"model {run.model} · suite {run.suite} · score {run.score}")
    for cat, val in run.by_category().items():
        print(f"  {cat:<10} {val}")
    for r in run.results:
        mark = "ok  " if r["score"] else "FAIL"
        print(f"  {mark} {r['id']:<14} {r['seconds']:>6.2f}s  {r['answer'][:70].replace(chr(10), ' ')}")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
