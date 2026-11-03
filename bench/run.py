"""Benchmark suite for any OpenAI-compatible endpoint (local gateway by default).

Suites:
  fast   routing, tool-call validity, strict JSON, one arithmetic task, honesty, concision
  long   the fast suite plus multi-step coding and ops prompts, scored on structure
  tools  tasks that are only solved by an actual tool call
  hard   routing, multi-step arithmetic and exact format - the ones the 4B fails
  code   a real temp workdir: the client executes the model's tools and the tests decide the score

Scoring lives in localai_gateway.benchscore, which the gateway's POST /v1/bench uses too:
one definition, two entry points (CLI here, HTTP there).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from localai_gateway.bench_exec import Workdir, check_with_tests, run_with_tools, summarize_transcript
from localai_gateway.benchscore import (  # noqa: F401
    CODE_TASKS,
    FAST,
    HARD_TASKS,
    TOOL_TASKS,
    BenchResult,
    BenchTask,
    score_patch_shape,
    score_route,
)

LONG: tuple[BenchTask, ...] = FAST + (
    BenchTask("refactor", "Refactor a blocking request handler into async, name the file and show the diff.",
              1.0, score_patch_shape, "coding"),
    BenchTask("test-first", "Write the failing pytest first, then the fix. Show the file and the code.",
              1.0, score_patch_shape, "coding"),
    BenchTask("rollback", "Describe a rollback plan for a bad production deploy, with the command that triggers it.",
              1.0, lambda a: 1.0 if __import__("re").search(r"vercel|promote|rollback|git revert", a, __import__("re").I) else 0.0,
              "ops"),
    BenchTask("perf", "The gateway is slow on the first request but fast on the second. Give the cause and the fix.",
              1.0, lambda a: 1.0 if __import__("re").search(r"cache|prefix|prefill|slot", a, __import__("re").I) else 0.0,
              "reasoning"),
)

SUITES: dict[str, tuple[BenchTask, ...]] = {"fast": FAST, "long": LONG, "tools": TOOL_TASKS,
                                            "hard": HARD_TASKS, "code": CODE_TASKS}


def complete(client: httpx.Client, base_url: str, model: str, prompt: str, max_tokens: int = 400,
             tools: list[dict] | None = None) -> tuple[dict, float]:
    """Returns the whole message (so tool calls can be scored) and the elapsed seconds."""
    started = time.perf_counter()
    body: dict = {"model": model, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "temperature": 0}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    r = client.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=body)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    content = msg.get("content") or ""
    if isinstance(content, list):
        msg["content"] = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return msg, time.perf_counter() - started


def run_code_task(base_url: str, model: str, task: BenchTask, max_tokens: int = 600,
                  timeout: float = 300.0) -> tuple[str, float, float]:
    """Client-side tool loop: this process executes the tools, so the tests are the only judge."""
    work = Workdir(task.files or {})
    started = time.perf_counter()

    async def workflow() -> list[dict]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async def call(body: dict) -> dict:
                payload = {**body, "model": model, "max_tokens": max_tokens, "temperature": 0}
                r = await client.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=payload)
                r.raise_for_status()
                return r.json()

            _, messages = await run_with_tools(call, {"messages": [{"role": "user", "content": task.prompt}]},
                                                work, check=check_with_tests(work), repairs=2)
            return messages

    try:
        messages = asyncio.run(workflow())
        passed, output = task.verify(work) if task.verify is not None else (False, "no check")
        answer = f"{summarize_transcript(messages)}\n\n[{'green' if passed else 'red'}]\n{output}"
        return answer, time.perf_counter() - started, 1.0 if passed else 0.0
    finally:
        work.cleanup()


def run_suite(base_url: str, model: str, suite: str = "fast", timeout: float = 300.0) -> BenchResult:
    rows: list[dict] = []
    with httpx.Client(timeout=timeout) as client:
        for task in SUITES[suite]:
            try:
                if task.verify is not None:  # the loop needs the tools on every round
                    answer, secs, score = run_code_task(base_url, model, task, timeout=timeout)
                else:
                    msg, secs = complete(client, base_url, model, task.prompt,
                                         tools=list(task.tools) or None)
                    answer = msg.get("content") or ""
                    score = float(task.score_message(msg)) if task.score_message else float(task.score(answer))
                error = None
            except Exception as exc:  # a broken task must not kill the run
                answer, secs, score, error = "", 0.0, 0.0, f"{type(exc).__name__}: {exc}"[:300]
            rows.append({
                "id": task.id, "category": task.category, "weight": task.weight,
                "score": score, "seconds": round(secs, 2), "answer": answer[:600], "error": error,
            })
    return BenchResult(model=model, suite=suite, rows=rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the local benchmark suite against an OpenAI-compatible endpoint")
    p.add_argument("--gateway", default="http://127.0.0.1:8090")
    p.add_argument("--model", required=True)
    p.add_argument("--suite", choices=sorted(SUITES), default="fast")
    p.add_argument("--out", default="bench/results")
    args = p.parse_args(argv)

    result = run_suite(args.gateway, args.model, args.suite)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = outdir / f"{args.model.replace('/', '_')}-{args.suite}-{stamp}.json"
    payload = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), **result.as_dict()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf8")

    print(f"model {result.model} · suite {result.suite} · score {result.score}")
    for cat, val in result.by_category().items():
        print(f"  {cat:<10} {val}")
    for r in result.rows:
        mark = "ok  " if r["score"] else "FAIL"
        print(f"  {mark} {r['id']:<14} {r['seconds']:>6.2f}s  {r['answer'][:70].replace(chr(10), ' ')}")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
