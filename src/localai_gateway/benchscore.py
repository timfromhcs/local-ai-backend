"""Deterministic fast benchmark, shared by the CLI (`bench.run`) and `POST /v1/bench`.

Scoring is plain Python with no LLM judge: routing words, tool-call shape, strict JSON,
arithmetic, honesty, concision. Same tasks everywhere, so a score is comparable across
models, machines and releases.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable


def score_route(answer: str, want: str) -> float:
    return 1.0 if re.search(rf"\b{want}\b", answer, re.I) else 0.0


def score_tool_call(answer: str) -> float:
    listing = re.search(r"(list_files?|list_dir)\s*\(", answer, re.I)
    reading = re.search(r"read_file\s*\(", answer, re.I)
    return 1.0 if listing and reading else 0.0


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
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(".", "").replace(",", ""))
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
    """A coding answer should name a file and include code or a diff."""
    has_file = bool(re.search(r"[\w./-]+\.(py|ts|tsx|js|json|yaml|yml|md)", answer))
    has_code = "```" in answer or bool(re.search(r"^\s*(diff --git|[-+]{1,3} )", answer, re.M))
    return 1.0 if has_file and has_code else 0.0


@dataclass(frozen=True)
class BenchTask:
    id: str
    prompt: str
    weight: float
    score: Callable[[str], float]
    category: str = "core"
    tools: tuple[dict, ...] = ()          # tool schemas the client offers; enables the duo loop
    score_message: Callable[[dict], float] | None = None  # when the answer is a tool call, not text


# Two small tools so a task can require an actual call instead of prose about one.
FILE_TOOLS = (
    {"type": "function", "function": {"name": "list_files", "description": "List files in a directory",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                     "required": ["path"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                     "required": ["path"]}}},
)


def score_expected_tool_calls(required: tuple[str, ...]) -> Callable[[dict], float]:
    """1.0 only when the message carries tool calls with exactly the expected names, in order."""
    def score(message: dict) -> float:
        calls = message.get("tool_calls") or []
        names = tuple(c.get("function", {}).get("name") for c in calls)
        return 1.0 if names == required else 0.0
    return score


TOOL_TASKS: tuple[BenchTask, ...] = (
    BenchTask(
        "tool-list-read",
        "List the files in C:\\work, then read package.json. Answer with tool calls only.",
        1.0,
        lambda a: 0.0,  # text alone never satisfies a tool task
        "tools",
        FILE_TOOLS,
        score_expected_tool_calls(("list_files", "read_file")),
    ),
    BenchTask(
        "tool-single",
        "Read C:\\work\\package.json and nothing else.",
        1.0,
        lambda a: 0.0,
        "tools",
        FILE_TOOLS,
        score_expected_tool_calls(("read_file",)),
    ),
)


FAST: tuple[BenchTask, ...] = (
    BenchTask("routing", "A task needs planning, implementation and verification. Answer with one word: "
                         "fast, duo or bigout.", 1.0, lambda a: score_route(a, "duo"), "routing"),
    BenchTask("routing-hard", "Migrate three services, keep every test green and prove rollback works. "
                              "One word: fast, duo or bigout.", 1.0, lambda a: score_route(a, "bigout"), "routing"),
    BenchTask("tool-call", "List the files in C:\\work, then read package.json. Reply with the tool calls only.",
              1.0, score_tool_call, "tools"),
    BenchTask("json", 'Return only JSON: {"model":"duo","loaded":true,"ram_mb":9000}',
              1.0, score_json, "format"),
    BenchTask("math", "A model decodes 5.9 tokens per second. How many tokens in 10 minutes? Number only.",
              1.0, lambda a: score_number(a, 3540), "reasoning"),
    BenchTask("honesty", "Can a browser run a 27B model at 50 tokens per second? Answer No or Yes, "
                        "then one short reason.", 1.0, score_starts_no, "honesty"),
    BenchTask("concise", "Explain in exactly one sentence what a KV cache does.", 0.5, score_one_sentence, "format"),
)

SUITES: dict[str, tuple[BenchTask, ...]] = {"fast": FAST, "tools": TOOL_TASKS}


@dataclass
class BenchResult:
    model: str
    suite: str
    rows: list[dict]

    @property
    def score(self) -> float:
        total = sum(r["weight"] for r in self.rows) or 1.0
        return round(sum(r["score"] * r["weight"] for r in self.rows) / total, 3)

    def by_category(self) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for row in self.rows:
            buckets.setdefault(row["category"], []).append(row["score"])
        return {k: round(sum(v) / len(v), 3) for k, v in buckets.items()}

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "suite": self.suite,
            "score": self.score,
            "by_category": self.by_category(),
            "rows": self.rows,
        }
