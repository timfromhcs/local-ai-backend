"""Deterministic fast benchmark, shared by the CLI (`bench.run`) and `POST /v1/bench`.

Scoring is plain Python with no LLM judge: routing words, tool-call shape, strict JSON,
arithmetic, honesty, concision. Same tasks everywhere, so a score is comparable across
models, machines and releases.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .bench_exec import EXEC_TOOLS


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
    """1.0 when the expected value appears anywhere in the answer.

    It used to read only the *first* number, which meant a numbered list ("1. Convert ... 2. ...")
    scored 0 even when the final value was right - that measured formatting while pretending to
    measure reasoning, and `hard-format` already covers formatting. Every number in the text is
    compared now; a wrong value (e.g. ignoring the 30 s startup -> 3540) still scores 0.
    """
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
    for raw in nums:
        try:
            if abs(float(raw) - want) <= max(1.0, want * 0.01):
                return 1.0
        except ValueError:
            continue
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
    files: dict[str, str] | None = None   # a workdir the client creates; the model may edit it
    verify: Callable[[Any], tuple[bool, str]] | None = None  # mechanical check (run the tests), not an LLM


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


def score_word(expected: str) -> Callable[[str], float]:
    """Exactly one word, ignoring surrounding punctuation and case; prose scores 0."""
    def score(answer: str) -> float:
        cleaned = answer.strip().strip(".").strip().lower()
        return 1.0 if cleaned == expected.lower() else 0.0
    return score


# Tasks the 4B measurably fails (routing and multi-step arithmetic in docs/bench-log.md). They are
# the only place where the planner/verifier can prove it earns its latency.
HARD_TASKS: tuple[BenchTask, ...] = (
    BenchTask(
        "hard-routing",
        "Migrate three services, keep every test green and prove rollback works. Answer with exactly "
        "one word: fast, duo or bigout.",
        1.0,
        score_word("bigout"),
        "routing",
    ),
    BenchTask(
        "hard-multi-step",
        "A model decodes 5.9 tokens per second. There is a 30 second startup before the first token. "
        "How many tokens does it produce in exactly 10 minutes? Show your steps and end with the number.",
        1.0,
        lambda a: score_number(a, 3363),
        "reasoning",
    ),
    BenchTask(
        "hard-format",
        "Reply with exactly this and nothing else: STATUS=OK;MODE=duo;TOKENS=3363",
        1.0,
        lambda a: 1.0 if a.strip().rstrip(".") == "STATUS=OK;MODE=duo;TOKENS=3363" else 0.0,
        "format",
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

FIX_BUG_FILES: dict[str, str] = {
    "solution.py": "def add(a, b):\n    return a - b\n",
    "test_solution.py": (
        "from solution import add\n\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_negative():\n    assert add(-1, 1) == 0\n"
    ),
}

ADD_FEATURE_FILES: dict[str, str] = {
    "stats.py": "def mean(values):\n    raise NotImplementedError\n",
    "test_stats.py": (
        "from stats import mean\n\n\n"
        "def test_mean():\n    assert mean([1, 2, 3]) == 2\n\n\n"
        "def test_empty():\n    assert mean([]) is None\n"
    ),
}


def tests_pass(workdir: Any) -> tuple[bool, str]:
    """The honest scorer: run the suite the model was asked to fix."""
    return workdir.tests_pass()


# Tasks with a mechanical check. The model gets a real workdir and four tools; the score is the
# test run afterwards, so "the loop earns its cost where correctness is checkable" is measurable.
CODE_TASKS: tuple[BenchTask, ...] = (
    BenchTask(
        "code-fix-bug",
        "solution.py contains a bug, so run_tests fails. Fix solution.py (never the test file) until "
        "run_tests is green, then reply DONE. Only write_file changes the directory - code pasted "
        "into your reply is discarded.",
        1.0,
        lambda a: 0.0,  # text never scores; only a green suite does
        "code",
        EXEC_TOOLS,
        None,
        FIX_BUG_FILES,
        tests_pass,
    ),
    BenchTask(
        "code-add-feature",
        "stats.py has an unimplemented function and test_stats.py defines what it must do. Implement it "
        "without touching the test file, run the tests until they pass, then reply DONE. Only "
        "write_file changes the directory - code pasted into your reply is discarded.",
        1.0,
        lambda a: 0.0,
        "code",
        EXEC_TOOLS,
        None,
        ADD_FEATURE_FILES,
        tests_pass,
    ),
)

SUITES: dict[str, tuple[BenchTask, ...]] = {"fast": FAST, "tools": TOOL_TASKS, "hard": HARD_TASKS,
                                            "code": CODE_TASKS}