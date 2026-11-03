"""A tiny sandboxed workdir so a bench task can be *verified*, not just read.

The previous suites score text. This one lets a model act: it gets `list_files`, `read_file`,
`write_file` and `run_tests` in a throwaway directory, and the score is produced by running the
tests afterwards - not by an LLM judge. That is the only shape in which the plan/execute/verify loop
can earn its cost (`docs/bench-log.md`).

Executing generated code is opt-in on purpose: the route refuses to run unless
`LOCALAI_BENCH_EXEC=1` is set, because the model writes files that pytest then imports.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

EXEC_TOOLS: tuple[dict, ...] = (
    {"type": "function", "function": {
        "name": "list_files", "description": "List files in the work directory",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "default": "."}},
                       "required": []}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file from the work directory",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Overwrite a text file in the work directory",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                        "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the pytest suite and return the output",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
)

MAX_FILE_CHARS = 6000
TEST_TIMEOUT_S = 90


@dataclass
class Workdir:
    """A temp directory with fixture files and the four tools a model may use on it."""

    files: dict[str, str] = field(default_factory=dict)
    root: Path = field(init=False)

    def __post_init__(self) -> None:
        # resolve once: on Windows a temp path can be a short name (RUNNER~1) and a later
        # resolve() of a child would then not compare equal to the root (CI caught this)
        self.root = Path(tempfile.mkdtemp(prefix="localai-bench-")).resolve()
        for name, content in self.files.items():
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    # ---- tools --------------------------------------------------------------------------------
    def list_files(self, path: str = ".") -> str:
        base = self._resolve(path)
        if not base.exists():
            return f"error: {path} does not exist"
        if base.is_file():
            return base.name
        names = sorted(p.relative_to(base).as_posix() + ("/" if p.is_dir() else "")
                       for p in base.iterdir() if p.name != "__pycache__")
        return "\n".join(names) or "(empty)"

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            return f"error: {path} is not a file"
        text = target.read_text(encoding="utf-8", errors="replace")
        return text[:MAX_FILE_CHARS] + ("\n...[truncated]" if len(text) > MAX_FILE_CHARS else "")

    def write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"written {path} ({len(content)} chars)"

    def run_tests(self) -> str:
        """Real pytest in the workdir: isolated, no user site, no repo on the path, hard timeout."""
        code, out = self._pytest()
        return f"exit={code}\n{out}"

    # ---- scoring ------------------------------------------------------------------------------
    def tests_pass(self) -> tuple[bool, str]:
        code, out = self._pytest()
        return code == 0, out

    def _pytest(self) -> tuple[int, str]:
        env = {k: os.environ[k] for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP") if k in os.environ}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
                cwd=self.root, env=env, capture_output=True, text=True, timeout=TEST_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return 124, f"pytest did not finish within {TEST_TIMEOUT_S}s"
        output = (proc.stdout or "") + (proc.stderr or "")
        lines = [ln for ln in output.strip().splitlines() if ln.strip()]
        return proc.returncode, "\n".join(lines[-25:])[:2500]

    # ---- dispatch -----------------------------------------------------------------------------
    def dispatch(self, name: str, arguments: Any) -> str:
        args = arguments if isinstance(arguments, dict) else _parse_args(arguments)
        try:
            if name == "list_files":
                return self.list_files(str(args.get("path") or "."))
            if name == "read_file":
                return self.read_file(str(args.get("path") or ""))
            if name == "write_file":
                return self.write_file(str(args.get("path") or ""), str(args.get("content") or ""))
            if name == "run_tests":
                return self.run_tests()
        except _OutsideRoot as exc:
            return f"error: {exc}"
        except OSError as exc:
            return f"error: {type(exc).__name__}: {exc}"
        return f"error: unknown tool {name!r}; available: {', '.join(t['function']['name'] for t in EXEC_TOOLS)}"

    def _resolve(self, path: str) -> Path:
        candidate = (self.root / (path or ".")).resolve()
        if candidate != self.root and not candidate.is_relative_to(self.root):
            raise _OutsideRoot(f"{path} is outside the work directory")
        return candidate

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class _OutsideRoot(Exception):
    pass


def _parse_args(arguments: Any) -> dict:
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def tool_loop_calls(tools: tuple[dict, ...] | list[dict]) -> list[dict]:
    """Schemas offered to the model; a task may restrict the set."""
    return list(tools)


CallFn = Callable[[dict], Awaitable[dict]]


async def run_with_tools(call: CallFn, body: dict, workdir: Workdir,
                         max_rounds: int = 8, max_steps: int = 12,
                         check: Callable[[], tuple[bool, str]] | None = None,
                         repairs: int = 0) -> tuple[dict, list[dict]]:
    """Drive model <-> tool exchanges until the model answers in text or the budget is spent.

    `call` takes a chat body and returns an OpenAI completion. Tool results become `role: tool`
    messages, so the *client* of the loop (this function) is what makes verification possible.

    With `check` + `repairs`, the client also refuses a finished answer while the mechanical check
    still fails: it feeds the failure back as a user message and lets the model try again. That is
    the difference between asking a model whether it is done and running the tests.
    """
    messages = list(body["messages"])
    steps = 0
    resp: dict = {}
    left = max(repairs, 0)
    for _ in range(max_rounds):
        # snapshot the conversation: the backend must see the state of this round, not a live list
        resp = await call({**body, "messages": list(messages)})
        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         **({"tool_calls": calls} if calls else {})})
        if not calls:
            if check is not None and left > 0:
                passed, output = check()
                if passed:
                    return resp, messages
                left -= 1
                messages.append({"role": "user", "content":
                                 "Your answer is not accepted: the tests still fail.\n\n"
                                 f"{output[-1200:]}\n\nUse the tools to fix it, then reply DONE."})
                continue
            return resp, messages
        for call_spec in calls:
            if steps >= max_steps:
                break
            steps += 1
            fn = call_spec.get("function") or {}
            result = workdir.dispatch(str(fn.get("name") or ""), fn.get("arguments"))
            messages.append({"role": "tool", "tool_call_id": call_spec.get("id") or f"c{steps}",
                             "content": result})
        if steps >= max_steps:
            break
    return resp, messages


def check_with_tests(workdir: Workdir) -> Callable[[], tuple[bool, str]]:
    """The mechanical check handed to `run_with_tools`: run the suite, no opinion involved."""
    return workdir.tests_pass


def summarize_transcript(messages: list[dict], limit: int = 900) -> str:
    """A compact, honest trace for the bench row: tools called, their outcome, what the model said."""
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            args = str(fn.get("arguments") or "")
            parts.append(f"-> {fn.get('name')}({args[:80]})")
        if m.get("role") == "tool":
            first = re.sub(r"\s+", " ", str(m.get("content") or ""))[:120]
            parts.append(f"   {first}")
        elif m.get("role") == "assistant" and not (m.get("tool_calls")):
            text = re.sub(r"\s+", " ", str(m.get("content") or ""))[:120]
            if text:
                parts.append(f"say: {text}")
    return "\n".join(parts)[:limit]
