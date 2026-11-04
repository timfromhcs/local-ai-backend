# Benchmark log (deterministic suite, real models)

Scores come from `POST /v1/bench` on the local gateway. Scoring is plain Python (no LLM judge), so
a number is reproducible and comparable. A low score means the **model** answered weakly, not that
the gateway broke — the two are kept apart on purpose.

## 2026-11-03 — Qwen3-4B-abliterated Q4_K_M, fast suite

Hardware: Ryzen 7 7735HS, 24 GB, Radeon 680M (Vulkan). Gateway on :8090, model loaded on demand.

| Task | Result | Seconds | Answer (trimmed) |
|---|---|---|---|
| routing | **0** | 177.5 | *(empty)* — cold start plus a model that answered nothing usable |
| routing-hard | 0 | 32.6 | "One word: **fast**." — wrong route (expected `bigout`) |
| tool-call | 0 | 8.8 | a JSON object naming the tool, not `list_files(...)` calls |
| json | 1 | 1.7 | exact JSON as requested |
| math | 0 | 7.4 | explained the arithmetic instead of returning a number |
| honesty | 1 | 2.2 | "No. A 27B model is too large…" |
| concise | 1 | 2.8 | one sentence about the KV cache |

**Weighted score 0.385** (routing 0.0, tools 0.0, format 1.0, reasoning 0.0, honesty 1.0).

### What this run actually taught us

1. **Thinking mode eats the budget.** The first run scored 0.077 because Qwen3 spent every token on
   reasoning and returned empty `content` for six of seven tasks. The bench route now sends
   `chat_template_kwargs.enable_thinking = false` by default; `"thinking": true` opts back in. That
   is a gateway bug fixed, not a model improvement.
2. **Instruction-following is the weak spot, not knowledge.** JSON, honesty and concision pass;
   routing and tool-call formatting fail. This is exactly the gap the `duo` model exists to cover
   (a bigger model plans/verifies, the 4B executes).
3. **Cold start dominates the wall clock.** 177 s for the first task (model load + generation),
   then single-digit seconds. A benchmark of a lazily loaded gateway must warm up first or report
   the cold path separately, or the number is meaningless.
4. **Scorers stay strict.** The tool-call task accepts `list_files(...)`/`read_file(...)` and not a
   JSON description of the same intent. Loosening it to make a model look better would defeat the
   gate; the observation is recorded instead.

### Reproduce

```powershell
localai-gateway --port 8090
curl -X POST http://127.0.0.1:8090/v1/bench `
  -H "content-type: application/json" `
  -d "{\"model\":\"qwen3-4b\",\"suite\":\"fast\",\"max_tokens\":160}"
```

## 2026-11-03 — duo vs bare 4B on the same four tasks

`scripts/measure-duo.ps1`, tasks `routing`, `tool-call`, `math`, `json`.

| Model | Score | Wall time | Per task |
|---|---|---|---|
| qwen3-4b | **0.25** | 24.2 s | routing 0 (12.4 s), tool-call 0 (2.5 s), json **1** (1.7 s), math 0 (7.2 s) |
| duo | **0.25** | **4.9 s** | routing 0 (0.6 s), tool-call 0 (0.3 s), json **1** (0.6 s), math 0 (3.4 s) |

### What this actually shows — and does not

`duo` is **five times faster** here and **exactly as correct**. That is not the plan/execute/verify
loop doing work: these tasks have no `tools`, and their prompts are short, so the mode controller
routes them to the **fast lane** (the 1.5B Jev model). The comparison therefore measures the fast
lane, which is the correct behaviour for "short prompt, no tools" — but it says nothing about duo's
planning loop.

That loop only engages when a request carries `tools`. Measuring it needs benchmark tasks with
tool schemas, which the current suite does not have. Recorded as the next honest step instead of
reporting "duo is 5x faster" as if it were a quality claim.

Three bugs were found by attempting this measurement, all fixed in PR #52:
1. `/v1/bench` rejected virtual models, so `duo` could not be measured at all.
2. Orphaned llama-server processes kept Vulkan memory after a gateway restart; the next model load
   died with `failed to allocate buffer for kv cache`. Added `Scheduler.shutdown()`.
3. `duo` answered 401 because the external Jev server is started with `--api-key local` and the
   gateway never sent the bearer header. External models now carry `api_key` in `models.yaml`.

## 2026-11-03 — tool-call suite: both models score 1.0

A `tools` suite was added precisely because the tasks above never engage duo's planning loop (no
tool schemas → fast lane). Two tasks now offer `list_files`/`read_file` and are scored on the
message's `tool_calls`, not on text; prose that merely describes a call scores 0.

| Model | Score | Wall time | Note |
|---|---|---|---|
| qwen3-4b | **1.0** (2/2) | 178.2 s | 176 s of that is the cold start, then 1.9 s |
| duo | **1.0** (2/2) | **3.7 s** | again the fast lane (Jev chose `fast`), not plan/execute/verify |

### What this run taught us

- The 4B model *can* emit correct tool calls when the client sends schemas; the earlier tool-call
  failure was a text-format task, not a capability gap.
- `duo` stays fast because its mode controller prefers the fast lane when the task is small. To
  measure the plan/execute/verify path a caller must be able to **force** the mode; that is the
  next open item, not something to claim from this number.
- While measuring, three further gateway bugs were fixed (PR #54): `ask_jev` sent no API key (401
  every check-up), the supervisor died on a lost Windows process handle and marked a healthy model
  as failed, and `bin/llama/llama-server.exe` vanished from this machine — the runtime path is now
  overridable via `LOCALAI_LLAMA_SERVER`, and `doctor` reports the effective path.

## 2026-11-03 — the loop, finally measured (forced mode)

`localai_mode` was added (PR #56) so the plan/execute/verify loop can be demanded instead of hoped
for. Same two tool tasks, all three lanes:

| Lane | Score | Wall time | What ran |
|---|---|---|---|
| qwen3-4b alone (cold) | 1.0 | 178 s | the executor only, cold start included |
| duo, automatic | 1.0 | **3.7 s** | fast lane (1.5B) — no planning |
| duo, `mode: "duo"` | 1.0 | 30.1 s | planner 27B → executor 4B → verifier 27B |

### Reading this honestly

On two simple tool tasks **all three lanes score 1.0**, so the loop buys no accuracy here — it
costs ~26 s of planner/verifier time. That is the expected shape: the loop exists for tasks where
the 4B fails, and this suite is not hard enough to show it. Reporting "duo is better" from these
numbers would be dishonest; reporting "the loop runs, is measurable, and costs 26 s on this
hardware" is the truth.

Next: tasks that the 4B actually fails (its weakest categories are routing and multi-step math
from the fast-suite run), where the verifier has something to correct.

## 2026-11-03 — hard suite: the loop does not rescue a capability gap

Three tasks were chosen from the fast-suite failures (`docs/hard-bench.json`):

- `hard-routing` — "migrate three services, keep tests green, prove rollback": one word, expected `bigout`
- `hard-multi-step` — 5.9 tokens/s, 30 s startup, 10 minutes: the number only, expected `3363`
- `hard-format` — reproduce `STATUS=OK;MODE=duo;TOKENS=3363` exactly

| Lane | Score | Wall time | What happened |
|---|---|---|---|
| qwen3-4b alone | **0.333** (1/3) | 13.1 s | format exact; routing answered `fast`; math explained, never stated |
| duo, automatic | **0.333** | 5.3 s | fast lane; routing answered in 0.13 s |
| duo, `mode: "duo"` | **0.333** | 18.5 s | planner 27B → executor 4B → verifier 27B, same wrong answers |

### The uncomfortable result

The forced loop **did** run (18.5 s vs 5.3 s automatic, and 30.5 s on a cold planner in a separate
single-prompt run), and it changed nothing: the planner's steps were followed by an executor that
still answered `fast`, and the verifier — which has no ground truth — replied `VERIFIED`. A plan
cannot add knowledge the 4B does not have, and a verifier without a test cannot notice a wrong
answer. Both routing tasks fail for the same reason; the arithmetic task fails because the model
writes the method and never the number.

So: on this suite the loop buys **zero accuracy for ~13 s extra**. What it can earn its cost on is a
task where correctness is *checkable* (run the tests, call the tool) — not a one-word answer that
needs world knowledge.

### Bug found while measuring

The first forced run scored 0.333 in 0.35 s per task: `localai_mode` was ignored whenever the task
offered **no tools**, because the no-tools shortcut ran before the forced-mode branch. Fixed in the
duo tests (`test_forced_mode_plans_even_without_tools`), together with a guard that never forwards
`tools: null` to a backend. Scores are strict on purpose; nothing was loosened to flatter a model.

### Infrastructure note

Two measurement runs failed with `backend exited with code 1 before becoming healthy` before any
score was produced: force-killing the gateway leaves `llama-server` children holding Vulkan memory,
and the next load dies with `ErrorOutOfDeviceMemory`. The gateway now keeps a small ledger of the
backends it spawned (`cache/backends.json`) and reaps exactly those whose owning process is gone at
the next start — it never touches processes it did not start.

## 2026-11-03 — code suite: the client, not the loop, is what verifies

The hard suite proved the point negatively (a verifier with no ground truth approves a wrong answer).
So this suite makes correctness mechanical: the client creates a temp workdir with a failing pytest
file, executes the model's `list_files` / `read_file` / `write_file` / `run_tests` calls, then runs
the tests itself. Green means green; no LLM judge anywhere. `POST /v1/bench` refuses the suite unless
`LOCALAI_BENCH_EXEC=1` is set, because it executes generated code in a temp directory.

| Run | qwen3-4b alone | duo forced (`mode: "duo"`) |
|---|---|---|
| before repair + protocol (2 runs) | 0.5 / 0.5 | 0.5 / 0.5 |
| after repair + protocol | **1.0** in 51.8 s | **1.0** in 66.4 s |
| repeat | **1.0** in 50.3 s | **1.0** in 65.2 s |

Two changes were made between those rows, and both live in the *client*:

1. **A bounded repair loop.** When the model declares itself done, the harness runs the tests; if
   they still fail it feeds the failure back ("your answer is not accepted") for up to two rounds.
   A model that says "DONE" over a red suite would otherwise be scored as finished.
2. **A protocol sentence in the task.** "Only `write_file` changes the directory — code pasted into
   your reply is discarded." Without it the 4B answered with prose and never touched a file; that is
   a behavioral failure, not a knowledge gap.

### Reading this honestly

The score moved from 0.5 to 1.0 because the *harness* got better at refusing unpaid claims — not
because either model learned something. On this suite `duo` scores the same as the 4B alone and
costs ~15 s more (planner + verifier on a 27B), exactly like the hard suite. The loop's value is
still unproven here; the demonstrated value is the mechanical check plus the repair round.

One caveat recorded on purpose: a single run is not evidence. A third 4B run in between scored 0.5
because the model chattered ("I will fix the add function…") and never called `write_file`; the two
repeat runs above scored 1.0. Variance is real and is reported, not averaged away.

## 2026-11-03 — a red test run now outranks the LLM verifier

The hard suite showed the failure mode: the verifier replied `VERIFIED` to an answer that was wrong,
because it has no ground truth. The repair round lived in the bench client only. It now lives in the
product too (`duo`, not just the harness):

- `duo` reads the newest *check* tool result (`run_tests`, `pytest`, `lint`, `build`, …) before it
  asks the verifier. Exit code ≠ 0 or a "N failed" summary is authoritative: the executor is called
  again with the failure as feedback, and the LLM is never consulted.
- A green result finishes immediately — no 27B opinion on something the tests already proved.
- After `max_verify_rounds` the failure is returned as-is; there is no path where a red suite ends
  in a confident answer. Opt-out: `DuoConfig(respect_tool_checks=False)`.

Measured on the code suite, duo forced, two runs: **1.0 in 65.0 s** and **1.0 in 73.1 s**, against
**66.4 s / 65.2 s** before the change. So: no measurable score or time change, and that is the honest
reading. What changed is the guarantee, not the number — a wrong answer can no longer be certified
by an opinion.

Tests: 9 new cases in `tests/test_duo_mechanical.py`, including "a non-check tool is not a verdict",
"a red exit code sends the executor back", "a green exit code skips the verifier" and the opt-out.

## 2026-11-03 — every answer says how it was checked (and a correction)

A `duo` answer used to look equally trustworthy whether it came from a verified loop or from the fast
lane. Non-streaming answers now carry `localai_verification`:

| `state` | Meaning |
|---|---|
| `mechanical` | a check tool reported success |
| `llm` | a model approved it - an opinion, not evidence |
| `none` | nothing was checked (fast lane, big-out, unreachable verifier/planner, rounds exhausted) |
| `pending` | the answer is a tool call; verification continues on the next request |
| `failed` | a check tool still reports a failure |

### Correction to the earlier numbers in this log

Writing that label exposed a real problem on this machine: the 27B planner/verifier cannot be loaded
by the current Vulkan build any more (`gguf_init_from_reader: failed to read tensor info`), and `duo`
**swallowed the error**, so a run that never planned looked exactly like one that did. Live proof
before the fix:

```
state: none
detail: verifier unreachable; planner unavailable: BackendError: bonsai-27b: backend exited with code 1
```

Consequences for this log, stated plainly:

- The **tools-suite forced run (30.1 s, 18:19)** and the **forced-mode comparison (docs/duo-forced.json)**
  happened before `bin/llama/llama-server.exe` vanished from this machine, so their planner/verifier
  calls are real.
- The **hard-suite forced run (18.5 s)** and the **code-suite duo runs (65.0 s / 73.1 s)** came after
  it. Their numbers are executor work plus the mechanical check; the planner and the LLM verifier did
  not contribute. The scores stand (the tests are the judge), but "the loop ran" does not.

The failure is no longer silent: a planner that cannot be reached is named in `localai_verification`
(`planner unavailable: …`). Making the 27B loadable again, or measuring a working planner/verifier
substitute, is the next open item - not something this log pretends is solved.

## 2026-11-04 — why the 27B cannot load, and an error message worth reading

Chased the `failed to read tensor info` from the previous cycle with the pristine build from
`llama-prism-b12128-d7b2896-bin-win-vulkan-x64.zip` (restored into `bin/llama/`, which is local and
gitignored). With that build the model is *readable*; the real failure is a VRAM allocation:

```
ggml_vulkan: Device memory allocation of size 1060751488 failed.
ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory
```

At `-ngl 99` and at `-ngl 60`, with `-c 4096`, while the fast-lane server (2.4 GB) and the desktop
compositor hold the shared carve-out. So the 27B planner/verifier does not fit on this machine right
now - a resource limit, not a broken file. What the gateway said about it was:

```
backend exited with code 1 before becoming healthy
```

That reads like a code bug and cost time twice. The supervisor now drains the child's stderr into a
bounded tail and puts the most useful error line into the exception (and therefore into
`localai_verification.detail`). Live, same request, after the change:

```
verifier unreachable; planner unavailable: BackendError: bonsai-27b: backend exited with code 1
before becoming healthy: ggml_vulkan: Device memory allocation of size 1060751488 failed.
```

Diagnosis is not a fix: until a smaller planner fits (or the fast-lane server yields VRAM), `duo`
runs without its planner/verifier and says so. The measured numbers in this log keep their earlier
caveat.

## 2026-11-04 - a planner that fits, and what it costs

With the fast-lane server stopped, the 27B loads fine (no OOM, health reached) - so the limit is
coexistence, not the model. It cannot share the carve-out with the 2.4 GB fast lane, which every
`duo` request needs for routing, so on this machine the 27B is not a usable planner right now.

That is why `DuoConfig.from_env()` exists: `LOCALAI_DUO_PLANNER` / `_VERIFIER` / `_EXECUTOR` / `_FAST`
name the roles without touching code. Measured with the 4B in all three roles (no extra VRAM):

| Lane | hard suite | Time |
|---|---|---|
| qwen3-4b alone | 0.333 | 11.7 s |
| duo, automatic | 0.333 | 7.1 s |
| duo forced, 4B planner + 4B verifier | **0.333** | **83.6 s** |

So the loop costs ~72 s and buys nothing when the planner is the same model. The one thing it did
improve is honest: `hard-format` now passes because the answer text is clean - the earlier 0 was our
own bug.

### Two fixes that this measurement forced

1. **The answer text is no longer annotated.** `_final` appended `[VERIFIED: ...]` to the content,
   which broke an exact-format task the executor had answered correctly (`STATUS=OK;MODE=duo;TOKENS=3363`
   plus a bracket note fails an equality check). The note now lives only in
   `localai_verification.detail`. Anyone parsing the old suffix gets a clean answer instead.
2. **The verifier is told to check explicit constraints** ("one word", "number only", "exactly this
   text"). It had approved an answer that broke the task's format because nothing in its prompt asked
   it to look.

### What is still true

`duo` cannot beat the executor without a stronger planner, and the only candidate here does not fit.
Any future claim that the loop helps must come with a planner that is actually different - otherwise
it is the same 4B talking to itself three times. The 11.7 s vs 83.6 s row is the cost of that.

## 2026-11-04 - the VRAM ceiling, measured from three sides

Goal: a planner that fits next to the fast lane. Answer: on this machine, none of the options do.

| Attempt | Result |
|---|---|
| 27B, `-c 16384 -ngl 99`, fast lane running | fails: 1.06 GB allocation |
| 27B, `-c 8192 -ngl 99`, fast lane running | fails: 1.06 GB allocation |
| 27B, `-c 4096 -ngl 40`, fast lane running | fails: 0.90 GB allocation |
| 27B, `-c 4096 -ngl 99`, fast lane stopped | loads and answers |

The shared carve-out is simply too small for a 27B planner, a 4B executor and the 1.5B fast lane at
once. `models.yaml` now says so in the model's own note, and the context reduction to 8k is recorded
as what it is - a smaller KV cache that is still not enough.

### The cost of a failure that repeats

Because the planner is asked on every `duo` request, a non-fitting model used to cost ~14-33 s *per
request*, each time re-spawning it, reading its stderr and failing identically. The supervisor now
caches a failed start for 90 s and answers with the reason immediately:

```
try 1: 33.2 s  planner unavailable: ... Device memory allocation ...
try 2:  3.2 s  (same reason, no spawn - what remains is the executor call)
```

The cache expires, so freeing VRAM (or fixing the flags) takes effect without restarting the gateway.
That is a latency fix with a test, not a claim that the model now fits.

