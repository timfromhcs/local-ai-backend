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
