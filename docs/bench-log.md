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
