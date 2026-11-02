# RSI Lab — real LoRA training, merge, benchmark gate and automatic post-chat improvement (design)

Date: 2026-11-02 · Status: approved to proceed under the user's autonomy delegation (sub-project 8, after gateway core)

## Intent
After every finished chat, turn the session log of a strong "teacher" model (Claude sessions like this one) into compact training data, train a LoRA on a *local trainable student*, evaluate it against a fixed benchmark gate, and **keep it only if it is better**. Mid-run checkpoints are evaluated too (keep-best / early stop). All of it local, resumable, rollback-able.

## Honest scope: what can and cannot be trained on this machine
| Model | Trainable here? | Plan |
|---|---|---|
| Bonsai-2-27B (ternary PTQ1) | **No** (27 B, quantized ternary, CPU-only torch, 24 GB shared RAM) | improved only by prompt/skill memory (real, prompt-level) and a distilled drafter |
| Jev: Qwen2.5-Coder-1.5B-Instruct (System 1) | **Yes**, CPU LoRA (r=8..16) | main student: tool-call / routing / code decisions |
| Drafter: Qwen3.5-0.8B (same tokenizer as Bonsai) | **Yes** | distill on Bonsai's own outputs, gate on acceptance rate/tok/s |
| Qwen3-4B | Marginal (bf16 ~8 GB + activations) | optional later |
The existing `lora_*`/`auto_finetune_chat_session` MCP tools are simulations (random matrices, hard-coded metrics) and are **not** used.

## Pipeline (each stage a small module with a CLI: `python -m rsi_lab <stage>`)
1. **harvest** – read teacher transcripts (`~/.claude/projects/**/*.jsonl`, OpenCode logs) incrementally (cursor file). Output: raw episodes.
2. **clean** – the "less tokens, fewer calls" requirement: drop failed/aborted episodes; keep episodes whose outcome was verified (tests passed / command exit 0 / user accepted); dedupe near-identical; truncate tool outputs to the informative head/tail; collapse retries into the final successful path; cap length (≤ 1536 tokens); **scrub secrets** (ghp_/Bearer/keys/e-mail, `.env` contents) and drop episodes whose context came from untrusted web content (prompt-injection / poisoning guard). Output: `sft.jsonl` of `{messages, tools}` with a provenance hash.
3. **split** – deterministic train / held-out by episode hash; held-out is never trained on; benchmark problems are blacklisted from training.
4. **train** – torch(CPU)+peft LoRA in a separate venv, loss on assistant/tool-call tokens only; checkpoint every N steps (mid-state); resume support; time- and RAM-boxed (refuses to run when the Bonsai server holds the RAM unless `--stop-bonsai`).
5. **merge** – `merge_and_unload` → HF dir → `convert_hf_to_gguf.py` (version matched to installed `gguf`) → `llama-quantize` to the student's current quant. Adapter versions stored under `rsi/adapters/<student>/<ts>/` with a manifest (data hash, metrics).
6. **evaluate (gate)** – run the benchmark suite against candidate and current champion on the same server settings (below). **Promote** only if: candidate score ≥ champion + margin on the primary metric, no category regresses by more than a tolerance, and tool-call JSON validity does not drop. Otherwise keep the champion and log why. Mid-state checkpoints compete on a short suite; the best one proceeds to the full gate.
7. **promote / rollback** – atomic switch of the `current` pointer used by the Jev server; one-command rollback.
8. **trigger** – a Claude Code `SessionEnd`/`Stop` hook runs `harvest` (cheap, <1 s) and enqueues a job; a low-priority Scheduled Task runs `clean→gate` only when idle (no foreground model server holding RAM) and at most once per N new verified episodes.

## Benchmark suite (researched; chosen for local feasibility at 5–30 tok/s)
| Suite | What it measures | How we run it here | Role |
|---|---|---|---|
| **BFCL** (`bfcl-eval`, Berkeley Function Calling v3/v4 AST categories: simple, multiple, parallel, irrelevance) | tool/function calling accuracy | `--skip-server-setup` against our OpenAI-compatible gateway endpoint (`LOCAL_SERVER_ENDPOINT/PORT`); AST scoring needs no code execution | **primary gate for agentic/tool skill** |
| **EvalPlus** HumanEval+ / MBPP+ (subset, e.g. 40+40 problems) | code generation correctness | OpenAI-compatible backend, sandboxed execution of tests | primary gate for code skill |
| Project **tool-call suite** (built from held-out teacher episodes) | JSON validity, correct tool + args on *our* tools | exact-match / schema check | regression guard |
| **SWE-bench Verified/Lite slice** via `mini-swe-agent` (+ `swebench` harness, Docker 29.x + WSL2 present) | real repo bug fixing | tiny slice (5–10 instances) as an occasional "nightly", **only for models fast enough (Bonsai)**; full 500 is infeasible at 5.8 tok/s | informational, not a gate |
| LiveCodeBench / Terminal-Bench / tau-bench | contamination-free coding / terminal agents / multi-turn tool use | evaluated for later; need cloud LLM user-simulators or heavy sandboxes | out of scope v1 |
Realistic expectation (stated up front): a 1.5 B student trained on a handful of sessions will move these scores only slightly; **the gate will reject most cycles by design**. That is the feature, not a bug.

## Interfaces / files
`rsi_lab/{harvest,clean,split,train,merge,gate,promote,cli}.py`, `rsi/` (data, adapters, runs, champions.json), `tests/rsi/` (unit: scrubber, cleaner, splitter, gate decision logic with synthetic scores; integration: tiny 0.5 B-class dry run end-to-end on 20 examples to prove train→merge→convert works). Benchmarks run behind `rsi_lab.bench.run(suite, endpoint) -> dict[str, float]`.

## Error handling / safety
Secrets never leave the scrubber (tested with canary strings); training refuses when free RAM < threshold; every stage idempotent with run manifests; promotion is atomic and reversible; no network calls except dataset/model downloads when explicitly enabled.

## Out of scope v1
RL / DPO from failures, multi-GPU/cloud training (a Runpod offload is possible later), training Bonsai.
