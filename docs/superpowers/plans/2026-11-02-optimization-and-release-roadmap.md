# Roadmap: optimize locally → perfect locally → GitHub → Vercel → README → merge main

Hardware: Ryzen 7 7735HS (8C/16T, AVX2), 24 GB DDR5 shared with the iGPU, Radeon 680M (RDNA2, 12 CU, Vulkan, **no matrix cores**, warp 32), Windows 11, 15.6 GB pagefile on SSD.
Rule set: no cloud subagents (user, 2026-11-02). Model stays Bonsai-2-27B MIXED-PTQ1-Q4_0 (user: optimize the runtime, not the model). Nothing leaves the machine until the Release Gate passes.

## Baselines measured on this machine (the numbers every change must beat)
| Metric | Value | Config |
|---|---|---|
| Generation | 5.82 tok/s | `-ngl 99 -fa on -ctk/-ctv q8_0 -t 8 --reasoning-budget 0` |
| Prompt (server, short) | 16.0 tok/s | same |
| Prompt (OpenCode, 13.6k tok) | 26 tok/s → ~9 min TTFT | same, 16k ctx |
| Load | 17 s (warm cache) / 97 s (cold, RAM pressure) | mmap |
| Draft: ngram-mod / ngram-map-k, novel text | 5.5 / 5.2 tok/s (slower) | `--spec-type`, n 1..12 |
Sources: PrismML Bonsai-demo community benchmarks (Vulkan PTQ1_0, `-ub 512 -b 2048`); llama.cpp #22534/#19408 (hybrid prefix checkpoints, TTFT 70 s → 1 s on repeats); llama.cpp #27360 (680M GTT/`--no-host`, mmap OOM workaround); #23973 (Vulkan UMA contiguous-write patch); sudoingx MTP graft README (MTP verify on 1.75 bpw kernels costs ~3x; best n-max 1, gain only +1.6..8 %).

## Phase 0 — Runtime optimization (measure, keep only what wins)
Each item: change one variable, `llama-bench`/server timings, keep if better, log in `docs/perf-log.md`. Acceptance: TTFT(OpenCode, 2nd request) < 20 s; first request < 3 min; tg ≥ 5.8 tok/s.
1. **Prompt size** (biggest lever): lean project OpenCode config (done: 10 MCPs/plugins off, project scope only). Measure resulting prompt tokens.
2. **Prefix cache for the hybrid model**: `--ctx-checkpoints 64..128 --checkpoint-every-n-tokens 4096 --swa-full --cache-reuse 256 --keep -1`; verify 2nd identical-prefix request skips prefill.
3. **Prefill kernel path**: `-ub` sweep (64..1024, running), `-b`, CPU prefill vs iGPU (`-ngl 0`) vs mixed (`-ngl 16/32`), `-t 8` vs 12/16, `GGML_VK_*` env (e.g. `GGML_VK_DISABLE_COOPMAT`, `GGML_VK_FORCE_MAX_ALLOCATION_SIZE`), Windows High-performance power plan + GPU clock.
4. **Memory**: KV q8_0 vs q4_0 (+kv-mean-center), `--no-mmap` vs mmap, `-ctk` quality check, UMA frame-buffer size note (BIOS), keep pagefile as SSD buffer.
5. **Speculation**: pretrained same-tokenizer drafter (Qwen3.5-0.8B Q8, already downloaded in `drafter/`), `--spec-type draft-simple`, n-max sweep 1..12; keep only if tok/s improves. Real LoRA distillation on Bonsai logs only if the pretrained drafter helps (train in CPU torch venv; **not** the simulated `lora_*` MCP tools).
6. **MoEfication**: not applicable to a dense quantized GGUF (needs retraining); documented, not built. Hardware-feasible equivalents are items 2–5.

## Phase 1 — Finish the gateway (Tasks 4–8 of the existing plan)
Loop: I write the tests first → local Bonsai (via OpenCode, lean config) implements until green → I run pytest natively and review the diff → fix loop (max 5 rounds, then I implement). No cloud agents.
- T4 Scheduler (lazy load, LRU, exclusive), T5 LLM adapter+app (default port **8090**, not 8080), T6 image adapter (flux2 experimental flag), T7 video/3D = `unsupported` with real errors (ONNX audio chain lacks `.data` weights), T8 self-heal e2e + runner + honest README table.

## Phase 2 — Integration on this machine
1. Gateway autostart at Windows logon via Scheduled Task (gateway only; **no model loaded** until first request).
2. OpenCode global default → `bonsai/bonsai-27b` through the gateway (edit after gateway proven; keep backup of the old config; move the plaintext tokens in the global config to env vars).
3. Local RSI: curated trajectories → compact skill file (real, prompt-level) injected into implementer prompts; log analysis script; drafter distillation per Phase 0.5.

## Phase 3 — Packaging
`pyproject` + console entry point `localai-gateway`, install script (PowerShell) that downloads runtimes (not models), `localai-gateway doctor` (hardware + model check), tests for install path, wheel build.

## Phase 4 — WebAssembly/Vercel demo (honest scope)
Browser cannot run 27B. Demo = static site: (a) in-browser tiny model (Qwen3.5-0.8B-class via WebGPU/wllama, measured), (b) "connect to your gateway" mode (CORS-safe localhost) for the real backend, (c) benchmark table from `docs/inventory.json`. Deploy with Vercel tooling only at the Release Gate.

## Release Gate (all must be true before any push)
1. `pytest -m "not real"` and `pytest -m real` green on this machine; perf-log shows Phase 0 acceptance met or the miss documented.
2. Secret scan clean (no `ghp_`/Bearer tokens/`.env`; `.gitignore` covers models, bin, drafter, .venv*, .superpowers).
3. README states only measured facts, unsupported models listed with real errors.
4. Then in order: create GitHub repo + push `feat/gateway` → CI green → build release (wheel + installer zip, no model files) → Vercel preview → production → README final → merge to `main` → tag.
Outward actions (repo creation, push, release, deploy, merge) are the ones the user pre-authorized for this sequence; each is done once, in this order, after the gate.
