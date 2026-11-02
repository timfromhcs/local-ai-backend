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

## Phase 0A — SOTA research result: "matrix cores / translation layer" on a Radeon 680M (what is and is not possible)
| Idea | Verdict | Evidence |
|---|---|---|
| Matrix cores via KHR_cooperative_matrix | **Not available.** RDNA2 has no WMMA hardware; only RDNA3/4 expose coopmat in Vulkan. Our device line says `matrix cores: none`. | llama.cpp Vulkan device log; int8-coopmat work (b11160, #27952) targets RDNA3/4 only |
| A translation/emulation layer that adds matrix cores | **Cannot create hardware.** A coopmat emulation runs on the same shader ALUs as today's `mul_mm`, so no speedup. Not built. | Vulkan coopmat spec/0cc4m: "matrix cores only used with coopmat; otherwise regular shader" |
| ROCm/HIP on gfx1035 (via HSA override) | **Rejected.** On RDNA2 APUs ROCm measured ~6x slower than Vulkan and ~3x slower than CPU (untuned rocBLAS: 76 % `fallback` kernels); `HSA_OVERRIDE_GFX_VERSION` unsupported on Windows. | unsloth ROCM_RDNA2_APU.md; llama.cpp build docs |
| Our actual ceiling | Prefill `pp512` ≈ 31 tok/s at 27 B ⇒ ~1.7 TFLOP/s effective vs. ~3.4 TFLOP/s FP32 peak (12 CU × 64 × 2 × ~2.2 GHz): **already ~50 % of peak**. Max realistic kernel gain ≈ 1.3–2x, not 10x. | measured `bench_ub.md` + arithmetic |
| What can still move the needle | (1) fewer prompt tokens (lean OpenCode config: done), (2) prefix checkpoints (TTFT 70 s → 1 s on repeats), (3) int8-dot / DOT2 / fp16 shader paths on `int dot: 1` (test env switches `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT`, `GGML_VK_DISABLE_DOT2`, `GGML_VK_DISABLE_F16`, `GGML_VK_FORCE_MMVQ`), (4) CPU+iGPU split prefill (`-ngl` mix, 8 Zen3 AVX2 cores), (5) newer upstream Vulkan shaders (RMS_NORM fusion +4 %, `GGML_VK_SUBALLOCATION_BLOCK_SIZE`) via a source build **only if** (3)/(4) show headroom, (6) AMD Adrenalin driver update. | research + env list in `ggml-vulkan.dll` |
Measured so far: `-ub` 64/128/256/512/1024 → 30.1/31.4/**31.7**/30.9/27.0 tok/s ⇒ use `-ub 256`.

## Phase 0B — "50–100 tok/s": what the physics allows on this machine (research + measured hardware facts)
Hardware fact (measured via WMI): RAM = **4 x 6 GB soldered LPDDR5X, 128-bit, configured 5500 MT/s** (rated 6400) ⇒ **88 GB/s theoretical**. Our decode of 5.9 tok/s x 8.2 GB/token = **48 GB/s effective (55 % of peak)**; realistic ceiling ~62-66 GB/s ⇒ **max ~8 tok/s for this 27 B model, no matter the software** (100 tok/s would need ~820 GB/s, 50 tok/s ~410 GB/s).
| Lever | Single-stream decode | Prefill | Verdict |
|---|---|---|---|
| Faster kernels / flags | ≤ +30 % (bandwidth wall) | already ~50 % of FP32 peak | done; see perf-log |
| Speculative decoding (draft model, MTP) | on a bandwidth-bound iGPU, verifying k tokens costs ~max(mem time, k/32 s): k≈4-5 is ~free ⇒ up to ~2-3x ⇒ **~12-15 tok/s realistic**, not 50 | – | worth building: pretrained Qwen3.5-0.8B drafter + our distillation; measure verify-cost curve first |
| Parallel slots (`-np 4..5`) | one weight read serves k streams: **aggregate ~25-30 tok/s** across parallel agents/requests (single request still ~6) | shared | build into gateway for multi-agent loops |
| MoE-class models (3B active): gpt-oss-20b measured 29 tok/s on a 780M/DDR5-5600 mini PC; Qwen3.6-35B-A3B ~50 on 128 GB Strix Halo | **~25-30 tok/s on this RAM** | – | different model (user asked to keep Bonsai) → optional extra "fast lane" |
| Sub-1 GB active weights (Qwen3.5-0.8B Q4, Jev 1.5B Q4) | **50-100 tok/s possible** (0.5 GB/token at ~48 GB/s ≈ 95 tok/s) | 300+ | use as System-1 / drafter / router, not as the 27 B replacement |
| Prefill 50+ tok/s on the 27 B | – | needs MMQ tile/dp4a work in `ggml-vulkan` (Strix Halo MMQ port: 2.3x; gfx10 dp4a inner loop alone: negative result) | kernel project, uncertain; only after the cheap wins |
| Higher RAM clock | BIOS/firmware-locked on this laptop (config 5500 < rated 6400) | – | not reachable from the OS |
Decision: keep Bonsai for quality, make *effective* speed higher by (1) tiny prompts (done), (2) prefix cache, (3) spec decoding, (4) parallel slots, (5) route simple/System-1 work to a sub-1 GB model in the gateway.

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
3. **RSI Lab** (own spec: `docs/superpowers/specs/2026-11-02-rsi-lab-design.md`): harvest teacher logs → clean/scrub → LoRA on trainable students (Jev Qwen2.5-Coder-1.5B, Qwen3.5-0.8B drafter; Bonsai-27B is not trainable here) → merge → GGUF → benchmark gate (BFCL, EvalPlus, project tool-call suite, SWE-bench slice informational) → promote only if better, with mid-checkpoint keep-best and rollback; triggered after each chat by a SessionEnd hook, trained when idle. Plus prompt-level skill memory for Bonsai.

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
