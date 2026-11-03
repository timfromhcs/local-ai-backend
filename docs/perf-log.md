# Performance log (Ryzen 7 7735HS, 24 GB, Radeon 680M Vulkan, Bonsai-2-27B MIXED-PTQ1-Q4_0)
Measured with llama-bench / llama-server timings on this machine, 2026-11-02. Higher is better.

| Experiment | Prefill pp (tok/s) | Decode tg (tok/s) | Verdict |
|---|---|---|---|
| iGPU, -ub 64 / 128 / 256 / 512 / 1024 (pp512) | 30.1 / 31.4 / **31.7** / 30.9 / 27.0 | – | use `-ub 256` |
| iGPU full offload, -ub 256, t=8 (pp512, tg32) | **32.1** | **5.90** | baseline |
| CPU only, t=8 (pp128, tg16) | 28.2 | 2.41 | prefill close, decode 2.4x slower |
| Mixed `-ngl 32` | 29.7 | 2.88 | worse than full iGPU |
| iGPU, t=16 | 23.8 | 5.85 | worse (SMT hurts) |
| Server, short prompt, reasoning off, -ngl 99 | 16.0 | 5.81 | (first server run, ub 512) |
| Server, -ngl 20, reasoning on | 6.3 | 2.11 | worse |
| Draft ngram-mod 1..12, novel text | – | 5.53 (0/12 accepted) | no gain; 7.29 only on exact repeat |
| Draft ngram-map-k 1..12, novel text | – | 5.2 | slower than baseline |
| Env `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` (pp256, tg24) | 26.8 (ref 32.9) | 3.19 (ref 5.96) | int-dot path IS used and valuable: keep default |
| Env `GGML_VK_DISABLE_F16=1` | 24.7 | 5.96 | fp16 path is used: keep default |
| Env `DISABLE_DOT2` / `FORCE_MMVQ` / `DISABLE_FUSION` | 32.6 / 32.7 / 32.2 | 6.03 / 5.86 / 5.92 | no effect (within noise) |
| Env `GGML_VK_PREFER_HOST_MEMORY=1` | 31.4 | 5.71 | slightly worse |
| ROCm/HIP on RDNA2 APU | – | – | rejected (research: ~6x slower than Vulkan) |
| Matrix cores / coopmat translation | – | – | impossible: RDNA2 has no WMMA hardware |

Hardware ceiling: pp ~32 tok/s x 27B x 2 FLOP = ~1.7 TFLOP/s of ~3.4 TFLOP/s FP32 peak (~50 %).

Conclusion: default Vulkan kernels with int-dot + fp16 are already the best path on this iGPU; remaining TTFT gains come from prompt size and prefix checkpoints, not kernel switches.

## TTFT: OpenCode prompt size (the dominant lever)
| Config | First-request prompt | Prefill at ~32 tok/s | 
|---|---|---|
| Global OpenCode (10 MCP servers incl. github --toolsets=all, plugin, AGENTS.md) | ~13,600 tokens | ~7-9 min |
| Lean project config (`opencode.jsonc`: MCPs/plugins off, project scope) | ~512 tokens | **~16 s** |
Speed-up ~26x on time-to-first-token without touching the model or the GPU path.

## Speed-path measurements (2026-11-03)
| Test | Result |
|---|---|
| Verify-cost curve, Bonsai (prompt batch of k tokens) | k=1 6.0, 2 8.3, 3 10.4, 4 8.9, 6 9.9, 8 10.3, 12 6.4, 16 11.2 tok/s (=166/242/290/449/604/775/1860/1427 ms per step) |
| Parallel slots, aggregate decode (`llama-batched-bench`) | B=1 5.53, B=2 5.93, B=4 7.49, B=6 8.46 tok/s; with prefill 8.9/9.7/12.6/13.5 |
| Qwen3.5-0.8B Q8_0 | pp128 1163 tok/s, **tg64 71.3 tok/s** |
| Qwen2.5-Coder-1.5B Q4_K_M (Jev) | pp128 550, **tg64 55.7** |
| Qwen3-4B Q4_K_M | pp128 201, **tg64 22.6** |
| RAM | 4x6 GB LPDDR5X soldered, 128-bit, configured 5500 MT/s (rated 6400) = 88 GB/s peak; Bonsai decode uses 48 GB/s effective |

## Correction + OpenCode prompt precompute (2026-11-03)
**Correction:** the earlier "TTFT 26x better" claim was wrong. The 512-token request was only OpenCode's *title* request; the real agent request is ~16.2k tokens and was unchanged by disabling MCP servers. Cause measured with a mock server: system prompt 51.8k chars of which `<available_skills>` = 38.0k chars (73 %), plus 13 tool schemas = 17.4k chars. Skills stay enabled (user decision); the prefix is precomputed instead.
| Case (Qwen3-4B Q4_K_M, 16.2k-token OpenCode prompt) | Time | Cache |
|---|---|---|
| No cache (first request, prefill ~4k tok/s... measured) | 70-190 s | 0 |
| Precomputed prefix in memory (second request) | **5.0 s total, 0.8 s prompt phase** | cache_n 16255 / new 10 |
| Fresh gateway start: load 4B + restore slot from SSD (1.2 GB) + request | **8.5 s total**, restore 0.6 s | cache_n 16235 / new 30 |
Design: `<env>` block moved to the first user message (chat templates render tool schemas right after the system text, so anything volatile in the system prompt kills the cached tool section); stable boundary = common prefix of two renderings; slot state saved/restored via `--slot-save-path` on SSD (`cache/slots/*.bin`).
Open check: first-ever cold compute showed cache_n 12141 on the first request (second request fine); re-verify on a clean cache.

## Speculative decoding measurement (2026-11-03)

Measured with `scripts/measure-speculative.ps1` (same binary, `-ngl 99 -fa on -t 8 -ub 256 -c 2048`,
greedy, 64 tokens, fresh prompt so the prefix cache cannot fake the prefill).

| Target | Config | Decode tok/s | Prefill tok/s | Draft acceptance |
|---|---|---|---|---|
| Qwen2.5-Coder 1.5B q4 | baseline | **63.83** | 307.0 | – |
| Qwen2.5-Coder 1.5B q4 | + 0.5B q3 drafter, `--spec-type draft-simple`, n-max 3 | **38.53 (-40 %)** | 152.0 | **57.1 % / 58.0 %**, mean len 2.33–2.74 |
| Qwen3-4B q4 | baseline | 22.28 | 105.8 | – |
| Qwen3-4B q4 | + Qwen3.5-0.8B q8 drafter | 22.19 | 158.4 | **speculation disabled** |

Two findings, both useful:

1. **Vocab mismatch is fatal, not slow.** `Qwen3.5-0.8B` cannot draft for `Qwen3-4B`; llama-server logs
   `the target and draft vocabs are not compatible` and disables speculation at load time. The identical
   tok/s in that row is therefore *not* evidence about speculation — it is evidence that the pair was
   rejected. Anyone benchmarking must check the log line `speculative decoding enabled|disabled`.
2. **On this RDNA2 iGPU, speculation loses even at 58 % acceptance.** The earlier estimate said
   break-even needs >= 46 % acceptance; that model assumed the drafter is almost free. In reality the
   0.5B draft and the 1.5B verify run on the same 12 CUs, so the extra forward passes cost more than the
   saved verify work: -40 % decode, -50 % prefill.

Decision: **speculation stays off** in `models.yaml` for this hardware. The measured escape hatch is
small-model routing (Jev 1.5B at ~56 tok/s, 0.5B at ~64 tok/s), not drafting for a bigger model.
Re-test only on a discrete GPU with spare compute, and re-check acceptance before believing a speedup.
