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
