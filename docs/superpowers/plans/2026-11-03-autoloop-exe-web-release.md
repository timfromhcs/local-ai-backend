# Plan: Autoloop (SWE-bench teacher→4B training) → solo EXE → WebGPU demo on Vercel → GitHub → README → main

Authority: user request 2026-11-03 ("erst Plan, dann autonom bis Ende"). Outward actions (GitHub repo + push, Vercel deploy, merge to main, release) are explicitly requested and are done once each, in order, only after the Release Gate.

## Facts that shape the plan (measured/checked on this machine)
- Docker 29.6.1 works (VM 16 GB / 16 CPUs), WSL2 Ubuntu present, a real SWE-bench image `sweb.eval.x86_64.sympy__sympy-20590` (4.05 GB) already exists. Node 22, npm, Vercel CLI 65.9, `gh` logged in (timfromhcs), `claude` CLI 2.1.316, 125 GB free.
- RAM is the shared bottleneck: gateway pair (27B 9 GB + 4B 3.6 GB) + Docker VM (up to 16 GB) + training cannot run together. The Autoloop therefore **serializes phases** and asks the scheduler to evict models before training.
- Realistic scores: a 4B model solves ~0-3 % of SWE-bench Verified. The loop's value is *relative* improvement on a fixed fast bench, kept only if better.
- Training here = LoRA on CPU in bf16/fp32 (no CUDA, bitsandbytes QLoRA not available on Windows/AMD). 4B LoRA ≈ minutes per few hundred tokens → datasets must be tiny and curated; runs are hours.

## Autoloop (RSI Lab v2) – "forever", checkpointed, shell-first
```
[teacher traces] → harvest → scrub → clean → split → TRAIN(LoRA, 4B, checkpoints every N steps)
      ↑                                                      ↓ merge → GGUF → quantize
  SWE tasks run in Docker (linux venv, always via shell) ←──  FAST BENCH (4B only) → gate: promote iff better, else discard
```
- Teacher sources: (1) Claude Code session logs (`~/.claude/projects`), (2) opt-in headless `claude -p` runs on SWE-bench instances (cloud, only when `--teacher claude` is passed), (3) local big model via `duo`. Only **verified** episodes (SWE-bench tests pass / pytest green) enter training.
- Student rollouts: the 4B solves SWE tasks through the gateway inside Docker (`mini`-style bash-only loop: the agent's only tool is `bash`), logging every step (tokens, seconds, exit codes, pass/fail) to `rsi/runs/<ts>/`.
- Fast bench (4B only, ~10 min): 20 tool-call cases + 20 HumanEval+/MBPP+ problems + 3 SWE-bench instances (sympy image present). Gate: promote iff primary ≥ champion + margin, no category drops > tolerance, tool-call validity never drops.
- Loop controller: state machine with a lock file, resumable, writes `rsi/state.json`; checkpoints every N optimizer steps are evaluated on a 5-minute mini-bench; best checkpoint competes in the full gate; failed/worse candidates are deleted (adapter kept only as metrics).

## Milestones (each ends green before the next)
M1 Modes: Jev-driven mode controller (fast / duo / small-in-big-out) with periodic check-ups; tests.
M2 RSI core: scrubber, harvester, cleaner, splitter, gate (tests already written in `rsi_pending/`) + loop state machine + dry-run end-to-end with a tiny model.
M3 Docker SWE runner: bash-only agent loop in a container, SWE-bench evaluation of a patch, run once for real on the local sympy instance with the 4B via the gateway.
M4 Train/merge/bench: peft LoRA (CPU) with checkpoints, merge→HF→GGUF→quantize, fast bench; real smoke run on a tiny dataset; then start the infinite loop as a background job.
M5 Solo EXE: PyInstaller one-file `localai-gateway.exe` (gateway + model picker + auto-download of runtimes, no model files inside), smoke-tested.
M6 Web demo: WebGPU/WASM in-browser LLM (WebLLM), hardware auto-detect gates (WebGPU adapter, memory, limits → model tier), "Zero to Hero" guide, "made with love by hcsmedia", connect-to-local-gateway mode; Playwright-verified; Vercel preview then production.
M7 Release Gate → GitHub: secret scan, tests green, README (honest, measured), CI (pytest + exe build), create repo `timfromhcs/local-ai-backend`, push, release with exe artifact, merge to `main`.

## Release Gate (all true before any outward action)
`pytest -m "not real"` green; real smoke tests recorded; secret scan clean (no `ghp_`, Bearer, `.env`, tokens in global OpenCode config not copied); `.gitignore` excludes models/bin/cache/drafter/.venv*/.superpowers; README states only measured facts.
