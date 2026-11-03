<p align="center">
  <img src="web/assets/logo.svg" width="96" alt="LocalAI Gateway logo" />
</p>

<h1 align="center">LocalAI Gateway</h1>

<p align="center">
  One OpenAI-compatible endpoint in front of the models already on your disk.<br />
  Lazy loading, a RAM budget, a plan/execute/verify <code>duo</code> model, and honest numbers.
</p>

<p align="center">
  <a href="https://local-ai-backend-blue.vercel.app"><b>Live demo</b></a> ·
  <a href="docs/perf-log.md">Performance log</a> ·
  <a href="docs/superpowers/plans/2026-11-03-mcp-stack-and-release.md">Plan &amp; status</a>
</p>

---

## What this is

A local gateway for GGUF models on Windows (Vulkan), written to be measured rather than
marketed. It gives your editor, scripts and MCP servers a single
`http://127.0.0.1:8090/v1` endpoint, and decides for you which model answers.

- **Lazy loading** — no model is loaded until the first request.
- **RAM budget** — a scheduler keeps the working set under a configurable limit and evicts
  the least recently used model.
- **Refcounts** — a model is never stopped mid-stream, even if the client disconnects.
- **`duo` virtual model** — the big model plans and verifies, the small model executes with
  tools, errors escalate deterministically instead of looping.
- **Prompt-prefix cache** — llama-server slot state is persisted to SSD and restored at start.
- **Single-file EXE** — the gateway ships without model weights, on purpose.

## Quick start

```powershell
# 1. build (no weights included)
powershell -ExecutionPolicy Bypass -File scripts/build-exe.ps1

# 2. see what this machine can actually run
dist\localai-gateway.exe --doctor

# 3. put your GGUF next to the exe, list it in models.yaml, then
dist\localai-gateway.exe --port 8090

# 4. any OpenAI client works
curl http://127.0.0.1:8090/v1/models
curl http://127.0.0.1:8090/v1/chat/completions ^
  -H "content-type: application/json" ^
  -d "{\"model\":\"duo\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"
```

Python source instead of the exe:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e .[dev]
.\.venv\Scripts\python -m localai_gateway --doctor
.\.venv\Scripts\python -m localai_gateway --port 8090
```

## Measured, not estimated

Ryzen 7 7735HS (8C/16T), 24 GB LPDDR5X, Radeon 680M (RDNA2, 12 CU, Vulkan), Windows 11,
llama.cpp Vulkan build. Full table and rejected experiments: [`docs/perf-log.md`](docs/perf-log.md).

| Metric | Value | Notes |
|---|---|---|
| Bonsai 2 27B q4 decode | **5.9 tok/s** | 88 GB/s theoretical RAM bandwidth is the wall |
| Same model, prefill | **~32 tok/s** | ≈50 % of FP32 peak; `-ub 256` won the sweep |
| OpenCode prompt, 16.2k tokens | **0.8 s** prompt phase, 5.0 s total | cached prefix reused |
| Same prompt, cold | **70–190 s** | why prefix caching is the biggest lever |
| Qwen3 4B q4 | **22.6 tok/s** | executor in the duo loop |
| Qwen2.5 Coder 1.5B q4 | **55.7 tok/s** | System-1 router |
| Unit tests | **133 passed** | `pytest -m "not real"`, no model downloads |

What is **not** possible here: 50–100 tok/s on a 27B model (needs ~410–820 GB/s), matrix-core
speedups (RDNA2 has none), or ROCm being faster (measured ~6× slower than Vulkan on this APU).
The gateway makes the model you have usable; it does not invent bandwidth.

## Architecture

```
client (editor / script / MCP)
        │  OpenAI chat.completions
        ▼
   gateway app  ──► mode controller ──► fast | duo | big-out
        │                                   │
        ▼                                   ▼
   scheduler: RAM budget · LRU · refcount  Jev :8080 (1.5B classifier)
        │
        ▼
   llama-server per model (:18xxx)  ──► slot state on SSD (prompt prefix)
```

Errors are structured and stable: `model_not_found` (404), `model_unavailable` (503),
`does_not_fit` (503), `backend_error` (503). CORS allows explicit origins plus
`*.vercel.app` previews — never a wildcard.

## The web demo

[`web/`](web/) is a static site (no build step) deployed at
**https://local-ai-backend-blue.vercel.app**:

- **Path A** runs a small model in the tab via **ONNX** (Transformers.js) on WebGPU, falling
  back to WebAssembly. GGUF is an inference format for llama.cpp; browsers run ONNX, so the
  demo uses ONNX and says so.
- **Path B** connects to your local gateway (`http://127.0.0.1:8090`) — the CORS contract is
  covered by tests.
- **Lab** trains a real softmax head on frozen ONNX embeddings (teacher-labelled routes
  `fast/duo/bigout`) with SGD, and reports accuracy before/after on held-out tasks. That is
  genuine training of genuine weights — of a head, not of the LLM.
- **EN / DE / FR**, no framework, 8 sections of long-form "Zero to Hero" documentation.

## Benchmark suite

```powershell
python -m bench.run --gateway http://127.0.0.1:8090 --model qwen3-4b --suite fast
python -m bench.run --gateway http://127.0.0.1:8090 --model qwen3-4b --suite long
```

Scoring is deterministic Python (routing words, tool-call shape, strict JSON, arithmetic,
honesty, concision, patch shape). No LLM judge, so results are cheap and repeatable.

## MCP stack (14 local servers)

The gateway is the model layer; the MCP layer is the agent's hands. All of it is local and
verified by a boot-time probe:

| Server | Tools | Role |
|---|---|---|
| github | 91 | repos, PRs, issues, actions |
| vercel | 241 | deployments, domains, logs |
| better-design | 45 | design systems, UI review |
| chrome-devtools | 32 | fast CDP browser, traces, console |
| playwright | 25 | E2E, DOM inspection |
| system-one-jev | 21 | fast decisions, routing |
| knowledge-db (Chroma) | 13 | local vector DB, seeded best practices |
| basic-memory | 12 | durable markdown knowledge base |
| memory | 9 | knowledge graph |
| blender | 9 | 3D, needs Blender running |
| local-crawl | 6 | local crawler: HTTP fast path → local Chromium, disk cache, robots.txt |
| ast-grep | 4 | structural search |
| context7 | 2 | library docs |
| fetch | 1 | single-page fetch |

Pinned installs (`~/.config/opencode/mcp`), absolute paths, `mcp-bootstrap.ps1` for repair,
`mcp-health.ps1` for a real `initialize` + `tools/list` probe, and Windows scheduled tasks
(`OpenCodeMCPHealth`, `OpenCodeJevGuard`) so a reboot cannot silently break the stack.

## Repository layout

```
src/localai_gateway/   gateway app, scheduler, supervisor, duo orchestration, modes, warmup
src/rsi_lab/           trajectory scrub/harvest/clean/split + promotion gate
bench/                 deterministic benchmark suite (fast + long)
web/                   static ONNX/WebGPU demo, EN/DE/FR, lab
packaging/             PyInstaller spec + entry point
scripts/               build-exe.ps1, bonsai-server.ps1, prefix capture
tests/                 133 hermetic tests (gateway, scheduler, duo, modes, RSI, CORS, bench)
docs/                  perf log, plans, design specs, inventory
models.yaml            model registry (files, memory estimates, measured flags)
```

Model files (`*.gguf`, `*.onnx`, `*.safetensors`), runtimes (`bin/`), the venv and caches are
git-ignored and never committed.

## Honest limits

- A 4B model does not reason like a frontier model. `duo` improves tool-call reliability, not
  intelligence.
- No browser can run a 27B model. The web demo is a demo, not a substitute for the gateway.
- Fine-tuning here means LoRA on CPU for small models; there is no CUDA and no bitsandbytes.
- The RSI lab (chat → training pairs → gated promotion) is partially built. The plan documents
  exactly which parts exist and which do not.
- `lance-video` and `slat-3d` are registered as **unsupported** with their real loader errors.

## Roadmap

See [`docs/superpowers/plans/2026-11-03-mcp-stack-and-release.md`](docs/superpowers/plans/2026-11-03-mcp-stack-and-release.md):
RSI capture MCP, LoRA distillation into the 1.5B router with a promotion gate, tiny
text-diffusion experiments on the iGPU, and a Docker SWE runner for verified episodes.

## Licence

Code: MIT. Model weights are **not** included and keep their own licences. The demo page and
its assets are part of this repository.
