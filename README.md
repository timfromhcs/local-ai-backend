<p align="center">
  <img src="web/assets/logo.svg" width="96" alt="LocalAI Gateway logo" />
</p>

<h1 align="center">LocalAI Gateway</h1>

<p align="center">
  One OpenAI-compatible endpoint in front of the models already on your disk.<br />
  Lazy loading, a RAM budget, a plan/execute/verify <code>duo</code> model, and honest numbers.
</p>

<p align="center">
  <a href="https://github.com/timfromhcs/local-ai-backend/actions/workflows/ci.yml"><img src="https://github.com/timfromhcs/local-ai-backend/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/timfromhcs/local-ai-backend/releases/latest"><img src="https://img.shields.io/github/v/release/timfromhcs/local-ai-backend" alt="Latest release" /></a>
  <a href="https://local-ai-backend-blue.vercel.app"><img src="https://img.shields.io/badge/demo-live-6366f1" alt="Live demo" /></a>
  <img src="https://img.shields.io/badge/python-3.13-3776ab" alt="Python 3.13" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT" />
</p>

<p align="center">
  <a href="https://local-ai-backend-blue.vercel.app"><b>Live demo</b></a> Â·
  <a href="docs/perf-log.md">Performance log</a> Â·
  <a href="docs/superpowers/plans/2026-11-03-v0.2.0-stable.md">Release plan</a>
</p>

---

## What this is

A local gateway for GGUF models on Windows (Vulkan), written to be measured rather than
marketed. It gives your editor, scripts and MCP servers a single
`http://127.0.0.1:8090/v1` endpoint, and decides for you which model answers.

- **Lazy loading** â€” no model is loaded until the first request.
- **RAM budget** â€” a scheduler keeps the working set under a configurable limit and evicts
  the least recently used model.
- **Refcounts** â€” a model is never stopped mid-stream, even if the client disconnects.
- **`duo` virtual model** â€” the big model plans and verifies, the small model executes with
  tools, errors escalate deterministically instead of looping.
- **Prompt-prefix cache** â€” llama-server slot state is persisted to SSD and restored at start.
- **Single-file EXE** â€” the gateway ships without model weights, on purpose.

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

Three more ways to run it:

```powershell
# container (API only, no GPU; models stay on the host)
docker build -f api/Dockerfile -t localai-gateway .
docker run -p 8090:8090 localai-gateway

# npm launcher: downloads the matching release binary and verifies its SHA256
npx @hcsmedia/localai-gateway --doctor

# MCP: expose the running gateway to any MCP client
LOCALAI_GATEWAY_URL=http://127.0.0.1:8090 node servers/localai-gateway-mcp/index.js
```

Or download the built binary from
[Releases](https://github.com/timfromhcs/local-ai-backend/releases/latest) and verify it:

```powershell
gh release download v0.3.0 -p localai-gateway.exe -p SHA256SUMS.txt
Get-FileHash .\localai-gateway.exe -Algorithm SHA256   # compare with SHA256SUMS.txt
.\localai-gateway.exe --version                        # 0.3.0
```

The EXE is unsigned, so Windows SmartScreen warns on first run ("More info â†’ Run anyway").
That is stated here instead of hidden; a code-signing certificate is not configured yet.

## Surfaces

| Surface | What runs there | Limit |
|---|---|---|
| Windows EXE / wheel | the gateway + llama-server (Vulkan) | needs your GGUF files |
| Docker (`api/Dockerfile`, `ghcr.io/timfromhcs/local-ai-backend`) | the gateway API/MCP | no GPU, no weights in the image |
| Vercel (static `web/`) | ONNX tier, **GGUF via WASM (wllama)**, lab | browser memory only |
| Phone browser | same page, responsive 320â†’2560 px | â‰¤ 1.5B q4 realistically |
| MCP (`servers/localai-gateway-mcp`) | status, models, chat, benchmark, doctor | talks to a running gateway |

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
| Same model, prefill | **~32 tok/s** | â‰ˆ50 % of FP32 peak; `-ub 256` won the sweep |
| OpenCode prompt, 16.2k tokens | **0.8 s** prompt phase, 5.0 s total | cached prefix reused |
| Same prompt, cold | **70â€“190 s** | why prefix caching is the biggest lever |
| Qwen3 4B q4 | **22.6 tok/s** | executor in the duo loop |
| Qwen2.5 Coder 1.5B q4 | **55.7 tok/s** | System-1 router |
| Unit tests | **154 passed** | `pytest -m "not real"`, no model downloads |

What is **not** possible here: 50â€“100 tok/s on a 27B model (needs ~410â€“820 GB/s), matrix-core
speedups (RDNA2 has none), or ROCm being faster (measured ~6Ã— slower than Vulkan on this APU).
The gateway makes the model you have usable; it does not invent bandwidth.

## Architecture

```
client (editor / script / MCP)
        â”‚  OpenAI chat.completions
        â–¼
   gateway app  â”€â”€â–º mode controller â”€â”€â–º fast | duo | big-out
        â”‚                                   â”‚
        â–¼                                   â–¼
   scheduler: RAM budget Â· LRU Â· refcount  Jev :8080 (1.5B classifier)
        â”‚
        â–¼
   llama-server per model (:18xxx)  â”€â”€â–º slot state on SSD (prompt prefix)
```

Errors are structured and stable: `model_not_found` (404), `model_unavailable` (503),
`does_not_fit` (503), `backend_error` (503). CORS allows explicit origins plus
`*.vercel.app` previews â€” never a wildcard.

## The web demo

[`web/`](web/) is a static site (no build step) deployed at
**https://local-ai-backend-blue.vercel.app**:

- **Path A â€” ONNX (WebGPU/WASM)**: a small ONNX model runs in the tab via Transformers.js.
- **Path C â€” GGUF (WASM)**: llama.cpp compiled to WebAssembly through **wllama**, loading real
  GGUF files (LFM2.5 350M, Qwen2.5 0.5B/1.5B, SmolLM2 360M) straight from Hugging Face. Same
  format as the local gateway, no server in the loop, cached in the browser. Single-threaded
  without COOP/COEP, which is why 0.5B is the honest sweet spot.
- **Path B â€” your local gateway**: connects to `http://127.0.0.1:8090`; the CORS contract is tested.
- **Lab**: trains a real softmax head on frozen ONNX embeddings (teacher-labelled routes) with SGD
  and reports accuracy before/after. Real training of real weights â€” of a head, not of the LLM.
- **EN / DE / FR**, no framework, responsive from 320 px, 8 sections of long-form documentation.

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
| local-crawl | 6 | local crawler: HTTP fast path â†’ local Chromium, disk cache, robots.txt |
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
tests/                 154 hermetic tests (gateway, scheduler, duo, modes, RSI, CORS, bench, docker/npm/mcp surfaces)
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
- The RSI lab (chat â†’ training pairs â†’ gated promotion) is partially built. The plan documents
  exactly which parts exist and which do not.
- `lance-video` and `slat-3d` are registered as **unsupported** with their real loader errors.

## Roadmap

See [`docs/superpowers/plans/2026-11-03-mcp-stack-and-release.md`](docs/superpowers/plans/2026-11-03-mcp-stack-and-release.md):
RSI capture MCP, LoRA distillation into the 1.5B router with a promotion gate, tiny
text-diffusion experiments on the iGPU, and a Docker SWE runner for verified episodes.

## Licence

Code: MIT. Model weights are **not** included and keep their own licences. The demo page and
its assets are part of this repository.

