<p align="center">
  <img src="web/assets/logo.svg" width="96" alt="LocalAI Gateway logo" />
</p>

<h1 align="center">LocalAI Gateway</h1>

<p align="center">
  One OpenAI-compatible endpoint in front of the models already on your disk.<br />
  Lazy loading, a RAM budget, SAT-based placement, device-aware autotuning, and honest numbers.
</p>

<p align="center">
  <a href="https://github.com/timfromhcs/local-ai-backend/actions/workflows/ci.yml"><img src="https://github.com/timfromhcs/local-ai-backend/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/timfromhcs/local-ai-backend/releases/latest"><img src="https://img.shields.io/github/v/release/timfromhcs/local-ai-backend" alt="Latest release" /></a>
  <a href="https://local-ai-backend-blue.vercel.app"><img src="https://img.shields.io/badge/demo-live-6366f1" alt="Live demo" /></a>
  <img src="https://img.shields.io/badge/python-3.13-3776ab" alt="Python 3.13" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT" />
</p>

<p align="center">
  <a href="https://local-ai-backend-blue.vercel.app"><b>Live demo</b></a> ·
  <a href="docs/perf-log.md">Performance log</a> ·
  <a href="docs/superpowers/plans/2026-11-08-v2.0-surfaces.md">v2.0 plan</a>
</p>

---

## What this is

A local gateway for GGUF models on Windows (Vulkan), written to be measured rather than
marketed. It gives editors, scripts and MCP clients a single `http://127.0.0.1:8090/v1`
endpoint and decides which model answers.

- **Lazy loading** — no model is loaded until the first request.
- **RAM budget** — a scheduler keeps the working set under a configured limit, evicts LRU.
- **Refcounts** — a model is never stopped mid-stream, even if the client disconnects.
- **`duo` virtual model** — big model plans and verifies, small model executes with tools.
- **SAT placement** — which models can be resident together, proven against brute force.
- **Device autotune** — iGPU / dGPU / CPU profiles with a reason per flag.
- **Prompt-prefix cache** — llama-server slot state persisted to SSD and restored at start.
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
.\localai-gateway.exe --version
```

The EXE is unsigned, so Windows SmartScreen warns on first run ("More info -> Run anyway").
That is stated here instead of hidden; no code-signing certificate is configured yet.

## Surfaces

| Surface | What runs there | Limit |
|---|---|---|
| Windows EXE / wheel | gateway + llama-server (Vulkan) | needs your GGUF files |
| Docker (`api/Dockerfile`, `ghcr.io/timfromhcs/local-ai-backend`) | gateway API + MCP | no GPU, no weights in the image |
| Vercel static `web/` | ONNX tier, GGUF via WASM (wllama), lab | browser memory only |
| Phone browser | same page, 320-2560 px, IndexedDB persistence | <= 1.5B q4 realistically |
| MCP `servers/localai-gateway-mcp` | status, models, chat, benchmark, doctor, placement, autotune | talks to a running gateway |

### Planning and autotuning

```powershell
# which models fit in RAM at the same time? (SAT solver, deterministic)
curl http://127.0.0.1:8090/v1/placement
curl "http://127.0.0.1:8090/v1/placement?budget_mb=8000&kinds=llm&pinned=jev-1.5b"

# device-aware llama.cpp flags, one reason per flag
curl "http://127.0.0.1:8090/v1/autotune?model=qwen3-4b&ctx=16384"
```

`placement` maximises the number of resident models under the budget, then uses as much of the
budget as possible. The objective and the cardinality encoding are cross-checked against brute
force on thousands of random instances. `autotune` distinguishes iGPU (shared memory), dGPU
(discrete VRAM) and CPU-only, and never promises kernel switches that measured as neutral here.

### Your data stays local

The web demo writes settings, the selected model and benchmark runs into **IndexedDB**, with
JSON export/import and a delete button. No analytics, no upload, no account.

## Measured, not estimated

Ryzen 7 7735HS (8C/16T), 24 GB LPDDR5X, Radeon 680M (RDNA2, 12 CU, Vulkan), Windows 11,
llama.cpp Vulkan build. Full table and rejected experiments: [`docs/perf-log.md`](docs/perf-log.md).

| Metric | Value | Notes |
|---|---|---|
| Bonsai 2 27B q4 decode | **5.9 tok/s** | 88 GB/s theoretical RAM bandwidth is the wall |
| Same model, prefill | **~32 tok/s** | about 50 % of FP32 peak; `-ub 256` won the sweep |
| OpenCode prompt, 16.2k tokens | **0.8 s** prompt phase, 5.0 s total | cached prefix reused |
| Same prompt, cold | **70-190 s** | why prefix caching is the biggest lever |
| Qwen3 4B q4 | **22.6 tok/s** | executor in the duo loop |
| Qwen2.5 Coder 1.5B q4 | **55.7 tok/s** | System-1 router |
| Tests | **180 passed** | `pytest -m "not real"`, no model downloads |

Not possible here: 50-100 tok/s on a 27B model (needs 410-820 GB/s), matrix-core speedups
(RDNA2 has none), or ROCm being faster (measured about 6x slower than Vulkan on this APU).

## Architecture

```
client (editor / script / MCP)
        |  OpenAI chat.completions
        v
   gateway app  --> mode controller --> fast | duo | big-out
        |                                  |
        v                                  v
   scheduler: RAM budget, LRU, refcount   Jev :8080 (1.5B classifier)
        |
        v
   llama-server per model (:18xxx)  --> slot state on SSD (prompt prefix)
```

Errors are structured and stable: `model_not_found` (404), `model_unavailable` (503),
`does_not_fit` (503), `no_placement` (409), `backend_error` (503). CORS allows explicit origins
plus `*.vercel.app` previews, never a wildcard.

## The web demo

[`web/`](web/) is a static site (no build step) at
**https://local-ai-backend-blue.vercel.app**:

- **Path A - ONNX (WebGPU/WASM)**: a small ONNX model runs in the tab via Transformers.js.
- **Path C - GGUF (WASM)**: llama.cpp compiled to WebAssembly through wllama, loading real GGUF
  files (LFM2.5 350M, Qwen2.5 0.5B/1.5B, SmolLM2 360M) from Hugging Face. Same format as the
  local gateway, no server, cached in the browser. Single-threaded without COOP/COEP.
- **Path B - your local gateway**: connects to `http://127.0.0.1:8090`; CORS is tested.
- **Lab**: trains a real softmax head on frozen ONNX embeddings with SGD and reports accuracy
  before/after. Real training of real weights - of a head, not of the LLM.
- **SEO**: canonical, `hreflang` for EN/DE/FR, `SoftwareApplication` + `FAQPage` JSON-LD,
  `robots.txt`, `sitemap.xml` - all validated by `scripts/check-web.mjs`.
- **EN / DE / FR** with 153 keys per locale, parity enforced by the web gate.

## Benchmark suite

```powershell
python -m bench.run --gateway http://127.0.0.1:8090 --model qwen3-4b --suite fast
python -m bench.run --gateway http://127.0.0.1:8090 --model qwen3-4b --suite long
```

Scoring is deterministic Python (routing words, tool-call shape, strict JSON, arithmetic,
honesty, concision, patch shape). No LLM judge, so results are cheap and repeatable.

## MCP stack

14 local servers plus the gateway's own MCP, all verified by a boot-time probe with real
`initialize` + `tools/list` handshakes. Highlights: github (91 tools), vercel (241),
better-design (45), chrome-devtools (32), playwright (25), system-one-jev (21),
knowledge-db/Chroma (13, seeded best practices), basic-memory (12), local-crawl (6),
localai-gateway (7), blender, memory, ast-grep, context7, fetch.

## Repository layout

```
src/localai_gateway/   gateway, scheduler, supervisor, duo, modes, warmup,
                       placement (SAT), hardware (autotune), config
src/rsi_lab/           trajectory scrub/harvest/clean/split + promotion gate
bench/                 deterministic benchmark suite (fast + long)
web/                   static ONNX/GGUF/WASM demo, EN/DE/FR, lab, IndexedDB, SEO files
api/                   Dockerfile + compose (API shell, no weights)
packages/npm/          launcher that downloads and verifies the release binary
servers/               localai-gateway-mcp
packaging/             PyInstaller spec + entry point
scripts/               build-exe, check-web, probes, bonsai-server
tests/                 180 hermetic tests (gateway, scheduler, duo, modes, RSI, CORS,
                       bench, placement, autotune, surfaces, release hygiene)
docs/                  perf log, plans, design specs, inventory
models.yaml            model registry (files, memory estimates, measured flags)
```

Model files (`*.gguf`, `*.onnx`, `*.safetensors`), runtimes (`bin/`), the venv and caches are
git-ignored and never committed.

## Honest limits

- A 4B model does not reason like a frontier model. `duo` improves tool-call reliability, not
  intelligence.
- Vercel containers have no GPU: they serve the API/MCP shell, not the 27B model.
- No browser runs a 27B model. The web demo is a demo, not a substitute for the gateway.
- Fine-tuning here means LoRA on CPU for small models; there is no CUDA and no bitsandbytes.
- The placement solver targets small instances; Kissat-WASM is the documented upgrade.
- `lance-video` and `slat-3d` are registered as **unsupported** with their real loader errors.

## Roadmap

See [`docs/superpowers/plans/2026-11-08-v2.0-surfaces.md`](docs/superpowers/plans/2026-11-08-v2.0-surfaces.md):
closed API contract with OpenAPI snapshots, publishing tokens for npm/PyPI, real-device mobile
pass, RSI capture MCP, and two consecutive green releases before the v1.0 tag.

## Licence

Code: MIT. Model weights are **not** included and keep their own licences. The demo page and
its assets are part of this repository.
