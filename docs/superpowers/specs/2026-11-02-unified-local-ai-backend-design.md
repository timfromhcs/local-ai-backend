# Unified Local AI Backend — Design (Sub-projects 1–3)

Date: 2026-11-02 · Status: draft, approved to proceed under user's auto-mode delegation

## Intent
One local server ("gateway") that exposes every model in this folder behind one
OpenAI-compatible API, loads/unloads models automatically within a memory budget, and
heals itself when a backend process dies. OpenCode (and any OpenAI client) uses it as a
provider. Later sub-projects (4–7: OpenCode/MCP/Blender integration, packaging, cloud
offload, Vercel WASM demo, release) build on this core and get their own specs.

Success = on the target machine (Ryzen 7 7735HS, 24 GB RAM, Radeon 680M, Vulkan, Windows 11):
- `POST /v1/chat/completions` answers from Qwen3-4B and Bonsai-27B, streaming works.
- `POST /v1/images/generations` returns an image from FLUX.2-klein.
- Video (Lance) and image-to-3D (SLAT) either work, or are marked `unsupported` with a
  recorded, reproducible reason (no silent failure, no fake success).
- Requesting a model that does not fit evicts the least-recently-used one; never OOMs the host.
- Killing a backend process mid-session is recovered automatically; a test proves it.

## Non-goals (this spec)
Cloud offload, PyPI/installer, WASM demo, Blender/MCP wiring, LoRA training, auth/multi-user.

## Facts from inventory (verified by reading file headers)
| Model | Arch | Size | Runtime candidate |
|---|---|---|---|
| Bonsai-2-27B Q4_0 | qwen35 | 8.5 GB | llama-prism (llama.cpp fork, Vulkan) |
| Qwen3-4B-abliterated Q4_K_M | qwen3 | 2.5 GB | llama-prism |
| bonsai-flux2-klein ternary Q2_K | FLUX.2 klein | 1.4 GB | sd-server (stable-diffusion.cpp, Vulkan) |
| Lance 3B Video Q5_K_M | lance | 5.5 GB | UNVERIFIED — arch may be unsupported |
| SLAT flow imgshape2tex 1.3B Q4_K_M | flux (TRELLIS-like) | 0.8 GB | UNVERIFIED — plus ss_dec_conv3d decoder |
| vae / diffusion_pytorch_model / ss_dec safetensors | VAE / decoders | 0.15–1.4 GB | companions of the above |
| dacvae_decoder, dit_single_step, t5_encoder, vision_encoder (.onnx) | unknown chain | ≤5.7 MB | UNVERIFIED — identify via onnxruntime |

Unverified rows are the project's main technical risk, so they are **tested first**.

## Architecture
Python 3.13, FastAPI + uvicorn, `httpx` for proxying, `psutil` for memory, `pydantic` config.

```
client (OpenCode / curl)
   │  OpenAI-compatible HTTP
   ▼
gateway (FastAPI)
 ├─ registry     models.yaml → ModelSpec(id, kind, files, runtime, est_mem_mb, args)
 ├─ scheduler    memory budget, LRU eviction, per-model lock, load/unload
 ├─ supervisor   spawns/monitors runtime processes, health probes, restart w/ backoff
 └─ adapters     one per kind, translate OpenAI API ⇄ runtime API
      ├─ llm      → llama-server  (/v1/chat/completions passthrough, SSE streaming)
      ├─ image    → sd-server     (/v1/images/generations)
      ├─ video    → runtime TBD by Phase 1 result
      └─ image3d  → runtime TBD by Phase 1 result (output: .glb/.obj)
```

Units and interfaces (each testable alone):
- `registry.load(path) -> dict[id, ModelSpec]`
- `Scheduler.ensure_loaded(id) -> Endpoint` (may evict; raises `DoesNotFit`)
- `Supervisor.start(spec) -> Process`, `.health(proc) -> bool`, `.stop(proc)`
- `Adapter.handle(request, endpoint) -> response|stream`

Memory policy: budget = 70% of physical RAM by default (config). Each spec carries
`est_mem_mb` that Phase 1 replaces with measured values. Models marked `exclusive`
(Bonsai-27B) evict everything else. Vulkan iGPU memory is shared with RAM, so one budget.

Error handling: unsupported/failed model → HTTP 503 with machine-readable `reason`;
backend crash → supervisor restarts (max 3, exponential backoff), request gets 503 + retry hint
if restart fails; no request ever hangs without a timeout.

## Phases (each ends green before the next starts)
1. **Smoketest harness** — script extracts zips to `bin/`, runs each model once, records
   load time, peak RAM, tokens/s or seconds/image into `docs/inventory.json`; identifies ONNX
   graphs (inputs/outputs) with onnxruntime. Output decides video/3D feasibility.
2. **Core** — registry, supervisor, scheduler, LLM + image adapters, `/v1/models`, `/healthz`.
3. **Video + 3D adapters** — only for models Phase 1 proved runnable; others stay `unsupported`.

## Testing
pytest. Unit tests with a fake runtime process (scheduler eviction, LRU, restart/backoff).
Integration tests (marked `@pytest.mark.real`) hit real models; they are what "green" means
for Phase 1–3. Self-healing loop: failing test → minimal patch → rerun, never whole-file rewrites.

## Open risks (stated honestly)
- Lance and SLAT may have no runnable runtime here → documented as unsupported, not faked.
- Bonsai-27B (8.5 GB) on a 12-CU iGPU will be slow; measured, not promised.
- Vulkan fork behavior (llama-prism) may differ from upstream flags; verified in Phase 1.
