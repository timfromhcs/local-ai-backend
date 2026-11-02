# Unified Local AI Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local FastAPI gateway that serves every model in `C:\LocalAIBACKEND` through one OpenAI-compatible API with memory-aware load/evict and self-healing backends.

**Architecture:** Gateway = registry + scheduler (RAM budget, LRU) + supervisor (spawn, health, restart) + per-modality adapters that proxy to `llama-server` (LLM) and `sd-server` (image). Video/3D adapters are built only if Phase 1 proves a runtime exists.

**Tech Stack:** Python 3.13, FastAPI, uvicorn, httpx, psutil, pydantic v2, PyYAML, onnxruntime (inspection only), pytest, pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-11-02-unified-local-ai-backend-design.md`

## Global Constraints
- Python 3.13; package `localai_gateway` under `src/`; models and binaries are never committed (see `.gitignore`).
- Memory budget default = 70% of physical RAM; `exclusive` models evict all others.
- Backend restart: max 3 attempts, exponential backoff (1s, 2s, 4s).
- Failure responses: HTTP 503 with JSON `{"error": {"code": str, "reason": str}}`; no request without a timeout.
- Unsupported models are reported as `unsupported` with a recorded reason, never faked.
- Commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus
1. Two concurrent requests for different models that cannot coexist → second waits/evicts, no deadlock, no double-spawn.
2. Client disconnects mid-stream → backend request cancelled, model not left locked.
3. Request for unknown model id → 404 `model_not_found`, lists valid ids.
4. Backend port already in use / process exits immediately at start → clean 503, no zombie.
5. Model file missing on disk → model listed as `unavailable`, other models unaffected.

## File Structure
- `pyproject.toml` — package + deps + pytest config
- `models.yaml` — registry (id, kind, files, runtime, est_mem_mb, exclusive, args)
- `src/localai_gateway/config.py` — `Settings`, `ModelSpec`
- `src/localai_gateway/registry.py` — YAML → `dict[str, ModelSpec]`, file-existence status
- `src/localai_gateway/supervisor.py` — process lifecycle + health + restart
- `src/localai_gateway/scheduler.py` — budget, LRU, per-model locks
- `src/localai_gateway/adapters/{llm,image,video,image3d}.py`
- `src/localai_gateway/app.py` — FastAPI routes
- `tools/smoketest.py`, `tools/onnx_inspect.py` — Phase 1 harness
- `tests/` — unit (fake runtime `tests/fake_runtime.py`) and `real`-marked integration

---

### Task 1: Scaffold, config, registry
**Files:** Create `pyproject.toml`, `models.yaml`, `src/localai_gateway/{__init__,config,registry}.py`, `tests/test_registry.py`
**Interfaces — Produces:**
- `ModelSpec(id: str, kind: Literal["llm","image","video","image3d"], files: list[Path], runtime: str, est_mem_mb: int, exclusive: bool=False, args: list[str]=[], status: Literal["ready","unavailable","unsupported"]="ready", reason: str|None=None)`
- `Settings(budget_mb: int, bin_dir: Path, model_dir: Path, port: int=8080)` with `Settings.default()` (70% RAM via psutil)
- `load_registry(path: Path, model_dir: Path) -> dict[str, ModelSpec]`

- [ ] **Step 1: Failing tests** in `tests/test_registry.py`: `test_loads_models_yaml` (ids include `qwen3-4b`, `bonsai-27b`, `flux2-klein`), `test_missing_file_marks_unavailable` (tmp dir without files → every spec `status=="unavailable"`, `reason` names the file), `test_default_budget_is_70_percent_ram` (within 1% of `psutil.virtual_memory().total*0.7/2**20`).
- [ ] **Step 2:** Run `pytest tests/test_registry.py -v` → FAIL (import error).
- [ ] **Step 3:** Implement the interfaces above; `models.yaml` entries use est_mem_mb: qwen3-4b 3500, bonsai-27b 11000 (`exclusive: true`), flux2-klein 3000, lance-video 8000, slat-3d 3000 (the last two start `unsupported`, reason "unverified in phase 1").
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Commit `feat: scaffold package, config and registry`.

### Task 2: Phase 1 smoketest harness
**Files:** Create `tools/smoketest.py`, `tools/onnx_inspect.py`, `tests/test_smoketest_parse.py`; output `docs/inventory.json`
**Interfaces — Produces:** `extract_runtimes(root: Path, bin_dir: Path) -> None` (unzips both vendor zips into `bin/llama` and `bin/sd`); `run_llm_probe(spec, bin_dir) -> ProbeResult`; `run_image_probe(...)`; `inspect_onnx(path: Path) -> dict` (`inputs`, `outputs` with names, shapes, dtypes); `ProbeResult(model_id, ok: bool, load_s: float|None, peak_rss_mb: float|None, metric: float|None, error: str|None)`.

- [ ] **Step 1:** Failing test `test_parse_llama_timings` — given a captured `llama-server` log string with `eval time = … tokens per second`, `parse_tps(log) == 12.34` (fixture in test).
- [ ] **Step 2:** Run → FAIL. **Step 3:** implement `parse_tps(log: str) -> float|None`. **Step 4:** PASS.
- [ ] **Step 5:** Implement harness. Probe each model once with a 30s load timeout x2 for 27B; sample peak RSS with psutil; run `inspect_onnx` on all four `.onnx` files (`pip install onnxruntime` first). Lance and SLAT: attempt `llama-server`/`sd-cli` load; on architecture error record the exact stderr as `error`.
- [ ] **Step 6:** Run `python tools/smoketest.py` → writes `docs/inventory.json`; every model has `ok` true or a recorded `error`. Update `models.yaml` `est_mem_mb` with measured values and flip `lance-video`/`slat-3d` status per result.
- [ ] **Step 7:** Commit `feat: phase 1 smoketest and measured inventory` (include `docs/inventory.json`).

### Task 3: Supervisor
**Files:** Create `src/localai_gateway/supervisor.py`, `tests/fake_runtime.py`, `tests/test_supervisor.py`
**Interfaces — Produces:** `class Supervisor` with `async start(spec, argv: list[str], port: int, health_url: str) -> ManagedProcess`, `async stop(mp)`, `async health(mp) -> bool`; `ManagedProcess(pid, port, started_at)`; `class BackendStartError(Exception)`. `fake_runtime.py` is an HTTP server with `/health`, honouring env `FAKE_CRASH_AFTER` seconds.

- [ ] **Step 1:** Failing tests: `test_start_waits_for_health`, `test_crash_is_restarted_up_to_3_times` (assert backoff 1,2,4 via injected sleep), `test_immediate_exit_raises_BackendStartError_no_zombie`, `test_port_in_use_raises`.
- [ ] **Step 2:** FAIL. **Step 3:** implement with `asyncio.create_subprocess_exec`, poll health every 0.25s to a 120s timeout (configurable). **Step 4:** PASS. **Step 5:** commit `feat: process supervisor with restart/backoff`.

### Task 4: Scheduler
**Files:** Create `src/localai_gateway/scheduler.py`, `tests/test_scheduler.py`
**Interfaces — Consumes:** `Supervisor`, `ModelSpec`, `Settings`. **Produces:** `class Scheduler(settings, registry, supervisor)` with `async ensure_loaded(model_id) -> Endpoint`, `async release(model_id)`, `loaded() -> list[str]`; `Endpoint(base_url: str)`; `DoesNotFit`, `ModelUnavailable` exceptions.

- [ ] **Step 1:** Failing tests with fake supervisor: `test_loads_when_fits`, `test_evicts_lru_when_over_budget`, `test_exclusive_evicts_all`, `test_too_big_raises_DoesNotFit`, `test_unavailable_raises`, `test_concurrent_requests_do_not_double_spawn` (asyncio.gather of 5 same-model calls → one start), `test_in_use_model_not_evicted_until_released`.
- [ ] **Step 2:** FAIL. **Step 3:** implement with one `asyncio.Lock` per model plus a global lock for eviction decisions; in-use refcount via `release`. **Step 4:** PASS. **Step 5:** commit `feat: memory-aware scheduler with LRU eviction`.

### Task 5: LLM adapter and app
**Files:** Create `src/localai_gateway/adapters/llm.py`, `src/localai_gateway/app.py`, `tests/test_app_llm.py`
**Interfaces — Produces:** `create_app(settings, registry, scheduler) -> FastAPI`; routes `GET /healthz`, `GET /v1/models` (lists all specs with status), `POST /v1/chat/completions` (stream and non-stream passthrough to llama-server).

- [ ] **Step 1:** Failing tests using the fake runtime as LLM backend: `test_models_lists_status`, `test_chat_nonstream`, `test_chat_stream_sse_chunks`, `test_unknown_model_404_model_not_found`, `test_client_disconnect_releases_model`, `test_backend_failure_returns_503_json`.
- [ ] **Step 2:** FAIL. **Step 3:** implement adapter with `httpx.AsyncClient` streaming, `release()` in `finally`. **Step 4:** PASS.
- [ ] **Step 5:** Real test (`@pytest.mark.real`): start gateway, POST chat to `qwen3-4b`, assert non-empty content; then to `bonsai-27b`, assert qwen3-4b was evicted (`/v1/models` shows loaded set). Run `pytest -m real -v` → PASS.
- [ ] **Step 6:** Commit `feat: gateway app and LLM adapter`.

### Task 6: Image adapter
**Files:** Create `src/localai_gateway/adapters/image.py`, `tests/test_app_image.py`
**Interfaces — Produces:** `POST /v1/images/generations` body `{model, prompt, size="512x512", n=1}` → `{"data":[{"b64_json": str}]}`.

- [ ] **Step 1:** Failing tests with fake sd runtime: `test_image_returns_b64`, `test_invalid_size_422`, `test_image_model_unavailable_503`.
- [ ] **Step 2–4:** FAIL → implement (sd-server request mapping verified against Phase 1 notes) → PASS.
- [ ] **Step 5:** Real test: generate 512x512 with `flux2-klein`; assert PNG magic bytes. **Step 6:** commit `feat: image generation adapter`.

### Task 7: Video and image-to-3D adapters (conditional)
**Files:** Create `adapters/video.py`, `adapters/image3d.py`, `tests/test_app_video3d.py`
Precondition: `docs/inventory.json` shows `ok: true` for `lance-video` / `slat-3d`. For each model that is not `ok`, the only work is a test asserting `POST` returns 503 `{"code":"unsupported","reason": <inventory error>}`, and the model stays `unsupported` in `models.yaml`.
**Interfaces — Produces:** `POST /v1/videos/generations` → `{"data":[{"b64_mp4": str}]}`; `POST /v1/3d/generations` (multipart image) → `{"data":[{"glb_b64": str}]}`.

- [ ] **Step 1:** Write failing tests per supported/unsupported case. **Step 2:** FAIL. **Step 3:** implement per inventory findings. **Step 4:** PASS (real tests for supported ones). **Step 5:** commit `feat: video and 3D adapters per phase 1 findings`.

### Task 8: Self-healing end-to-end and runner
**Files:** Create `src/localai_gateway/__main__.py`, `scripts/run.ps1`, `tests/test_selfheal_e2e.py`, `README.md` (honest status table copied from `docs/inventory.json`)
- [ ] **Step 1:** Failing test `test_backend_killed_midsession_recovers` (real, qwen3-4b): load, `psutil.Process(pid).kill()`, next chat request succeeds within 120s.
- [ ] **Step 2:** FAIL. **Step 3:** wire supervisor restart into scheduler (`ensure_loaded` re-checks `health` before returning endpoint); `python -m localai_gateway` starts uvicorn on port 8080. **Step 4:** PASS; run full `pytest -v` and `pytest -m real -v` green. **Step 5:** commit `feat: self-healing e2e, runner and README`.

---
## Self-review
Spec coverage: phases 1/2/3 → Tasks 2 / 1,3–6,8 / 7; memory policy → Task 4; error format → Tasks 3,5; honest-unsupported → Tasks 2,7. Types consistent across tasks (`ModelSpec`, `Endpoint`, `Supervisor.start`). Sub-projects 4–7 intentionally out of scope (own specs).
