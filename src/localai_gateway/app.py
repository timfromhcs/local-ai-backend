"""FastAPI gateway: OpenAI-compatible routes over the scheduler; `duo` (runtime virtual) is orchestrated, not loaded."""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__

from .adapters.llm import BackendError, forward_chat, iter_stream, open_stream
from .benchscore import SUITES as BENCH_SUITES, BenchResult
from .config import Settings, config_file
from .doctor import report
from .duo import Duo, DuoConfig, to_sse_chunks
from .exposure import max_body_bytes
from .hardware import detect_device, profile_for
from .modes import ModeController
from .placement import Unsatisfiable, solve_placement
from .registry import ModelSpec
from .scheduler import DoesNotFit, ModelUnavailable
from .warmup import normalize_for_cache
from .supervisor import BackendStartError
from rsi_lab.capture import CaptureError, CaptureLog


def error(status: int, code: str, reason: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "reason": reason, **extra}}, status_code=status)


DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8090",
    "http://127.0.0.1:8090",
)


def cors_origins() -> list[str]:
    """Explicit origins only (never '*'), overridable with LOCALAI_CORS_ORIGINS=a,b,c."""
    raw = os.environ.get("LOCALAI_CORS_ORIGINS", "")
    items = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    return items or list(DEFAULT_CORS_ORIGINS)


def create_app(settings: Settings, registry: dict[str, ModelSpec], scheduler: Any,
               http_client: httpx.AsyncClient | None = None, duo_config: DuoConfig | None = None) -> FastAPI:
    holder: dict[str, Any] = {"client": http_client}
    pending: set[asyncio.Task] = set()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        owned = holder["client"] is None
        if owned:
            holder["client"] = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0))
        try:
            yield
        finally:
            if owned:
                await holder["client"].aclose()
            # stop backends on graceful shutdown; otherwise they keep the GPU memory and the
            # next start dies with a Vulkan allocation error
            try:
                await scheduler.shutdown()
            except Exception:  # pragma: no cover - shutdown must never mask the real error
                pass

    app = FastAPI(title="localai-gateway", lifespan=lifespan)
    # The static demo (Vercel preview/production) and local dev servers call this gateway
    # from the browser. Explicit origins + vercel.app previews, never "*".
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_origin_regex=r"https://[a-z0-9][a-z0-9.-]*\.vercel\.app",
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        max_age=86400,
    )
    app.state.scheduler, app.state.registry = scheduler, registry

    def release_later(model_id: str) -> None:
        """Release without awaiting: safe inside cancelled streaming generators (client disconnect)."""
        task = asyncio.ensure_future(scheduler.release(model_id))
        pending.add(task)
        task.add_done_callback(pending.discard)

    def prepare(model_id: str, body: dict) -> dict:
        """Per-model request prep: thinking stays ON (model default; the caller may override); stable cacheable prefix."""
        b = {**body}
        spec = registry.get(model_id)
        return normalize_for_cache(b) if spec is not None and spec.warm_prefix else b

    async def acquire(model_id: str):
        try:
            return await scheduler.ensure_loaded(model_id), None
        except KeyError:
            return None, error(404, "model_not_found", f"unknown model {model_id!r}", valid_ids=sorted(registry))
        except ModelUnavailable as exc:
            return None, error(503, "model_unavailable", str(exc))
        except DoesNotFit as exc:
            return None, error(503, "does_not_fit", str(exc))
        except BackendStartError as exc:
            return None, error(503, "backend_error", str(exc))

    async def call_model(model_id: str, body: dict) -> dict:
        """Non-stream call used by Duo and the benchmark; always releases."""
        ep, err = await acquire(model_id)
        if err is not None:
            raise BackendError(f"{model_id}: {err.body.decode()[:200]}")
        try:
            return await forward_chat(ep.base_url, {**prepare(model_id, body), "stream": False}, holder["client"], getattr(ep, "api_key", None))
        finally:
            await scheduler.release(model_id)

    cfg = duo_config or DuoConfig()
    captures = CaptureLog(settings.model_dir / "rsi")

    async def ask_jev(prompt: str) -> str | None:
        """Mode check-up on the fast external model (Jev); any failure => heuristics decide."""
        spec = registry.get(cfg.fast)
        if spec is None or spec.status != "ready":
            return None
        ep, err = await acquire(cfg.fast)
        if err is not None:
            return None
        try:
            r = await forward_chat(ep.base_url, {"model": cfg.fast, "max_tokens": 6, "temperature": 0,
                                                 "messages": [{"role": "user", "content": prompt}]},
                                   holder["client"], getattr(ep, "api_key", None))
            return (r["choices"][0]["message"].get("content") or "").strip()
        finally:
            await scheduler.release(cfg.fast)

    duo = Duo(call_model, cfg, modes=ModeController(ask_jev))

    @app.middleware("http")
    async def version_header(request: Request, call_next):
        limit = max_body_bytes()
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            return error(413, "payload_too_large",
                         f"request body exceeds {limit // (1024 * 1024)} MB", limit_bytes=limit)
        response = await call_next(request)
        response.headers["X-LocalAI-Version"] = __version__
        return response

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/doctor")
    async def doctor_route() -> JSONResponse:
        """Same report as `localai-gateway --doctor`, as JSON."""
        return JSONResponse(report(settings, registry, config_file("models.yaml")).as_dict())

    @app.get("/v1/models")
    async def models() -> dict:
        loaded = set(scheduler.loaded())
        return {"object": "list", "data": [
            {"id": s.id, "object": "model", "kind": s.kind, "status": s.status, "reason": s.reason,
             "runtime": s.runtime, "loaded": s.id in loaded} for s in registry.values()]}

    @app.get("/v1/placement")
    async def placement(budget_mb: int | None = None, kinds: str = "llm", pinned: str = "") -> JSONResponse:
        """SAT-based answer to "which models fit in RAM at the same time"."""
        budget = budget_mb or settings.budget_mb
        required = tuple(k.strip() for k in kinds.split(",") if k.strip()) or ("llm",)
        pins = tuple(p.strip() for p in pinned.split(",") if p.strip())
        try:
            result = solve_placement(registry, budget_mb=budget, required_kinds=required, pinned=pins)
        except Unsatisfiable as exc:
            return error(409, "no_placement", str(exc), budget_mb=budget, kinds=list(required))
        return JSONResponse({
            "models": list(result.models),
            "memory_mb": result.memory_mb,
            "budget_mb": budget,
            "required_kinds": list(result.required_kinds),
            "why": list(result.why),
        })

    @app.post("/v1/bench")
    async def bench(request: Request) -> JSONResponse:
        """Run the deterministic fast/long suite against one model. Scoring has no LLM judge."""
        try:
            body = await request.json()
        except ValueError:
            return error(400, "bad_request", "body is not valid JSON")
        model_id = body.get("model") if isinstance(body, dict) else None
        if not model_id:
            return error(422, "missing_model", "field 'model' is required")
        suite = body.get("suite", "fast")
        if suite not in BENCH_SUITES:
            return error(400, "unknown_suite", f"suite must be one of {sorted(BENCH_SUITES)}")
        spec = registry.get(model_id)
        if spec is None:
            return error(404, "model_not_found", f"unknown model {model_id!r}", valid_ids=sorted(registry))
        if spec.status != "ready":
            return error(503, "model_unavailable", spec.reason or spec.status)
        if spec.kind != "llm":
            return error(400, "wrong_kind", f"model {model_id!r} is a {spec.kind} model")

        max_tokens = int(body.get("max_tokens", 400))
        thinking = bool(body.get("thinking", False))
        wanted = body.get("tasks")
        tasks = [t for t in BENCH_SUITES[suite] if not wanted or t.id in wanted]
        if not tasks:
            return error(400, "unknown_task", f"no task matched {wanted!r}",
                         valid_ids=[t.id for t in BENCH_SUITES[suite]])
        virtual = spec.runtime == "virtual"
        rows: list[dict] = []
        for task in tasks:
            started = time.perf_counter()
            request = {
                "model": model_id,
                "messages": [{"role": "user", "content": task.prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
            }
            if not thinking:
                # measured: Qwen3 spends the whole budget on reasoning and returns empty content,
                # which scores 0 for every task. Scoring does not need a reasoning trace.
                request["chat_template_kwargs"] = {"enable_thinking": False}
            if task.tools:
                request["tools"] = list(task.tools)
                request["tool_choice"] = "auto"
            try:
                # virtual models (duo) are orchestrated by the app, not loaded by the scheduler
                resp = await (duo.chat(request) if virtual else call_model(model_id, request))
                msg = resp["choices"][0]["message"]
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                if task.score_message is not None:
                    score = float(task.score_message(msg))
                else:
                    score = float(task.score(content))
                err = None
            except Exception as exc:  # one broken task must not kill the run
                content, score, err = "", 0.0, f"{type(exc).__name__}: {exc}"[:300]
            rows.append({
                "id": task.id, "category": task.category, "weight": task.weight,
                "score": score, "seconds": round(time.perf_counter() - started, 2),
                "answer": content[:600], "error": err,
            })

        if all(r["error"] for r in rows):
            return error(502, "bench_failed", "every task failed", first=rows[0]["error"])
        return JSONResponse(BenchResult(model=model_id, suite=suite, rows=rows).as_dict())

    @app.post("/v1/rsi/sessions")
    async def rsi_start(request: Request) -> JSONResponse:
        """Start (or resume) a capture session. Append-only JSONL, scrubbed on write."""
        body = await request.json() if await request.body() else {}
        session = (body or {}).get("session") or f"rsi-{int(time.time())}"
        try:
            events = captures.events(session)
        except CaptureError as exc:
            return error(400, "capture_error", str(exc))
        return JSONResponse({"session": session, "events": len(events), "root": str(captures.root)})

    @app.post("/v1/rsi/events")
    async def rsi_event(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except ValueError:
            return error(400, "bad_request", "body is not valid JSON")
        session = body.pop("session", None) if isinstance(body, dict) else None
        if not session:
            return error(422, "missing_session", "field 'session' is required")
        try:
            record = captures.append(session, body)
        except CaptureError as exc:
            return error(400, "capture_error", str(exc))
        return JSONResponse({"ok": True, "session": session, "recorded": record["type"]})

    @app.post("/v1/rsi/finalize")
    async def rsi_finalize(request: Request) -> JSONResponse:
        body = await request.json()
        session = (body or {}).get("session")
        if not session:
            return error(422, "missing_session", "field 'session' is required")
        try:
            return JSONResponse(captures.finalize(session))
        except CaptureError as exc:
            return error(400, "capture_error", str(exc))

    @app.get("/v1/rsi/pairs")
    async def rsi_pairs(session: str) -> JSONResponse:
        try:
            return JSONResponse({"session": session, "pairs": captures.pairs(session)})
        except CaptureError as exc:
            return error(400, "capture_error", str(exc))

    @app.get("/v1/rsi/stats")
    async def rsi_stats() -> JSONResponse:
        return JSONResponse(captures.stats())

    @app.get("/v1/autotune")
    async def autotune(model: str, ctx: int = 16384) -> JSONResponse:
        """Device-aware llama.cpp flags for one registered model."""
        spec = registry.get(model)
        if spec is None:
            return error(404, "model_not_found", f"unknown model {model!r}", valid_ids=sorted(registry))
        if spec.est_mem_mb <= 0:
            return error(422, "no_memory_estimate", f"{model!r} has no est_mem_mb, cannot autotune")
        profile = profile_for(detect_device(), spec.est_mem_mb, ctx=ctx)
        return JSONResponse(profile.as_dict())

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return error(400, "bad_request", "body is not valid JSON")
        mid = body.get("model") if isinstance(body, dict) else None
        if not mid:
            return error(422, "missing_model", "field 'model' is required")
        spec = registry.get(mid)
        if spec is None:
            return error(404, "model_not_found", f"unknown model {mid!r}", valid_ids=sorted(registry))
        if spec.status != "ready":
            return error(503, "model_unavailable", spec.reason or spec.status)
        if spec.kind != "llm":
            return error(400, "wrong_kind", f"model {mid!r} is a {spec.kind} model; use its own endpoint")
        stream = bool(body.get("stream"))

        if spec.runtime == "virtual":
            try:
                result = await duo.chat(body)
            except (BackendError, KeyError, IndexError) as exc:
                return error(503, "backend_error", str(exc))
            if stream:
                return StreamingResponse(to_sse_chunks(result, mid), media_type="text/event-stream")
            return JSONResponse(result)

        ep, err = await acquire(mid)
        if err is not None:
            return err
        client = holder["client"]
        body = prepare(mid, body)
        if not stream:
            try:
                return JSONResponse(await forward_chat(ep.base_url, body, client, getattr(ep, "api_key", None)))
            except BackendError as exc:
                return error(503, "backend_error", str(exc))
            finally:
                release_later(mid)
        try:
            resp = await open_stream(ep.base_url, body, client, getattr(ep, "api_key", None))
        except BackendError as exc:
            release_later(mid)
            return error(503, "backend_error", str(exc))

        async def gen() -> AsyncIterator[bytes]:
            try:
                async for chunk in iter_stream(resp):
                    yield chunk
            finally:
                release_later(mid)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
