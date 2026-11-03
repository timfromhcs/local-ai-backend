"""FastAPI gateway: OpenAI-compatible routes over the scheduler; `duo` (runtime virtual) is orchestrated, not loaded."""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .adapters.llm import BackendError, forward_chat, iter_stream, open_stream
from .config import Settings
from .duo import Duo, DuoConfig, to_sse_chunks
from .modes import ModeController
from .registry import ModelSpec
from .scheduler import DoesNotFit, ModelUnavailable
from .warmup import normalize_for_cache
from .supervisor import BackendStartError


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
        """Non-stream call used by Duo for each role; always releases."""
        ep, err = await acquire(model_id)
        if err is not None:
            raise BackendError(f"{model_id}: {err.body.decode()[:200]}")
        try:
            return await forward_chat(ep.base_url, {**prepare(model_id, body), "stream": False}, holder["client"])
        finally:
            await scheduler.release(model_id)

    cfg = duo_config or DuoConfig()

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
                                                 "messages": [{"role": "user", "content": prompt}]}, holder["client"])
            return (r["choices"][0]["message"].get("content") or "").strip()
        finally:
            await scheduler.release(cfg.fast)

    duo = Duo(call_model, cfg, modes=ModeController(ask_jev))

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict:
        loaded = set(scheduler.loaded())
        return {"object": "list", "data": [
            {"id": s.id, "object": "model", "kind": s.kind, "status": s.status, "reason": s.reason,
             "runtime": s.runtime, "loaded": s.id in loaded} for s in registry.values()]}

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
                return JSONResponse(await forward_chat(ep.base_url, body, client))
            except BackendError as exc:
                return error(503, "backend_error", str(exc))
            finally:
                release_later(mid)
        try:
            resp = await open_stream(ep.base_url, body, client)
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
