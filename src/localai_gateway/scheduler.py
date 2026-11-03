"""Memory-aware model scheduler: lazy load, LRU eviction, exclusive models, in-use refcounts."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .config import Settings
from .registry import ModelSpec
from .supervisor import free_port

log = logging.getLogger("localai_gateway.scheduler")


class DoesNotFit(Exception):
    pass


class ModelUnavailable(Exception):
    pass


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    api_key: str | None = None


@dataclass
class _Entry:
    mp: Any
    port: int
    mem: int
    exclusive: bool
    in_use: int = 0


def default_command(settings: Settings):
    def build(spec: ModelSpec, port: int) -> tuple[list[str], str]:
        exe = settings.llama_path() if spec.runtime == "llama" else settings.sd_path()
        argv = [str(exe), "--host" if spec.runtime == "llama" else "--listen-ip", "127.0.0.1"]
        argv += ["--port" if spec.runtime == "llama" else "--listen-port", str(port)]
        if spec.runtime == "llama":
            argv += ["--model", str(spec.files[0]), "--slot-save-path", str(settings.cache_dir)]
        argv += list(spec.args)
        return argv, f"http://127.0.0.1:{port}/health"
    return build


class Scheduler:
    def __init__(self, settings: Settings, registry: dict[str, ModelSpec], supervisor: Any,
                 build_command: Callable[[ModelSpec, int], tuple[list[str], str]] | None = None,
                 port_allocator: Callable[[], int] | None = None, wait_timeout_s: float = 120.0,
                 on_started: Callable[[ModelSpec, Endpoint], Awaitable[None]] | None = None) -> None:
        self.settings, self.registry, self.supervisor = settings, registry, supervisor
        self._build = build_command or default_command(settings)
        self._alloc = port_allocator or free_port
        self._timeout = wait_timeout_s
        self._on_started = on_started
        self._cond = asyncio.Condition()
        self._loaded: dict[str, _Entry] = {}      # insertion order = LRU order (oldest first)
        self._loading: dict[str, asyncio.Future] = {}
        self._reserved: dict[str, tuple[int, bool]] = {}  # models being started: (mem, exclusive)

    def loaded(self) -> list[str]:
        return list(self._loaded)

    def _used_mb(self) -> int:
        return sum(e.mem for e in self._loaded.values()) + sum(m for m, _ in self._reserved.values())

    async def ensure_loaded(self, model_id: str) -> Endpoint:
        spec = self.registry[model_id]  # KeyError for unknown ids
        if spec.status != "ready":
            raise ModelUnavailable(spec.reason or spec.status)
        if spec.runtime == "external":  # already running elsewhere (e.g. the Jev server): no spawn, no budget
            if not spec.endpoint:
                raise ModelUnavailable(f"{model_id}: external model without endpoint")
            return Endpoint(spec.endpoint.rstrip("/"), api_key=spec.api_key)
        if spec.runtime == "virtual":
            raise ModelUnavailable(f"{model_id} is a virtual model; it is orchestrated by the gateway, not loaded")
        if spec.est_mem_mb > self.settings.budget_mb:
            raise DoesNotFit(f"{model_id} needs {spec.est_mem_mb} MB, budget is {self.settings.budget_mb} MB")
        fresh = False  # True right after we (or a concurrent caller) started it: supervisor just verified health
        while True:
            async with self._cond:
                entry = self._loaded.get(model_id)
                if entry is not None:
                    if not fresh and not await self.supervisor.health(entry.mp):
                        await self._drop(model_id)
                        continue
                    entry.in_use += 1
                    self._loaded.pop(model_id)
                    self._loaded[model_id] = entry  # touch -> most recently used
                    return Endpoint(f"http://127.0.0.1:{entry.port}")
                fut = self._loading.get(model_id)
                if fut is None:
                    await self._make_room(spec)
                    fut = asyncio.get_running_loop().create_future()
                    self._loading[model_id] = fut
                    self._reserved[model_id] = (spec.est_mem_mb, spec.exclusive)
                    starter = True
                else:
                    starter = False
            if starter:
                await self._start(spec, fut)
            else:
                await asyncio.shield(fut)
            fresh = True  # loop: pick up the freshly loaded entry without a redundant health probe

    async def _make_room(self, spec: ModelSpec) -> None:
        """Called with the condition held; evicts idle LRU entries until `spec` fits, else waits."""
        async def fits_or_evict() -> bool:
            if any(x for x in self._reserved.values() if x[1]) or (spec.exclusive and self._reserved):
                return False
            victims = [i for i, e in self._loaded.items() if e.exclusive or spec.exclusive]
            if any(self._loaded[i].in_use for i in victims):
                return False
            for i in victims:
                await self._drop(i)
            while self._used_mb() + spec.est_mem_mb > self.settings.budget_mb:
                idle = next((i for i, e in self._loaded.items() if e.in_use == 0), None)
                if idle is None:
                    return False
                await self._drop(idle)
            return True

        deadline = asyncio.get_running_loop().time() + self._timeout
        while not await fits_or_evict():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise DoesNotFit(f"busy: cannot free memory for {spec.id} within {self._timeout}s")
            try:
                await asyncio.wait_for(self._cond.wait(), remaining)
            except asyncio.TimeoutError:
                raise DoesNotFit(f"busy: cannot free memory for {spec.id} within {self._timeout}s") from None

    async def _drop(self, model_id: str) -> None:
        entry = self._loaded.pop(model_id)
        try:
            await self.supervisor.stop(entry.mp)
        finally:
            self._cond.notify_all()

    async def _start(self, spec: ModelSpec, fut: asyncio.Future) -> None:
        port = self._alloc()
        argv, health_url = self._build(spec, port)
        try:
            mp = await self.supervisor.start(spec, argv, port, health_url)
        except BaseException as exc:
            async with self._cond:
                self._loading.pop(spec.id, None)
                self._reserved.pop(spec.id, None)
                self._cond.notify_all()
            if not fut.done():
                fut.set_exception(exc)
                fut.exception()  # mark retrieved: waiters re-raise, avoid "never retrieved" warnings
            raise
        if self._on_started is not None:  # e.g. restore/precompute the OpenCode prefix BEFORE the model becomes visible
            try:
                await self._on_started(spec, Endpoint(f"http://127.0.0.1:{port}"))
            except Exception:  # a failing warm-up must never prevent the model from serving
                log.exception("on_started hook failed for %s", spec.id)
        async with self._cond:
            self._loaded[spec.id] = _Entry(mp=mp, port=port, mem=spec.est_mem_mb, exclusive=spec.exclusive)
            self._loading.pop(spec.id, None)
            self._reserved.pop(spec.id, None)
            self._cond.notify_all()
        if not fut.done():
            fut.set_result(None)

    async def release(self, model_id: str) -> None:
        async with self._cond:
            entry = self._loaded.get(model_id)
            if entry is not None and entry.in_use > 0:
                entry.in_use -= 1
            self._cond.notify_all()

    async def shutdown(self) -> None:
        """Stop every loaded backend. Without this, a killed gateway leaves llama-server
        processes holding Vulkan device memory and the next start fails with an allocation error
        (observed: 204 MB KV cache could not be allocated)."""
        async with self._cond:
            entries = list(self._loaded.items())
            self._loaded.clear()
            self._cond.notify_all()
        for _, entry in entries:
            try:
                await self.supervisor.stop(entry.mp)
            except Exception:  # shutdown must never raise
                log.exception("failed to stop backend while shutting down")
