"""Scheduler contract tests (written first; the implementation must satisfy exactly these).

Scheduler(settings, registry, supervisor, build_command=None, port_allocator=None, wait_timeout_s=120.0)
  - build_command(spec, port) -> (argv: list[str], health_url: str); default builds llama-server / sd-server argv
  - port_allocator() -> int; default localai_gateway.supervisor.free_port
  - ensure_loaded(model_id) -> Endpoint  (increments in-use refcount; caller MUST release)
  - release(model_id) -> None
  - loaded() -> list[str]  (least recently used first)
Errors: KeyError (unknown id), ModelUnavailable (status != ready, message = spec.reason), DoesNotFit.
"""
import asyncio
from pathlib import Path

import pytest

from localai_gateway.config import Settings
from localai_gateway.registry import ModelSpec
from localai_gateway.scheduler import DoesNotFit, Endpoint, ModelUnavailable, Scheduler
from localai_gateway.supervisor import BackendStartError


def spec(mid, mem, exclusive=False, status="ready", reason=None):
    return ModelSpec(id=mid, kind="llm", files=[Path(mid)], runtime="llama", est_mem_mb=mem,
                     exclusive=exclusive, status=status, reason=reason)


class FakeMP:
    def __init__(self, port):
        self.port = port
        self.pid = 1000 + port


class FakeSupervisor:
    def __init__(self, start_delay=0.02):
        self.started, self.stopped, self.healthy = [], [], True
        self.fail_next = False
        self.start_delay = start_delay

    async def start(self, spec, argv, port, health_url):
        await asyncio.sleep(self.start_delay)
        if self.fail_next:
            self.fail_next = False
            raise BackendStartError("boom")
        self.started.append(spec.id)
        return FakeMP(port)

    async def stop(self, mp):
        self.stopped.append(mp.port)

    async def health(self, mp):
        return self.healthy


def make(specs, budget=10000, wait_timeout_s=2.0):
    ports = iter(range(18100, 18200))
    sup = FakeSupervisor()
    sched = Scheduler(
        Settings(budget_mb=budget, bin_dir=Path("bin"), model_dir=Path(".")),
        {s.id: s for s in specs},
        sup,
        build_command=lambda s, port: (["fake", s.id], f"http://127.0.0.1:{port}/health"),
        port_allocator=lambda: next(ports),
        wait_timeout_s=wait_timeout_s,
    )
    return sched, sup


async def test_loads_when_fits():
    sched, sup = make([spec("a", 3000)])
    ep = await sched.ensure_loaded("a")
    assert isinstance(ep, Endpoint) and ep.base_url == "http://127.0.0.1:18100"
    assert sched.loaded() == ["a"] and sup.started == ["a"]


async def test_second_call_reuses_running_backend():
    sched, sup = make([spec("a", 3000)])
    e1 = await sched.ensure_loaded("a")
    e2 = await sched.ensure_loaded("a")
    assert e1 == e2 and sup.started == ["a"]


async def test_evicts_lru_when_over_budget():
    sched, sup = make([spec("a", 4000), spec("b", 4000), spec("c", 4000)], budget=10000)
    for mid in ("a", "b"):
        await sched.ensure_loaded(mid)
        await sched.release(mid)
    await sched.ensure_loaded("c")  # 12000 > 10000 -> evict LRU = a
    assert sched.loaded() == ["b", "c"]
    assert sup.stopped == [18100]  # a's port


async def test_recently_used_is_not_evicted():
    sched, _ = make([spec("a", 4000), spec("b", 4000), spec("c", 4000)], budget=10000)
    for mid in ("a", "b"):
        await sched.ensure_loaded(mid)
        await sched.release(mid)
    await sched.ensure_loaded("a")  # touch a -> b is now LRU
    await sched.release("a")
    await sched.ensure_loaded("c")
    assert sched.loaded() == ["a", "c"]


async def test_exclusive_evicts_all():
    sched, sup = make([spec("a", 2000), spec("b", 2000), spec("big", 8000, exclusive=True)], budget=10000)
    for mid in ("a", "b"):
        await sched.ensure_loaded(mid)
        await sched.release(mid)
    await sched.ensure_loaded("big")
    assert sched.loaded() == ["big"] and len(sup.stopped) == 2


async def test_loading_other_model_evicts_exclusive():
    sched, _ = make([spec("a", 2000), spec("big", 8000, exclusive=True)], budget=10000)
    await sched.ensure_loaded("big")
    await sched.release("big")
    await sched.ensure_loaded("a")
    assert sched.loaded() == ["a"]


async def test_too_big_raises_DoesNotFit():
    sched, sup = make([spec("huge", 20000)], budget=10000)
    with pytest.raises(DoesNotFit):
        await sched.ensure_loaded("huge")
    assert sup.started == [] and sched.loaded() == []


async def test_unavailable_raises_with_reason():
    sched, _ = make([spec("x", 100, status="unsupported", reason="unknown model architecture")])
    with pytest.raises(ModelUnavailable, match="unknown model architecture"):
        await sched.ensure_loaded("x")


async def test_unknown_model_raises_KeyError():
    sched, _ = make([spec("a", 100)])
    with pytest.raises(KeyError):
        await sched.ensure_loaded("nope")


async def test_concurrent_requests_do_not_double_spawn():
    sched, sup = make([spec("a", 3000)])
    eps = await asyncio.gather(*(sched.ensure_loaded("a") for _ in range(5)))
    assert len({e.base_url for e in eps}) == 1 and sup.started == ["a"]
    for _ in range(5):
        await sched.release("a")


async def test_in_use_model_not_evicted_until_released():
    sched, sup = make([spec("a", 6000), spec("b", 6000)], budget=10000, wait_timeout_s=5.0)
    await sched.ensure_loaded("a")  # in use, never released yet
    waiter = asyncio.create_task(sched.ensure_loaded("b"))
    await asyncio.sleep(0.2)
    assert not waiter.done() and sup.stopped == []  # b waits, a is NOT killed under its user
    await sched.release("a")
    ep = await asyncio.wait_for(waiter, 3)
    assert ep.base_url and sched.loaded() == ["b"] and len(sup.stopped) == 1


async def test_wait_timeout_raises_DoesNotFit_when_busy_forever():
    sched, _ = make([spec("a", 6000), spec("b", 6000)], budget=10000, wait_timeout_s=0.3)
    await sched.ensure_loaded("a")
    with pytest.raises(DoesNotFit, match="busy"):
        await sched.ensure_loaded("b")
    assert sched.loaded() == ["a"]


async def test_unhealthy_backend_is_replaced_on_next_request():
    sched, sup = make([spec("a", 3000)])
    await sched.ensure_loaded("a")
    await sched.release("a")
    sup.healthy = False
    await asyncio.sleep(0)
    ep = await sched.ensure_loaded("a")
    sup.healthy = True
    assert sup.started == ["a", "a"] and len(sup.stopped) == 1 and ep.base_url.endswith("18101")


async def test_failed_start_leaks_nothing():
    sched, sup = make([spec("a", 6000), spec("b", 6000)], budget=10000)
    sup.fail_next = True
    with pytest.raises(BackendStartError):
        await sched.ensure_loaded("a")
    assert sched.loaded() == []
    await sched.ensure_loaded("b")  # budget not blocked by the failed attempt
    assert sched.loaded() == ["b"]


async def test_release_unknown_or_not_loaded_is_noop():
    sched, _ = make([spec("a", 100)])
    await sched.release("a")
    await sched.release("nope")


async def test_external_model_returns_fixed_endpoint_without_spawning():
    ext = ModelSpec(id="jev", kind="llm", files=[], runtime="external", est_mem_mb=0, endpoint="http://127.0.0.1:8080/")
    sched, sup = make([ext, spec("a", 100)])
    ep = await sched.ensure_loaded("jev")
    assert ep.base_url == "http://127.0.0.1:8080" and sup.started == [] and sched.loaded() == []
    await sched.release("jev")  # no-op, must not raise


async def test_virtual_model_is_not_loadable():
    v = ModelSpec(id="duo", kind="llm", files=[], runtime="virtual", est_mem_mb=0)
    sched, _ = make([v])
    with pytest.raises(ModelUnavailable, match="virtual"):
        await sched.ensure_loaded("duo")


async def test_two_models_stay_resident_together_when_budget_allows():
    sched, sup = make([spec("big", 9000), spec("small", 3600)], budget=16800)
    await sched.ensure_loaded("big"); await sched.release("big")
    await sched.ensure_loaded("small"); await sched.release("small")
    await sched.ensure_loaded("big")
    assert sorted(sched.loaded()) == ["big", "small"] and sup.started == ["big", "small"] and sup.stopped == []
