"""Shutdown contract: a stopped gateway must not leave backends holding GPU memory.

Real failure this prevents: after killing the gateway, orphaned llama-server processes kept
204 MB of Vulkan memory each, and the next model start failed with
"failed to allocate buffer for kv cache".
"""
import pytest

from localai_gateway.config import Settings
from localai_gateway.registry import ModelSpec
from localai_gateway.scheduler import Scheduler


class _Supervisor:
    def __init__(self):
        self.stopped = []

    async def start(self, spec, argv, port, health_url):
        return f"mp-{spec.id}"

    async def stop(self, mp):
        self.stopped.append(mp)

    async def health(self, mp):
        return True


class _Endpoint:
    base_url = "http://127.0.0.1:1"


@pytest.fixture()
def scheduler(tmp_path):
    settings = Settings(budget_mb=20000, bin_dir=tmp_path, model_dir=tmp_path)
    registry = {
        "a": ModelSpec(id="a", kind="llm", files=[], runtime="llama", est_mem_mb=1000),
        "b": ModelSpec(id="b", kind="llm", files=[], runtime="llama", est_mem_mb=1000),
    }
    sup = _Supervisor()
    return Scheduler(settings, registry, sup, build_command=lambda s, p: (["x"], "http://127.0.0.1:1/health"),
                     port_allocator=iter([18100, 18101]).__next__), sup


async def test_shutdown_stops_every_loaded_backend(scheduler):
    sched, sup = scheduler
    await sched.ensure_loaded("a")
    await sched.ensure_loaded("b")
    assert set(sched.loaded()) == {"a", "b"}

    await sched.shutdown()
    assert sched.loaded() == []
    assert sorted(sup.stopped) == ["mp-a", "mp-b"]


async def test_shutdown_is_idempotent_and_survives_failures(tmp_path):
    settings = Settings(budget_mb=20000, bin_dir=tmp_path, model_dir=tmp_path)
    registry = {"a": ModelSpec(id="a", kind="llm", files=[], runtime="llama", est_mem_mb=1000)}

    class Failing(_Supervisor):
        async def stop(self, mp):
            raise OSError("already gone")

    sched = Scheduler(settings, registry, Failing(), build_command=lambda s, p: (["x"], "h"),
                      port_allocator=lambda: 18102)
    await sched.ensure_loaded("a")
    await sched.shutdown()   # must not raise
    await sched.shutdown()   # second call is a no-op
    assert sched.loaded() == []


def test_app_shutdown_calls_scheduler():
    """The lifespan must wire the shutdown, otherwise the scheduler method is dead code."""
    import inspect

    from localai_gateway import app as app_module

    source = inspect.getsource(app_module.create_app)
    assert "scheduler.shutdown()" in source
