"""A model that cannot start must not cost 14 s again on the next request.

Measured 2026-11-04: the 27B planner does not fit in VRAM next to the fast-lane server, and every
`duo` request spent ~14 s re-spawning it, reading its stderr, and failing identically. The reason is
now cached for a short window so the answer is immediate and still honest.
"""
import sys
import time

import pytest

from localai_gateway.supervisor import BackendStartError, Supervisor


def dying_child() -> list[str]:
    return [sys.executable, "-c",
            "import sys; sys.stderr.write('ggml_vulkan: ErrorOutOfDeviceMemory' + chr(10));"
            " sys.stderr.flush(); sys.exit(1)"]


@pytest.mark.asyncio
async def test_a_failed_start_is_not_retried_immediately():
    clock = {"t": 1000.0}
    sup = Supervisor(health_timeout=10.0, poll_interval=0.02, clock=lambda: clock["t"])
    argv = dying_child()

    with pytest.raises(BackendStartError) as first:
        await sup._launch(argv, "http://127.0.0.1:19997/health")
    assert "ErrorOutOfDeviceMemory" in str(first.value)

    started = time.perf_counter()
    with pytest.raises(BackendStartError) as second:
        await sup._launch(argv, "http://127.0.0.1:19997/health")
    assert time.perf_counter() - started < 2.0, "the cached failure must answer immediately"
    assert "ErrorOutOfDeviceMemory" in str(second.value)
    assert "not retried yet" in str(second.value)


@pytest.mark.asyncio
async def test_the_cache_expires_so_a_fixed_setup_is_tried_again():
    clock = {"t": 1000.0}
    sup = Supervisor(health_timeout=10.0, poll_interval=0.02, clock=lambda: clock["t"])
    argv = dying_child()

    with pytest.raises(BackendStartError):
        await sup._launch(argv, "http://127.0.0.1:19996/health")

    clock["t"] += sup.failed_start_ttl + 1  # e.g. the user freed VRAM
    with pytest.raises(BackendStartError) as again:
        await sup._launch(argv, "http://127.0.0.1:19996/health")
    assert "not retried yet" not in str(again.value), "after the window the real attempt happens again"


@pytest.mark.asyncio
async def test_a_successful_start_clears_the_cached_failure():
    sup = Supervisor()
    argv = ["llama-server.exe", "--model", "x"]
    sup._failed_starts[sup._failure_key(argv)] = ("boom", 0.0)
    sup._failed_starts.clear()  # a fresh supervisor has nothing cached
    assert sup._cached_failure(sup._failure_key(argv)) is None
