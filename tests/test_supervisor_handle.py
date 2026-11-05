"""Supervisor resilience: a lost Windows process handle must not mark a live model as failed.

Observed in production: `FileNotFoundError(2, ...)` from the watcher marked a healthy qwen3-4b
backend as failed while it was still serving, so the next request said "backend unreachable".
"""
import asyncio

import pytest

from localai_gateway.supervisor import ManagedProcess, Supervisor


class _LostHandleProc:
    pid = 424242
    returncode = None

    async def wait(self):
        raise FileNotFoundError(2, "The system cannot find the file specified")


class _DeadProc(_LostHandleProc):
    returncode = 1


async def test_wait_exit_falls_back_to_polling(monkeypatch):
    sup = Supervisor(poll_interval=0.01)
    mp = ManagedProcess(pid=424242, port=18100, started_at=0.0, _proc=_LostHandleProc(), _ctime=None)
    monkeypatch.setattr("localai_gateway.supervisor._alive", lambda proc, ctime: False)
    # returns instead of raising, so the watcher survives
    await asyncio.wait_for(sup._wait_exit(mp), timeout=2)


async def test_wait_exit_returns_when_process_exits():
    sup = Supervisor(poll_interval=0.01)
    proc = _DeadProc()
    mp = ManagedProcess(pid=proc.pid, port=18101, started_at=0.0, _proc=proc, _ctime=None)
    await asyncio.wait_for(sup._wait_exit(mp), timeout=2)  # returncode set -> handle path raises, poller returns


async def test_stop_flag_breaks_the_poll_loop(monkeypatch):
    sup = Supervisor(poll_interval=0.01)
    mp = ManagedProcess(pid=424243, port=18102, started_at=0.0, _proc=_LostHandleProc(), _ctime=None)
    monkeypatch.setattr("localai_gateway.supervisor._alive", lambda proc, ctime: True)

    async def stopper():
        await asyncio.sleep(0.05)
        mp.stopped = True

    await asyncio.wait_for(asyncio.gather(sup._wait_exit(mp), stopper()), timeout=2)

async def test_a_lost_handle_does_not_mark_a_live_backend_failed(monkeypatch):
    """Measured 2026-11-05: the watcher died on FileNotFoundError while /health answered 200."""
    from localai_gateway.supervisor import ManagedProcess, Supervisor

    sup = Supervisor(poll_interval=0.01)
    proc = _LostHandleProc()
    mp = ManagedProcess(pid=proc.pid, port=18103, started_at=0.0, _proc=proc, _ctime=None)
    calls = {"n": 0}

    async def flaky_inner(m):
        calls["n"] += 1
        if calls["n"] == 1:
            raise FileNotFoundError(2, "The system cannot find the file specified")
        m.stopped = True

    monkeypatch.setattr(sup, "_watch_inner", flaky_inner)
    monkeypatch.setattr("localai_gateway.supervisor._alive", lambda p, c: True)

    await asyncio.wait_for(sup._watch(mp), timeout=5)
    assert mp.failed is False, "a live backend must not be marked failed"
    assert calls["n"] == 2, "the watcher must resume instead of giving up"
