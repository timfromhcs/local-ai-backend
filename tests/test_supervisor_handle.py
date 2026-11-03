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
