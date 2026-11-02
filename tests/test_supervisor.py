import asyncio
import socket
import sys
import time
from pathlib import Path

import psutil
import pytest

from localai_gateway.supervisor import BackendStartError, Supervisor, free_port

FAKE = str(Path(__file__).parent / "fake_runtime.py")


def fake_argv(port: int) -> list[str]:
    return [sys.executable, FAKE, "--port", str(port)]


def url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"


async def until(pred, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.05)
    return False


def test_free_port_in_range():
    assert 18100 <= free_port() <= 18199


async def test_start_waits_for_health():
    sup = Supervisor()
    port = free_port()
    mp = await sup.start(None, fake_argv(port), port, url(port))
    try:
        assert psutil.pid_exists(mp.pid)
        assert mp.port == port
        assert await sup.health(mp) is True
    finally:
        await sup.stop(mp)
    assert not psutil.pid_exists(mp.pid)
    assert await sup.health(mp) is False


async def test_crash_is_restarted_up_to_3_times(monkeypatch):
    monkeypatch.setenv("FAKE_CRASH_AFTER", "0.5")
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    sup = Supervisor(sleep=fake_sleep)
    port = free_port()
    mp = await sup.start(None, fake_argv(port), port, url(port))
    try:
        assert await until(lambda: mp.failed)
        assert sleeps == [1, 2, 4]
        assert mp.restarts == 3
    finally:
        await sup.stop(mp)


async def test_immediate_exit_raises_BackendStartError_no_zombie():
    sup = Supervisor()
    port = free_port()
    with pytest.raises(BackendStartError):
        await sup.start(None, [sys.executable, "-c", "import sys; sys.exit(3)"], port, url(port))
    assert sup.last_pid is not None
    assert not psutil.pid_exists(sup.last_pid)


async def test_health_timeout_kills_process():
    sup = Supervisor(health_timeout=1.0)
    port = free_port()
    argv = [sys.executable, "-c", "import time; time.sleep(60)"]
    with pytest.raises(BackendStartError):
        await sup.start(None, argv, port, url(port))
    assert not psutil.pid_exists(sup.last_pid)


async def test_port_in_use_raises():
    port = free_port()
    s = socket.socket()
    s.bind(("127.0.0.1", port))
    s.listen()
    try:
        sup = Supervisor()
        with pytest.raises(BackendStartError):
            await sup.start(None, fake_argv(port), port, url(port))
        assert sup.last_pid is None
    finally:
        s.close()
