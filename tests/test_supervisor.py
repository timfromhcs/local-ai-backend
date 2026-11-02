import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from localai_gateway import supervisor as sv
from localai_gateway.supervisor import BackendStartError, Supervisor, free_port

FAKE = str(Path(__file__).parent / "fake_runtime.py")


def fake_argv(port: int) -> list[str]:
    return [sys.executable, FAKE, "--port", str(port)]


def url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"


async def until(pred, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.05)
    return False


async def noop_sleep(_s):
    return None


def test_free_port_in_range():
    assert 18100 <= free_port() <= 18199


async def test_start_waits_for_health(monkeypatch):
    monkeypatch.setenv("FAKE_BIND_DELAY", "1.0")  # port opens only after 1s: start must really wait
    sup = Supervisor()
    port = free_port()
    t0 = time.monotonic()
    mp = await sup.start(None, fake_argv(port), port, url(port))
    elapsed = time.monotonic() - t0
    try:
        assert elapsed >= 0.9
        assert psutil.pid_exists(mp.pid)
        assert mp.port == port
        assert await sup.health(mp) is True
    finally:
        await sup.stop(mp)
    assert not psutil.pid_exists(mp.pid)
    assert await sup.health(mp) is False


async def test_crash_is_restarted_up_to_3_times(monkeypatch):
    monkeypatch.setenv("FAKE_CRASH_AFTER", "2.0")
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
        assert mp.failed is True
        assert await sup.health(mp) is False
    finally:
        await sup.stop(mp)


async def test_restart_produces_healthy_process(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_CRASH_AFTER", "2.0")
    monkeypatch.setenv("FAKE_CRASH_ONCE_FILE", str(tmp_path / "crashed"))
    sup = Supervisor(sleep=noop_sleep)
    port = free_port()
    mp = await sup.start(None, fake_argv(port), port, url(port))
    old = mp.pid
    try:
        assert await until(lambda: mp.pid != old)
        assert mp.restarts == 1 and not mp.failed
        assert psutil.pid_exists(mp.pid)
        assert await sup.health(mp) is True
    finally:
        await sup.stop(mp)


async def test_budget_resets_after_stable_uptime(monkeypatch):
    monkeypatch.setenv("FAKE_CRASH_AFTER", "2.0")
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    ticks = [0.0]

    def clock():  # every reading is 100s later: each life counts as >= 60s stable
        ticks[0] += 100.0
        return ticks[0]

    sup = Supervisor(sleep=fake_sleep, clock=clock, stable_after_s=60.0)
    port = free_port()
    mp = await sup.start(None, fake_argv(port), port, url(port))
    try:
        assert await until(lambda: len(sleeps) >= 4)  # more than the 3-attempt budget
        assert not mp.failed
        assert sleeps[:4] == [1, 1, 1, 1]
    finally:
        await sup.stop(mp)


async def test_watch_spawn_failure_sets_error(monkeypatch, caplog):
    monkeypatch.setenv("FAKE_CRASH_AFTER", "2.0")
    sup = Supervisor(sleep=noop_sleep)
    real = sup._spawn
    calls = []

    async def flaky(argv):
        calls.append(1)
        if len(calls) > 1:
            raise OSError("boom")
        return await real(argv)

    sup._spawn = flaky
    port = free_port()
    with caplog.at_level("ERROR", logger="localai_gateway.supervisor"):
        mp = await sup.start(None, fake_argv(port), port, url(port))
        try:
            assert await until(lambda: mp.failed)
            assert "boom" in mp.error and "OSError" in mp.error
            assert "boom" in caplog.text
            assert await sup.health(mp) is False
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


async def test_cancel_during_start_kills_tree(monkeypatch):
    monkeypatch.setenv("FAKE_BIND_DELAY", "30")
    monkeypatch.setenv("FAKE_SPAWN_CHILD", "1")
    sup = Supervisor()
    port = free_port()
    task = asyncio.create_task(sup.start(None, fake_argv(port), port, url(port)))
    assert await until(lambda: sup.last_pid is not None)
    pid = sup.last_pid
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await until(lambda: not psutil.pid_exists(pid), timeout=10)


async def test_cancel_during_restart_kills_new_process(monkeypatch):
    monkeypatch.setenv("FAKE_CRASH_AFTER", "2.0")
    sup = Supervisor(sleep=noop_sleep)
    port = free_port()
    mp = await sup.start(None, fake_argv(port), port, url(port))
    first = mp.pid
    # make restarted lives slow to bind so stop() lands mid-restart
    monkeypatch.setenv("FAKE_BIND_DELAY", "30")
    assert await until(lambda: mp._pending is not None)
    new_pid = mp._pending[0].pid
    assert new_pid != first
    await sup.stop(mp)
    assert not psutil.pid_exists(new_pid)
    assert not psutil.pid_exists(first)


async def test_stop_kills_descendants(monkeypatch):
    monkeypatch.setenv("FAKE_SPAWN_CHILD", "1")
    sup = Supervisor()
    port = free_port()
    mp = await sup.start(None, fake_argv(port), port, url(port))
    kids = psutil.Process(mp.pid).children(recursive=True)
    assert kids, "fake runtime should have spawned a child"
    pids = [mp.pid] + [k.pid for k in kids]
    await sup.stop(mp)
    for pid in pids:
        assert not psutil.pid_exists(pid)


async def test_kill_helper_noop_for_exited_proc(monkeypatch):
    proc = await asyncio.create_subprocess_exec(sys.executable, "-c", "pass")
    ctime = psutil.Process(proc.pid).create_time()
    await proc.wait()

    def boom(*a, **k):
        raise AssertionError("must not touch psutil for an exited proc")

    monkeypatch.setattr(sv.psutil, "Process", boom)
    await sv._kill_tree(proc, ctime)  # no-op, must not raise


async def test_kill_helper_skips_reused_pid():
    proc = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(60)")
    try:
        real = psutil.Process(proc.pid).create_time()
        await asyncio.to_thread(sv._kill_tree_sync, proc.pid, real + 1000)  # wrong ctime = reused pid
        await asyncio.sleep(0.3)
        assert psutil.pid_exists(proc.pid) and proc.returncode is None
    finally:
        await sv._kill_tree(proc, real)
    assert not psutil.pid_exists(proc.pid)


async def test_death_right_after_healthy_is_start_error():
    # server answers /health once then the process dies: _alive must reject it
    sup = Supervisor(health_timeout=5.0)
    port = free_port()
    code = (
        "import http.server,os,threading\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(s):\n"
        "        s.send_response(200); s.send_header('Content-Length','2'); s.end_headers(); s.wfile.write(b'ok'); s.wfile.flush()\n"
        "        os._exit(1)\n"
        "    def log_message(s,*a): pass\n"
        f"http.server.HTTPServer(('127.0.0.1',{port}),H).serve_forever()\n"
    )
    with pytest.raises(BackendStartError):
        await sup.start(None, [sys.executable, "-c", code], port, url(port))
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
