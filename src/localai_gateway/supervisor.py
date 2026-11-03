"""Process supervisor: spawn a backend, wait for health, restart on crash with backoff."""
from __future__ import annotations

import asyncio
import logging
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx
import psutil

PORT_RANGE = range(18100, 18200)
BACKOFF = (1, 2, 4)
log = logging.getLogger("localai_gateway.supervisor")


class BackendStartError(Exception):
    pass


def _port_free(port: int) -> bool:
    """True when nothing is *listening* on the port.

    On Linux a plain bind fails with EADDRINUSE while sockets from our own health
    checks sit in TIME_WAIT, so every restart was rejected with "port in use on
    restart" and burned the backoff budget (found by CI on ubuntu-latest).
    SO_REUSEADDR fixes POSIX. It is deliberately NOT set on Windows, where it would
    let us bind a port that is actively listened on.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if sys.platform != "win32":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def free_port() -> int:
    """Pick a free port from 18100-18199 (never 8080)."""
    for port in PORT_RANGE:
        if _port_free(port):
            return port
    raise BackendStartError("no free port in 18100-18199")


@dataclass
class ManagedProcess:
    pid: int
    port: int
    started_at: float
    restarts: int = 0
    failed: bool = False
    stopped: bool = False
    error: str = ""
    _proc: Any = field(default=None, repr=False)
    _ctime: float | None = field(default=None, repr=False)
    _pending: Any = field(default=None, repr=False)  # (proc, ctime) of an in-flight restart
    _argv: list[str] = field(default_factory=list, repr=False)
    _health_url: str = field(default="", repr=False)
    _monitor: Any = field(default=None, repr=False)


def _ctime_of(pid: int) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error:
        return None


def _alive(proc: Any, ctime: float | None) -> bool:
    """True only if proc has not exited and the OS process is the one we spawned."""
    if proc.returncode is not None:
        return False
    try:
        p = psutil.Process(proc.pid)
        if ctime is not None and abs(p.create_time() - ctime) > 0.001:
            return False
        return p.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _kill_tree_sync(pid: int, ctime: float | None) -> None:
    try:
        root = psutil.Process(pid)
        if ctime is not None and abs(root.create_time() - ctime) > 0.001:
            return  # PID was reused by someone else: never kill it
        procs = root.children(recursive=True) + [root]
    except psutil.Error:
        return
    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=5)


async def _kill_tree(proc: Any, ctime: float | None) -> None:
    """Kill proc's tree, but only if proc is still running and is the process we spawned."""
    if proc.returncode is not None:
        return
    await asyncio.to_thread(_kill_tree_sync, proc.pid, ctime)
    try:
        await asyncio.wait_for(proc.wait(), 5)
    except asyncio.TimeoutError:
        pass


class Supervisor:
    """Spawns backends, waits for /health, restarts crashed ones with backoff.

    Restart policy: up to len(backoff) restart attempts (sleeps 1, 2, 4 by default, via the
    injectable ``sleep``). The attempt budget resets after ``stable_after_s`` (60s) of
    continuous healthy uptime, measured with the injectable ``clock``. When the budget is
    exhausted, or restarting raises unexpectedly, ``mp.failed`` is set and ``mp.error`` holds
    the cause.
    """

    def __init__(
        self,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        health_timeout: float = 120.0,
        poll_interval: float = 0.25,
        backoff: tuple[float, ...] = BACKOFF,
        stable_after_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sleep = sleep  # backoff sleeps only (injectable); health polling uses asyncio.sleep
        self.health_timeout = health_timeout
        self.poll_interval = poll_interval
        self.backoff = backoff
        self.stable_after_s = stable_after_s
        self._clock = clock
        self.last_pid: int | None = None

    async def _spawn(self, argv: list[str]) -> tuple[Any, float | None]:
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = await asyncio.create_subprocess_exec(*argv, **kwargs)
        self.last_pid = proc.pid
        return proc, _ctime_of(proc.pid)

    async def _wait_healthy(self, proc: Any, ctime: float | None, health_url: str) -> None:
        deadline = time.monotonic() + self.health_timeout
        async with httpx.AsyncClient(timeout=2.0) as client:
            while True:
                if proc.returncode is not None:
                    raise BackendStartError(f"backend exited with code {proc.returncode} before becoming healthy")
                try:
                    if (await client.get(health_url)).status_code == 200:
                        await asyncio.sleep(0.1)  # grace: catch a child that dies right after answering
                        if not _alive(proc, ctime):
                            raise BackendStartError("backend died right after reporting healthy")
                        return
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    raise BackendStartError(f"backend not healthy after {self.health_timeout}s")
                await asyncio.sleep(self.poll_interval)

    async def _launch(self, argv: list[str], health_url: str, mp: ManagedProcess | None = None):
        """Spawn + wait healthy. On ANY failure (incl. cancellation) the new tree is killed."""
        proc, ctime = await self._spawn(argv)
        if mp is not None:
            mp._pending = (proc, ctime)
        try:
            await self._wait_healthy(proc, ctime, health_url)
        except BaseException:
            await _kill_tree(proc, ctime)
            raise
        finally:
            if mp is not None:
                mp._pending = None
        return proc, ctime

    async def start(self, spec: Any, argv: list[str], port: int, health_url: str) -> ManagedProcess:
        self.last_pid = None
        if not _port_free(port):
            raise BackendStartError(f"port {port} already in use")
        proc, ctime = await self._launch(argv, health_url)
        mp = ManagedProcess(pid=proc.pid, port=port, started_at=time.time(), _proc=proc,
                            _ctime=ctime, _argv=list(argv), _health_url=health_url)
        mp._monitor = asyncio.create_task(self._watch(mp))
        return mp

    async def _wait_exit(self, mp: ManagedProcess) -> None:
        """Wait for the process to exit.

        On Windows `asyncio` can raise FileNotFoundError when the OS already reaped the handle
        while the process is still serving (seen as a watcher crash that marked a healthy model
        as failed). Fall back to polling psutil instead of treating the model as dead.
        """
        try:
            await mp._proc.wait()
            return
        except (FileNotFoundError, ProcessLookupError, OSError):
            log.warning("process handle for pid %s was lost; polling liveness instead", mp.pid)
        while not mp.stopped and _alive(mp._proc, mp._ctime):
            await asyncio.sleep(self.poll_interval)

    async def _watch(self, mp: ManagedProcess) -> None:
        try:
            await self._watch_inner(mp)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # spawn OSError etc.
            mp.failed = True
            mp.error = repr(e)
            log.error("supervisor watcher for port %s failed: %r", mp.port, e)

    async def _watch_inner(self, mp: ManagedProcess) -> None:
        attempts = 0
        healthy_since = self._clock()
        while not mp.stopped:
            await self._wait_exit(mp)
            if mp.stopped:
                return
            if self._clock() - healthy_since >= self.stable_after_s:
                attempts = 0  # was stable long enough: fresh budget
            while True:
                if attempts >= len(self.backoff):
                    mp.failed = True
                    mp.error = f"restart budget exhausted after {attempts} attempts"
                    log.error("backend on port %s: %s", mp.port, mp.error)
                    return
                await self._sleep(self.backoff[attempts])
                attempts += 1
                if mp.stopped:
                    return
                mp.restarts += 1
                try:
                    if not _port_free(mp.port):
                        raise BackendStartError("port in use on restart")
                    proc, ctime = await self._launch(mp._argv, mp._health_url, mp)
                except BackendStartError:
                    continue
                mp._proc, mp._ctime, mp.pid, mp.started_at = proc, ctime, proc.pid, time.time()
                healthy_since = self._clock()
                break

    async def stop(self, mp: ManagedProcess) -> None:
        mp.stopped = True
        if mp._monitor is not None and mp._monitor is not asyncio.current_task():
            mp._monitor.cancel()  # in-flight restart is killed by _launch's cancellation handler
            await asyncio.gather(mp._monitor, return_exceptions=True)
        if mp._pending is not None:
            await _kill_tree(*mp._pending)
        if mp._proc is not None:
            await _kill_tree(mp._proc, mp._ctime)

    async def health(self, mp: ManagedProcess) -> bool:
        if mp.stopped or mp.failed or mp._proc.returncode is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                return (await client.get(mp._health_url)).status_code == 200
        except httpx.HTTPError:
            return False
