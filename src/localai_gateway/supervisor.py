"""Process supervisor: spawn a backend, wait for health, restart on crash with backoff."""
from __future__ import annotations

import asyncio
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


class BackendStartError(Exception):
    pass


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
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
    _proc: Any = field(default=None, repr=False)
    _argv: list[str] = field(default_factory=list, repr=False)
    _health_url: str = field(default="", repr=False)
    _monitor: Any = field(default=None, repr=False)


def _kill_tree_sync(pid: int) -> None:
    try:
        root = psutil.Process(pid)
        procs = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=5)


async def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    await asyncio.to_thread(_kill_tree_sync, proc.pid)
    try:
        await asyncio.wait_for(proc.wait(), 5)
    except asyncio.TimeoutError:
        pass


class Supervisor:
    def __init__(
        self,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        health_timeout: float = 120.0,
        poll_interval: float = 0.25,
        backoff: tuple[float, ...] = BACKOFF,
    ) -> None:
        self._sleep = sleep  # backoff sleeps only (injectable); health polling uses asyncio.sleep
        self.health_timeout = health_timeout
        self.poll_interval = poll_interval
        self.backoff = backoff
        self.last_pid: int | None = None

    async def _spawn(self, argv: list[str]) -> asyncio.subprocess.Process:
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = await asyncio.create_subprocess_exec(*argv, **kwargs)
        self.last_pid = proc.pid
        return proc

    async def _wait_healthy(self, proc: asyncio.subprocess.Process, health_url: str) -> None:
        """Poll health until OK; kill tree and raise BackendStartError on exit or timeout."""
        deadline = time.monotonic() + self.health_timeout
        async with httpx.AsyncClient(timeout=2.0) as client:
            while True:
                if proc.returncode is not None:
                    await _kill_tree(proc)
                    raise BackendStartError(f"backend exited with code {proc.returncode} before becoming healthy")
                try:
                    if (await client.get(health_url)).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    await _kill_tree(proc)
                    raise BackendStartError(f"backend not healthy after {self.health_timeout}s")
                await asyncio.sleep(self.poll_interval)

    async def start(self, spec: Any, argv: list[str], port: int, health_url: str) -> ManagedProcess:
        self.last_pid = None
        if not _port_free(port):
            raise BackendStartError(f"port {port} already in use")
        proc = await self._spawn(argv)
        await self._wait_healthy(proc, health_url)
        mp = ManagedProcess(pid=proc.pid, port=port, started_at=time.time(),
                            _proc=proc, _argv=list(argv), _health_url=health_url)
        mp._monitor = asyncio.create_task(self._watch(mp))
        return mp

    async def _watch(self, mp: ManagedProcess) -> None:
        """Restart on crash: up to len(backoff) attempts, sleeping 1,2,4 before each."""
        attempts = 0
        while not mp.stopped:
            await mp._proc.wait()
            if mp.stopped:
                return
            while True:
                if attempts >= len(self.backoff):
                    mp.failed = True
                    return
                await self._sleep(self.backoff[attempts])
                attempts += 1
                if mp.stopped:
                    return
                mp.restarts = attempts
                try:
                    if not _port_free(mp.port):
                        raise BackendStartError("port in use on restart")
                    proc = await self._spawn(mp._argv)
                    await self._wait_healthy(proc, mp._health_url)
                except BackendStartError:
                    continue
                mp._proc, mp.pid, mp.started_at = proc, proc.pid, time.time()
                break

    async def stop(self, mp: ManagedProcess) -> None:
        mp.stopped = True
        proc = mp._proc
        if mp._monitor is not None and mp._monitor is not asyncio.current_task():
            mp._monitor.cancel()
            await asyncio.gather(mp._monitor, return_exceptions=True)
        if proc is not None:
            await _kill_tree(proc)

    async def health(self, mp: ManagedProcess) -> bool:
        if mp.stopped or mp.failed or mp._proc.returncode is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                return (await client.get(mp._health_url)).status_code == 200
        except httpx.HTTPError:
            return False
