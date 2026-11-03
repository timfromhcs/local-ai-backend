"""Backends outlive a killed gateway and keep the GPU memory.

A graceful shutdown stops every child (``Scheduler.shutdown``), but a killed process (Ctrl+C in the
wrong console, a crashed parent, Task Manager) leaves ``llama-server`` children running. The next
start then dies with ``vk::Device::allocateMemory: ErrorOutOfDeviceMemory`` - observed here as two
failed measurement runs in a row.

This module keeps a tiny ledger of the backends we spawned, so the next gateway can kill exactly
those whose owner is gone. It never touches processes it did not start (the external Jev server, a
second gateway's backends, anything the user runs).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable

import psutil

log = logging.getLogger("localai_gateway.orphans")

_alive_fn: Callable[[int, float | None], bool]
_kill_fn: Callable[[int], None]


def _alive(pid: int, ctime: float | None) -> bool:
    """True when the process exists and is the one recorded (guards against PID reuse)."""
    try:
        p = psutil.Process(pid)
    except psutil.Error:
        return False
    try:
        if ctime is not None and abs(p.create_time() - ctime) > 0.001:
            return False
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _kill(pid: int) -> None:
    try:
        psutil.Process(pid).kill()
    except psutil.Error:
        pass


def _ctime(pid: int) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error:
        return None


def ledger_path(root: Path) -> Path:
    return root / "backends.json"


def _read(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict) and isinstance(e.get("pid"), int)]


def _write(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def record(path: Path, *, pid: int, gateway_pid: int, label: str = "") -> None:
    """Remember a backend we just spawned (no-op when no ledger path is configured)."""
    entries = [e for e in _read(path) if e["pid"] != pid]
    entries.append({
        "pid": pid,
        "ctime": _ctime(pid),
        "gateway_pid": gateway_pid,
        "gateway_ctime": _ctime(gateway_pid),
        "label": label[:120],
        "at": round(time.time(), 3),
    })
    try:
        _write(path, entries)
    except OSError as exc:  # a read-only checkout must not break model startup
        log.debug("cannot write backend ledger %s: %r", path, exc)


def forget(path: Path, pid: int) -> None:
    entries = _read(path)
    keep = [e for e in entries if e["pid"] != pid]
    if len(keep) == len(entries):
        return
    try:
        _write(path, keep)
    except OSError as exc:
        log.debug("cannot update backend ledger %s: %r", path, exc)


def reap(path: Path, *, gateway_pid: int,
         alive: Callable[[int, float | None], bool] = _alive,
         kill: Callable[[int], None] = _kill) -> list[dict]:
    """Kill backends whose owning gateway is gone; keep entries we cannot judge.

    Returns the killed entries so a caller (or a test) can report what happened.
    """
    entries, keep, killed = _read(path), [], []
    for e in entries:
        owner = e.get("gateway_pid")
        if owner == gateway_pid:
            keep.append(e)  # ours: this gateway stops its own backends
            continue
        if isinstance(owner, int) and alive(owner, e.get("gateway_ctime")):
            keep.append(e)  # another live gateway still owns it
            continue
        if alive(e["pid"], e.get("ctime")):
            kill(e["pid"])
            killed.append(e)
            log.warning("reaped orphaned backend pid %s (%s)", e["pid"], e.get("label") or "unknown")
    try:
        _write(path, keep)
    except OSError as exc:
        log.debug("cannot update backend ledger %s: %r", path, exc)
    return killed
