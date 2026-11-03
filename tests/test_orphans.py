"""The ledger that lets the next gateway kill backends left by a killed one.

Observed twice on this machine: after force-killing the gateway, llama-server children kept their
Vulkan memory, and the next load died with ErrorOutOfDeviceMemory. The ledger must kill those, and
must never kill a process it did not start.
"""
import json

import pytest

from localai_gateway.orphans import forget, ledger_path, reap, record


def entries(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_record_and_forget(tmp_path):
    path = ledger_path(tmp_path)
    record(path, pid=111, gateway_pid=999, label="llama-server")
    assert [e["pid"] for e in entries(path)] == [111]
    assert entries(path)[0]["gateway_pid"] == 999

    record(path, pid=111, gateway_pid=999, label="llama-server again")  # no duplicate
    assert len(entries(path)) == 1

    forget(path, 111)
    assert entries(path) == []


def test_reap_kills_backends_whose_gateway_is_dead(tmp_path):
    path = ledger_path(tmp_path)
    record(path, pid=111, gateway_pid=999)
    record(path, pid=222, gateway_pid=999)
    killed: list[int] = []

    dead_owners = reap(path, gateway_pid=1000,
                       alive=lambda pid, ctime: pid in (111, 222), kill=killed.append)

    assert sorted(e["pid"] for e in dead_owners) == [111, 222]
    assert killed == [111, 222]
    assert entries(path) == [], "reaped entries must not stay in the ledger"


def test_reap_leaves_a_live_gateways_backends_alone(tmp_path):
    path = ledger_path(tmp_path)
    record(path, pid=111, gateway_pid=999)
    record(path, pid=333, gateway_pid=999)
    killed: list[int] = []

    # gateway 999 is alive, so both of its backends are its business
    reap(path, gateway_pid=1000,
         alive=lambda pid, ctime: pid in (111, 333, 999), kill=killed.append)

    assert killed == []
    assert [e["pid"] for e in entries(path)] == [111, 333]


def test_reap_kills_when_the_owner_died_but_the_backend_lives(tmp_path):
    path = ledger_path(tmp_path)
    record(path, pid=333, gateway_pid=999)
    killed: list[int] = []

    reap(path, gateway_pid=1000, alive=lambda pid, ctime: pid == 333, kill=killed.append)

    assert killed == [333]
    assert entries(path) == []


def test_reap_keeps_our_own_entries_and_skips_dead_pids(tmp_path):
    path = ledger_path(tmp_path)
    record(path, pid=444, gateway_pid=1000)   # this gateway's own backend
    record(path, pid=555, gateway_pid=999)    # owner gone, process already gone
    killed: list[int] = []

    result = reap(path, gateway_pid=1000,
                  alive=lambda pid, ctime: pid == 444, kill=killed.append)

    assert result == []
    assert killed == []
    assert [e["pid"] for e in entries(path)] == [444]


def test_broken_ledger_is_ignored(tmp_path):
    path = ledger_path(tmp_path)
    path.write_text("not json", encoding="utf-8")
    assert reap(path, gateway_pid=1, alive=lambda pid, ctime: True, kill=lambda pid: None) == []
    record(path, pid=7, gateway_pid=1)  # rewriting a corrupt ledger still works
    assert [e["pid"] for e in entries(path)] == [7]


def test_unwritable_ledger_does_not_raise(tmp_path):
    path = tmp_path / "nope" / "backends.json"
    path.parent.mkdir()
    path.parent.chmod(0o500) if hasattr(path.parent, "chmod") else None
    record(path, pid=1, gateway_pid=2)  # must not raise on failure


def test_alive_guard_rejects_pid_reuse():
    import os

    from localai_gateway.orphans import _alive

    assert _alive(os.getpid(), None) is True
    assert _alive(os.getpid(), 0.0) is False, "a different create_time means a different process"
    assert _alive(2**31 - 1, None) is False
