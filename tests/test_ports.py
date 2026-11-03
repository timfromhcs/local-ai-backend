"""Port accounting must not confuse TIME_WAIT sockets with a running listener.

Regression for the CI failure on ubuntu-latest: after a crash, sockets from our own
health checks sit in TIME_WAIT and a plain bind failed, so every restart was
rejected with "port in use on restart" and burned the whole backoff budget.
"""
import socket
import sys

import pytest

from localai_gateway.supervisor import _port_free, free_port


def test_port_with_listener_is_not_free():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        assert _port_free(port) is False


def test_port_is_free_again_after_listener_closes():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        assert _port_free(port) is False
    assert _port_free(port) is True


def test_time_wait_socket_does_not_block_the_port():
    """A closed connection leaves TIME_WAIT state; the port must still be usable."""
    if sys.platform == "win32":
        pytest.skip("Windows does not reproduce TIME_WAIT bind failures the same way")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            conn, _ = srv.accept()
            conn.close()
        # client close leaves the server side in TIME_WAIT for this port
    assert _port_free(port) is True


def test_free_port_is_in_range_and_unique_per_call():
    a, b = free_port(), free_port()
    assert 18100 <= a <= 18199 and 18100 <= b <= 18199
