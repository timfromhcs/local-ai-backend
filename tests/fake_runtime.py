"""Tiny fake model runtime for supervisor tests: GET /health -> 200.

Usage: python fake_runtime.py --port N
Env FAKE_CRASH_AFTER=<seconds>: exit(1) that many seconds after startup.
"""
import argparse
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        code = 200 if self.path == "/health" else 404
        self.send_response(code)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok" if code == 200 else b"no")

    def log_message(self, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    crash = os.environ.get("FAKE_CRASH_AFTER")
    if crash:
        def _die():
            time.sleep(float(crash))
            os._exit(1)
        threading.Thread(target=_die, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
