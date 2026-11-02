"""Tiny fake model runtime for supervisor tests: GET /health -> 200.

Usage: python fake_runtime.py --port N
Env:
  FAKE_CRASH_AFTER=<s>      exit(1) that many seconds after the port opens
  FAKE_CRASH_ONCE_FILE=<p>  with FAKE_CRASH_AFTER: only crash if <p> does not exist yet (creates it)
  FAKE_BIND_DELAY=<s>       sleep before opening the port (simulates slow model load)
  FAKE_SPAWN_CHILD=1        spawn a long-sleeping child process (to test tree kill)
"""
import argparse
import os
import subprocess
import sys
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
    delay = os.environ.get("FAKE_BIND_DELAY")
    if delay:
        time.sleep(float(delay))
    if os.environ.get("FAKE_SPAWN_CHILD") == "1":
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    crash = os.environ.get("FAKE_CRASH_AFTER")
    once = os.environ.get("FAKE_CRASH_ONCE_FILE")
    if once:
        if os.path.exists(once):
            crash = None
        else:
            open(once, "w").close()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    if crash:
        def _die():
            time.sleep(float(crash))
            os._exit(1)
        threading.Thread(target=_die, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
