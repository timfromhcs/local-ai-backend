"""Capture the prompt prefix OpenCode really sends (system prompt incl. skills list + tool schemas) -> cache/opencode_prefix.json.

Runs `opencode run` once against a local mock OpenAI server (no GPU, no model), so the prefix always matches the CURRENT
skills/tools/config. Re-run after changing skills, MCP servers or OpenCode config; the gateway re-warms automatically
because the cache key changes with the prefix.

usage: python scripts/capture_opencode_prefix.py [--cwd DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def serve(captured: list[dict]) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured.append(body)
            chunks = [{"role": "assistant"}, {"content": "OK"}, {}]
            if body.get("stream"):
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                for d in chunks:
                    c = {"id": "m", "object": "chat.completion.chunk", "created": 0, "model": "m",
                         "choices": [{"index": 0, "delta": d, "finish_reason": None if d else "stop"}]}
                    self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                data = json.dumps({"id": "m", "object": "chat.completion", "created": 0, "model": "m", "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}]}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def log_message(self, *a):  # noqa: D401
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwd", default=str(ROOT))
    ap.add_argument("--out", default=str(ROOT / "cache" / "opencode_prefix.json"))
    args = ap.parse_args()
    captured: list[dict] = []
    srv = serve(captured)
    port = srv.server_address[1]
    cfg = {"providers": {"capture": {"package": "@ai-sdk/openai-compatible", "name": "capture",
           "settings": {"baseURL": f"http://127.0.0.1:{port}/v1", "apiKey": "x"},
           "models": {"m": {"name": "m", "limit": {"context": 65536, "output": 2048}}}}}}
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "capture.json"
        cfg_path.write_text(json.dumps(cfg))
        env = {**os.environ, "OPENCODE_CONFIG": str(cfg_path)}
        subprocess.run(["opencode", "run", "--standalone", "--auto", "-m", "capture/m", "hello"],
                       cwd=args.cwd, env=env, timeout=180, capture_output=True, shell=(os.name == "nt"))
    srv.shutdown()
    main_req = max((b for b in captured if b.get("tools")), key=lambda b: len(json.dumps(b)), default=None)
    if main_req is None:
        print("no agent request captured (does OPENCODE_CONFIG work in this OpenCode version?)")
        return 1
    system = next(m for m in main_req["messages"] if m["role"] == "system")
    prefix = {"messages": [system, {"role": "user", "content": "hello"}], "tools": main_req["tools"]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prefix, ensure_ascii=False), encoding="utf8")
    sys_chars = len(system["content"]) if isinstance(system["content"], str) else len(json.dumps(system["content"]))
    print(f"captured: system {sys_chars} chars, {len(prefix['tools'])} tools -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
