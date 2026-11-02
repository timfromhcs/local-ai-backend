"""Phase 1 smoketest: probe every model with the vendor runtimes, record real results.

Usage:
    python tools/smoketest.py                 # everything
    python tools/smoketest.py --only qwen3-4b # one model (merged into docs/inventory.json)
    python tools/smoketest.py --only onnx     # only the onnx inspection
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

LOAD_TIMEOUT_S = 120
LOAD_TIMEOUT_27B_S = 300
REQUEST_TIMEOUT_S = 60
IMAGE_REQUEST_TIMEOUT_S = 180
SERVER_PROCS = ("llama-server", "sd-server", "sd-cli", "llama-cli")


@dataclass
class ProbeResult:
    model_id: str
    ok: bool
    load_s: float | None = None
    peak_rss_mb: float | None = None
    metric: float | None = None  # tokens/s (llm) or generation seconds (image)
    error: str | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- pure helpers
_EVAL_RE = re.compile(r"^\s*eval time\s*=.*?([\d.]+)\s+tokens per second", re.IGNORECASE | re.MULTILINE)


def parse_tps(log: str) -> float | None:
    """Tokens/s of generation from a llama.cpp timing log (not the prompt-eval line)."""
    m = None
    for m in _EVAL_RE.finditer(log):
        pass
    return float(m.group(1)) if m else None


def error_tail(log: str, n: int = 12) -> str:
    lines = [ln.rstrip() for ln in log.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


_GENERIC = re.compile(
    r"new_sd_ctx_t failed|model metadata validation failed|exiting due to model loading error|"
    r"failed to load model|failed to create_context|failed to create context|cleaning up|"
    r"common_init_|common_fit_params|server_run_i",
    re.I,
)
_ERR = re.compile(r"error|failed|unknown|not found|too long|not in model metadata", re.I)


def first_error_line(log: str) -> str:
    """Root-cause line: the last error-looking line that is not a generic wrapper message."""
    lines = [ln.strip() for ln in log.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if _ERR.search(ln) and not _GENERIC.search(ln):
            return ln
    return lines[-1] if lines else "no output"


PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


def validate_png(b64: str) -> tuple[int, int] | None:
    """(width, height) if b64 decodes to a PNG with a sane IHDR, else None."""
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        return None
    if len(raw) < 33 or raw[:8] != PNG_MAGIC or raw[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", raw[16:24])
    return (w, h) if w > 0 and h > 0 else None


def round_up_100(mb: float) -> int:
    return int(-(-mb // 100) * 100)


def free_port() -> int:
    """Free port in 18080-18199 (8080 belongs to the user's own server)."""
    for port in range(18080, 18200):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port in 18080-18199")


# ---------------------------------------------------------------- runtimes
def extract_runtimes(root: Path, bin_dir: Path) -> None:
    for pattern, sub in (("llama-prism-*.zip", "llama"), ("sd-master-*.zip", "sd")):
        dest = Path(bin_dir) / sub
        if dest.exists() and any(dest.glob("*.exe")):
            continue
        zips = sorted(Path(root).glob(pattern))
        if not zips:
            raise FileNotFoundError(f"no {pattern} in {root}")
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(dest)


# ---------------------------------------------------------------- process mgmt
def kill_tree(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        parent = psutil.Process(proc.pid)
        victims = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for p in victims:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(victims, timeout=10)


def stray_children() -> list[str]:
    """Descendants of THIS process still alive (never inspects/kills other servers)."""
    return [f"{c.name()}:{c.pid}" for c in psutil.Process().children(recursive=True)]


class RssSampler(threading.Thread):
    """Samples summed RSS (MB) of a process tree; keeps the peak."""

    def __init__(self, pid: int):
        super().__init__(daemon=True)
        self.pid, self.peak, self._stop_evt = pid, 0.0, threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                parent = psutil.Process(self.pid)
                total = sum(p.memory_info().rss for p in [parent] + parent.children(recursive=True))
                self.peak = max(self.peak, total / 2**20)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            self._stop_evt.wait(0.25)

    def stop(self) -> float:
        self._stop_evt.set()
        self.join(timeout=2)
        return self.peak


class Server:
    """A spawned server whose output goes to a log file; always killed as a tree on close()."""

    def __init__(self, exe: Path, args: list[str], tag: str):
        self.log_path = Path(tempfile.gettempdir()) / f"smoke_{tag}_{int(time.time())}.log"
        self._fh = open(self.log_path, "wb")
        try:
            self.proc = subprocess.Popen(
                [str(exe), *args], stdout=self._fh, stderr=subprocess.STDOUT, cwd=str(exe.parent),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
            )
        except Exception:
            self._fh.close()
            raise
        self.sampler = RssSampler(self.proc.pid)
        self.sampler.start()
        self.t0 = time.time()

    def log(self) -> str:
        self._fh.flush()
        return self.log_path.read_text(encoding="utf-8", errors="replace")

    def wait_ready(self, check, timeout: float) -> tuple[bool, str | None]:
        """Poll check() until true. Returns (ok, failure_kind), kind in {exited, timeout}."""
        end = time.time() + timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                return False, "exited"
            try:
                if check():
                    return True, None
            except Exception:
                pass
            time.sleep(0.5)
        return False, "timeout"

    def close(self) -> float:
        peak = self.sampler.stop()
        kill_tree(self.proc)
        self._fh.close()
        return peak


# ---------------------------------------------------------------- probes
def _llm_attempt(spec, bin_dir: Path, ngl: int, load_timeout: float) -> ProbeResult:
    port = free_port()
    exe = Path(bin_dir) / "llama" / "llama-server.exe"
    args = ["--model", str(spec.files[0]), "--port", str(port), "--host", "127.0.0.1", "-ngl", str(ngl), "-c", "2048"]
    srv = Server(exe, args, f"{spec.id}_ngl{ngl}")
    base = f"http://127.0.0.1:{port}"
    extra: dict = {"ngl": ngl, "ctx": 2048, "runtime": "llama-server", "unit": "tokens/s", "log": str(srv.log_path)}
    try:
        ok, kind = srv.wait_ready(lambda: httpx.get(base + "/health", timeout=2).status_code == 200, load_timeout)
        load_s = round(time.time() - srv.t0, 2)
        if not ok:
            reason = f"load timeout after {load_timeout}s" if kind == "timeout" else first_error_line(srv.log())
            extra.update(failure=kind, tail=error_tail(srv.log()))
            return ProbeResult(spec.id, False, load_s, None, None, reason, extra)
        t1 = time.time()
        r = httpx.post(
            base + "/v1/chat/completions", timeout=REQUEST_TIMEOUT_S,
            json={"messages": [{"role": "user", "content": "Say hello in five words."}], "max_tokens": 32, "stream": False},
        )
        r.raise_for_status()
        body = r.json()
        tps = (body.get("timings") or {}).get("predicted_per_second")
        extra["tps_source"] = "timings" if tps else "log"
        if not tps:
            time.sleep(0.5)
            tps = parse_tps(srv.log())
        extra["reply"] = (body["choices"][0]["message"].get("content") or "")[:120]
        extra["request_s"] = round(time.time() - t1, 2)
        return ProbeResult(spec.id, True, load_s, None, round(tps, 2) if tps else None, None, extra)
    except Exception as e:
        extra["tail"] = error_tail(srv.log())
        return ProbeResult(spec.id, False, round(time.time() - srv.t0, 2), None, None, f"{type(e).__name__}: {e}", extra)
    finally:
        extra["peak_rss_mb"] = round(srv.close(), 1)


def run_llm_probe(spec, bin_dir: Path) -> ProbeResult:
    timeout = LOAD_TIMEOUT_27B_S if spec.exclusive else LOAD_TIMEOUT_S
    res = _llm_attempt(spec, bin_dir, 99, timeout)
    if not res.ok and res.extra.get("failure") == "exited":
        first = res
        res = _llm_attempt(spec, bin_dir, 0, timeout)
        res.extra["ngl99_error"] = first.error
        res.extra["ngl99_tail"] = first.extra.get("tail")
    res.peak_rss_mb = res.extra.get("peak_rss_mb")
    return res


def run_image_probe(spec, bin_dir: Path, label: str, sd_args: list[str], load_timeout: float = LOAD_TIMEOUT_S) -> ProbeResult:
    port = free_port()
    exe = Path(bin_dir) / "sd" / "sd-server.exe"
    srv = Server(exe, [*sd_args, "--listen-ip", "127.0.0.1", "--listen-port", str(port), "-v"], f"{spec.id}_{label.split()[0]}")
    base = f"http://127.0.0.1:{port}"
    extra: dict = {"runtime": "sd-server", "variant": label, "args": sd_args, "unit": "gen_seconds", "log": str(srv.log_path)}

    def listening() -> bool:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True

    try:
        ok, kind = srv.wait_ready(listening, load_timeout)
        load_s = round(time.time() - srv.t0, 2)
        if not ok:
            reason = f"load timeout after {load_timeout}s" if kind == "timeout" else first_error_line(srv.log())
            extra.update(failure=kind, tail=error_tail(srv.log()))
            return ProbeResult(spec.id, False, load_s, None, None, reason, extra)
        t1 = time.time()
        r = httpx.post(
            base + "/sdapi/v1/txt2img", timeout=IMAGE_REQUEST_TIMEOUT_S,
            json={"prompt": "a red apple on a table", "width": 256, "height": 256, "steps": 4, "seed": 1},
        )
        gen_s = round(time.time() - t1, 2)
        r.raise_for_status()
        imgs = r.json().get("images") or []
        if not imgs:
            extra["tail"] = error_tail(srv.log())
            return ProbeResult(spec.id, False, load_s, None, None, "server returned no image", extra)
        extra["image_b64_len"] = len(imgs[0])
        dims = validate_png(imgs[0])
        if dims is None:
            extra["tail"] = error_tail(srv.log())
            return ProbeResult(spec.id, False, load_s, None, None, "server returned data that is not a valid PNG", extra)
        extra["png_size"] = list(dims)
        extra["output_validated"] = True
        return ProbeResult(spec.id, True, load_s, None, gen_s, None, extra)
    except Exception as e:
        extra["tail"] = error_tail(srv.log())
        return ProbeResult(spec.id, False, round(time.time() - srv.t0, 2), None, None, f"{type(e).__name__}: {e}", extra)
    finally:
        extra["peak_rss_mb"] = round(srv.close(), 1)


def _image_variants(spec, model_dir: Path) -> list[tuple[str, list[str]]]:
    f = [str(p) for p in spec.files]
    if spec.id == "flux2-klein":
        vae = str(Path(model_dir) / "vae.safetensors")
        qwen = str(Path(model_dir) / "mlabonne_Qwen3-4B-abliterated-Q4_K_M.gguf")
        return [
            ("diffusion+vae", ["--diffusion-model", f[0], "--vae", vae, "--offload-to-cpu"]),
            ("EXPERIMENT diffusion+vae+qwen3-4b-as-llm", ["--diffusion-model", f[0], "--vae", vae, "--llm", qwen, "--offload-to-cpu"]),
            ("EXPERIMENT diffusers-vae+qwen3-4b-as-llm", ["--diffusion-model", f[0], "--vae", str(Path(model_dir) / "diffusion_pytorch_model.safetensors"), "--llm", qwen, "--offload-to-cpu"]),
        ]
    if spec.id == "slat-3d":
        return [("diffusion-model", ["--diffusion-model", f[0]]), ("model", ["-m", f[0]])]
    return [("model", ["-m", f[0]]), ("diffusion-model", ["--diffusion-model", f[0]])]


def probe_model(spec, bin_dir: Path, model_dir: Path) -> dict:
    """Run every applicable attempt for one model. Returns the inventory record."""
    if spec.exclusive:
        size_mb = sum(f.stat().st_size for f in spec.files) / 2**20
        need = int(size_mb + 1500)
        for attempt in range(2):
            avail = psutil.virtual_memory().available / 2**20
            if avail >= need:
                break
            if attempt == 0:
                time.sleep(20)
        else:
            err = f"insufficient free RAM at probe time: available {avail:.0f} MB < needed {need} MB (model file {size_mb:.0f} MB + 1500)"
            return {"model_id": spec.id, "ok": False, "load_s": None, "peak_rss_mb": None, "metric": None,
                    "metric_unit": None, "error": err, "attempts": []}
    attempts: list[ProbeResult] = []
    if spec.kind == "llm" or spec.id in ("lance-video", "slat-3d"):
        attempts.append(run_llm_probe(spec, bin_dir))
    if spec.kind != "llm" and not (attempts and attempts[0].ok):
        for label, args in _image_variants(spec, model_dir):
            r = run_image_probe(spec, bin_dir, label, args)
            r.peak_rss_mb = r.extra.get("peak_rss_mb")
            attempts.append(r)
            if r.ok:
                break
    winner = next((a for a in attempts if a.ok), attempts[0])
    return {
        "model_id": spec.id,
        "ok": winner.ok,
        "load_s": winner.load_s,
        "peak_rss_mb": winner.peak_rss_mb if winner.ok else max((a.peak_rss_mb or 0) for a in attempts) or None,
        "est_mem_mb": round_up_100(winner.peak_rss_mb) if winner.ok and winner.peak_rss_mb else None,
        "metric": winner.metric,
        "metric_unit": winner.extra.get("unit") if winner.ok else None,
        "error": None if winner.ok else winner.error,
        "attempts": [asdict(a) for a in attempts],
    }


# ---------------------------------------------------------------- main
def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_inventory(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"models": {}, "onnx": [], "runtimes": {}}


def main() -> int:
    from localai_gateway.registry import load_registry
    from tools.onnx_inspect import inspect_onnx

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="model id, or 'onnx'")
    ap.add_argument("--out", default=str(ROOT / "docs" / "inventory.json"))
    ns = ap.parse_args()
    out = Path(ns.out)
    registry = load_registry(ROOT / "models.yaml", ROOT)
    if ns.only and ns.only != "onnx" and ns.only not in registry:
        print(f"unknown --only id {ns.only!r}; known: onnx, {', '.join(registry)}", file=sys.stderr)
        return 2
    inv = _load_inventory(out)
    bin_dir = ROOT / "bin"
    extract_runtimes(ROOT, bin_dir)
    inv["runtimes"] = {"llama-server": "build 12128 (d7b289683)", "sd-server": "commit 3be2378"}

    if ns.only in (None, "onnx"):
        inv["onnx"] = [inspect_onnx(p) for p in sorted(ROOT.glob("*.onnx"))]
        by_hash: dict[str, list[str]] = {}
        for o in inv["onnx"]:
            by_hash.setdefault(o["sha256"], []).append(o["file"])
        inv["onnx_duplicates"] = [v for v in by_hash.values() if len(v) > 1]
    if ns.only != "onnx":
        for mid, spec in registry.items():
            if ns.only and mid != ns.only:
                continue
            print(f"== probing {mid}", flush=True)
            try:
                rec = probe_model(spec, bin_dir, ROOT)
            finally:
                stray = stray_children()
                if stray:
                    print("STRAY PROCESSES:", stray, flush=True)
            inv["models"][mid] = rec
            print(json.dumps({k: rec[k] for k in ("ok", "load_s", "peak_rss_mb", "metric", "error")}, indent=1), flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(out, inv)
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(out, inv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
