"""python -m localai_gateway  ->  OpenAI-compatible gateway on 127.0.0.1:8090 (no model is loaded until first request)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys

import httpx
import psutil
import uvicorn

from .app import create_app
from .config import PROJECT_ROOT, Settings, config_file
from .registry import load_registry
from .scheduler import Scheduler
from .supervisor import Supervisor
from .warmup import Warmer, make_hook


def build_app(settings: Settings | None = None):
    settings = settings or Settings.default()
    if os.environ.get("LOCALAI_PORT"):
        settings.port = int(os.environ["LOCALAI_PORT"])
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry(config_file("models.yaml"), settings.model_dir)
    prefix_file = settings.cache_dir.parent / "opencode_prefix.json"

    def load_prefix() -> dict | None:
        return json.loads(prefix_file.read_text(encoding="utf8")) if prefix_file.exists() else None

    warmer = Warmer(settings.cache_dir, httpx.AsyncClient())
    scheduler = Scheduler(settings, registry, Supervisor(), on_started=make_hook(warmer, load_prefix))
    return settings, create_app(settings, registry, scheduler)


def _gpus() -> list[str]:
    """Best-effort GPU list. Never raises: the doctor must work on any machine."""
    if platform.system() != "Windows":
        return []
    try:
        import subprocess

        raw = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"],
            text=True, timeout=20,
        )
        data = json.loads(raw or "[]")
        if isinstance(data, dict):
            data = [data]
        return [f"{g.get('Name', '?')} ({(g.get('AdapterRAM') or 0) / 2**20:.0f} MB)" for g in data]
    except Exception:
        return []


def doctor() -> int:
    """Print what this machine can actually run. Exit 1 if no LLM is ready."""
    settings = Settings.default()
    registry = load_registry(config_file("models.yaml"), settings.model_dir)
    gpus = _gpus()

    print("localai-gateway doctor")
    print(f"  python          : {sys.version.split()[0]} ({'frozen exe' if getattr(sys, 'frozen', False) else 'source'})")
    print(f"  project root    : {PROJECT_ROOT}")
    print(f"  models.yaml     : {config_file('models.yaml')}")
    print(f"  ram             : {psutil.virtual_memory().total / 2**30:.1f} GB total, budget {settings.budget_mb} MB")
    print(f"  cpu             : {platform.processor() or platform.machine()} ({psutil.cpu_count()} threads)")
    for g in gpus:
        print(f"  gpu             : {g}")
    print(f"  llama-server    : {settings.bin_dir / 'llama' / 'llama-server.exe'} "
          f"{'found' if (settings.bin_dir / 'llama' / 'llama-server.exe').exists() else 'MISSING'}")
    print(f"  sd-server       : {settings.bin_dir / 'sd' / 'sd-server.exe'} "
          f"{'found' if (settings.bin_dir / 'sd' / 'sd-server.exe').exists() else 'MISSING'}")
    print("  models:")
    ready = 0
    for spec in registry.values():
        mem = f"{spec.est_mem_mb} MB" if spec.est_mem_mb else "-"
        print(f"    {spec.id:<18} {spec.kind:<8} {spec.status:<12} {mem:>9}  {spec.reason or ''}".rstrip())
        ready += spec.status == "ready"
    print(f"  ready models    : {ready}")
    if ready == 0:
        print("\nNo runnable model found. Put GGUF files next to this binary (see models.yaml) - "
              "the exe deliberately ships without model weights.")
        return 1
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="localai-gateway", description="Local OpenAI-compatible model gateway")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="port (default 8090, env LOCALAI_PORT)")
    parser.add_argument("--doctor", action="store_true", help="print hardware/model status and exit")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(__version__)
        return
    if args.doctor:
        raise SystemExit(doctor())

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings, app = build_app()
    if args.port:
        settings.port = args.port
    uvicorn.run(app, host=args.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
