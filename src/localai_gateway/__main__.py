"""python -m localai_gateway  ->  OpenAI-compatible gateway on 127.0.0.1:8090 (no model is loaded until first request)."""
from __future__ import annotations

import argparse
import json
import logging
import os

import httpx
import uvicorn

from .app import create_app
from .config import PROJECT_ROOT, Settings, config_file
from .doctor import render, report
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


def doctor() -> int:
    """Print what this machine can actually run. Exit 1 if no LLM is ready."""
    settings = Settings.default()
    registry = load_registry(config_file("models.yaml"), settings.model_dir)
    result = report(settings, registry, config_file("models.yaml"))
    print(render(result))
    if result.ready == 0:
        print("\nNo runnable model found. Put GGUF files next to this binary (see models.yaml) - "
              "the exe deliberately ships without model weights.")
        return 1
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="localai-gateway", description="Local OpenAI-compatible model gateway")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="port (default 8090, env LOCALAI_PORT)")
    parser.add_argument("--doctor", action="store_true", help="print hardware/model status and exit")
    parser.add_argument("--self-test", action="store_true", help="exercise the API in-process (no port, no model)")
    parser.add_argument("--check-update", action="store_true", help="ask GitHub whether a newer release exists (only network call)")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(__version__)
        return
    if args.check_update:
        from .update import main as check_update

        raise SystemExit(check_update())
    if args.self_test:
        from .selftest import main as self_test

        raise SystemExit(self_test())
    if args.doctor:
        raise SystemExit(doctor())

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings, app = build_app()
    if args.port:
        settings.port = args.port
    from .exposure import PublicBindRefused, resolve_host

    try:
        host = resolve_host(args.host)
    except PublicBindRefused as exc:
        print(f"error: {exc}", flush=True)
        raise SystemExit(2) from None
    uvicorn.run(app, host=host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
