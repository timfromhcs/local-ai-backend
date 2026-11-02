"""python -m localai_gateway  ->  OpenAI-compatible gateway on 127.0.0.1:8090 (no model is loaded until first request)."""
from __future__ import annotations

import json
import logging
import os

import httpx
import uvicorn

from .app import create_app
from .config import PROJECT_ROOT, Settings
from .registry import load_registry
from .scheduler import Scheduler
from .supervisor import Supervisor
from .warmup import Warmer, make_hook


def build_app():
    settings = Settings.default()
    if os.environ.get("LOCALAI_PORT"):
        settings.port = int(os.environ["LOCALAI_PORT"])
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry(PROJECT_ROOT / "models.yaml", settings.model_dir)
    prefix_file = settings.cache_dir.parent / "opencode_prefix.json"

    def load_prefix() -> dict | None:
        return json.loads(prefix_file.read_text(encoding="utf8")) if prefix_file.exists() else None

    warmer = Warmer(settings.cache_dir, httpx.AsyncClient())
    scheduler = Scheduler(settings, registry, Supervisor(), on_started=make_hook(warmer, load_prefix))
    return settings, create_app(settings, registry, scheduler)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings, app = build_app()
    uvicorn.run(app, host="127.0.0.1", port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
