from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    id: str
    kind: Literal["llm", "image", "video", "image3d"]
    files: list[Path]
    runtime: str
    est_mem_mb: int
    exclusive: bool = False
    args: list[str] = Field(default_factory=list)
    status: Literal["ready", "unavailable", "unsupported"] = "ready"
    reason: str | None = None
    experimental: bool = False
    notes: str | None = None
    warm_prefix: bool = False  # precompute + restore the OpenCode prompt prefix at every server start
    endpoint: str | None = None  # runtime "external": an already running OpenAI-compatible server (never spawned)
    api_key: str | None = None  # runtime "external": bearer token that server requires (e.g. llama-server --api-key)


def load_registry(path: Path, model_dir: Path) -> dict[str, ModelSpec]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    registry: dict[str, ModelSpec] = {}
    for entry in data.get("models", []):
        entry = dict(entry)
        entry["files"] = [Path(model_dir) / f for f in entry.get("files", [])]
        spec = ModelSpec(**entry)
        if spec.status == "ready":
            missing = [f.name for f in spec.files if not f.exists()]
            if missing:
                spec.status = "unavailable"
                spec.reason = "missing file(s): " + ", ".join(missing)
        registry[spec.id] = spec
    return registry
