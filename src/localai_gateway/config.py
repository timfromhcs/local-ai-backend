from __future__ import annotations

from pathlib import Path

import psutil
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    budget_mb: int
    bin_dir: Path
    model_dir: Path
    port: int = 8090  # 8080 belongs to the Jev server
    cache_dir: Path = PROJECT_ROOT / "cache" / "slots"  # SSD: llama-server slot states (precomputed OpenCode prefix)

    @classmethod
    def default(cls) -> "Settings":
        total_mb = psutil.virtual_memory().total / 2**20
        return cls(
            budget_mb=int(total_mb * 0.7),
            bin_dir=PROJECT_ROOT / "bin",
            model_dir=PROJECT_ROOT,
        )
