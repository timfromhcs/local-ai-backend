from __future__ import annotations

import os
import sys
from pathlib import Path

import psutil
from pydantic import BaseModel


def _pick_root(here: Path, cwd: Path, home: str | None) -> Path:
    """Pick the writable project root for a non-frozen install.

    - LOCALAI_HOME always wins (containers, portable layouts).
    - A source checkout has models.yaml two levels above config.py.
    - A pip-installed wheel does NOT live in a checkout, so falling back to
      parents[2] pointed at site-packages and tried to create /usr/local/lib/python3.13/cache
      (found by the container smoke test). Use the working directory instead.
    """
    if home:
        return Path(home)
    candidate = here.parents[2]
    return candidate if (candidate / "models.yaml").exists() else cwd


def _project_root() -> Path:
    """Source tree in dev, the exe's folder when frozen, cwd for site-packages."""
    if getattr(sys, "frozen", False):
        return Path(os.environ.get("LOCALAI_HOME", Path(sys.executable).parent))
    return _pick_root(Path(__file__).resolve(), Path.cwd(), os.environ.get("LOCALAI_HOME"))


def bundle_dir() -> Path:
    """Read-only files shipped inside the one-file exe (e.g. models.yaml)."""
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else _project_root()


def config_file(name: str) -> Path:
    """Prefer the user-editable file next to the exe, fall back to the bundled copy."""
    local = _project_root() / name
    if local.exists():
        return local
    return bundle_dir() / name


PROJECT_ROOT = _project_root()


class Settings(BaseModel):
    budget_mb: int
    bin_dir: Path
    model_dir: Path
    port: int = 8090  # 8080 belongs to the Jev server
    cache_dir: Path = PROJECT_ROOT / "cache" / "slots"  # SSD: llama-server slot states (precomputed OpenCode prefix)
    llama_server: Path | None = None  # LOCALAI_LLAMA_SERVER: use a runtime from elsewhere (e.g. a shared Vulkan build)
    sd_server: Path | None = None      # LOCALAI_SD_SERVER

    def llama_path(self) -> Path:
        return self.llama_server or (self.bin_dir / "llama" / "llama-server.exe")

    def sd_path(self) -> Path:
        return self.sd_server or (self.bin_dir / "sd" / "sd-server.exe")

    @classmethod
    def default(cls) -> "Settings":
        total_mb = psutil.virtual_memory().total / 2**20
        return cls(
            budget_mb=int(total_mb * 0.7),
            bin_dir=PROJECT_ROOT / "bin",
            model_dir=PROJECT_ROOT,
            llama_server=Path(os.environ["LOCALAI_LLAMA_SERVER"]) if os.environ.get("LOCALAI_LLAMA_SERVER") else None,
            sd_server=Path(os.environ["LOCALAI_SD_SERVER"]) if os.environ.get("LOCALAI_SD_SERVER") else None,
        )
