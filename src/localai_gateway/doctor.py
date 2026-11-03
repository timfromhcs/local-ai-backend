"""Machine and model report, shared by the CLI (`--doctor`) and the HTTP route (`GET /doctor`)."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil

from .config import Settings
from .registry import ModelSpec


@dataclass(frozen=True)
class DoctorReport:
    version: str
    frozen: bool
    project_root: str
    models_yaml: str
    ram_total_gb: float
    budget_mb: int
    cpu_threads: int
    cpu_physical: int
    platform: str
    gpus: tuple[str, ...]
    llama_server: str
    sd_server: str
    models: tuple[dict[str, Any], ...]
    ready: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def gpu_names() -> tuple[str, ...]:
    """Best-effort GPU list. Never raises: doctor must work on any machine."""
    if platform.system() != "Windows":
        return ()
    try:
        raw = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"],
            text=True, timeout=20,
        )
        data = json.loads(raw or "[]")
        if isinstance(data, dict):
            data = [data]
        return tuple(f"{g.get('Name', '?')} ({(g.get('AdapterRAM') or 0) / 2**20:.0f} MB)" for g in data)
    except Exception:
        return ()


def report(settings: Settings, registry: dict[str, ModelSpec], models_yaml: Path | None = None) -> DoctorReport:
    from . import __version__

    rows = []
    ready = 0
    for spec in registry.values():
        rows.append({
            "id": spec.id,
            "kind": spec.kind,
            "status": spec.status,
            "est_mem_mb": spec.est_mem_mb,
            "reason": spec.reason,
        })
        ready += spec.status == "ready"
    return DoctorReport(
        version=__version__,
        frozen=bool(getattr(sys, "frozen", False)),
        project_root=str(settings.model_dir),
        models_yaml=str(models_yaml or (settings.model_dir / "models.yaml")),
        ram_total_gb=round(psutil.virtual_memory().total / 2**30, 1),
        budget_mb=settings.budget_mb,
        cpu_threads=psutil.cpu_count() or 1,
        cpu_physical=psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        platform=f"{platform.system()} {platform.release()}",
        gpus=gpu_names(),
        llama_server=str(settings.llama_path()),
        sd_server=str(settings.sd_path()),
        models=tuple(rows),
        ready=ready,
    )


def render(r: DoctorReport) -> str:
    """Human-readable CLI output. Same numbers as the HTTP route."""
    lines = [
        f"localai-gateway {r.version} doctor",
        f"  python          : {'frozen exe' if r.frozen else 'source'}",
        f"  project root    : {r.project_root}",
        f"  models.yaml     : {r.models_yaml}",
        f"  ram             : {r.ram_total_gb} GB total, budget {r.budget_mb} MB",
        f"  cpu             : {r.cpu_physical} cores / {r.cpu_threads} threads",
        f"  platform        : {r.platform}",
        f"  llama-server    : {r.llama_server} {'found' if Path(r.llama_server).exists() else 'MISSING'}",
        f"  sd-server       : {r.sd_server} {'found' if Path(r.sd_server).exists() else 'MISSING'}",
        "  models:",
    ]
    for g in r.gpus:
        lines.insert(7, f"  gpu             : {g}")
    for m in r.models:
        mem = f"{m['est_mem_mb']} MB" if m["est_mem_mb"] else "-"
        lines.append(f"    {m['id']:<18} {m['kind']:<8} {m['status']:<12} {mem:>9}  {m['reason'] or ''}".rstrip())
    lines.append(f"  ready models    : {r.ready}")
    return "\n".join(lines)
