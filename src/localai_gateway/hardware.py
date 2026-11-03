"""Device-aware llama.cpp launch profiles.

The gateway used to hardcode flags per model in models.yaml. That is fine for one laptop and wrong
for everything else. This module derives flags from the device class:

    iGPU (shared memory) -> full offload, more threads, moderate batch
    dGPU (discrete VRAM) -> offload only what fits VRAM with headroom
    CPU only             -> no offload, small batch, no mmap tricks

Every returned flag carries a one-line reason, because a profile nobody can explain is a guess.
Measured constraints from this repo (docs/perf-log.md): on the RDNA2 iGPU, kernel env switches gave
nothing, `-ub 256` beat 64/128/512/1024, and `-t 16` was slower than `-t 8`.
"""
from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psutil


class DeviceClass(str, Enum):
    IGPU_UMA = "igpu_uma"
    DGPU = "dgpu"
    CPU = "cpu"


@dataclass(frozen=True)
class Device:
    name: str
    device_class: DeviceClass
    vram_mb: int
    ram_mb: int
    physical_cores: int
    platform: str

    @property
    def shared_memory(self) -> bool:
        return self.device_class is DeviceClass.IGPU_UMA


@dataclass(frozen=True)
class Profile:
    args: tuple[str, ...]
    why: tuple[str, ...]
    device: Device
    model_mb: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device.name,
            "class": self.device.device_class.value,
            "model_mb": self.model_mb,
            "args": list(self.args),
            "why": list(self.why),
        }


def _threads(device: Device) -> int:
    # measured: 8 threads beat 16 on an 8C/16T Zen3+ with an iGPU doing the math
    return max(1, min(8, device.physical_cores or 4))


def profile_for(device: Device, model_mb: int, ctx: int = 16384) -> Profile:
    """Derive llama.cpp flags for one model on one device."""
    if model_mb <= 0:
        raise ValueError("model_mb must be positive")
    t = _threads(device)
    why: list[str] = []
    args: list[str] = [f"-c {ctx}", "-fa on", "-ctk q8_0", "-ctv q8_0"]

    if device.device_class is DeviceClass.CPU:
        args += ["-ngl 0", f"-t {t}", "-ub 128", "-b 1024"]
        why.append(f"cpu-only: no offload, {t} threads (physical cores, capped at 8)")
        why.append("small batch 128: less scratch memory, decode is bandwidth-bound anyway")
    elif device.device_class is DeviceClass.DGPU:
        usable = int(device.vram_mb * 0.85)
        per_layer = max(1.0, model_mb / 32)
        layers = max(0, min(32, int(usable / per_layer)))
        if layers >= 32:
            args += ["-ngl 99", f"-t {t}", "-ub 256", "-b 2048"]
            why.append(f"dgpu: whole model fits in {usable} MB usable VRAM, offload all layers")
        else:
            args += [f"-ngl {layers}", f"-t {t}", "-ub 256", "-b 2048"]
            why.append(
                f"dgpu: {model_mb} MB does not fit {usable} MB usable VRAM, offload {layers}/32 layers"
            )
    else:  # IGPU_UMA
        budget = int(device.ram_mb * 0.75)
        if model_mb + 1024 <= budget:
            args += ["-ngl 99", f"-t {t}", "-ub 256", "-b 2048"]
            why.append(f"igpu (shared RAM): full offload, {model_mb} MB + 1 GB headroom within {budget} MB")
        else:
            args += ["-ngl 20", f"-t {t}", "-ub 256", "-b 2048"]
            why.append(f"igpu (shared RAM): model {model_mb} MB is tight for {budget} MB, partial offload")

    if device.ram_mb and device.ram_mb * 0.7 < model_mb * 1.35:
        args.append("--no-mmap")
        why.append("ram is tight: let llama-server own the weights instead of the page cache")
    else:
        why.append("mmap stays on: the page cache is free memory for a model that is used repeatedly")

    why.append(f"context {ctx}, flash attention, q8_0 KV (q4_0 measurably broke tool calling)")
    return Profile(tuple(args), tuple(why), device, model_mb)


def _windows_gpus() -> list[dict[str, Any]]:
    try:
        raw = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress"],
            text=True, timeout=20,
        )
        data = __import__("json").loads(raw or "[]")
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def detect_device(gpus: list[dict[str, Any]] | None = None) -> Device:
    """Best-effort detection. `gpus` can be injected so the logic is testable."""
    ram_mb = int(psutil.virtual_memory().total / 2**20)
    cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 4
    system = platform.system()

    if gpus is None:
        gpus = _windows_gpus() if system == "Windows" else []

    # explicit "GPU" entries beat virtual display adapters (Parsec, RDP)
    candidates = [g for g in gpus if g.get("Name") and "virtual" not in str(g.get("Name", "")).lower()]
    if not candidates:
        return Device("cpu", DeviceClass.CPU, 0, ram_mb, cores, system)

    best = max(candidates, key=lambda g: int(g.get("AdapterRAM") or 0))
    name = str(best.get("Name") or "gpu")
    vram_mb = int((best.get("AdapterRAM") or 0) / 2**20)
    lowered = name.lower()
    integrated = any(k in lowered for k in ("radeon(tm) graphics", "radeon graphics", "intel", "iris", "vega", "apu"))
    # Windows reports a carved-out adapter RAM for iGPUs (e.g. 4 GB of 20 GB shared)
    if integrated or (0 < vram_mb <= 8192 and ram_mb >= 16384 and "geforce" not in lowered and "rx " not in lowered):
        return Device(name, DeviceClass.IGPU_UMA, vram_mb, ram_mb, cores, system)
    return Device(name, DeviceClass.DGPU, vram_mb, ram_mb, cores, system)
