"""Self-healing runtime: `bin/llama/llama-server.exe` has disappeared twice on this machine.

No antivirus event, no git action (bin/ is ignored), and only that one file vanished while its DLLs
stayed behind. The cause is unknown; what is knowable is that a missing launcher turns into a 90 s
timeout and a confusing 503. The shipped zip sits in the project root, so the gateway can restore the
runtime itself instead of failing.
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

log = logging.getLogger("localai_gateway.runtime")

MEMBER = "llama-server.exe"


def restore_from_zip(zip_path: Path, target: Path) -> str:
    """Extract the whole runtime next to `target`. Returns a short note for the caller."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            if MEMBER not in names:
                return f"{zip_path.name} does not contain {MEMBER}"
            target.parent.mkdir(parents=True, exist_ok=True)
            for name in names:
                if name.endswith("/"):
                    continue
                # flat archives only: the zip ships the binaries at its root
                if "/" in name and not name.startswith(("bin/", "lib/")):
                    continue
                data = zf.read(name)
                out = target.parent / Path(name).name
                out.write_bytes(data)
        return f"restored {MEMBER} from {zip_path.name}"
    except (OSError, zipfile.BadZipFile) as exc:
        return f"could not restore {MEMBER} from {zip_path.name}: {type(exc).__name__}: {exc}"


def ensure_runtime(exe: Path, zip_path: Path | None,
                   fallback: Path | None = None) -> tuple[Path, str]:
    """(path, note). Restores the launcher when it is missing and a zip is available.

    Measured 2026-11-05: writing `bin/llama/llama-server.exe` fails with PermissionError on this
    machine while every sibling writes fine, and the file disappears again - something outside the
    repository guards that exact path. A second location keeps the stack working; the note says
    where the runtime ended up.
    """
    if exe.exists():
        return exe, ""
    if zip_path is None or not Path(zip_path).exists():
        return exe, f"{exe} is missing and no runtime zip was found"
    note = restore_from_zip(Path(zip_path), exe)
    if exe.exists():
        log.warning("%s", note)
        return exe, note
    if fallback is not None:
        alt = Path(fallback) / exe.name
        alt_note = restore_from_zip(Path(zip_path), alt)
        if alt.exists():
            combined = f"{note}; using {alt} instead"
            log.warning("%s", combined)
            return alt, combined
        note = f"{note}; fallback failed: {alt_note}"
    log.error("%s", note)
    return exe, note


def default_zip(root: Path) -> Path | None:
    """The runtime archive shipped next to the project, if any."""
    candidates = sorted(root.glob("llama-*vulkan*.zip")) or sorted(root.glob("llama-*.zip"))
    return candidates[0] if candidates else None
