"""CycloneDX 1.5 SBOM from the installed distributions (stdlib only, no network).

    python scripts/sbom.py                 -> dist/sbom.cdx.json
    python scripts/sbom.py --out sbom.json

Why not a library: the release workflow must not depend on a package that can disappear, and
the SBOM only needs name/version/licence/purl for the components we actually ship.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def component(dist: metadata.Distribution) -> dict | None:
    try:
        name = dist.metadata["Name"]
        version = dist.version
    except Exception:
        return None
    if not name or not version:
        return None
    licences = []
    try:
        raw = dist.metadata.get_all("License") or []
        for item in raw:
            if item and len(item) < 120:
                licences.append({"license": {"name": item}})
    except Exception:
        pass
    return {
        "type": "library",
        "name": name,
        "version": version,
        "purl": f"pkg:pypi/{name.lower().replace('_', '-')}@{version}",
        **({"licenses": licences} if licences else {}),
    }


def _root_version(root_name: str) -> str:
    """Prefer the working tree version: an editable install can be stale."""
    source = Path(__file__).resolve().parents[1] / "src" / "localai_gateway" / "__init__.py"
    try:
        for line in source.read_text(encoding="utf8").splitlines():
            if line.startswith("__version__"):
                return line.split('"')[1]
    except OSError:
        pass
    try:
        return metadata.version(root_name)
    except metadata.PackageNotFoundError:
        return "0.0.0"


def build(root_name: str = "localai-gateway") -> dict:
    seen: dict[str, dict] = {}
    for dist in metadata.distributions():
        comp = component(dist)
        if comp:
            seen.setdefault(comp["purl"], comp)
    components = sorted(seen.values(), key=lambda c: (c["name"].lower(), c["version"]))

    root_version = _root_version(root_name)

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tools": [{"vendor": "localai-backend", "name": "scripts/sbom.py", "version": "1.0.0"}],
            "component": {
                "type": "application",
                "name": root_name,
                "version": root_version,
                "purl": f"pkg:pypi/{root_name}@{root_version}",
            },
            "properties": [
                {"name": "python", "value": platform.python_version()},
                {"name": "platform", "value": f"{platform.system()} {platform.release()}"},
                {"name": "machine", "value": platform.machine()},
            ],
        },
        "components": components,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write a CycloneDX SBOM for the installed environment")
    p.add_argument("--out", default="dist/sbom.cdx.json")
    p.add_argument("--name", default="localai-gateway")
    args = p.parse_args(argv)

    data = build(args.name)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")
    print(f"wrote {out} ({len(data['components'])} components, {data['metadata']['component']['version']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
