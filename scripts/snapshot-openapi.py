"""Writes docs/openapi.json from the real app, so the contract snapshot is never hand-edited."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from localai_gateway.app import create_app  # noqa: E402
from localai_gateway.config import Settings  # noqa: E402


class _Scheduler:
    def loaded(self):
        return []

    async def ensure_loaded(self, model_id):
        raise KeyError(model_id)

    async def release(self, model_id):
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Writes docs/openapi.json from the real app, so the contract snapshot is never hand-edited.")
    p.add_argument("--check", action="store_true", help="fail if the snapshot is stale")
    args = p.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    settings = Settings(budget_mb=5000, bin_dir=root, model_dir=root)
    app = create_app(settings, {}, _Scheduler())
    spec = TestClient(app).get("/openapi.json").json()
    out = root / "docs" / "openapi.json"
    rendered = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    if args.check:
        current = out.read_text(encoding="utf8") if out.exists() else ""
        if current != rendered:
            print("docs/openapi.json is stale - run: python scripts/snapshot-openapi.py")
            return 1
        print("docs/openapi.json is up to date")
        return 0
    out.write_text(rendered, encoding="utf8")
    print(f"wrote {out} ({len(spec['paths'])} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
