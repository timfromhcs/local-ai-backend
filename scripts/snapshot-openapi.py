"""Writes docs/openapi.json from the real app, so the contract snapshot is never hand-edited."""
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


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    settings = Settings(budget_mb=5000, bin_dir=root, model_dir=root)
    app = create_app(settings, {}, _Scheduler())
    spec = TestClient(app).get("/openapi.json").json()
    out = root / "docs" / "openapi.json"
    out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print(f"wrote {out} ({len(spec['paths'])} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
