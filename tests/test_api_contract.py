"""API contract: the paths and the error shape are frozen, the version is visible, and the
`/doctor` route returns the same report as the CLI."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import localai_gateway
from localai_gateway.app import create_app
from localai_gateway.config import Settings
from localai_gateway.registry import ModelSpec

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "docs" / "openapi.json"

STABLE_PATHS = [
    "/healthz",
    "/doctor",
    "/v1/models",
    "/v1/chat/completions",
    "/v1/placement",
    "/v1/autotune",
]


class _Scheduler:
    def loaded(self):
        return []

    async def ensure_loaded(self, model_id):
        raise KeyError(model_id)

    async def release(self, model_id):
        return None


@pytest.fixture()
def client(tmp_path):
    settings = Settings(budget_mb=5000, bin_dir=tmp_path, model_dir=tmp_path)
    registry = {"m": ModelSpec(id="m", kind="llm", files=[], runtime="llama", est_mem_mb=1000)}
    return TestClient(create_app(settings, registry, _Scheduler()))


def test_every_response_carries_the_version(client):
    r = client.get("/healthz")
    assert r.headers["x-localai-version"] == localai_gateway.__version__


def test_stable_paths_exist(client):
    spec = client.get("/openapi.json").json()
    for path in STABLE_PATHS:
        assert path in spec["paths"], f"stable path removed: {path}"


def test_openapi_matches_snapshot(client):
    """Fail on unannounced API drift; refresh with scripts/snapshot-openapi.py."""
    current = client.get("/openapi.json").json()
    assert SNAPSHOT.is_file(), (
        "docs/openapi.json is missing - generate it with: python scripts/snapshot-openapi.py"
    )
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf8"))
    assert sorted(snapshot["paths"]) == sorted(current["paths"]), (
        "API paths changed; update docs/openapi.json and the client surface deliberately"
    )
    for path, ops in snapshot["paths"].items():
        assert sorted(ops) == sorted(current["paths"][path]), f"methods changed on {path}"


def test_doctor_route_shape(client):
    body = client.get("/doctor").json()
    assert body["version"] == localai_gateway.__version__
    assert body["budget_mb"] == 5000
    assert isinstance(body["gpus"], list)
    assert body["models"][0]["id"] == "m"
    assert body["ready"] == 1
    for key in ["ram_total_gb", "cpu_threads", "cpu_physical", "platform", "llama_server", "sd_server"]:
        assert key in body


def test_error_schema_is_stable(client):
    r = client.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "model_not_found"
    assert "reason" in err
    assert "valid_ids" in err


def test_placement_conflict_schema(client):
    r = client.get("/v1/placement", params={"budget_mb": 10})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "no_placement"
