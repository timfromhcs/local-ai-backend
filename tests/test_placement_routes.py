"""Placement and autotune HTTP contract (the MCP tools call these routes)."""
import pytest
from fastapi.testclient import TestClient

from localai_gateway.app import create_app
from localai_gateway.config import Settings
from localai_gateway.registry import ModelSpec


class _Scheduler:
    def loaded(self):
        return []

    async def ensure_loaded(self, model_id):
        raise KeyError(model_id)

    async def release(self, model_id):
        return None


def _registry():
    return {
        "big": ModelSpec(id="big", kind="llm", files=[], runtime="llama", est_mem_mb=9000),
        "mid": ModelSpec(id="mid", kind="llm", files=[], runtime="llama", est_mem_mb=3600),
        "small": ModelSpec(id="small", kind="llm", files=[], runtime="llama", est_mem_mb=800),
        "img": ModelSpec(id="img", kind="image", files=[], runtime="sd", est_mem_mb=1000),
        "gone": ModelSpec(id="gone", kind="llm", files=[], runtime="llama", est_mem_mb=100,
                          status="unavailable", reason="missing file"),
    }


@pytest.fixture()
def client(tmp_path):
    settings = Settings(budget_mb=5000, bin_dir=tmp_path, model_dir=tmp_path)
    return TestClient(create_app(settings, _registry(), _Scheduler()))


def test_placement_returns_the_largest_fitting_set(client):
    r = client.get("/v1/placement")
    assert r.status_code == 200
    body = r.json()
    assert set(body["models"]) == {"mid", "small"}
    assert body["memory_mb"] == 4400
    assert body["budget_mb"] == 5000
    assert body["why"]


def test_placement_respects_explicit_budget_and_kinds(client):
    body = client.get("/v1/placement", params={"budget_mb": 2000, "kinds": "llm"}).json()
    assert body["models"] == ["small"]
    assert body["memory_mb"] == 800


def test_placement_includes_pinned_other_kind(client):
    body = client.get("/v1/placement", params={"kinds": "llm", "pinned": "img"}).json()
    assert "img" in body["models"]


def test_placement_conflict_is_structured(client):
    r = client.get("/v1/placement", params={"budget_mb": 50})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "no_placement"


def test_autotune_returns_explainable_profile(client):
    body = client.get("/v1/autotune", params={"model": "mid"}).json()
    assert body["class"] in {"igpu_uma", "dgpu", "cpu"}
    assert body["model_mb"] == 3600
    assert isinstance(body["args"], list) and body["args"]
    assert len(body["why"]) >= 2


def test_autotune_unknown_model(client):
    r = client.get("/v1/autotune", params={"model": "nope"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_autotune_without_memory_estimate(client):
    r = client.get("/v1/autotune", params={"model": "small", "ctx": 4096})
    assert r.status_code == 200
    assert "-c 4096" in r.json()["args"]
