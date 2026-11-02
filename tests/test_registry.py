from pathlib import Path

import psutil

from localai_gateway.config import Settings
from localai_gateway.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent
YAML = ROOT / "models.yaml"


def test_loads_models_yaml():
    reg = load_registry(YAML, ROOT)
    assert {"qwen3-4b", "bonsai-27b", "flux2-klein"} <= set(reg)
    assert reg["bonsai-27b"].exclusive is False          # duo keeps planner and executor resident together
    assert reg["duo"].runtime == "virtual" and reg["jev-1.5b"].runtime == "external"
    assert reg["jev-1.5b"].endpoint == "http://127.0.0.1:8080"
    assert "--jinja" in reg["qwen3-4b"].args and "--jinja" in reg["bonsai-27b"].args
    assert reg["qwen3-4b"].runtime == "llama"
    assert reg["flux2-klein"].runtime == "sd"
    assert reg["lance-video"].status == "unsupported"
    assert reg["slat-3d"].status == "unsupported"
    assert "too long" in reg["lance-video"].reason
    assert "unknown model architecture" in reg["slat-3d"].reason


def test_missing_file_marks_unavailable(tmp_path):
    reg = load_registry(YAML, tmp_path)
    ready_ids = {"qwen3-4b", "bonsai-27b", "flux2-klein"}
    for mid in ready_ids:
        spec = reg[mid]
        assert spec.status == "unavailable"
        assert spec.reason
        for f in spec.files:
            assert f.name in spec.reason
    # unsupported is preserved
    assert reg["lance-video"].status == "unsupported"
    assert reg["slat-3d"].status == "unsupported"


def test_default_budget_is_70_percent_ram():
    expected = psutil.virtual_memory().total * 0.7 / 2**20
    assert abs(Settings.default().budget_mb - expected) / expected < 0.01
