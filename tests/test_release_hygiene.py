"""Release hygiene: one version, one source of truth, and a web demo that is shippable."""
import json
import re
from pathlib import Path

import localai_gateway

ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_pyproject():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf8")
    version = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)
    assert version == localai_gateway.__version__


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", localai_gateway.__version__)


def test_vercel_config_pins_other_preset():
    cfg = json.loads((ROOT / "web" / "vercel.json").read_text(encoding="utf8"))
    assert cfg["framework"] is None, "framework must be null, otherwise Vercel runs an npm build"
    assert cfg["outputDirectory"] == "."
    assert cfg["buildCommand"] == ""
    assert cfg["installCommand"] == ""


def test_web_demo_files_exist():
    for rel in ["web/index.html", "web/styles.css", "web/app.js", "web/i18n.js",
                "web/assets/logo.svg", "web/assets/hero.svg", "web/assets/diagram.svg"]:
        assert (ROOT / rel).is_file(), f"missing {rel}"


def test_workflows_exist_and_are_valid_yaml():
    yaml = __import__("yaml")
    for name in ["ci.yml", "release.yml"]:
        path = ROOT / ".github" / "workflows" / name
        assert path.is_file(), f"missing workflow {name}"
        data = yaml.safe_load(path.read_text(encoding="utf8"))
        assert "jobs" in data and data["jobs"], f"{name} has no jobs"


def test_no_model_weights_are_tracked():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf8")
    for pattern in ["*.gguf", "*.safetensors", "*.onnx", "bin/", "dist/"]:
        assert pattern in gitignore, f"gitignore must exclude {pattern}"
