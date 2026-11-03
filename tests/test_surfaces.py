"""Distribution surfaces: container, npm launcher, MCP server, workflows."""
import json
import re
from pathlib import Path

import yaml

import localai_gateway

ROOT = Path(__file__).resolve().parents[1]


def test_container_files_exist():
    assert (ROOT / "api" / "Dockerfile").is_file()
    assert (ROOT / "api" / "docker-compose.yml").is_file()
    assert (ROOT / ".dockerignore").is_file()


def test_container_stays_model_free():
    dockerfile = (ROOT / "api" / "Dockerfile").read_text(encoding="utf8")
    assert "gguf" not in dockerfile.lower(), "the image must not copy model weights"
    assert "USER gateway" in dockerfile, "must not run as root"
    assert "HEALTHCHECK" in dockerfile
    assert "0.0.0.0" in dockerfile


def test_dockerignore_excludes_weights_and_caches():
    text = (ROOT / ".dockerignore").read_text(encoding="utf8")
    for pattern in ["*.gguf", "*.onnx", "bin", ".venv", "web"]:
        assert pattern in text


def test_npm_launcher_version_matches_package():
    pkg = json.loads((ROOT / "packages" / "npm" / "package.json").read_text(encoding="utf8"))
    assert pkg["version"] == localai_gateway.__version__
    assert pkg["bin"]["localai-gateway"] == "bin/cli.js"
    for rel in ["bin/cli.js", "scripts/install.js"]:
        assert (ROOT / "packages" / "npm" / rel).is_file()


def test_npm_installer_verifies_checksums():
    src = (ROOT / "packages" / "npm" / "scripts" / "install.js").read_text(encoding="utf8")
    assert "SHA256SUMS.txt" in src
    assert "createHash(\"sha256\")" in src
    assert "LOCALAI_SKIP_DOWNLOAD" in src


def test_mcp_server_registers_expected_tools():
    src = (ROOT / "servers" / "localai-gateway-mcp" / "index.js").read_text(encoding="utf8")
    for tool in ["gateway_status", "list_models", "chat", "benchmark", "doctor"]:
        assert f'name: "{tool}"' in src, f"MCP tool missing: {tool}"
    assert "@modelcontextprotocol/sdk" in src


def test_all_workflows_parse():
    files = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    assert len(files) >= 5
    for path in files:
        data = yaml.safe_load(path.read_text(encoding="utf8"))
        assert data.get("jobs"), f"{path.name} has no jobs"


def test_release_workflow_publishes_every_surface():
    text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf8")
    assert "action-gh-release" in text
    docker = (ROOT / ".github" / "workflows" / "docker.yml").read_text(encoding="utf8")
    assert "ghcr.io" in docker
    pypi = (ROOT / ".github" / "workflows" / "publish-pypi.yml").read_text(encoding="utf8")
    assert "pypa/gh-action-pypi-publish" in pypi and "id-token: write" in pypi
    npm = (ROOT / ".github" / "workflows" / "publish-npm.yml").read_text(encoding="utf8")
    assert "npm publish" in npm


def test_version_is_semver_and_matches_pyproject():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf8")
    version = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)
    assert version == localai_gateway.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
