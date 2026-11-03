"""Supply-chain contract: SBOM shape, clean secret scan, no unpinned release assets."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sbom  # noqa: E402


def test_sbom_is_valid_cyclonedx():
    data = sbom.build()
    assert data["bomFormat"] == "CycloneDX"
    assert data["specVersion"] == "1.5"
    assert data["serialNumber"].startswith("urn:uuid:")
    root = data["metadata"]["component"]
    assert root["name"] == "localai-gateway"
    assert root["version"] == __import__("localai_gateway").__version__
    assert len(data["components"]) >= 10, "an SBOM without dependencies is decoration"
    purls = {c["purl"] for c in data["components"]}
    assert any("pkg:pypi/fastapi@" in p for p in purls)
    for comp in data["components"]:
        assert comp["type"] == "library"
        assert comp["name"] and comp["version"] and comp["purl"]


def test_sbom_is_json_serialisable_and_sorted():
    data = sbom.build()
    assert json.loads(json.dumps(data))["components"] == data["components"]
    names = [c["name"].lower() for c in data["components"]]
    assert names == sorted(names)


def test_repository_secret_scan_is_clean():
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "scan_secrets.py")],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_secret_scanner_detects_a_real_token(tmp_path):
    """The scanner must fail on a token, not just pass on a clean tree."""
    (tmp_path / "leak.txt").write_text("token ghp_" + "A" * 36 + "\n", encoding="utf8")
    sys.path.insert(0, str(ROOT / "scripts"))
    import scan_secrets

    findings = scan_secrets.scan(tmp_path)
    assert findings and findings[0]["kind"] == "github-token"


def test_allow_marker_keeps_test_fixtures_usable(tmp_path):
    (tmp_path / "fixture.txt").write_text("token ghp_" + "A" * 36 + "  # secret-scan: allow\n", encoding="utf8")
    import scan_secrets

    assert scan_secrets.scan(tmp_path) == []


def test_release_workflow_publishes_an_sbom():
    text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf8")
    assert "sbom" in text.lower()
    assert "SHA256SUMS" in text


def test_attestation_workflow_is_separate_and_least_privilege():
    path = ROOT / ".github" / "workflows" / "attest.yml"
    assert path.is_file(), "attestation must not live inside release.yml (a broken action would block releases)"
    text = path.read_text(encoding="utf8")
    assert "attest-build-provenance" in text or "actions/attest" in text
    assert "id-token: write" in text and "attestations: write" in text
    assert "contents: read" in text
