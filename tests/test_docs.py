"""Release documentation contract: the README must stay honest and navigable."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf8")


def test_readme_has_the_required_sections():
    for heading in ["## What this is", "## Quick start", "## Surfaces", "## Measured, not estimated",
                    "## Architecture", "## The web demo", "## Benchmark suite", "## MCP stack",
                    "## Troubleshooting", "## Honest limits", "## Licence"]:
        assert heading in README, f"README missing {heading}"


def test_readme_links_security_and_contributing():
    assert "[SECURITY.md](SECURITY.md)" in README
    assert "[CONTRIBUTING.md](CONTRIBUTING.md)" in README
    for name in ["SECURITY.md", "CONTRIBUTING.md"]:
        assert (ROOT / name).is_file(), f"{name} missing"


def test_readme_has_no_todo_or_placeholder():
    assert "TODO" not in README
    assert "TBD" not in README
    assert "lorem ipsum" not in README.lower()


def test_readme_verify_block_is_real():
    """The verify commands must be copy-pasteable and include the attestation step."""
    block = re.search(r"```powershell\n(.*?)```", README, re.S)
    assert block, "no powershell block found"
    text = block.group(1)
    assert "localai-gateway.exe" in text
    assert "attestation verify" in README


def test_security_policy_states_what_is_out_of_scope():
    policy = (ROOT / "SECURITY.md").read_text(encoding="utf8")
    assert "Out of scope" in policy and "No authentication" in policy
    assert "7 days" in policy, "a policy without a response window is decoration"


def test_contributing_lists_the_real_gates():
    doc = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf8")
    for gate in ["pytest -m", "check-web.mjs", "audit-web.mjs", "scan_secrets.py", "--self-test"]:
        assert gate in doc, f"CONTRIBUTING missing gate: {gate}"
