"""Update check: semver parsing, comparison, and a check that fails soft without network."""
import json

from localai_gateway import __version__
from localai_gateway.update import check, is_newer, parse_version, render


def release(tag="v9.9.9", assets=("localai-gateway.exe", "sbom.cdx.json")):
    return json.dumps({
        "tag_name": tag,
        "html_url": f"https://github.com/timfromhcs/local-ai-backend/releases/tag/{tag}",
        "assets": [{"name": a} for a in assets],
    }).encode()


def test_parse_version_accepts_v_prefix_and_suffix():
    assert parse_version("v0.5.0") == (0, 5, 0)
    assert parse_version("0.5.0") == (0, 5, 0)
    assert parse_version("v1.2.3-beta.1") == (1, 2, 3)
    assert parse_version("nightly") is None


def test_is_newer_compares_numerically_not_lexically():
    assert is_newer("v0.10.0", "0.9.9") is True
    assert is_newer("v0.5.0", "0.5.0") is False
    assert is_newer("v0.4.9", "0.5.0") is False
    assert is_newer("garbage", "0.5.0") is False


def test_check_reports_an_update_with_assets():
    report = check("0.5.0", fetch=lambda url: release("v0.6.0"))
    assert report["status"] == "update_available"
    assert report["latest"] == "0.6.0"
    assert "localai-gateway.exe" in report["assets"]
    assert "0.5.0 -> 0.6.0" in render(report)


def test_check_reports_up_to_date():
    report = check(__version__, fetch=lambda url: release(f"v{__version__}"))
    assert report["status"] == "up_to_date"
    assert "up to date" in render(report)


def test_check_fails_soft_when_offline():
    def boom(url):
        raise OSError("no route to host")

    report = check("0.5.0", fetch=boom)
    assert report["status"] == "unavailable"
    assert "no route to host" in report["reason"]
    assert "never automatic" in report["hint"]
    assert "could not check" in render(report)


def test_check_handles_unparseable_tags_and_bad_json():
    def bad_tag(url):
        return json.dumps({"tag_name": "nightly"}).encode()

    def bad_json(url):
        return b"<html>rate limited</html>"

    assert check("0.5.0", fetch=bad_tag)["status"] == "unavailable"
    assert check("0.5.0", fetch=bad_json)["status"] == "unavailable"


def test_check_uses_the_public_release_endpoint():
    seen = {}

    def spy(url):
        seen["url"] = url
        return release()

    check("0.5.0", fetch=spy)
    assert seen["url"].startswith("https://api.github.com/repos/timfromhcs/local-ai-backend/releases/latest")
