"""Where does an installed gateway write its cache?

Regression for the container smoke test: a pip-installed wheel used to resolve the
project root to `.../site-packages`, so creating `cache/slots` failed with
PermissionError on /usr/local/lib/python3.13/cache.
"""
from pathlib import Path

from localai_gateway.config import _pick_root


def test_localai_home_wins(tmp_path):
    here = tmp_path / "site-packages" / "localai_gateway" / "config.py"
    here.parent.mkdir(parents=True)
    here.touch()
    assert _pick_root(here, tmp_path / "cwd", str(tmp_path / "home")) == tmp_path / "home"


def test_source_checkout_is_detected_by_models_yaml(tmp_path):
    root = tmp_path / "repo"
    here = root / "src" / "localai_gateway" / "config.py"
    here.parent.mkdir(parents=True)
    here.touch()
    (root / "models.yaml").write_text("models: []", encoding="utf8")
    assert _pick_root(here, tmp_path / "cwd", None) == root


def test_installed_wheel_falls_back_to_cwd(tmp_path):
    site = tmp_path / "venv" / "lib" / "python3.13" / "site-packages"
    here = site / "localai_gateway" / "config.py"
    here.parent.mkdir(parents=True)
    here.touch()
    cwd = tmp_path / "app"
    cwd.mkdir()
    assert _pick_root(here, cwd, None) == cwd
    assert not (site / "models.yaml").exists()
