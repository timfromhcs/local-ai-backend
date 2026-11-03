"""--self-test contract: the EXE must be able to prove itself without a model or a port."""
from localai_gateway.selftest import CHECKS, render, run


def test_self_test_passes_with_an_empty_registry():
    report = run()
    assert report["ok"] is True, report
    assert len(report["checks"]) == len(CHECKS)
    for check in report["checks"]:
        assert check["status"] == check["expected"], check


def test_self_test_reports_counts_for_discovery_routes():
    checks = {c["path"]: c for c in run()["checks"]}
    assert "model(s)" in checks["/v1/models"]["detail"]
    assert "path(s)" in checks["/openapi.json"]["detail"]


def test_self_test_does_not_open_a_port(monkeypatch):
    """Guard: if someone later adds a uvicorn.run here, this test fails."""
    import uvicorn

    called = {"run": 0}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: called.__setitem__("run", called["run"] + 1))
    run()
    assert called["run"] == 0


def test_render_is_human_readable_and_fails_loudly():
    report = run()
    text = render(report)
    assert "self-test: PASS" in text
    assert text.count("ok  ") == len(CHECKS)

    broken = {"ok": False, "checks": [{"method": "GET", "path": "/x", "status": 500,
                                       "expected": 200, "ok": False, "detail": "boom"}]}
    assert "self-test: FAIL" in render(broken)
