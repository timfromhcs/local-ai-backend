"""A failed backend start must say *why*, not just the exit code.

Measured on this machine (2026-11-04, 27B planner): the gateway reported
"backend exited with code 1 before becoming healthy" while the child's own stderr said
`ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory`. The fix is diagnostic, not
cosmetic: without the reason the failure looks like a code bug instead of a VRAM limit.
"""
import sys

import pytest

from localai_gateway.supervisor import BackendStartError, Supervisor, _stderr_hint

def dying_child(message: str) -> list[str]:
    return [sys.executable, "-c",
            f"import sys; sys.stderr.write({message!r} + chr(10)); sys.stderr.flush(); sys.exit(1)"]


@pytest.mark.asyncio
async def test_a_dying_backend_reports_its_own_error_line():
    sup = Supervisor(health_timeout=10.0, poll_interval=0.05)
    argv = dying_child("ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory")
    with pytest.raises(BackendStartError) as exc:
        await sup._launch(argv, "http://127.0.0.1:19999/health")
    text = str(exc.value)
    assert "exited with code 1" in text
    assert "ErrorOutOfDeviceMemory" in text, text


@pytest.mark.asyncio
async def test_a_child_that_never_answers_includes_the_hint_too():
    sup = Supervisor(health_timeout=1.0, poll_interval=0.05, sleep=lambda _s: __import__("asyncio").sleep(0))
    argv = dying_child("llama_model_load: error loading model: unknown architecture")
    # the process exits immediately here as well, but the timeout path must also carry the hint
    with pytest.raises(BackendStartError) as exc:
        await sup._launch(argv, "http://127.0.0.1:19998/health")
    assert "unknown architecture" in str(exc.value)


def test_hint_prefers_an_error_line_over_a_banner():
    tail = ["I srv  llama_server: initializing ...", "E srv  load_model: failed to load model",
            "I srv  cleaning up before exit"]
    assert _stderr_hint(tail) == "E srv  load_model: failed to load model"


def test_hint_falls_back_to_the_last_line_and_tolerates_empty():
    assert _stderr_hint(["building", "ready"]) == "ready"
    assert _stderr_hint([]) == ""
    assert _stderr_hint(None) == ""
