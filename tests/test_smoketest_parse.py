import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.smoketest import parse_tps  # noqa: E402

LOG = """\
prompt eval time =      85.12 ms /     9 tokens (    9.46 ms per token,   105.73 tokens per second)
       eval time =    1623.40 ms /    20 runs   (   81.17 ms per token,    12.34 tokens per second)
      total time =    1708.52 ms /    29 tokens
"""


def test_parse_llama_timings():
    assert parse_tps(LOG) == 12.34


def test_parse_tps_none_when_absent():
    assert parse_tps("nothing here") is None


import base64  # noqa: E402

from tools.smoketest import first_error_line, validate_png  # noqa: E402

PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAfbLI3wAAAABJRU5ErkJggg=="
)


def test_validate_png_valid():
    assert validate_png(PNG_1X1) == (1, 1)


def test_validate_png_garbage_and_empty():
    assert validate_png("not base64 !!") is None
    assert validate_png(base64.b64encode(b"hello world, not a png at all, definitely").decode()) is None
    assert validate_png("") is None


SD_LOG = """\
[I] Version: Flux.2 klein  --- diffusion_engine.cpp:975
[E] Conditioner model tensor 'text_encoders.llm.model.norm.weight' not in model metadata --- model_manager.cpp:765
[E] model metadata validation failed --- diffusion_engine.cpp:1366
[E] new_sd_ctx_t failed --- main.cpp:105
"""

LLAMA_LOG = """\
0.00.115 E llama_model_load: error loading model: unknown model architecture: 'flux'
0.00.115 E llama_model_load_from_file_impl: failed to load model
0.00.115 E srv    load_model: failed to load model, 'x.gguf'
0.00.117 E srv  server_run_i: exiting due to model loading error
"""


def test_first_error_line_skips_generic_sd_wrapper():
    assert "not in model metadata" in first_error_line(SD_LOG)


def test_first_error_line_llama_root_cause():
    assert "unknown model architecture: 'flux'" in first_error_line(LLAMA_LOG)
