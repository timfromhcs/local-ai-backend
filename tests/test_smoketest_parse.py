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
