"""Scorer contract: the benchmark must be deterministic and never call an LLM judge."""
import json

from bench.run import FAST, LONG
from localai_gateway.benchscore import (
    BenchResult,
    score_json,
    score_number,
    score_one_sentence,
    score_patch_shape,
    score_route,
    score_starts_no,
    score_tool_call,
)


def test_route_matching_is_case_insensitive_and_word_bounded():
    assert score_route("Duo.", "duo") == 1.0
    assert score_route("the answer is duopoly", "duo") == 0.0
    assert score_route("fast", "duo") == 0.0


def test_tool_call_needs_both_calls():
    assert score_tool_call("list_files('C:/work')\nread_file('package.json')") == 1.0
    assert score_tool_call("list_files('C:/work')") == 0.0


def test_json_scoring_accepts_fenced_output_and_rejects_wrong_values():
    assert score_json('```json\n{"model":"duo","loaded":true,"ram_mb":9000}\n```') == 1.0
    assert score_json('{"model":"duo","loaded":false}') == 0.0
    assert score_json("not json at all") == 0.0


def test_number_scoring_tolerates_thousand_separators():
    assert score_number("3,540 tokens", 3540) == 1.0
    assert score_number("about 3540", 3540) == 1.0
    assert score_number("100", 3540) == 0.0


def test_honesty_and_concision():
    assert score_starts_no("No. A browser cannot hold 27B weights.") == 1.0
    assert score_starts_no("Yes, it can.") == 0.0
    assert score_one_sentence("It stores attention keys and values so they are not recomputed.") == 1.0
    assert score_one_sentence("One. Two. Three. Four.") == 0.0


def test_patch_shape_requires_file_and_code():
    assert score_patch_shape("server/app.py\n```python\nasync def f(): ...\n```") == 1.0
    assert score_patch_shape("just rename it") == 0.0


def test_result_aggregates_weighted_score_and_categories():
    result = BenchResult(model="m", suite="fast", rows=[
        {"id": "a", "category": "routing", "weight": 1.0, "score": 1.0},
        {"id": "b", "category": "routing", "weight": 1.0, "score": 0.0},
        {"id": "c", "category": "format", "weight": 1.0, "score": 1.0},
    ])
    assert result.score == 0.667
    assert result.by_category() == {"routing": 0.5, "format": 1.0}
    assert result.as_dict()["score"] == 0.667


def test_suites_are_non_empty_and_have_unique_ids():
    for suite in (FAST, LONG):
        ids = [t.id for t in suite]
        assert ids and len(ids) == len(set(ids))
    assert len(LONG) > len(FAST)


def test_suite_is_json_serialisable():
    payload = {"suite": [{"id": t.id, "category": t.category, "weight": t.weight} for t in FAST]}
    assert json.loads(json.dumps(payload))["suite"][0]["id"] == "routing"
