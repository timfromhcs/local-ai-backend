"""Golden eval gate: the deterministic scorers must not drift silently.

Complements the unit tests (which check intent) with a versioned dataset that pins the concrete
outcome for representative answers. A change in a scorer has to be a deliberate edit of
bench/golden.json, which then shows up in review.
"""
import json
from pathlib import Path

import pytest

from localai_gateway.benchscore import (
    score_json,
    score_number,
    score_one_sentence,
    score_patch_shape,
    score_route,
    score_starts_no,
    score_tool_call,
)

GOLDEN = Path(__file__).resolve().parents[1] / "bench" / "golden.json"

SCORERS = {
    "route": lambda answer, arg: score_route(answer, arg),
    "tool_call": lambda answer, arg: score_tool_call(answer),
    "json": lambda answer, arg: score_json(answer),
    "number": lambda answer, arg: score_number(answer, arg),
    "starts_no": lambda answer, arg: score_starts_no(answer),
    "one_sentence": lambda answer, arg: score_one_sentence(answer),
    "patch_shape": lambda answer, arg: score_patch_shape(answer),
}


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text(encoding="utf8"))


def test_golden_file_is_well_formed(golden):
    assert golden["version"] >= 1
    assert len(golden["cases"]) == golden["baseline"]["case_count"]
    ids = [c["id"] for c in golden["cases"]]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for case in golden["cases"]:
        assert case["scorer"] in SCORERS, f"unknown scorer {case['scorer']}"
        assert "answer" in case and "expected" in case


def test_every_case_matches_its_pinned_score(golden):
    drift = []
    for case in golden["cases"]:
        got = SCORERS[case["scorer"]](case["answer"], case.get("arg"))
        if abs(got - case["expected"]) > 1e-9:
            drift.append((case["id"], case["expected"], got))
    assert not drift, f"scorer drift: {drift}"


def test_aggregate_matches_the_baseline(golden):
    total = sum(SCORERS[c["scorer"]](c["answer"], c.get("arg")) for c in golden["cases"])
    assert total == golden["baseline"]["sum_expected"]


def test_golden_covers_both_outcomes_per_scorer(golden):
    """A gate with only positive cases would pass while scoring everything 1.0."""
    by_scorer: dict[str, set[float]] = {}
    for case in golden["cases"]:
        by_scorer.setdefault(case["scorer"], set()).add(case["expected"])
    for name, outcomes in by_scorer.items():
        assert outcomes == {0.0, 1.0}, f"{name} needs a passing and a failing case"
