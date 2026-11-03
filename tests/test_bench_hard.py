"""Hard suite: tasks the 4B demonstrably fails, where the planner/verifier can earn its cost."""
import pytest

from localai_gateway.benchscore import HARD_TASKS, score_number, score_word


def test_word_scorer_requires_exactly_one_word():
    score = score_word("bigout")
    assert score("bigout") == 1.0
    assert score("  BigOut.  ") == 1.0
    assert score("bigout, because three services") == 0.0
    assert score("fast") == 0.0


def test_multi_step_task_needs_the_number_not_the_working():
    task = next(t for t in HARD_TASKS if t.id == "hard-multi-step")
    assert task.score("3363") == 1.0
    assert task.score("3363 tokens") == 1.0
    # the 4B explained the arithmetic instead; that must not pass
    assert task.score("First convert 10 minutes to 600 seconds, then multiply by 5.9") == 0.0
    assert task.score("3540") == 0.0, "ignoring the 30 s startup must fail"


def test_format_task_is_exact():
    task = next(t for t in HARD_TASKS if t.id == "hard-format")
    assert task.score("STATUS=OK;MODE=duo;TOKENS=3363") == 1.0
    assert task.score("STATUS=OK;MODE=duo;TOKENS=3363.") == 1.0
    assert task.score("STATUS=OK; MODE=duo; TOKENS=3363") == 0.0


def test_every_hard_task_is_scored_and_unique():
    ids = [t.id for t in HARD_TASKS]
    assert len(ids) == len(set(ids))
    for task in HARD_TASKS:
        assert task.category in {"routing", "reasoning", "format"}
        assert task.weight > 0
        assert callable(task.score)


def test_number_scorer_tolerates_separators():
    assert score_number("3,363", 3363) == 1.0
    assert score_number("about 3363", 3363) == 1.0
