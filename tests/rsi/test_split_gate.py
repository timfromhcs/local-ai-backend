from rsi_lab.gate import decide
from rsi_lab.split import split_episodes


def eps(n):
    return [{"session_id": f"s{i}", "prompt": f"task {i}"} for i in range(n)]


def test_split_is_deterministic_and_disjoint():
    a1, h1 = split_episodes(eps(200), held_frac=0.2)
    a2, h2 = split_episodes(eps(200), held_frac=0.2)
    assert a1 == a2 and h1 == h2
    assert not ({e["session_id"] for e in a1} & {e["session_id"] for e in h1})
    assert 20 <= len(h1) <= 60 and len(a1) + len(h1) == 200


def test_split_stable_when_more_episodes_arrive():
    _, held_small = split_episodes(eps(100))
    _, held_big = split_episodes(eps(300))
    assert {e["session_id"] for e in held_small} <= {e["session_id"] for e in held_big}


def test_split_blacklist_never_in_train():
    bl = {"task 3", "task 4"}
    train, held = split_episodes(eps(50), blacklist_prompts=bl)
    assert all(e["prompt"] not in bl for e in train)


CH = {"bfcl": 0.50, "evalplus": 0.40, "toolcall_json": 0.90}


def test_gate_promotes_clear_improvement():
    d = decide({"bfcl": 0.53, "evalplus": 0.40, "toolcall_json": 0.91}, CH, primary="bfcl")
    assert d.promote and any("bfcl" in r for r in d.reasons)


def test_gate_rejects_below_margin():
    d = decide({"bfcl": 0.505, "evalplus": 0.40, "toolcall_json": 0.90}, CH, primary="bfcl", margin=0.01)
    assert not d.promote


def test_gate_rejects_regression_in_other_category():
    d = decide({"bfcl": 0.60, "evalplus": 0.30, "toolcall_json": 0.90}, CH, primary="bfcl", tolerance=0.02)
    assert not d.promote and any("evalplus" in r for r in d.reasons)


def test_gate_rejects_toolcall_validity_drop_even_within_tolerance():
    d = decide({"bfcl": 0.60, "evalplus": 0.40, "toolcall_json": 0.89}, CH, primary="bfcl", tolerance=0.05,
               never_drop=["toolcall_json"])
    assert not d.promote


def test_gate_missing_metric_is_rejection():
    d = decide({"bfcl": 0.60}, CH, primary="bfcl")
    assert not d.promote and any("missing" in r for r in d.reasons)


def test_gate_first_champion_promotes_when_none_exists():
    assert decide({"bfcl": 0.1}, None, primary="bfcl").promote
