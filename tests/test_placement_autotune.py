"""SAT placement and device autotune contracts."""
import random
from itertools import combinations

import pytest

from localai_gateway.hardware import Device, DeviceClass, detect_device, profile_for
from localai_gateway.placement import Placement, Unsatisfiable, at_least_k, dpll, solve_placement
from localai_gateway.registry import ModelSpec


def spec(id_, mb, kind="llm", status="ready", reason=None):
    return ModelSpec(id=id_, kind=kind, files=[], runtime="llama", est_mem_mb=mb, status=status, reason=reason)


# ------------------------------------------------------------------ SAT core
def test_dpll_satisfies_simple_cnf():
    assignment = dpll([[1, 2], [-1, 2]], 2)
    assert assignment is not None
    assert assignment[2] is True


def test_dpll_detects_unsatisfiable():
    assert dpll([[1], [-1]], 1) is None


def test_at_least_k_matches_bruteforce():
    """The cardinality encoding is the risky part; brute force every assignment for n<=4."""
    for n in range(1, 5):
        vars_ = list(range(1, n + 1))
        for k in range(0, n + 2):
            for bits in range(2 ** n):
                chosen = {v for v in vars_ if bits & (1 << (v - 1))}
                counter = [n + 1]
                clauses = at_least_k(vars_, k, counter)
                # fix the variables to this assignment and ask for satisfiability
                fixed = [[v] if v in chosen else [-v] for v in vars_]
                got = dpll(clauses + fixed, counter[0] - 1) is not None
                expected = len(chosen) >= k
                assert got == expected, (n, k, bits, clauses)


# ------------------------------------------------------------------ placement
def test_placement_picks_the_largest_fitting_set():
    registry = {
        "big": spec("big", 9000),
        "mid": spec("mid", 3600),
        "small": spec("small", 800),
    }
    p = solve_placement(registry, budget_mb=5000)
    assert isinstance(p, Placement)
    assert set(p.models) == {"mid", "small"}
    assert p.memory_mb == 4400
    assert any("budget" in line for line in p.why)


def test_placement_never_exceeds_budget_on_random_instances():
    rng = random.Random(7)
    for _ in range(60):
        n = rng.randint(1, 7)
        registry = {f"m{i}": spec(f"m{i}", rng.choice([300, 800, 1500, 3600, 9000])) for i in range(n)}
        budget = rng.choice([500, 2000, 5000, 12000])
        try:
            p = solve_placement(registry, budget_mb=budget)
        except Unsatisfiable:
            assert all(m.est_mem_mb > budget for m in registry.values())
            continue
        assert p.memory_mb <= budget
        assert p.models, "an empty placement must raise instead of returning nothing"


def test_placement_matches_bruteforce_optimum_on_small_sets():
    """Objective: maximum number of models, then maximum memory used (documented in the docstring)."""
    rng = random.Random(11)
    for _ in range(40):
        n = rng.randint(2, 8)
        sizes = [rng.choice([300, 900, 1500, 3600]) for _ in range(n)]
        registry = {f"m{i}": spec(f"m{i}", sizes[i]) for i in range(n)}
        budget = rng.choice([1200, 2500, 6000])

        feasible = [
            (len(combo), sum(sizes[i] for i in combo))
            for r in range(1, n + 1)
            for combo in combinations(range(n), r)
            if sum(sizes[i] for i in combo) <= budget
        ]
        if not feasible:
            with pytest.raises(Unsatisfiable):
                solve_placement(registry, budget_mb=budget)
            continue
        best = max(feasible)  # (count, memory)
        p = solve_placement(registry, budget_mb=budget)
        assert (len(p.models), p.memory_mb) == best


def test_placement_honours_pinned_and_required_kinds():
    registry = {
        "llm": spec("llm", 1000),
        "img": spec("img", 1000, kind="image"),
        "other": spec("other", 5000),
    }
    p = solve_placement(registry, budget_mb=2500, required_kinds=("llm",), pinned=("img",))
    assert set(p.models) == {"llm", "img"}


def test_placement_rejects_unready_pinned_model():
    registry = {"x": spec("x", 1000, status="unavailable", reason="missing file")}
    with pytest.raises(Unsatisfiable):
        solve_placement(registry, budget_mb=5000, pinned=("x",))


def test_placement_raises_when_nothing_fits():
    registry = {"big": spec("big", 9000)}
    with pytest.raises(Unsatisfiable):
        solve_placement(registry, budget_mb=100)


def test_placement_ignores_unsupported_models():
    registry = {
        "ok": spec("ok", 500),
        "bad": spec("bad", 500, status="unsupported", reason="loader error"),
    }
    p = solve_placement(registry, budget_mb=2000)
    assert p.models == ("ok",)


# ------------------------------------------------------------------ hardware
def device(cls, vram=0, ram=24000, cores=8):
    return Device("test-gpu" if cls is not DeviceClass.CPU else "cpu", cls, vram, ram, cores, "Windows")


def test_igpu_full_offload_when_it_fits():
    p = profile_for(device(DeviceClass.IGPU_UMA, vram=4096, ram=24000), 3600)
    assert "-ngl 99" in p.args
    assert "-t 8" in p.args
    assert "-ub 256" in p.args
    assert any("shared RAM" in w for w in p.why)


def test_igpu_partial_offload_when_tight():
    p = profile_for(device(DeviceClass.IGPU_UMA, ram=16000), 14000)
    assert "-ngl 20" in p.args
    assert "--no-mmap" in p.args  # 16000*0.7=11200 < 14000*1.35
    assert any("tight" in w for w in p.why)


def test_dgpu_offloads_only_what_fits_vram():
    p = profile_for(device(DeviceClass.DGPU, vram=8000, ram=32000), 9000)
    ngl = [a for a in p.args if a.startswith("-ngl")][0]
    assert ngl != "-ngl 99"
    assert 0 <= int(ngl.split()[1]) < 32


def test_dgpu_full_offload_when_model_fits():
    p = profile_for(device(DeviceClass.DGPU, vram=24000, ram=64000), 9000)
    assert "-ngl 99" in p.args


def test_cpu_profile_has_no_offload():
    p = profile_for(device(DeviceClass.CPU, ram=32000, cores=16), 9000)
    assert "-ngl 0" in p.args
    assert "-t 8" in p.args
    assert "-ub 128" in p.args


def test_profile_is_explainable_and_rejects_bad_input():
    p = profile_for(device(DeviceClass.IGPU_UMA), 3600)
    assert len(p.why) >= 2
    assert p.as_dict()["class"] == "igpu_uma"
    with pytest.raises(ValueError):
        profile_for(device(DeviceClass.IGPU_UMA), 0)


def test_detect_device_prefers_real_gpu_over_virtual_display():
    gpus = [
        {"Name": "Parsec Virtual Display Adapter", "AdapterRAM": 0},
        {"Name": "AMD Radeon(TM) Graphics", "AdapterRAM": 4293918720},
    ]
    d = detect_device(gpus)
    assert d.device_class is DeviceClass.IGPU_UMA
    assert d.ram_mb > 0


def test_detect_device_picks_discrete_gpu():
    gpus = [{"Name": "NVIDIA GeForce RTX 4070", "AdapterRAM": 12 * 1024**3}]
    d = detect_device(gpus)
    assert d.device_class is DeviceClass.DGPU
    assert d.vram_mb >= 12000


def test_detect_device_without_gpu_is_cpu():
    d = detect_device([])
    assert d.device_class is DeviceClass.CPU
    assert d.physical_cores >= 1
