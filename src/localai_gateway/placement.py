"""SAT-based model placement.

The question: given a RAM budget and a registry of models, which models can be resident at the
same time? This is a small constraint problem, so we encode it as CNF and solve it with a
deterministic DPLL implementation (unit propagation + pure literals). No dependency, no network.

Honest scope: the instances here are tiny (â‰¤ 20 models). Kissat compiled to WASM is the documented
upgrade if the registry ever grows; the API below would not change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Sequence

from .registry import ModelSpec


class Unsatisfiable(Exception):
    """No placement satisfies the constraints (e.g. not even one model fits)."""


@dataclass(frozen=True)
class Placement:
    models: tuple[str, ...]
    memory_mb: int
    required_kinds: tuple[str, ...] = ("llm",)
    why: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ids(self) -> list[str]:
        return list(self.models)


# --------------------------------------------------------------------------- CNF + DPLL
CNF = list[list[int]]


def dpll(clauses: CNF, n_vars: int, max_decisions: int = 200_000) -> dict[int, bool] | None:
    """Return a satisfying assignment or None. Deterministic: variables in ascending order."""
    decisions = 0

    def propagate(assign: dict[int, bool], clauses: CNF) -> dict[int, bool] | None:
        while True:
            changed = False
            for clause in clauses:
                unassigned, satisfied = [], False
                for lit in clause:
                    var, val = abs(lit), assign.get(abs(lit))
                    if val is None:
                        unassigned.append(lit)
                    elif val == (lit > 0):
                        satisfied = True
                        break
                if satisfied:
                    continue
                if not unassigned:
                    return None  # conflict
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    assign[abs(lit)] = lit > 0
                    changed = True
            if not changed:
                return assign

    def pure_literals(assign: dict[int, bool], clauses: CNF) -> dict[int, bool]:
        polarity: dict[int, set[bool]] = {}
        for clause in clauses:
            if any(assign.get(abs(l)) == (l > 0) for l in clause if abs(l) in assign):
                continue
            for lit in clause:
                if abs(lit) in assign:
                    continue
                polarity.setdefault(abs(lit), set()).add(lit > 0)
        for var, pol in polarity.items():
            if len(pol) == 1:
                assign[var] = pol.pop()
        return assign

    def search(assign: dict[int, bool]) -> dict[int, bool] | None:
        nonlocal decisions
        a = propagate(dict(assign), clauses)
        if a is None:
            return None
        a = pure_literals(a, clauses)
        a = propagate(a, clauses)
        if a is None:
            return None
        if all(any(a.get(abs(l)) == (l > 0) for l in c if abs(l) in a) for c in clauses):
            if all(abs(l) in a for c in clauses for l in c):
                return a
        if decisions >= max_decisions:
            return None
        free = next((v for v in range(1, n_vars + 1) if v not in a), None)
        if free is None:
            return a
        for value in (True, False):
            decisions += 1
            branch = dict(a)
            branch[free] = value
            got = search(branch)
            if got is not None:
                return got
        return None

    return search({})


def at_most_k(literals: Sequence[int], k: int, counter: list[int]) -> CNF:
    """Sinz sequential-counter encoding of sum(literals) <= k."""
    n = len(literals)
    if k >= n:
        return []          # trivially true
    if k < 0:
        return [[1], [-1]]  # trivially false
    if k == 0:
        return [[-lit] for lit in literals]  # every literal must be false

    s: dict[tuple[int, int], int] = {}
    for i in range(1, n):
        for j in range(1, min(i, k) + 1):
            s[(i, j)] = counter[0]
            counter[0] += 1

    clauses: CNF = []
    for i, lit in enumerate(literals, start=1):
        if i < n:
            clauses.append([-lit, s[(i, 1)]])
            for j in range(2, min(i, k) + 1):
                clauses.append([-lit, -s[(i - 1, j - 1)], s[(i, j)]])
                if (i - 1, j) in s:
                    clauses.append([-s[(i - 1, j)], s[(i, j)]])
        if 2 <= i <= n - 1 and (i - 1, 1) in s:
            clauses.append([-s[(i - 1, 1)], s[(i, 1)]])
        if i >= 2 and (i - 1, k) in s:
            clauses.append([-lit, -s[(i - 1, k)]])  # would be the (k+1)-th true literal
    return [c for c in clauses if c]


def at_least_k(vars_: Sequence[int], k: int, counter: list[int]) -> CNF:
    """sum(vars) >= k  ==  sum(-vars) <= n-k, which is the direction Sinz encodes exactly."""
    n = len(vars_)
    if k <= 0:
        return []
    if k > n:
        return [[1], [-1]]
    return at_most_k([-v for v in vars_], n - k, counter)


# --------------------------------------------------------------------------- placement
def _candidate_models(
    registry: dict[str, ModelSpec], required_kinds: Iterable[str], pinned: Iterable[str] = ()
) -> list[ModelSpec]:
    kinds, pinned = set(required_kinds), set(pinned)
    out = [
        m for m in registry.values()
        if m.status == "ready" and (m.kind in kinds or m.id in pinned)
    ]
    return sorted(out, key=lambda m: (-m.est_mem_mb, m.id))


def _memory_clauses(models: list[ModelSpec], budget_mb: int, index: dict[str, int]) -> CNF:
    """Models that exceed the budget on their own, plus every minimal over-budget combination."""
    clauses: CNF = [[-index[m.id]] for m in models if m.est_mem_mb > budget_mb]
    for size in range(2, len(models) + 1):
        for combo in combinations(models, size):
            if sum(m.est_mem_mb for m in combo) <= budget_mb:
                continue
            if any(sum(m.est_mem_mb for m in smaller) > budget_mb
                   for smaller in combinations(combo, size - 1)):
                continue  # not minimal: already covered
            clauses.append([-index[m.id] for m in combo])
    return clauses


def solve_placement(
    registry: dict[str, ModelSpec],
    budget_mb: int,
    required_kinds: Sequence[str] = ("llm",),
    pinned: Sequence[str] = (),
    max_models: int | None = None,
) -> Placement:
    """Find the largest set of ready models of the required kinds that fits the budget."""
    candidates = _candidate_models(registry, required_kinds, pinned)
    if not candidates:
        raise Unsatisfiable(f"no ready model of kind(s) {list(required_kinds)}")

    for name in pinned:
        spec = registry.get(name)
        if spec is None:
            raise Unsatisfiable(f"pinned model {name!r} is not registered")
        if spec.status != "ready":
            raise Unsatisfiable(f"pinned model {name!r} is {spec.status}: {spec.reason}")
        if spec.est_mem_mb > budget_mb:
            raise Unsatisfiable(f"pinned model {name!r} needs {spec.est_mem_mb} MB > budget {budget_mb} MB")

    index = {m.id: i + 1 for i, m in enumerate(candidates)}
    base: CNF = _memory_clauses(candidates, budget_mb, index)

    for kind in required_kinds:
        clause = [index[m.id] for m in candidates if m.kind == kind]
        if not clause:
            raise Unsatisfiable(f"no ready model of kind {kind!r} fits")
        base.append(clause)

    for name in pinned:
        if name in index:
            base.append([index[name]])

    upper = min(len(candidates), max_models or len(candidates))
    counter = [len(candidates) + 1]
    best: dict[int, bool] | None = None
    best_k = 0
    for k in range(upper, 0, -1):
        counter[0] = len(candidates) + 1
        extra = at_least_k([index[m.id] for m in candidates], k, counter)
        model = dpll(base + extra, counter[0])
        if model is not None:
            best, best_k = model, k
            break

    if best is None:
        raise Unsatisfiable(f"no placement fits {budget_mb} MB")

    # Tie-break: among placements with the maximum number of models, use as much of the
    # budget as possible. Greedy forcing in descending memory order keeps it deterministic.
    exact = at_least_k([index[m.id] for m in candidates], best_k, counter)
    exact += at_most_k([index[m.id] for m in candidates], best_k, counter)
    max_var = counter[0] - 1
    forced: CNF = []
    for candidate in sorted(candidates, key=lambda m: (-m.est_mem_mb, m.id)):
        trial = forced + [[index[candidate.id]]]
        got = dpll(base + exact + trial, max_var)
        if got is not None:
            forced = trial
            best = got

    chosen = [m for m in candidates if best.get(index[m.id], False)]
    why = tuple(
        [f"{m.id}: {m.est_mem_mb} MB ({m.kind}, ready)" for m in chosen]
        + [f"budget: {sum(m.est_mem_mb for m in chosen)} / {budget_mb} MB used"]
        + ([f"pinned: {', '.join(pinned)}"] if pinned else [])
    )
    return Placement(tuple(m.id for m in chosen), sum(m.est_mem_mb for m in chosen), tuple(required_kinds), why)

