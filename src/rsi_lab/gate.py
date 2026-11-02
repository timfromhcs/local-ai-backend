"""Promotion gate: keep a candidate only if it is measurably better and nothing important regresses."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Decision:
    promote: bool
    reasons: list[str] = field(default_factory=list)


def decide(candidate: dict[str, float], champion: dict[str, float] | None, primary: str, margin: float = 0.01,
           tolerance: float = 0.02, never_drop: list[str] | None = None) -> Decision:
    if champion is None:
        return Decision(True, [f"no champion yet: {primary}={candidate.get(primary)} becomes the baseline"])
    reasons: list[str] = []
    for name in champion:
        if name not in candidate:
            reasons.append(f"missing metric {name} in candidate")
    if reasons:
        return Decision(False, reasons)
    gain = candidate[primary] - champion[primary]
    if gain < margin:
        reasons.append(f"{primary} gain {gain:+.3f} below margin {margin}")
    for name, base in champion.items():
        if name == primary:
            continue
        drop = base - candidate[name]
        if name in (never_drop or []) and drop > 1e-9:
            reasons.append(f"{name} must not drop but fell {drop:.3f}")
        elif drop > tolerance:
            reasons.append(f"{name} regressed {drop:.3f} (> tolerance {tolerance})")
    if reasons:
        return Decision(False, reasons)
    return Decision(True, [f"{primary} improved {gain:+.3f}"])
