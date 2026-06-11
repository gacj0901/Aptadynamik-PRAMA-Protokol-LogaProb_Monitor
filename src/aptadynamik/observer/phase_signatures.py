"""Experimental phase-signature helpers for PRAMA observer series.

This module is intentionally narrow. It provides only early scalar helpers for
phase-transition exploration over observed numeric trajectories:

- discontinuity_score
- critical_slowing_score
- transition_warning

It does not implement hysteresis or structural-target effects yet.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _finite_series(series: Iterable[float]) -> List[float]:
    values: List[float] = []
    for value in series:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if x == x and x not in (float("inf"), float("-inf")):
            values.append(x)
    return values


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)


def _lag1_autocorrelation(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    left = values[:-1]
    right = values[1:]
    ml = mean(left)
    mr = mean(right)
    num = sum((a - ml) * (b - mr) for a, b in zip(left, right))
    den_l = sum((a - ml) ** 2 for a in left)
    den_r = sum((b - mr) ** 2 for b in right)
    den = (den_l * den_r) ** 0.5
    if den <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, num / den))


def discontinuity_score(series: Iterable[float]) -> float:
    """Return the largest one-step jump relative to normal local movement.

    A score near 0 means no detectable jump. Values above 1 indicate that the
    largest jump exceeds the average absolute step. This is a descriptive score,
    not proof of a phase transition.
    """
    values = _finite_series(series)
    if len(values) < 2:
        return 0.0
    steps = [abs(b - a) for a, b in zip(values, values[1:])]
    if not steps:
        return 0.0
    avg_step = mean(steps)
    if avg_step <= 0.0:
        return 0.0
    return max(steps) / avg_step


def critical_slowing_score(series: Iterable[float], window: int = 6) -> float:
    """Estimate rising variance/autocorrelation in the tail of a series.

    The score compares early and late rolling windows. Positive values indicate
    that the tail has more variance and/or autocorrelation than the beginning,
    a proxy for critical slowing. It is intentionally conservative and depends
    only on observed numeric trajectories.
    """
    values = _finite_series(series)
    w = max(2, int(window))
    if len(values) < w * 2:
        return 0.0
    early = values[:w]
    late = values[-w:]
    variance_gain = _variance(late) - _variance(early)
    ac_gain = _lag1_autocorrelation(late) - _lag1_autocorrelation(early)
    return max(0.0, variance_gain) + max(0.0, ac_gain)


def transition_warning(
    series: Iterable[float],
    discontinuity_threshold: float = 2.5,
    slowing_threshold: float = 0.25,
    window: int = 6,
) -> Dict[str, Any]:
    """Return a compact warning payload from discontinuity and slowing scores."""
    d_score = discontinuity_score(series)
    c_score = critical_slowing_score(series, window=window)
    discontinuity = d_score >= float(discontinuity_threshold)
    critical_slowing = c_score >= float(slowing_threshold)
    return {
        "transition_warning": bool(discontinuity or critical_slowing),
        "discontinuity_score": d_score,
        "critical_slowing_score": c_score,
        "discontinuity_detected": discontinuity,
        "critical_slowing_detected": critical_slowing,
        "window": max(2, int(window)),
        "note": "Experimental observer signal; not hysteresis or structural-target analysis.",
    }
