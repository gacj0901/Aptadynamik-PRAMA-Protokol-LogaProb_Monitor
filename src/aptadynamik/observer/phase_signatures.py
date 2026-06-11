"""Experimental phase-signature helpers for PRAMA observer series.

Status: experimental / observer layer.

This module consumes PRAMA ProbLog output rows and returns diagnostic signatures
only. It does not modify PRAMA Protokol Core, regime classification, viability,
or thresholds.

Implemented scope:
- discontinuity detection
- passive critical slowing indicators: rolling variance, lag-1 autocorrelation,
  and recovery latency

Out of scope for this module:
- hysteresis inference
- structural target inference
- semantic judges
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

PRAMA_SIGNAL_FIELDS = (
    "delta_instant",
    "xi_norm",
    "viability_margin",
    "instant_viability_margin",
    "rigidity",
    "entropy_norm",
    "uncertainty",
)


def _finite_series(series: Iterable[float]) -> List[float]:
    values: List[float] = []
    for value in series:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
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


def extract_signal(rows: Sequence[Dict[str, Any]], field: str) -> List[float]:
    """Extract one finite numeric PRAMA signal from output rows."""
    return _finite_series(row.get(field) for row in rows)


def rolling_variance(series: Iterable[float], window: int = 6) -> List[float]:
    """Return rolling sample variance over finite values.

    The returned list starts at the first complete window. An empty list means
    there is not enough observed history for this passive indicator.
    """
    values = _finite_series(series)
    w = max(2, int(window))
    if len(values) < w:
        return []
    return [_variance(values[i : i + w]) for i in range(0, len(values) - w + 1)]


def rolling_lag1_autocorrelation(series: Iterable[float], window: int = 6) -> List[float]:
    """Return rolling lag-1 autocorrelation over finite values."""
    values = _finite_series(series)
    w = max(3, int(window))
    if len(values) < w:
        return []
    return [_lag1_autocorrelation(values[i : i + w]) for i in range(0, len(values) - w + 1)]


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
    a proxy for passive critical slowing. It is intentionally conservative and
    depends only on observed numeric trajectories.
    """
    values = _finite_series(series)
    w = max(3, int(window))
    if len(values) < w * 2:
        return 0.0
    early = values[:w]
    late = values[-w:]
    variance_gain = _variance(late) - _variance(early)
    ac_gain = _lag1_autocorrelation(late) - _lag1_autocorrelation(early)
    return max(0.0, variance_gain) + max(0.0, ac_gain)


def recovery_latency(
    rows: Sequence[Dict[str, Any]],
    margin_field: str = "instant_viability_margin",
    crossed_field: str = "instant_threshold_crossed",
) -> Optional[int]:
    """Return units from first local crossing to first later instant recovery.

    Recovery is observed when the instant margin becomes positive after a local
    crossing. If no local crossing or no later recovery exists, return None.
    """
    first_crossing: Optional[int] = None
    for idx, row in enumerate(rows):
        crossed = bool(row.get(crossed_field, False))
        margin = row.get(margin_field)
        try:
            margin_value = float(margin)
        except (TypeError, ValueError):
            margin_value = None
        if first_crossing is None and (crossed or (margin_value is not None and margin_value <= 0.0)):
            first_crossing = idx
            continue
        if first_crossing is not None and margin_value is not None and margin_value > 0.0:
            return idx - first_crossing
    return None


def signal_diagnostics(series: Iterable[float], window: int = 6) -> Dict[str, Any]:
    """Return diagnostic signatures for one numeric PRAMA signal."""
    values = _finite_series(series)
    rv = rolling_variance(values, window=window)
    ac = rolling_lag1_autocorrelation(values, window=window)
    return {
        "n": len(values),
        "discontinuity_score": discontinuity_score(values),
        "critical_slowing_score": critical_slowing_score(values, window=window),
        "rolling_variance": rv,
        "rolling_lag1_autocorrelation": ac,
        "last_rolling_variance": rv[-1] if rv else None,
        "last_lag1_autocorrelation": ac[-1] if ac else None,
    }


def prama_phase_diagnostics(
    rows: Sequence[Dict[str, Any]],
    fields: Sequence[str] = PRAMA_SIGNAL_FIELDS,
    window: int = 6,
) -> Dict[str, Any]:
    """Analyze PRAMA output rows and return observer-layer diagnostics only."""
    field_payload = {
        field: signal_diagnostics(extract_signal(rows, field), window=window)
        for field in fields
    }
    warning = transition_warning(
        extract_signal(rows, "instant_viability_margin") or extract_signal(rows, "viability_margin"),
        window=window,
    )
    return {
        "status": "experimental_observer_layer",
        "n_rows": len(rows),
        "fields": list(fields),
        "signals": field_payload,
        "recovery_latency": recovery_latency(rows),
        "transition_warning": warning,
        "scope": "diagnostic signatures only; no regime, viability, or threshold changes",
        "out_of_scope": ["hysteresis", "structural_target", "semantic_judges"],
    }


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
        "window": max(3, int(window)),
        "note": "Experimental observer signal; not hysteresis, structural-target, or semantic-judge analysis.",
    }
