from __future__ import annotations

import math
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence


EPSILON = 1e-12


def _as_floats(values: Sequence[float]) -> List[float]:
    return [float(value) for value in values]


def _local_lambda(mu_window: Sequence[float]) -> float:
    values = _as_floats(mu_window)
    if len(values) < 3:
        return 0.0
    xs = values[:-1]
    ys = values[1:]
    mx = mean(xs)
    my = mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= EPSILON:
        # A flat displaced series is the stuck-system failure mode. Treat it as
        # near-unit-root so restitution capacity collapses instead of looking healthy.
        if max(values) - min(values) <= 1e-6 and abs(mean(values)) > 1e-6:
            return 0.999
        return 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return max(-0.999, min(0.999, slope))


def restitution_passive(mu_window: Sequence[float], mu_star: float = 0.0) -> float:
    centered = [float(value) - float(mu_star) for value in mu_window]
    lambda_local = _local_lambda(centered)
    return max(0.0, min(1.0, 1.0 - abs(lambda_local)))


def paramend_original(mu_series: Sequence[float], eps: float = 1e-4, C: float = 1e4) -> List[float]:
    values = _as_floats(mu_series)
    fatigue: List[float] = []
    previous: Optional[float] = None
    for value in values:
        if previous is None:
            delta = eps
        else:
            delta = max(abs(value - previous), eps)
        fatigue.append(min(abs(value) / delta, C))
        previous = value
    return fatigue


def compute_restitution_series(mu_series: Sequence[float], win: int = 12) -> List[float]:
    values = _as_floats(mu_series)
    series: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - win + 1)
        series.append(restitution_passive(values[start : idx + 1]))
    return series


def paramend_rayleigh(
    mu_series: Sequence[float],
    mu_star: float = 0.0,
    win: int = 12,
    eps: float = 1e-3,
) -> List[float]:
    values = _as_floats(mu_series)
    fatigue: List[float] = []
    for idx, value in enumerate(values):
        start = max(0, idx - win + 1)
        g = restitution_passive(values[start : idx + 1], mu_star=mu_star)
        fatigue.append(abs(value - mu_star) / max(g, eps))
    return fatigue


def _summary_value(summary: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in summary and summary[key] is not None:
            return float(summary[key])
    return default


def estimate_mu_from_turn(turn: Dict[str, Any]) -> float:
    """Estimate mu from a PRAMA Monitor turn summary.

    This is a first-pass proxy when no full-session normalization context is
    available:

    mu = avg_uncertainty + entropy_std + entropy_range_normalized - avg_rigidity

    The preferred session-level path is estimate_mu_series_from_turns(), which
    min-max normalizes viability across the full turn series and uses
    mu = 1 - normalized_viability.
    """
    summary = turn.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("turn missing required summary mapping for mu estimation")
    avg_uncertainty = _summary_value(summary, "avg_uncertainty")
    avg_rigidity = _summary_value(summary, "avg_rigidity")
    entropy_std = _summary_value(summary, "max_entropy_std", "entropy_std")
    entropy_range = _summary_value(summary, "max_entropy_range", "entropy_range")
    entropy_range_normalized = max(0.0, min(1.0, entropy_range))
    return avg_uncertainty + entropy_std + entropy_range_normalized - avg_rigidity


def _turn_viability(turn: Dict[str, Any]) -> float:
    summary = turn.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("turn missing required summary mapping for viability normalization")
    return _summary_value(summary, "avg_rigidity") - _summary_value(summary, "avg_uncertainty")


def estimate_mu_series_from_turns(turns: Sequence[Dict[str, Any]]) -> List[float]:
    if not turns:
        return []
    viabilities = [_turn_viability(turn) for turn in turns]
    low = min(viabilities)
    high = max(viabilities)
    if high - low <= EPSILON:
        return [estimate_mu_from_turn(turn) for turn in turns]
    return [1.0 - ((value - low) / (high - low)) for value in viabilities]


def estimate_mu_star_from_baseline(
    turns: Sequence[Dict[str, Any]],
    baseline_regimes: Optional[Sequence[str]] = None,
) -> float:
    if not turns:
        return 0.0
    regimes = set(baseline_regimes or ("R0", "R1", "baseline", "control", "control_neutral"))
    mu_series = estimate_mu_series_from_turns(turns)
    baseline_indices: List[int] = []
    for idx, turn in enumerate(turns):
        labels = [
            turn.get("regime"),
            turn.get("final_regime"),
            turn.get("condition"),
            turn.get("perturbation_type"),
        ]
        summary = turn.get("summary")
        if isinstance(summary, dict):
            labels.extend([summary.get("regime"), summary.get("final_regime")])
        if any(str(label) in regimes for label in labels if label is not None):
            baseline_indices.append(idx)

    if not baseline_indices:
        cutoff = max(1, math.ceil(len(turns) * 0.20))
        baseline_indices = list(range(cutoff))
    return mean(mu_series[idx] for idx in baseline_indices)


def detect_early_warning(mu_series: Sequence[float], g_threshold: float = 0.3) -> Dict[str, Any]:
    values = _as_floats(mu_series)
    if len(values) < 2:
        return {"detected": False, "early_warning_turn": -1, "half_displacement_turn": -1, "min_restitution_g": None}
    mu_star = mean(values[: max(1, math.ceil(len(values) * 0.20))])
    displacement = [abs(value - mu_star) for value in values]
    max_displacement = max(displacement)
    if max_displacement < 0.15:
        g_series = compute_restitution_series(values)
        return {
            "detected": False,
            "early_warning_turn": -1,
            "half_displacement_turn": -1,
            "min_restitution_g": min(g_series) if g_series else None,
        }
    half_displacement = max_displacement * 0.5
    half_turn = next((idx for idx, value in enumerate(displacement) if value >= half_displacement), -1)
    g_series = compute_restitution_series(values)
    warning_turn = next((idx for idx, value in enumerate(g_series) if value < g_threshold), -1)
    return {
        "detected": bool(warning_turn >= 0 and half_turn >= 0 and warning_turn < half_turn),
        "early_warning_turn": warning_turn if warning_turn >= 0 and (half_turn < 0 or warning_turn < half_turn) else -1,
        "half_displacement_turn": half_turn,
        "min_restitution_g": min(g_series) if g_series else None,
    }


def rayleigh_summary(mu_series: Sequence[float], mu_star: Optional[float] = None, win: int = 12) -> Dict[str, Any]:
    values = _as_floats(mu_series)
    if not values:
        return {
            "paramend_rayleigh_median": None,
            "paramend_rayleigh_max": None,
            "min_restitution_g": None,
            "early_warning_turn": -1,
            "mu_star": None,
        }
    reference = mean(values[: max(1, math.ceil(len(values) * 0.20))]) if mu_star is None else float(mu_star)
    fatigue = paramend_rayleigh(values, mu_star=reference, win=win)
    g_series = compute_restitution_series(values, win=win)
    warning = detect_early_warning(values)
    return {
        "paramend_rayleigh_median": median(fatigue),
        "paramend_rayleigh_max": max(fatigue),
        "min_restitution_g": min(g_series) if g_series else None,
        "early_warning_turn": warning["early_warning_turn"],
        "mu_star": reference,
    }


def synthetic_paramend_signals(n: int = 60) -> Dict[str, List[float]]:
    healthy_quiet = [0.05 + 0.002 * math.sin(i / 3.0) for i in range(n)]
    dying_stuck = [0.85 + 0.0002 * math.sin(i / 5.0) for i in range(n)]
    transition_real: List[float] = []
    for i in range(n):
        if i < n // 3:
            transition_real.append(0.06 + 0.002 * math.sin(i))
        elif i < (2 * n) // 3:
            transition_real.append(0.06 + ((i - n // 3) / (n // 3)) * 0.55)
        else:
            transition_real.append(0.62 + 0.004 * math.sin(i / 4.0))
    noise = [0.12 + 0.08 * math.sin(i * 1.7) for i in range(n)]
    return {
        "healthy_quiet": healthy_quiet,
        "dying_stuck": dying_stuck,
        "transition_real": transition_real,
        "noise": noise,
    }


def synthetic_paramend_validation() -> Dict[str, Dict[str, Any]]:
    signals = synthetic_paramend_signals()
    result: Dict[str, Dict[str, Any]] = {}
    for name, series in signals.items():
        fatigue = paramend_rayleigh(series, mu_star=0.0)
        original = paramend_original(series)
        g_series = compute_restitution_series(series)
        warning = detect_early_warning(series)
        result[name] = {
            "rayleigh_median": median(fatigue),
            "rayleigh_max": max(fatigue),
            "original_median": median(original),
            "original_max": max(original),
            "min_restitution_g": min(g_series),
            "early_warning": warning,
        }
    return result
