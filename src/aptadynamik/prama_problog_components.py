from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence

from aptadynamik.prama_protokol_core import (
    DEFAULT_LAMBDA0,
    DEFAULT_MEMORY_BETA,
    DEFAULT_MIN_POST_CROSSING_UNITS,
    DEFAULT_MIN_TURNS_FOR_REGIME,
    DEFAULT_MIN_WINDOWS_FOR_REGIME,
    DEFAULT_THETA0,
    ORGANIZED_STABILITY_ASSESSMENT,
    ORGANIZED_STABILITY_REGIME,
    boundary_pressure_from_margin,
    clamp01,
    classify_boundary,
    classify_regime,
    collapse_xi_norm,
    normalize_regime_label,
    normalize_trajectory_assessment,
    pulsation_subtype_from_state,
    trajectory_assessment_from_regime,
    viability_status,
)


SUBSTRATE_BLIND_WARNING = (
    "This module is substrate-blind: it observes generative geometry from "
    "logprobs, not material cost. It does not measure energy use, thermal "
    "pressure, GPU load, memory pressure, latency, or physical friction."
)

COHERENCE_VIABILITY_NOTE = (
    "Coherence is a threshold. Viability is a gradient. Viability is measured "
    "here as dynamic acople under admissible variation."
)

ECHO_NOTE = (
    "Structural repetition is not full coherence. In the PRAMA framework, "
    "coherence is not mere regularity; it is structural persistence under flow."
)

@dataclass
class TurnReading:
    turn_index: int
    valid_token_count: int
    invalid_token_count: int
    logprob_valid: bool
    insufficient_data: bool
    micro_raw: Optional[float]
    micro_health: Optional[float]
    macro_health: Optional[float]
    activity_raw: Optional[float]
    activity_structural: float
    activity_effective: float
    activity: float
    acople: Optional[float]
    acople_effective: Optional[float]
    delta_instant: Optional[float]
    xi_accumulated: Optional[float]
    xi_norm: Optional[float]
    lambda_remaining: Optional[float]
    theta_dynamic: Optional[float]
    viability_margin: Optional[float]
    accumulated_viability_margin: Optional[float]
    instant_viability_margin: Optional[float]
    instant_threshold_crossed: bool
    instant_recovered: bool
    compression_gap: Optional[float]
    distance_to_threshold: Optional[float]
    xi_exceeds_theta: bool
    threshold_crossed: bool
    viability_status: str
    boundary_pressure: Optional[float]
    boundary_side: str
    micro_drop: Optional[float]
    micro_excess: Optional[float]
    # Compatibility aliases for older runner consumers.
    viability: Optional[float]
    delta: Optional[float]
    xi: Optional[float]
    lam: Optional[float]
    theta: Optional[float]


def is_valid_logprob(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) <= 0.0 and abs(float(value)) < 100.0


def token_surprise(logprob: float) -> float:
    return -float(logprob)


def token_surprise_series(token_logprobs: Sequence[float]) -> List[float]:
    return [token_surprise(logprob) for logprob in token_logprobs]


def turn_micro_from_surprises(surprises: Sequence[float]) -> float:
    if not surprises:
        return 0.0
    return max(surprises) - min(surprises)


def turn_mean_surprise(surprises: Sequence[float]) -> float:
    if not surprises:
        return 0.0
    return mean(surprises)


def centered_health(micro_raw: float, baseline_micro: float) -> float:
    """Return health centered on the frozen viable micro-amplitude baseline.

    Values below and above the baseline are both penalized; the maximum is near
    the baseline. This measures generative geometry, not semantic quality.

    Known limitation (see PATCH_NOTES): this linear form hard-zeros at
    micro_raw >= 2 * baseline, saturating delta_instant at its maximum and
    destroying gradation before the dynamic core. It is also asymmetric in
    ratio space (2x baseline -> 0.0 while 0.5x baseline -> 0.5). Retained as
    the legacy default for reproducibility of v0.2.x published results.
    """
    baseline = max(float(baseline_micro), 1e-9)
    deviation = abs(float(micro_raw) - baseline) / baseline
    return clamp01(1.0 - deviation)


def log_ratio_health(micro_raw: float, baseline_micro: float, scale: float = 1.0) -> float:
    """Multiplicatively symmetric health on the baseline-relative amplitude ratio.

        health = exp(-|ln(micro_raw / baseline)| / scale)

    Properties (contrast with centered_health):
    - symmetric in ratio space: health(2*b) == health(b/2). Amplitude is a
      multiplicative magnitude; equal ratios deserve equal penalty.
    - never hard-zeros for finite positive amplitude, so delta_instant keeps
      gradation and the dynamic core receives a gradient, not a clipped step.
    - scale calibrates tolerance: for r >= 1, health(r * b) = r ** (-1 / scale).
      With scale = 1.0, a 2x deviation scores 0.5 and a 6.4x deviation scores
      ~0.156; with the legacy linear form both score exactly 0.0.

    The scale parameter should be anchored on a healthy reference corpus.
    """
    baseline = max(float(baseline_micro), 1e-9)
    value = max(float(micro_raw), 1e-9)
    s = max(float(scale), 1e-9)
    return clamp01(math.exp(-abs(math.log(value / baseline)) / s))


MICRO_HEALTH_MODES = ("linear", "log_ratio")


def micro_health_from_mode(
    micro_raw: float,
    baseline_micro: float,
    mode: str = "linear",
    scale: float = 1.0,
) -> float:
    """Dispatch micro_health computation by mode.

    "linear" is the legacy centered_health (default, preserves v0.2.x outputs).
    "log_ratio" is the gradation-preserving form recommended going forward.
    """
    if mode == "log_ratio":
        return log_ratio_health(micro_raw, baseline_micro, scale=scale)
    if mode == "linear":
        return centered_health(micro_raw, baseline_micro)
    raise ValueError(f"unknown micro_health_mode: {mode!r}; expected one of {MICRO_HEALTH_MODES}")


def macro_continuity(previous_mean_surprise: Optional[float], current_mean_surprise: float) -> float:
    if previous_mean_surprise is None:
        return 1.0
    return 1.0 / (1.0 + abs(float(current_mean_surprise) - float(previous_mean_surprise)))


def micro_drop(micro_raw: float, baseline_micro: float) -> float:
    baseline = max(float(baseline_micro), 1e-9)
    return clamp01((baseline - float(micro_raw)) / baseline)


def micro_excess(micro_raw: float, baseline_micro: float) -> float:
    baseline = max(float(baseline_micro), 1e-9)
    return max(0.0, (float(micro_raw) - baseline) / baseline)


def extract_turn_logprobs_with_counts(turn: Dict[str, Any]) -> Dict[str, Any]:
    tokens = turn.get("tokens", [])
    if not isinstance(tokens, list):
        return {"logprobs": [], "valid_token_count": 0, "invalid_token_count": 0}
    values: List[float] = []
    invalid = 0
    for token in tokens:
        if not isinstance(token, dict):
            invalid += 1
            continue
        value = token.get("top1_logprob", token.get("logprob"))
        if is_valid_logprob(value):
            values.append(float(value))
        else:
            invalid += 1
    return {"logprobs": values, "valid_token_count": len(values), "invalid_token_count": invalid}


def baseline_from_turns(
    turns: Sequence[Dict[str, Any]],
    calib_window: Optional[int] = None,
    baseline_stat: str = "mean",
    min_calib_tokens: int = 0,
) -> Dict[str, Any]:
    """Compute the frozen micro-amplitude baseline over the calibration window.

    baseline_stat: "mean" (legacy default) or "median" (robust to a single
    atypical calibration turn).
    min_calib_tokens: turns with fewer valid tokens are excluded from the
    calibration set (0 = legacy behavior, no exclusion). Near-deterministic
    openings (short greetings) compress micro_raw and, if they dominate the
    baseline, induce spurious micro_excess / DISSOLUTION attribution on every
    subsequent content turn.
    """
    if calib_window is not None:
        n_calib = max(1, min(int(calib_window), len(turns)))
        method = "explicit_calib_window"
        warning = None
    else:
        n_calib = max(1, math.ceil(len(turns) * 0.25))
        method = "first_25_percent_fallback"
        warning = "No --calib-window provided; first 25% of turns used as frozen neutral baseline."
    micros: List[float] = []
    excluded_short = 0
    for turn in turns[:n_calib]:
        extracted = extract_turn_logprobs_with_counts(turn)
        if extracted["valid_token_count"] < 2:
            continue
        if min_calib_tokens > 0 and extracted["valid_token_count"] < int(min_calib_tokens):
            excluded_short += 1
            continue
        micros.append(turn_micro_from_surprises(token_surprise_series(extracted["logprobs"])))
    if baseline_stat == "median":
        baseline_micro = median(micros) if micros else 0.0
    elif baseline_stat == "mean":
        baseline_micro = mean(micros) if micros else 0.0
    else:
        raise ValueError(f"unknown baseline_stat: {baseline_stat!r}; expected 'mean' or 'median'")
    warnings: List[str] = [warning] if warning else []
    if excluded_short:
        warnings.append(
            f"{excluded_short} calibration turn(s) excluded for having fewer than "
            f"{int(min_calib_tokens)} valid tokens."
        )
    if len(micros) < 2:
        warnings.append(
            "Baseline computed from fewer than 2 contributing turns; a single "
            "near-deterministic opening can anchor the baseline too low and induce "
            "spurious micro_excess (DISSOLUTION) attribution on subsequent turns. "
            "Prefer a wider calibration window or min_calib_tokens > 0."
        )
    return {
        "baseline_micro": baseline_micro,
        "baseline_n_calib": n_calib,
        "baseline_method": method,
        "baseline_stat": baseline_stat,
        "baseline_contributing_turns": len(micros),
        "baseline_excluded_short_turns": excluded_short,
        "baseline_warning": " ".join(warnings) if warnings else None,
    }


def raw_activity_from_turn(turn: Dict[str, Any]) -> Optional[float]:
    value = turn.get("activity")
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return clamp01(float(value))


def structural_activity(micro_raw: Optional[float], baseline_micro: float) -> float:
    if micro_raw is None:
        return 0.0
    baseline = max(float(baseline_micro), 0.0)
    return clamp01(float(micro_raw) / (baseline + 1e-3))


def effective_activity(activity_raw: Optional[float], activity_structural: float) -> float:
    if activity_raw is None:
        return clamp01(activity_structural)
    return min(clamp01(activity_structural), clamp01(activity_raw))


def measure(
    turns: Sequence[Dict[str, Any]],
    calib_window: Optional[int] = None,
    micro_health_mode: str = "linear",
    micro_health_scale: float = 1.0,
    baseline_stat: str = "mean",
    min_calib_tokens: int = 0,
    theta0: float = DEFAULT_THETA0,
    lambda0: float = DEFAULT_LAMBDA0,
    memory_beta: float = DEFAULT_MEMORY_BETA,
    critical_margin: float = 0.05,
    side_threshold: float = 0.05,
    side_margin: float = 0.03,
    min_turns_for_regime: int = DEFAULT_MIN_TURNS_FOR_REGIME,
    min_windows_for_regime: int = DEFAULT_MIN_WINDOWS_FOR_REGIME,
    min_post_crossing_units: int = DEFAULT_MIN_POST_CROSSING_UNITS,
    crossing_index_scope: str = "turn",
) -> Dict[str, Any]:
    baseline = baseline_from_turns(
        turns,
        calib_window=calib_window,
        baseline_stat=baseline_stat,
        min_calib_tokens=min_calib_tokens,
    )
    baseline_micro = float(baseline["baseline_micro"])
    rows: List[Dict[str, Any]] = []
    previous_mean: Optional[float] = None
    xi_accumulated = 0.0

    for idx, turn in enumerate(turns):
        turn_index = int(turn.get("turn_index", idx))
        extracted = extract_turn_logprobs_with_counts(turn)
        logprobs = extracted["logprobs"]
        valid_token_count = extracted["valid_token_count"]
        invalid_token_count = extracted["invalid_token_count"]
        activity_raw = raw_activity_from_turn(turn)
        if valid_token_count < 2:
            activity_structural = 0.0
            activity_effective = effective_activity(activity_raw, activity_structural)
            rows.append(
                asdict(
                    TurnReading(
                        turn_index=turn_index,
                        valid_token_count=valid_token_count,
                        invalid_token_count=invalid_token_count,
                        logprob_valid=False,
                        insufficient_data=True,
                        micro_raw=None,
                        micro_health=None,
                        macro_health=None,
                        activity_raw=activity_raw,
                        activity_structural=activity_structural,
                        activity_effective=activity_effective,
                        activity=activity_effective,
                        acople=None,
                        acople_effective=None,
                        delta_instant=None,
                        xi_accumulated=None,
                        xi_norm=None,
                        lambda_remaining=None,
                        theta_dynamic=None,
                        viability_margin=None,
                        accumulated_viability_margin=None,
                        instant_viability_margin=None,
                        instant_threshold_crossed=False,
                        instant_recovered=False,
                        compression_gap=None,
                        distance_to_threshold=None,
                        xi_exceeds_theta=False,
                        threshold_crossed=False,
                        viability_status="UNRESOLVED",
                        boundary_pressure=None,
                        boundary_side="UNRESOLVED",
                        micro_drop=None,
                        micro_excess=None,
                        viability=None,
                        delta=None,
                        xi=None,
                        lam=None,
                        theta=None,
                    )
                )
            )
            continue

        surprises = token_surprise_series(logprobs)
        micro_raw = turn_micro_from_surprises(surprises)
        mean_surprise = turn_mean_surprise(surprises)
        micro_health_value = micro_health_from_mode(
            micro_raw, baseline_micro, mode=micro_health_mode, scale=micro_health_scale
        )
        macro_health = macro_continuity(previous_mean, mean_surprise)
        acople_raw = min(micro_health_value, macro_health)
        activity_structural = structural_activity(micro_raw, baseline_micro)
        activity_effective = effective_activity(activity_raw, activity_structural)
        delta_instant = (1.0 - acople_raw) * activity_effective
        acople_effective = 1.0 - delta_instant
        xi_accumulated = (float(memory_beta) * xi_accumulated) + delta_instant
        xi_norm = xi_accumulated / (1.0 + xi_accumulated)
        lambda_remaining = clamp01(lambda0 * (1.0 - xi_norm))
        theta_dynamic = float(theta0) * lambda_remaining
        viability_margin = theta_dynamic - xi_norm
        instant_viability_margin = theta_dynamic - delta_instant
        instant_threshold_crossed = delta_instant > theta_dynamic
        instant_recovered = instant_viability_margin > 0.0
        xi_exceeds_theta = xi_norm >= theta_dynamic
        threshold_crossed = idx > 0 and xi_exceeds_theta
        status = viability_status(viability_margin, critical_margin)
        if idx == 0 and status == "THRESHOLD_CROSSED":
            status = "NEAR_THRESHOLD"
        boundary_pressure = boundary_pressure_from_margin(viability_margin, critical_margin)
        drop = micro_drop(micro_raw, baseline_micro)
        excess = micro_excess(micro_raw, baseline_micro)
        boundary_side = classify_boundary(
            status,
            boundary_pressure,
            drop,
            excess,
            1.0 - macro_health,
            side_threshold=side_threshold,
            side_margin=side_margin,
        )
        rows.append(
            asdict(
                TurnReading(
                    turn_index=turn_index,
                    valid_token_count=valid_token_count,
                    invalid_token_count=invalid_token_count,
                    logprob_valid=True,
                    insufficient_data=False,
                    micro_raw=micro_raw,
                    micro_health=micro_health_value,
                    macro_health=macro_health,
                    activity_raw=activity_raw,
                    activity_structural=activity_structural,
                    activity_effective=activity_effective,
                    activity=activity_effective,
                    acople=acople_raw,
                    acople_effective=acople_effective,
                    delta_instant=delta_instant,
                    xi_accumulated=xi_accumulated,
                    xi_norm=xi_norm,
                    lambda_remaining=lambda_remaining,
                    theta_dynamic=theta_dynamic,
                    viability_margin=viability_margin,
                    accumulated_viability_margin=viability_margin,
                    instant_viability_margin=instant_viability_margin,
                    instant_threshold_crossed=instant_threshold_crossed,
                    instant_recovered=instant_recovered,
                    compression_gap=None,
                    distance_to_threshold=viability_margin,
                    xi_exceeds_theta=xi_exceeds_theta,
                    threshold_crossed=threshold_crossed,
                    viability_status=status,
                    boundary_pressure=boundary_pressure,
                    boundary_side=boundary_side,
                    micro_drop=drop,
                    micro_excess=excess,
                    viability=viability_margin,
                    delta=delta_instant,
                    xi=xi_accumulated,
                    lam=lambda_remaining,
                    theta=theta_dynamic,
                )
            )
        )
        previous_mean = mean_surprise

    valid_rows = [row for row in rows if row.get("logprob_valid")]
    final = valid_rows[-1] if valid_rows else {}
    margins = [row["viability_margin"] for row in valid_rows if row.get("viability_margin") is not None]
    critical_turns = [
        row["turn_index"]
        for row in valid_rows
        if row.get("turn_index") != 0 and row.get("viability_status") in {"THRESHOLD_CROSSED", "NEAR_THRESHOLD"}
    ]
    regime = classify_regime(
        rows,
        crossing_index_scope=crossing_index_scope,
        min_turns_for_regime=min_turns_for_regime,
        min_windows_for_regime=min_windows_for_regime,
        min_post_crossing_units=min_post_crossing_units,
    )
    trajectory_threshold_crossed = any(row.get("threshold_crossed") for row in valid_rows)
    trajectory_xi_exceeds_theta = any(
        row.get("xi_exceeds_theta") and row.get("turn_index") != 0 for row in valid_rows
    )
    final_instant_recovered = bool(final.get("instant_recovered", False))
    final_instant_threshold_crossed = bool(final.get("instant_threshold_crossed", False))
    recovered_finally = final_instant_recovered
    relapsed_after_recovery = bool(regime["recovery_observed"] and final_instant_threshold_crossed)
    pulsation_subtype = pulsation_subtype_from_state(
        regime["regime_label"],
        bool(regime["recovery_observed"]),
        final_instant_recovered,
        final_instant_threshold_crossed,
    )
    return {
        "substrate_blind": True,
        "material_cost_measured": False,
        "requires_exogenous_telemetry_for_material_cost": True,
        "baseline_micro": baseline_micro,
        "baseline_n_calib": baseline["baseline_n_calib"],
        "baseline_method": baseline["baseline_method"],
        "baseline_stat": baseline["baseline_stat"],
        "baseline_contributing_turns": baseline["baseline_contributing_turns"],
        "baseline_excluded_short_turns": baseline["baseline_excluded_short_turns"],
        "baseline_warning": baseline["baseline_warning"],
        "micro_health_mode": micro_health_mode,
        "micro_health_scale": micro_health_scale if micro_health_mode == "log_ratio" else None,
        "theta0": theta0,
        "lambda0": lambda0,
        "memory_beta": memory_beta,
        "collapse_threshold": collapse_xi_norm(theta0, lambda0),
        "critical_margin": critical_margin,
        "valid_turns": len(valid_rows),
        "invalid_turns": len(rows) - len(valid_rows),
        "turns_with_insufficient_data": sum(1 for row in rows if row.get("insufficient_data")),
        "micro_scale_definition": (
            "micro_raw is unbounded surprise amplitude; micro_health is centered on baseline_micro."
            if micro_health_mode == "linear"
            else "micro_raw is unbounded surprise amplitude; micro_health = exp(-|ln(micro_raw/baseline_micro)|/scale), multiplicatively symmetric and gradation-preserving."
        ),
        "final_viability": final.get("viability_margin"),
        "min_viability": min(margins) if margins else None,
        "final_accumulated_viability_margin": final.get("accumulated_viability_margin"),
        "final_instant_viability_margin": final.get("instant_viability_margin"),
        "final_distance_to_threshold": final.get("viability_margin"),
        "min_distance_to_threshold": min(margins) if margins else None,
        "threshold_crossed": trajectory_threshold_crossed,
        "final_threshold_crossed": bool(final.get("threshold_crossed", False)),
        "xi_exceeds_theta": trajectory_xi_exceeds_theta,
        "final_xi_exceeds_theta": bool(final.get("xi_exceeds_theta", False)),
        "final_instant_threshold_crossed": final_instant_threshold_crossed,
        "final_instant_recovered": final_instant_recovered,
        "recovered_finally": recovered_finally,
        "relapsed_after_recovery": relapsed_after_recovery,
        "pulsation_subtype": pulsation_subtype,
        "boundary_side": final.get("boundary_side"),
        "boundary_pressure": final.get("boundary_pressure"),
        "viability_status": final.get("viability_status", "UNRESOLVED"),
        "critical_turns": critical_turns,
        "regime_label": regime["regime_label"],
        "regime_description": regime["regime_description"],
        "recovery_observed": regime["recovery_observed"],
        "first_crossing_turn": regime["first_crossing_turn"],
        "threshold_crossing_ratio": regime["threshold_crossing_ratio"],
        "persistent_crossing_ratio": regime["persistent_crossing_ratio"],
        "post_crossing_recovery_turns": regime["post_crossing_recovery_turns"],
        "local_threshold_cascade": regime["local_threshold_cascade"],
        "crossing_index_scope": regime["crossing_index_scope"],
        "first_crossing_window": regime["first_crossing_window"],
        "trajectory_assessment": trajectory_assessment_from_regime(regime["regime_label"]),
        "turns": rows,
        "notes": {
            "scope": SUBSTRATE_BLIND_WARNING,
            "viability": COHERENCE_VIABILITY_NOTE,
            "echo": ECHO_NOTE,
            "compression_gap": (
                "compression_gap is reserved for the gap between task-required support "
                "and response-expressed support. It is None until an exogenous task channel exists."
            ),
            "acople": (
                "acople = acople_raw; acople_effective is the effective decoupling incorporated "
                "into accumulated Xi. Low-activity ACKs may have low acople_raw and high "
                "acople_effective without contradiction."
            ),
            "activity": (
                "activity_raw records any incoming activity field. activity_structural is derived "
                "from micro_raw relative to baseline_micro. activity_effective is the structural "
                "activity used by delta_instant and never exceeds activity_structural."
            ),
        },
    }


def demo() -> None:
    turns = [
        {"turn_index": 0, "tokens": [{"top1_logprob": -0.7}, {"top1_logprob": -1.1}], "activity": 1.0},
        {"turn_index": 1, "tokens": [{"top1_logprob": -0.9}, {"top1_logprob": -1.0}], "activity": 0.0},
        {"turn_index": 2, "tokens": [{"top1_logprob": -0.2}, {"top1_logprob": -1.8}], "activity": 1.0},
    ]
    result = measure(turns, calib_window=1)
    print("PRAMA ProbLog Components v0.2.1 demo")
    print("collapse_xi_norm(theta0, lambda0) = (theta0 * lambda0) / (1 + theta0 * lambda0)")
    print("with lambda0 = 1, this reduces to theta0 / (1 + theta0)")
    print(f"collapse_xi_norm(0.35, 1.0) = {collapse_xi_norm(0.35, 1.0):.6f}")
    print(f"final_viability_margin = {result['final_viability']:.6f}")
    print(f"threshold_crossed = {result['threshold_crossed']}")


if __name__ == "__main__":
    demo()
