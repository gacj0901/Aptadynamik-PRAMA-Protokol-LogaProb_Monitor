from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence


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
    activity: float
    acople: Optional[float]
    acople_effective: Optional[float]
    delta_instant: Optional[float]
    xi_accumulated: Optional[float]
    xi_norm: Optional[float]
    lambda_remaining: Optional[float]
    theta_dynamic: Optional[float]
    viability_margin: Optional[float]
    compression_gap: Optional[float]
    distance_to_threshold: Optional[float]
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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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
    """
    baseline = max(float(baseline_micro), 1e-9)
    deviation = abs(float(micro_raw) - baseline) / baseline
    return clamp01(1.0 - deviation)


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


def collapse_xi_norm(theta0: float, lambda0: float = 1.0) -> float:
    """Return normalized accumulated stress at the initial dynamic threshold.

    General form: (theta0 * lambda0) / (1 + theta0 * lambda0).
    With lambda0 = 1, this reduces to theta0 / (1 + theta0).
    """
    product = max(0.0, float(theta0) * float(lambda0))
    return product / (1.0 + product)


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


def baseline_from_turns(turns: Sequence[Dict[str, Any]], calib_window: Optional[int] = None) -> Dict[str, Any]:
    if calib_window is not None:
        n_calib = max(1, min(int(calib_window), len(turns)))
        method = "explicit_calib_window"
        warning = None
    else:
        n_calib = max(1, math.ceil(len(turns) * 0.25))
        method = "first_25_percent_fallback"
        warning = "No --calib-window provided; first 25% of turns used as frozen neutral baseline."
    micros: List[float] = []
    for turn in turns[:n_calib]:
        extracted = extract_turn_logprobs_with_counts(turn)
        if extracted["valid_token_count"] >= 2:
            micros.append(turn_micro_from_surprises(token_surprise_series(extracted["logprobs"])))
    return {
        "baseline_micro": mean(micros) if micros else 0.0,
        "baseline_n_calib": n_calib,
        "baseline_method": method,
        "baseline_warning": warning,
    }


def activity_from_turn(turn: Dict[str, Any], valid_token_count: int, activity_token_scale: int = 8) -> float:
    if "activity" in turn:
        return clamp01(float(turn["activity"]))
    return clamp01(valid_token_count / max(float(activity_token_scale), 1.0))


def boundary_pressure_from_margin(viability_margin: Optional[float], critical_margin: float) -> Optional[float]:
    if viability_margin is None:
        return None
    margin = max(float(critical_margin), 1e-9)
    return clamp01(1.0 - (float(viability_margin) / margin))


def viability_status(viability_margin: Optional[float], critical_margin: float) -> str:
    if viability_margin is None:
        return "UNRESOLVED"
    if viability_margin <= 0.0:
        return "THRESHOLD_CROSSED"
    if viability_margin <= critical_margin:
        return "NEAR_THRESHOLD"
    return "VIABLE"


def classify_boundary(
    status: str,
    boundary_pressure: Optional[float],
    condensation_pressure: float,
    dissolution_pressure: float,
    decoupling_pressure: float,
    side_threshold: float = 0.05,
    side_margin: float = 0.03,
) -> str:
    if status == "VIABLE" and boundary_pressure == 0.0:
        return "CENTERED"
    pressures = {
        "CONDENSATION": clamp01(condensation_pressure),
        "DISSOLUTION": clamp01(dissolution_pressure),
        "DECOUPLING": clamp01(decoupling_pressure),
    }
    ranked = sorted(pressures.items(), key=lambda item: item[1], reverse=True)
    winner, max_pressure = ranked[0]
    second_pressure = ranked[1][1] if len(ranked) > 1 else 0.0
    if max_pressure < side_threshold or (max_pressure - second_pressure) < side_margin:
        return "UNRESOLVED"
    return winner


def measure(
    turns: Sequence[Dict[str, Any]],
    calib_window: Optional[int] = None,
    theta0: float = 0.35,
    lambda0: float = 1.0,
    memory_beta: float = 0.65,
    critical_margin: float = 0.05,
    side_threshold: float = 0.05,
    side_margin: float = 0.03,
) -> Dict[str, Any]:
    baseline = baseline_from_turns(turns, calib_window=calib_window)
    baseline_micro = float(baseline["baseline_micro"])
    rows: List[Dict[str, Any]] = []
    previous_mean: Optional[float] = None
    xi_accumulated = 0.0

    for idx, turn in enumerate(turns):
        extracted = extract_turn_logprobs_with_counts(turn)
        logprobs = extracted["logprobs"]
        valid_token_count = extracted["valid_token_count"]
        invalid_token_count = extracted["invalid_token_count"]
        activity = activity_from_turn(turn, valid_token_count)
        if valid_token_count < 2:
            rows.append(
                asdict(
                    TurnReading(
                        turn_index=int(turn.get("turn_index", idx)),
                        valid_token_count=valid_token_count,
                        invalid_token_count=invalid_token_count,
                        logprob_valid=False,
                        insufficient_data=True,
                        micro_raw=None,
                        micro_health=None,
                        macro_health=None,
                        activity=activity,
                        acople=None,
                        acople_effective=None,
                        delta_instant=None,
                        xi_accumulated=None,
                        xi_norm=None,
                        lambda_remaining=None,
                        theta_dynamic=None,
                        viability_margin=None,
                        compression_gap=None,
                        distance_to_threshold=None,
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
        micro_health_value = centered_health(micro_raw, baseline_micro)
        macro_health = macro_continuity(previous_mean, mean_surprise)
        acople_raw = min(micro_health_value, macro_health)
        delta_instant = (1.0 - acople_raw) * activity
        acople_effective = 1.0 - delta_instant
        xi_accumulated = (float(memory_beta) * xi_accumulated) + delta_instant
        xi_norm = xi_accumulated / (1.0 + xi_accumulated)
        lambda_remaining = clamp01(lambda0 * (1.0 - xi_norm))
        theta_dynamic = float(theta0) * lambda_remaining
        viability_margin = theta_dynamic - xi_norm
        status = viability_status(viability_margin, critical_margin)
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
        threshold_crossed = viability_margin <= 0.0
        rows.append(
            asdict(
                TurnReading(
                    turn_index=int(turn.get("turn_index", idx)),
                    valid_token_count=valid_token_count,
                    invalid_token_count=invalid_token_count,
                    logprob_valid=True,
                    insufficient_data=False,
                    micro_raw=micro_raw,
                    micro_health=micro_health_value,
                    macro_health=macro_health,
                    activity=activity,
                    acople=acople_raw,
                    acople_effective=acople_effective,
                    delta_instant=delta_instant,
                    xi_accumulated=xi_accumulated,
                    xi_norm=xi_norm,
                    lambda_remaining=lambda_remaining,
                    theta_dynamic=theta_dynamic,
                    viability_margin=viability_margin,
                    compression_gap=None,
                    distance_to_threshold=viability_margin,
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
        if row.get("viability_status") in {"THRESHOLD_CROSSED", "NEAR_THRESHOLD"}
    ]
    return {
        "substrate_blind": True,
        "material_cost_measured": False,
        "requires_exogenous_telemetry_for_material_cost": True,
        "baseline_micro": baseline_micro,
        "baseline_n_calib": baseline["baseline_n_calib"],
        "baseline_method": baseline["baseline_method"],
        "baseline_warning": baseline["baseline_warning"],
        "theta0": theta0,
        "lambda0": lambda0,
        "collapse_threshold": collapse_xi_norm(theta0, lambda0),
        "critical_margin": critical_margin,
        "valid_turns": len(valid_rows),
        "invalid_turns": len(rows) - len(valid_rows),
        "turns_with_insufficient_data": sum(1 for row in rows if row.get("insufficient_data")),
        "micro_scale_definition": "micro_raw is unbounded surprise amplitude; micro_health is centered on baseline_micro.",
        "final_viability": final.get("viability_margin"),
        "min_viability": min(margins) if margins else None,
        "final_distance_to_threshold": final.get("viability_margin"),
        "min_distance_to_threshold": min(margins) if margins else None,
        "threshold_crossed": bool(final.get("threshold_crossed", False)),
        "boundary_side": final.get("boundary_side"),
        "boundary_pressure": final.get("boundary_pressure"),
        "viability_status": final.get("viability_status", "UNRESOLVED"),
        "critical_turns": critical_turns,
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
        },
    }


def demo() -> None:
    turns = [
        {"turn_index": 0, "tokens": [{"top1_logprob": -0.7}, {"top1_logprob": -1.1}], "activity": 1.0},
        {"turn_index": 1, "tokens": [{"top1_logprob": -0.9}, {"top1_logprob": -1.0}], "activity": 0.0},
        {"turn_index": 2, "tokens": [{"top1_logprob": -0.2}, {"top1_logprob": -1.8}], "activity": 1.0},
    ]
    result = measure(turns, calib_window=1)
    print("PRAMA Components v0.2.1 demo")
    print("collapse_xi_norm(theta0, lambda0) = (theta0 * lambda0) / (1 + theta0 * lambda0)")
    print("with lambda0 = 1, this reduces to theta0 / (1 + theta0)")
    print(f"collapse_xi_norm(0.35, 1.0) = {collapse_xi_norm(0.35, 1.0):.6f}")
    print(f"final_viability_margin = {result['final_viability']:.6f}")
    print(f"threshold_crossed = {result['threshold_crossed']}")


if __name__ == "__main__":
    demo()
