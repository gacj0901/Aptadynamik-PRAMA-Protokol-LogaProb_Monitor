from __future__ import annotations

import math
from statistics import mean
from typing import Any, Dict, List, Sequence


SUBSTRATE_BLIND_WARNING = (
    "This module is substrate-blind: it observes generative geometry from "
    "logprobs, not material cost. It does not measure energy use, thermal "
    "pressure, GPU load, memory pressure, latency, or physical friction."
)

COHERENCE_VIABILITY_NOTE = (
    "Coherence is a threshold. Viability is a gradient. Viability is measured "
    "here as the weakest band of micro/macro coupling. La coherencia es umbral. "
    "La viabilidad es gradiente. La viabilidad se mide aquí como la banda más "
    "débil del acoplamiento micro/macro."
)

ECHO_NOTE = (
    "Structural repetition is not full coherence. In the PRAMA framework, "
    "coherence is not mere regularity; it is structural persistence under flow. "
    "Repetition without living coupling is echo, not viable coherence. "
    "La repetición estructural no equivale a coherencia plena. En PRAMA, "
    "coherencia no es mera regularidad; es persistencia estructural bajo flujo. "
    "La repetición sin acoplamiento vivo es eco, no coherencia viable."
)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def token_surprise(logprob: float) -> float:
    return -float(logprob)


def is_valid_logprob(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x)) and float(x) <= 0.0 and abs(float(x)) < 100.0


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


def macro_continuity(previous_mean_surprise: float | None, current_mean_surprise: float) -> float:
    if previous_mean_surprise is None:
        return 1.0
    return 1.0 / (1.0 + abs(float(current_mean_surprise) - float(previous_mean_surprise)))


def micro_health(micro_raw: float, baseline_micro: float) -> float:
    baseline = max(float(baseline_micro), 1e-9)
    micro_raw = max(float(micro_raw), 0.0)
    if micro_raw <= baseline:
        return clamp01(micro_raw / baseline)
    return clamp01(1.0 - ((micro_raw - baseline) / baseline))


def generative_structural_debt(micro_raw: float, baseline_micro: float) -> float:
    """Generative structural debt proxy derived from logprob geometry.

    micro_drop is not material cost. It is a generative structural debt proxy
    derived from token-level surprise geometry.
    """
    baseline = max(float(baseline_micro), 1e-9)
    return clamp01((baseline - float(micro_raw)) / baseline)


def micro_excess(micro_raw: float, baseline_micro: float, eps: float = 1e-9) -> float:
    baseline = max(float(baseline_micro), eps)
    return max(0.0, (float(micro_raw) - baseline) / baseline)


def viability_status(viability: float | None, collapse_threshold: float, critical_margin: float) -> str:
    if viability is None:
        return "UNRESOLVED"
    if viability <= collapse_threshold:
        return "THRESHOLD_CROSSED"
    if viability <= collapse_threshold + critical_margin:
        return "NEAR_THRESHOLD"
    return "VIABLE"


def boundary_pressure_from_viability(
    viability: float | None,
    collapse_threshold: float,
    critical_margin: float,
) -> float | None:
    if viability is None:
        return None
    margin = max(float(critical_margin), 1e-9)
    return clamp01(1.0 - ((float(viability) - float(collapse_threshold)) / margin))


def classify_boundary(
    viability_status_value: str,
    boundary_pressure: float | None,
    micro_drop: float,
    micro_excess_value: float,
    macro_health: float,
    side_threshold: float = 0.05,
    side_margin: float = 0.03,
) -> Dict[str, Any]:
    if viability_status_value == "VIABLE" and boundary_pressure == 0.0:
        return {
            "boundary_side": "CENTERED",
            "condensation_pressure": clamp01(micro_drop),
            "dissolution_pressure": clamp01(micro_excess_value),
            "decoupling_pressure": clamp01(1.0 - macro_health),
        }
    pressures = {
        "CONDENSATION": clamp01(micro_drop),
        "DISSOLUTION": clamp01(micro_excess_value),
        "DECOUPLING": clamp01(1.0 - macro_health),
    }
    ranked = sorted(pressures.items(), key=lambda item: item[1], reverse=True)
    boundary_side, max_pressure = ranked[0]
    second_pressure = ranked[1][1] if len(ranked) > 1 else 0.0
    if max_pressure < side_threshold or (max_pressure - second_pressure) < side_margin:
        boundary_side = "UNRESOLVED"
    return {
        "boundary_side": boundary_side,
        "condensation_pressure": float(pressures["CONDENSATION"]),
        "dissolution_pressure": float(pressures["DISSOLUTION"]),
        "decoupling_pressure": float(pressures["DECOUPLING"]),
    }


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
        value = None
        if "top1_logprob" in token:
            value = token["top1_logprob"]
        elif "logprob" in token:
            value = token["logprob"]
        if is_valid_logprob(value):
            values.append(float(value))
        else:
            invalid += 1
    return {"logprobs": values, "valid_token_count": len(values), "invalid_token_count": invalid}


def extract_turn_logprobs(turn: Dict[str, Any]) -> List[float]:
    return extract_turn_logprobs_with_counts(turn)["logprobs"]


def baseline_from_turns(
    turns: Sequence[Dict[str, Any]],
    calib_window: int | None = None,
) -> Dict[str, Any]:
    if calib_window is not None:
        n_calib = max(1, min(int(calib_window), len(turns)))
        method = "explicit_calib_window"
        warning = None
    else:
        n_calib = max(1, math.ceil(len(turns) * 0.25))
        method = "first_25_percent_fallback"
        warning = "No --calib-window provided; first 25% of turns used as frozen neutral baseline."
    micros = []
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


def structural_diagnostics(
    micro_raw: float,
    macro_health: float,
    micro_health_value: float,
    micro_drop: float,
    micro_excess_value: float,
    collapse_threshold: float,
    diagnostic_threshold: float,
) -> Dict[str, Any]:
    macro_high_score = clamp01((macro_health - 0.75) / 0.25)
    macro_broken_score = clamp01(1.0 - macro_health)
    micro_dead_score = clamp01(
        ((1.0 - diagnostic_threshold) - micro_health_value) / max(1.0 - diagnostic_threshold, 1e-9)
    )
    micro_alive_score = clamp01(
        (micro_health_value - diagnostic_threshold) / max(1.0 - diagnostic_threshold, 1e-9)
    )
    rig = min(macro_high_score, micro_dead_score, 1.0 if micro_raw <= 1e-9 else 0.0)
    eco = min(macro_high_score, micro_dead_score, micro_drop)
    alu = min(micro_alive_score, macro_broken_score)
    turb_base = clamp01(micro_excess_value)
    if macro_health > 0.85 and micro_health_value <= collapse_threshold:
        turb = turb_base
    else:
        turb = clamp01(turb_base * 0.7 * min(macro_high_score, micro_dead_score))
    return {
        "rig": float(rig),
        "eco": float(eco),
        "alu": float(alu),
        "turb": float(turb),
    }


def measure(
    turns: Sequence[Dict[str, Any]],
    calib_window: int | None = None,
    collapse_threshold: float = 0.35,
    diagnostic_threshold: float = 0.60,
    critical_margin: float = 0.05,
    side_threshold: float = 0.05,
    side_margin: float = 0.03,
) -> Dict[str, Any]:
    baseline = baseline_from_turns(turns, calib_window=calib_window)
    baseline_micro = float(baseline["baseline_micro"])
    rows: List[Dict[str, Any]] = []
    previous_mean: float | None = None

    for idx, turn in enumerate(turns):
        extracted = extract_turn_logprobs_with_counts(turn)
        logprobs = extracted["logprobs"]
        valid_token_count = extracted["valid_token_count"]
        invalid_token_count = extracted["invalid_token_count"]
        insufficient_data = valid_token_count < 2
        if insufficient_data:
            rows.append(
                {
                    "turn_index": int(turn.get("turn_index", idx)),
                    "valid_token_count": valid_token_count,
                    "invalid_token_count": invalid_token_count,
                    "logprob_valid": False,
                    "insufficient_data": True,
                    "micro_raw": None,
                    "micro_health": None,
                    "macro_health": None,
                    "acople": None,
                    "micro_drop": None,
                    "micro_excess": None,
                    "viability": None,
                    "distance_to_threshold": None,
                    "threshold_crossed": False,
                    "boundary_side": "UNRESOLVED",
                    "boundary_pressure": None,
                    "condensation_pressure": None,
                    "dissolution_pressure": None,
                    "decoupling_pressure": None,
                    "viability_status": "UNRESOLVED",
                    "rig": 0.0,
                    "eco": 0.0,
                    "alu": 0.0,
                    "turb": 0.0,
                    "baseline_micro": baseline_micro,
                    "baseline_n_calib": baseline["baseline_n_calib"],
                }
            )
            continue
        surprises = token_surprise_series(logprobs)
        micro_raw = turn_micro_from_surprises(surprises)
        mean_surprise = turn_mean_surprise(surprises)
        macro = macro_continuity(previous_mean, mean_surprise)
        micro_health_value = micro_health(micro_raw, baseline_micro)
        macro_health = macro
        acople = min(micro_health_value, macro_health)
        micro_drop = generative_structural_debt(micro_raw, baseline_micro)
        micro_excess_value = micro_excess(micro_raw, baseline_micro)
        viability = acople
        distance_to_threshold = viability - collapse_threshold
        threshold_crossed = viability <= collapse_threshold
        status = viability_status(viability, collapse_threshold, critical_margin)
        boundary_pressure = boundary_pressure_from_viability(viability, collapse_threshold, critical_margin)
        boundary = classify_boundary(
            status,
            boundary_pressure,
            micro_drop,
            micro_excess_value,
            macro_health,
            side_threshold=side_threshold,
            side_margin=side_margin,
        )
        diagnostics = structural_diagnostics(
            micro_raw,
            macro_health,
            micro_health_value,
            micro_drop,
            micro_excess_value,
            collapse_threshold,
            diagnostic_threshold,
        )
        rows.append(
            {
                "turn_index": int(turn.get("turn_index", idx)),
                "valid_token_count": valid_token_count,
                "invalid_token_count": invalid_token_count,
                "logprob_valid": True,
                "insufficient_data": False,
                "micro_raw": micro_raw,
                "micro_health": micro_health_value,
                "macro_health": macro_health,
                "acople": acople,
                "micro_drop": micro_drop,
                "micro_excess": micro_excess_value,
                "viability": viability,
                "distance_to_threshold": distance_to_threshold,
                "threshold_crossed": threshold_crossed,
                "boundary_side": boundary["boundary_side"],
                "boundary_pressure": boundary_pressure,
                "condensation_pressure": boundary["condensation_pressure"],
                "dissolution_pressure": boundary["dissolution_pressure"],
                "decoupling_pressure": boundary["decoupling_pressure"],
                "viability_status": status,
                "rig": diagnostics["rig"],
                "eco": diagnostics["eco"],
                "alu": diagnostics["alu"],
                "turb": diagnostics["turb"],
                "baseline_micro": baseline_micro,
                "baseline_n_calib": baseline["baseline_n_calib"],
            }
        )
        previous_mean = mean_surprise

    valid_rows = [row for row in rows if row.get("logprob_valid")]
    final = valid_rows[-1] if valid_rows else {}
    viability_values = [row["viability"] for row in valid_rows if row.get("viability") is not None]
    distance_values = [row["distance_to_threshold"] for row in valid_rows if row.get("distance_to_threshold") is not None]
    critical_turns = [
        row["turn_index"]
        for row in valid_rows
        if row.get("viability_status") in {"THRESHOLD_CROSSED", "NEAR_THRESHOLD"}
    ]
    invalid_turns = len(rows) - len(valid_rows)
    return {
        "substrate_blind": True,
        "material_cost_measured": False,
        "requires_exogenous_telemetry_for_material_cost": True,
        "baseline_micro": baseline_micro,
        "baseline_n_calib": baseline["baseline_n_calib"],
        "baseline_method": baseline["baseline_method"],
        "baseline_warning": baseline["baseline_warning"],
        "collapse_threshold": collapse_threshold,
        "diagnostic_threshold": diagnostic_threshold,
        "critical_margin": critical_margin,
        "side_threshold": side_threshold,
        "side_margin": side_margin,
        "valid_turns": len(valid_rows),
        "invalid_turns": invalid_turns,
        "turns_with_insufficient_data": sum(1 for row in rows if row.get("insufficient_data")),
        "micro_scale_definition": "micro_raw is unbounded surprise amplitude; micro and micro_health are normalized to [0,1] against baseline_micro.",
        "final_viability": final.get("viability"),
        "min_viability": min(viability_values) if viability_values else None,
        "distance_to_threshold": final.get("distance_to_threshold"),
        "final_distance_to_threshold": final.get("distance_to_threshold"),
        "min_distance_to_threshold": min(distance_values) if distance_values else None,
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
        },
    }
