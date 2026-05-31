from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

from aptadynamik.observer.paramend_rayleigh import (
    estimate_mu_series_from_turns,
    estimate_mu_star_from_baseline,
    rayleigh_summary,
)


EPSILON = 1e-9


def _as_floats(series: Sequence[float]) -> List[float]:
    return [float(value) for value in series]


def lag1_ac(series: Sequence[float]) -> float:
    values = _as_floats(series)
    if len(values) < 3:
        return 0.0
    xs = values[:-1]
    ys = values[1:]
    mx = mean(xs)
    my = mean(ys)
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    if denom <= EPSILON:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def variance(series: Sequence[float]) -> float:
    values = _as_floats(series)
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / len(values)


def _adjacent_drops(values: Sequence[float]) -> List[float]:
    return [float(a) - float(b) for a, b in zip(values, values[1:])]


def _transition_turn(values: Sequence[float]) -> int:
    if len(values) < 2:
        return -1
    drops = _adjacent_drops(values)
    return max(range(len(drops)), key=lambda idx: abs(drops[idx])) + 1


def sig_discontinuity(series: Sequence[float]) -> Dict[str, Any]:
    values = _as_floats(series)
    if len(values) < 2:
        return {
            "triggered": False,
            "discontinuity_ratio": 0.0,
            "strongest_transition_turn": -1,
            "max_jump": 0.0,
            "local_drops": [],
        }

    jumps = [abs(b - a) for a, b in zip(values, values[1:])]
    local_drops = _adjacent_drops(values)
    max_jump = max(jumps)
    observed_range = max(values) - min(values)
    ratio = max_jump / (observed_range + EPSILON)
    transition_turn = jumps.index(max_jump) + 1
    return {
        "triggered": bool(max_jump >= 0.25 and ratio >= 0.45),
        "discontinuity_ratio": ratio,
        "strongest_transition_turn": transition_turn,
        "max_jump": max_jump,
        "local_drops": local_drops,
    }


def sig_hysteresis(up_series: Sequence[float], down_series: Sequence[float]) -> Dict[str, Any]:
    up = _as_floats(up_series)
    down = _as_floats(down_series)
    count = min(len(up), len(down))
    if count == 0:
        return {"triggered": False, "hysteresis_area": 0.0}

    aligned_down = list(reversed(down))[:count]
    aligned_up = up[:count]
    area = sum(abs(a - b) for a, b in zip(aligned_up, aligned_down)) / count
    return {"triggered": bool(area >= 0.05), "hysteresis_area": area}


def sig_critical_slowing(series: Sequence[float], window: int = 6) -> Dict[str, Any]:
    values = _as_floats(series)
    if len(values) < max(window * 2, 4):
        return {
            "triggered": False,
            "critical_slowing_score": 0.0,
            "autocorrelation_gain": 0.0,
            "variance_gain": 0.0,
        }

    early = values[:window]
    late = values[-window:]
    autocorrelation_gain = lag1_ac(late) - lag1_ac(early)
    variance_gain = variance(late) - variance(early)
    score = max(0.0, autocorrelation_gain) + max(0.0, variance_gain)

    drops = _adjacent_drops(values)
    if drops:
        largest_drop = max(drops)
        if largest_drop > 0.20:
            drop_idx = drops.index(largest_drop)
            recovery_target = values[drop_idx + 1] + largest_drop * 0.5
            recovery_steps = 0
            for value in values[drop_idx + 1 :]:
                recovery_steps += 1
                if value >= recovery_target:
                    break
            score += recovery_steps / len(values)

    return {
        "triggered": bool(score >= 0.08),
        "critical_slowing_score": score,
        "autocorrelation_gain": autocorrelation_gain,
        "variance_gain": variance_gain,
    }


def sig_structural_target(
    peripheral_series: Sequence[float],
    constitutive_series: Sequence[float],
) -> Dict[str, Any]:
    peripheral = _as_floats(peripheral_series)
    constitutive = _as_floats(constitutive_series)
    if len(peripheral) < 2 or len(constitutive) < 2:
        return {
            "triggered": False,
            "structural_target_shift": 0.0,
            "peripheral_transition_turn": -1,
            "constitutive_transition_turn": -1,
        }

    peripheral_turn = _transition_turn(peripheral)
    constitutive_turn = _transition_turn(constitutive)
    peripheral_disc = sig_discontinuity(peripheral)["discontinuity_ratio"]
    constitutive_disc = sig_discontinuity(constitutive)["discontinuity_ratio"]
    shift = float(peripheral_turn - constitutive_turn)
    if shift == 0 and constitutive_disc > peripheral_disc:
        shift = constitutive_disc - peripheral_disc

    return {
        "triggered": bool(shift > 0),
        "structural_target_shift": shift,
        "peripheral_transition_turn": peripheral_turn,
        "constitutive_transition_turn": constitutive_turn,
    }


def _require_mapping(parent: Dict[str, Any], key: str, context: str) -> Dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context} missing required mapping '{key}'")
    return value


def _require_number(parent: Dict[str, Any], key: str, context: str) -> float:
    if key not in parent:
        raise ValueError(f"{context} missing required numeric field '{key}'")
    try:
        return float(parent[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} field '{key}' must be numeric") from exc


def _summary_value(summary: Dict[str, Any], preferred: str, fallback: str, context: str) -> float:
    if preferred in summary:
        return _require_number(summary, preferred, context)
    if fallback in summary:
        return _require_number(summary, fallback, context)
    raise ValueError(f"{context} missing required numeric field '{preferred}'")


def viability_from_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    context = f"turn {turn.get('turn_index', '<unknown>')}"
    summary = _require_mapping(turn, "summary", context)

    if "turn_index" not in turn:
        raise ValueError("turn missing required field 'turn_index'")
    if "token_count" not in turn:
        raise ValueError(f"{context} missing required numeric field 'token_count'")

    turn_index = int(_require_number(turn, "turn_index", context))
    token_count = int(_require_number(turn, "token_count", context))
    avg_rigidity = _require_number(summary, "avg_rigidity", context)
    avg_uncertainty = _require_number(summary, "avg_uncertainty", context)
    avg_entropy_norm = _require_number(summary, "avg_entropy_norm", context)
    entropy_range = _summary_value(summary, "max_entropy_range", "entropy_range", context)
    entropy_std = _summary_value(summary, "max_entropy_std", "entropy_std", context)

    return {
        "turn_index": turn_index,
        "viability": avg_rigidity - avg_uncertainty,
        "avg_rigidity": avg_rigidity,
        "avg_uncertainty": avg_uncertainty,
        "avg_entropy_norm": avg_entropy_norm,
        "entropy_range": entropy_range,
        "entropy_std": entropy_std,
        "token_count": token_count,
    }


def load_raw_session(path: str | Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def viability_series_from_raw(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns = raw.get("turns")
    if not isinstance(turns, list):
        raise ValueError("raw session missing required list 'turns'")
    return [viability_from_turn(turn) for turn in turns]


def paramend_rayleigh_fields_from_turns(turns: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not turns:
        return {
            "paramend_rayleigh_median": None,
            "paramend_rayleigh_max": None,
            "min_restitution_g": None,
            "early_warning_turn": -1,
            "mu_star": None,
        }
    mu_series = estimate_mu_series_from_turns(turns)
    mu_star = estimate_mu_star_from_baseline(turns)
    return rayleigh_summary(mu_series, mu_star=mu_star)


def synthetic_fold_series(n: int = 72) -> Dict[str, List[float]]:
    half = n // 2
    up: List[float] = []
    down: List[float] = []
    for i in range(half):
        pressure = i / max(half - 1, 1)
        value = 0.88 - 0.14 * pressure
        if pressure > 0.58:
            value -= 0.42
        value += 0.035 * math.sin(i / 1.8)
        up.append(value)
    for i in range(half):
        pressure = 1.0 - i / max(half - 1, 1)
        value = 0.42 + 0.10 * pressure
        if pressure < 0.28:
            value += 0.28
        value += 0.055 * math.sin(i / 1.7)
        down.append(value)
    return {"up": up, "down": down, "combined": up + down}


def synthetic_smooth_series(n: int = 72) -> Dict[str, List[float]]:
    half = n // 2
    up = [0.82 - 0.12 * (i / max(half - 1, 1)) + 0.006 * math.sin(i / 5.0) for i in range(half)]
    down = [0.70 + 0.12 * (i / max(half - 1, 1)) + 0.006 * math.sin(i / 5.0) for i in range(half)]
    return {"up": up, "down": down, "combined": up + down}


def synthetic_validation() -> Dict[str, Dict[str, Any]]:
    fold = synthetic_fold_series()
    smooth = synthetic_smooth_series()
    return {
        "fold": {
            "discontinuity": sig_discontinuity(fold["up"]),
            "hysteresis": sig_hysteresis(fold["up"], fold["down"]),
            "critical_slowing": sig_critical_slowing(fold["combined"]),
        },
        "smooth": {
            "discontinuity": sig_discontinuity(smooth["up"]),
            "hysteresis": sig_hysteresis(smooth["up"], smooth["down"]),
            "critical_slowing": sig_critical_slowing(smooth["combined"]),
        },
    }
