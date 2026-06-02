from __future__ import annotations

import math
from statistics import mean
from typing import Any, Dict, List, Sequence


BASELINE_WARNING = (
    "r0,u0 are an operational viable-point assumption estimated from this model/session, "
    "not an ontological truth."
)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def viability_legacy(rigidity: float, uncertainty: float) -> float:
    return float(rigidity) - float(uncertainty)


def viability_corrected(
    rigidity: float,
    uncertainty: float,
    r0: float,
    u0: float,
    scale: float = 0.35,
) -> float:
    """Non-monotone geometry proxy anchored to an operational model baseline.

    r0 and u0 are treated as a viable operating point estimated from baseline
    turns. They are methodological assumptions, not ontological truths.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    rigidity_penalty = abs(float(rigidity) - float(r0))
    uncertainty_penalty = max(0.0, float(uncertainty) - float(u0))
    return clamp01(1.0 - ((rigidity_penalty + uncertainty_penalty) / scale))


def _summary(turn: Dict[str, Any]) -> Dict[str, Any]:
    summary = turn.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"turn {turn.get('turn_index', '<unknown>')} missing required summary mapping")
    return summary


def _turn_regime_labels(turn: Dict[str, Any]) -> List[str]:
    labels = [
        turn.get("regime"),
        turn.get("final_regime"),
        turn.get("condition"),
        turn.get("phase"),
        turn.get("perturbation_type"),
    ]
    summary = turn.get("summary")
    if isinstance(summary, dict):
        labels.extend([summary.get("regime"), summary.get("final_regime"), summary.get("condition")])
    return [str(label) for label in labels if label is not None]


def _baseline_indices(turns: Sequence[Dict[str, Any]]) -> tuple[List[int], str]:
    baseline_labels = {"R0", "R1", "baseline", "control", "control_neutral"}
    labeled = [
        idx
        for idx, turn in enumerate(turns)
        if any(label in baseline_labels for label in _turn_regime_labels(turn))
    ]
    if labeled:
        return labeled, "labeled_R0_R1_or_control"
    cutoff = max(1, math.ceil(len(turns) * 0.20))
    return list(range(cutoff)), "first_20_percent_fallback"


def estimate_viability_baseline(turns: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not turns:
        return {
            "r0": 0.0,
            "u0": 0.0,
            "method": "empty_session_fallback",
            "warning": BASELINE_WARNING,
        }
    indices, method = _baseline_indices(turns)
    rigidities = []
    uncertainties = []
    for idx in indices:
        summary = _summary(turns[idx])
        rigidities.append(float(summary.get("avg_rigidity", 0.0)))
        uncertainties.append(float(summary.get("avg_uncertainty", 0.0)))
    return {
        "r0": mean(rigidities) if rigidities else 0.0,
        "u0": mean(uncertainties) if uncertainties else 0.0,
        "method": method,
        "warning": BASELINE_WARNING,
    }


def corrected_viability_rows(
    turns: Sequence[Dict[str, Any]],
    scale: float = 0.35,
) -> List[Dict[str, Any]]:
    baseline = estimate_viability_baseline(turns)
    rows = []
    for turn in turns:
        summary = _summary(turn)
        rigidity = float(summary.get("avg_rigidity", 0.0))
        uncertainty = float(summary.get("avg_uncertainty", 0.0))
        legacy = viability_legacy(rigidity, uncertainty)
        corrected = viability_corrected(rigidity, uncertainty, baseline["r0"], baseline["u0"], scale=scale)
        rows.append(
            {
                "turn_index": int(turn.get("turn_index", len(rows))),
                "viability": legacy,
                "viability_legacy": legacy,
                "viability_corrected": corrected,
                "corrected_fatigue": 1.0 - corrected,
                "avg_rigidity": rigidity,
                "avg_uncertainty": uncertainty,
                "baseline_r0": baseline["r0"],
                "baseline_u0": baseline["u0"],
                "baseline_method": baseline["method"],
                "baseline_warning": baseline["warning"],
                "viability_scale": scale,
            }
        )
    return rows
