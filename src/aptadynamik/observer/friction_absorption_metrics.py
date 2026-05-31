from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional


DEFAULT_THRESHOLDS = {
    "commitment_near_zero": 0.05,
    "commitment_nonzero": 0.15,
    "recombination_high": 0.60,
    "recombination_low": 0.45,
    "surprise_low": 0.35,
    "surprise_high": 0.60,
    "elaboration_increased": 1.0,
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    left = _tokens(a)
    right = _tokens(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def compute_commitment_shift(before: str, after: str) -> float:
    """Placeholder delta_C: lexical distance from tracked commitment to response."""
    return 1.0 - _jaccard(before, after)


def compute_recombination(response: str, history: Iterable[str]) -> float:
    """Placeholder R: maximum lexical reuse/recombination against prior context."""
    history_items = [item for item in history if item]
    if not history_items:
        return 0.0
    return max(_jaccard(response, item) for item in history_items)


def compute_surprise_from_prama_turn(turn: Dict[str, Any]) -> float:
    summary = turn.get("summary")
    if isinstance(summary, dict):
        for key in ("avg_entropy_norm", "entropy_range", "max_entropy_range"):
            if key in summary:
                return float(summary[key])

    windows = turn.get("windows")
    if isinstance(windows, list) and windows:
        values = []
        for window in windows:
            if not isinstance(window, dict):
                continue
            if "entropy_norm" in window:
                values.append(float(window["entropy_norm"]))
            elif "entropy_range" in window:
                values.append(float(window["entropy_range"]))
        if values:
            return sum(values) / len(values)
    return 0.0


def compute_elaboration(response: str) -> float:
    token_count = len(re.findall(r"\S+", response))
    return float(token_count if token_count else len(response))


def classify_absorption_or_friction(
    delta_C: float,
    R: float,
    S: float,
    E: float,
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    absorption = (
        abs(delta_C) <= limits["commitment_near_zero"]
        and R >= limits["recombination_high"]
        and S <= limits["surprise_low"]
        and E >= limits["elaboration_increased"]
    )
    if absorption:
        return "absorption"

    friction = (
        abs(delta_C) >= limits["commitment_nonzero"]
        and R <= limits["recombination_low"]
        and S >= limits["surprise_high"]
    )
    if friction:
        return "friction"

    if abs(delta_C) <= limits["commitment_near_zero"] and S <= limits["surprise_low"]:
        return "neutral"

    return "ambiguous"
