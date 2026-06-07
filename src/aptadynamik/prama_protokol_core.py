from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_THETA0 = 0.35
DEFAULT_LAMBDA0 = 1.0
DEFAULT_MEMORY_BETA = 0.65
DEFAULT_MIN_TURNS_FOR_REGIME = 3
DEFAULT_MIN_WINDOWS_FOR_REGIME = 12
DEFAULT_MIN_POST_CROSSING_UNITS = 2
ORGANIZED_STABILITY_REGIME = "II_ORGANIZED_STABILITY"
ORGANIZED_STABILITY_ASSESSMENT = "VIABLE_ORGANIZED_STABILITY"
LEGACY_REGIME_ALIASES = {
    "II_ORGANIZED_EQUILIBRIUM": ORGANIZED_STABILITY_REGIME,
}
LEGACY_ASSESSMENT_ALIASES = {
    "VIABLE_ORGANIZED_EQUILIBRIUM": ORGANIZED_STABILITY_ASSESSMENT,
}


@dataclass
class SessionReading:
    regime_label: str
    regime_description: str
    recovery_observed: bool
    first_crossing_turn: Optional[int]
    threshold_crossing_ratio: float
    persistent_crossing_ratio: float
    post_crossing_recovery_turns: List[int]
    local_threshold_cascade: bool
    crossing_index_scope: str
    first_crossing_window: Optional[int]


def normalize_regime_label(value: str | None) -> str | None:
    if value is None:
        return None
    return LEGACY_REGIME_ALIASES.get(value, value)


def normalize_trajectory_assessment(value: str | None) -> str | None:
    if value is None:
        return None
    return LEGACY_ASSESSMENT_ALIASES.get(value, value)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def collapse_xi_norm(theta0: float, lambda0: float = 1.0) -> float:
    """Return normalized accumulated stress at the initial dynamic threshold.

    General form: (theta0 * lambda0) / (1 + theta0 * lambda0).
    With lambda0 = 1, this reduces to theta0 / (1 + theta0).
    """
    product = max(0.0, float(theta0) * float(lambda0))
    return product / (1.0 + product)


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


def classify_regime(
    turns: Sequence[Dict[str, Any]],
    crossing_index_scope: str = "turn",
    min_turns_for_regime: int = DEFAULT_MIN_TURNS_FOR_REGIME,
    min_windows_for_regime: int = DEFAULT_MIN_WINDOWS_FOR_REGIME,
    min_post_crossing_units: int = DEFAULT_MIN_POST_CROSSING_UNITS,
) -> Dict[str, Any]:
    valid_turns = [turn for turn in turns if turn.get("logprob_valid")]
    if not valid_turns:
        return asdict(
            SessionReading(
                regime_label="CALIBRATING",
                regime_description="insufficient history for aptadynamic regime classification",
                recovery_observed=False,
                first_crossing_turn=None,
                threshold_crossing_ratio=0.0,
                persistent_crossing_ratio=0.0,
                post_crossing_recovery_turns=[],
                local_threshold_cascade=False,
                crossing_index_scope=crossing_index_scope,
                first_crossing_window=None,
            )
        )

    crossing_turns = [turn for turn in valid_turns if turn.get("threshold_crossed")]
    first_crossing_turn = crossing_turns[0]["turn_index"] if crossing_turns else None
    threshold_crossing_ratio = len(crossing_turns) / len(valid_turns)
    local_threshold_cascade = len(crossing_turns) >= min_post_crossing_units
    first_crossing_window = (
        int(first_crossing_turn)
        if crossing_index_scope == "token_window" and first_crossing_turn is not None
        else None
    )
    enough_history = (
        len(valid_turns) >= min_windows_for_regime
        if crossing_index_scope == "token_window"
        else len(valid_turns) >= min_turns_for_regime
    )
    post_crossing_units = (
        len([turn for turn in valid_turns if first_crossing_turn is not None and turn["turn_index"] >= first_crossing_turn])
        if first_crossing_turn is not None
        else 0
    )
    if not enough_history or (first_crossing_turn is not None and post_crossing_units < min_post_crossing_units):
        return asdict(
            SessionReading(
                regime_label="CALIBRATING",
                regime_description="insufficient history for aptadynamic regime classification",
                recovery_observed=False,
                first_crossing_turn=int(first_crossing_turn) if first_crossing_turn is not None else None,
                threshold_crossing_ratio=threshold_crossing_ratio,
                persistent_crossing_ratio=0.0,
                post_crossing_recovery_turns=[],
                local_threshold_cascade=local_threshold_cascade,
                crossing_index_scope=crossing_index_scope,
                first_crossing_window=first_crossing_window,
            )
        )

    final_instant_margin = valid_turns[-1].get("instant_viability_margin")
    avg_activity_effective = mean(float(turn.get("activity_effective") or 0.0) for turn in valid_turns)
    avg_acople_raw = mean(float(turn.get("acople") or 0.0) for turn in valid_turns)

    if first_crossing_turn is None:
        if len(valid_turns) >= 3 and avg_activity_effective < 0.05 and avg_acople_raw < 0.25:
            return asdict(
                SessionReading(
                    regime_label="I_SUBCRITICAL_DISSOLUTION",
                    regime_description="persistent low structural activity and low raw acople without formal over-threshold oscillation",
                    recovery_observed=False,
                    first_crossing_turn=None,
                    threshold_crossing_ratio=threshold_crossing_ratio,
                    persistent_crossing_ratio=0.0,
                    post_crossing_recovery_turns=[],
                    local_threshold_cascade=local_threshold_cascade,
                    crossing_index_scope=crossing_index_scope,
                    first_crossing_window=first_crossing_window,
                )
            )
        return asdict(
            SessionReading(
                regime_label=ORGANIZED_STABILITY_REGIME,
                regime_description="no formal threshold crossing; organized dynamic stability is conserved",
                recovery_observed=False,
                first_crossing_turn=None,
                threshold_crossing_ratio=threshold_crossing_ratio,
                persistent_crossing_ratio=0.0,
                post_crossing_recovery_turns=[],
                local_threshold_cascade=local_threshold_cascade,
                crossing_index_scope=crossing_index_scope,
                first_crossing_window=first_crossing_window,
            )
        )

    post_crossing = [turn for turn in valid_turns if turn["turn_index"] >= first_crossing_turn]
    post_recovery = [
        turn
        for turn in valid_turns
        if turn["turn_index"] > first_crossing_turn and bool(turn.get("instant_recovered"))
    ]
    post_crossing_recovery_turns = [int(turn["turn_index"]) for turn in post_recovery]
    recovery_observed = bool(post_crossing_recovery_turns)
    persistent_crossing_ratio = (
        sum(1 for turn in post_crossing if turn.get("instant_threshold_crossed")) / len(post_crossing)
        if post_crossing
        else 0.0
    )

    if recovery_observed:
        label = "III_STRUCTURAL_PULSATION"
        description = "threshold crossing followed by recovery; trajectory operates as bounded structural pulsation"
    elif (
        persistent_crossing_ratio >= 0.80
        and final_instant_margin is not None
        and float(final_instant_margin) <= 0.0
    ):
        label = "IV_ENTROPIC_COLLAPSE"
        description = "persistent threshold crossing without observed recovery; terminal drift is operationally indicated"
    else:
        label = "III_STRUCTURAL_PULSATION"
        description = "threshold crossing without confirmed terminal collapse"

    return asdict(
        SessionReading(
            regime_label=label,
            regime_description=description,
            recovery_observed=recovery_observed,
            first_crossing_turn=int(first_crossing_turn),
            threshold_crossing_ratio=threshold_crossing_ratio,
            persistent_crossing_ratio=persistent_crossing_ratio,
            post_crossing_recovery_turns=post_crossing_recovery_turns,
            local_threshold_cascade=local_threshold_cascade,
            crossing_index_scope=crossing_index_scope,
            first_crossing_window=first_crossing_window,
        )
    )


def trajectory_assessment_from_regime(regime_label: str) -> str:
    regime_label = normalize_regime_label(regime_label) or regime_label
    if regime_label == "CALIBRATING":
        return "INSUFFICIENT_HISTORY"
    if regime_label == "III_STRUCTURAL_PULSATION":
        return "THRESHOLD_CROSSED_STRUCTURAL_PULSATION"
    if regime_label == "IV_ENTROPIC_COLLAPSE":
        return "ENTROPIC_COLLAPSE"
    if regime_label == ORGANIZED_STABILITY_REGIME:
        return ORGANIZED_STABILITY_ASSESSMENT
    if regime_label == "I_SUBCRITICAL_DISSOLUTION":
        return "SUBCRITICAL_DISSOLUTION"
    return "UNRESOLVED_APTADYNAMIC_REGIME"


def pulsation_subtype_from_state(
    regime_label: str,
    recovery_observed: bool,
    final_instant_recovered: bool,
    final_instant_threshold_crossed: bool,
) -> Optional[str]:
    if regime_label != "III_STRUCTURAL_PULSATION":
        return None
    recovered_finally = bool(final_instant_recovered)
    relapsed_after_recovery = bool(recovery_observed and final_instant_threshold_crossed)
    if recovered_finally:
        return "RECOVERED_FINAL"
    if relapsed_after_recovery:
        return "RELAPSED_AFTER_RECOVERY"
    if recovery_observed:
        return "ACTIVE_PULSATION"
    return None
