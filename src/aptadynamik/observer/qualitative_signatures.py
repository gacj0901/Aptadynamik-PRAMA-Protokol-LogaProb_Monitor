from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence


CONSTITUTIVE_KEYWORDS = (
    "must",
    "cannot",
    "contradict",
    "constraint",
    "role",
    "identity",
    "system",
    "ignore",
    "rules",
    "always",
    "never",
    "exactly",
)


@dataclass
class SignatureResult:
    signature: str
    triggered: bool
    score: float
    threshold: float
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "triggered": self.triggered,
            "score": round(self.score, 6),
            "threshold": round(self.threshold, 6),
            "detail": self.detail,
        }


def viability_from_window(window: Dict[str, Any]) -> float:
    return float(window.get("rigidity", 0.0)) - float(window.get("uncertainty", 0.0))


def load_session_raw(path: str | Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def session_to_observed_series(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for turn in session.get("turns", []):
        user_message = turn.get("user_message", "")
        target = classify_prompt_target(user_message)
        turn_index = int(turn.get("turn_index", len(rows)))
        for window in turn.get("windows", []):
            row = {
                "session_id": session.get("session_id", ""),
                "model": session.get("model", ""),
                "turn_index": turn_index,
                "window_index": int(window.get("window_index", len(rows))),
                "pressure": pressure_proxy(user_message, turn_index),
                "target": target,
                "rigidity": float(window.get("rigidity", 0.0)),
                "uncertainty": float(window.get("uncertainty", 0.0)),
                "viability": viability_from_window(window),
            }
            rows.append(row)
    return rows


def pressure_proxy(user_message: str, turn_index: int) -> float:
    text = user_message.lower()
    keyword_load = sum(1 for word in CONSTITUTIVE_KEYWORDS if word in text)
    return min(1.0, 0.1 * turn_index + 0.2 * keyword_load)


def classify_prompt_target(user_message: str) -> str:
    text = user_message.lower()
    if any(word in text for word in CONSTITUTIVE_KEYWORDS):
        return "constitutive"
    return "peripheral"


def adjacent_jumps(values: Sequence[float]) -> List[float]:
    return [abs(b - a) for a, b in zip(values, values[1:])]


def detect_discontinuity(values: Sequence[float], threshold: float = 0.35) -> SignatureResult:
    jumps = adjacent_jumps(values)
    score = max(jumps) if jumps else 0.0
    return SignatureResult(
        signature="discontinuity",
        triggered=score >= threshold,
        score=score,
        threshold=threshold,
        detail="maximum adjacent viability/rigidity jump",
    )


def detect_hysteresis(
    pressures: Sequence[float],
    responses: Sequence[float],
    threshold: float = 0.05,
) -> SignatureResult:
    if len(pressures) < 4 or len(pressures) != len(responses):
        return SignatureResult("hysteresis", False, 0.0, threshold, "insufficient pressure sweep")

    midpoint = len(pressures) // 2
    up = list(zip(pressures[:midpoint], responses[:midpoint]))
    down = list(zip(pressures[midpoint:], responses[midpoint:]))
    if not up or not down:
        return SignatureResult("hysteresis", False, 0.0, threshold, "missing up/down branches")
    if up[-1][0] <= up[0][0] or down[-1][0] >= down[0][0]:
        return SignatureResult("hysteresis", False, 0.0, threshold, "requires rising and falling pressure branches")

    up_mid = mean([r for p, r in up if p >= 0.35]) if any(p >= 0.35 for p, _ in up) else mean(r for _, r in up)
    down_mid = mean([r for p, r in down if p >= 0.35]) if any(p >= 0.35 for p, _ in down) else mean(r for _, r in down)
    score = abs(up_mid - down_mid)
    return SignatureResult(
        signature="hysteresis",
        triggered=score >= threshold,
        score=score,
        threshold=threshold,
        detail="branch separation between rising and falling pressure sweeps",
    )


def lag1_autocorrelation(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    xs = list(values[:-1])
    ys = list(values[1:])
    mx = mean(xs)
    my = mean(ys)
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def detect_critical_slowing(values: Sequence[float], threshold: float = 0.08) -> SignatureResult:
    if len(values) < 8:
        return SignatureResult("critical_slowing", False, 0.0, threshold, "insufficient series length")
    midpoint = len(values) // 2
    early = values[:midpoint]
    late = values[midpoint:]
    autocorr_gain = lag1_autocorrelation(late) - lag1_autocorrelation(early)
    variance_gain = variance(late) - variance(early)
    drops = [(idx, values[idx] - values[idx + 1]) for idx in range(len(values) - 1)]
    drop_idx, max_drop = max(drops, key=lambda item: item[1])
    recovery_score = 0.0
    if max_drop > 0.25:
        pre_value = values[drop_idx]
        half_recovery = values[drop_idx + 1] + max_drop * 0.5
        recovery_steps = 0
        for value in values[drop_idx + 1 :]:
            recovery_steps += 1
            if value >= half_recovery or value >= pre_value:
                break
        recovery_score = recovery_steps / len(values)
    score = max(autocorr_gain, 0.0) + max(variance_gain, 0.0) + recovery_score
    return SignatureResult(
        signature="critical_slowing",
        triggered=score >= threshold,
        score=score,
        threshold=threshold,
        detail="late autocorrelation/variance gain",
    )


def detect_structural_target_effect(rows: Sequence[Dict[str, Any]], threshold: float = 0.10) -> SignatureResult:
    peripheral = [float(row["viability"]) for row in rows if row.get("target") == "peripheral"]
    constitutive = [float(row["viability"]) for row in rows if row.get("target") == "constitutive"]
    if not peripheral or not constitutive:
        return SignatureResult(
            "structural_target_effect",
            False,
            0.0,
            threshold,
            "requires both peripheral and constitutive prompts",
        )
    score = mean(peripheral) - mean(constitutive)
    return SignatureResult(
        signature="structural_target_effect",
        triggered=score >= threshold,
        score=score,
        threshold=threshold,
        detail="peripheral viability minus constitutive viability",
    )


def evaluate_series(rows: Sequence[Dict[str, Any]], value_key: str = "viability") -> List[SignatureResult]:
    values = [float(row.get(value_key, 0.0)) for row in rows]
    pressures = [float(row.get("pressure", idx / max(len(rows) - 1, 1))) for idx, row in enumerate(rows)]
    return [
        detect_discontinuity(values),
        detect_hysteresis(pressures, values),
        detect_critical_slowing(values),
        detect_structural_target_effect(rows),
    ]


def smooth_synthetic_rows(n: int = 60) -> List[Dict[str, Any]]:
    rows = []
    for i in range(n):
        pressure = i / max(n - 1, 1)
        viability = 0.78 - 0.18 * pressure + 0.01 * math.sin(i / 5.0)
        rows.append(
            {
                "session_id": "synthetic_smooth",
                "model": "synthetic_smooth",
                "turn_index": i,
                "window_index": i,
                "pressure": pressure,
                "target": "peripheral",
                "rigidity": viability + 0.1,
                "uncertainty": 0.1,
                "viability": viability,
            }
        )
    return rows


def fold_synthetic_rows(n: int = 80) -> List[Dict[str, Any]]:
    rows = []
    half = n // 2
    for i in range(n):
        if i < half:
            pressure = i / max(half - 1, 1)
            viability = 0.86 - 0.18 * pressure
            if pressure > 0.62:
                viability -= 0.45
            target = "peripheral" if i % 5 else "constitutive"
        else:
            pressure = 1.0 - ((i - half) / max(half - 1, 1))
            viability = 0.36 + 0.08 * pressure
            if pressure < 0.32:
                viability += 0.28
            target = "constitutive" if i % 4 == 0 else "peripheral"
        viability += 0.08 * math.sin(i / 2.0)
        rows.append(
            {
                "session_id": "synthetic_fold",
                "model": "synthetic_fold",
                "turn_index": i,
                "window_index": i,
                "pressure": pressure,
                "target": target,
                "rigidity": viability + 0.12,
                "uncertainty": 0.12,
                "viability": viability,
            }
        )
    return rows


def synthetic_validation() -> Dict[str, List[SignatureResult]]:
    return {
        "fold": evaluate_series(fold_synthetic_rows()),
        "smooth": evaluate_series(smooth_synthetic_rows()),
    }


def load_sessions_from_path(path: str | Path) -> List[Dict[str, Any]]:
    root = Path(path)
    if root.is_file():
        files = [root]
    else:
        files = sorted(root.rglob("session_*_raw.json"))
    return [load_session_raw(file) for file in files]


def rows_from_sessions(sessions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for session in sessions:
        rows.extend(session_to_observed_series(session))
    return rows
