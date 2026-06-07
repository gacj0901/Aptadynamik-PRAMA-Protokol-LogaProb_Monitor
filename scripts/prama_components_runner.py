from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Dict, Sequence

from aptadynamik.prama_components import (
    COHERENCE_VIABILITY_NOTE,
    DEFAULT_LAMBDA0,
    DEFAULT_MEMORY_BETA,
    DEFAULT_THETA0,
    ECHO_NOTE,
    SUBSTRATE_BLIND_WARNING,
    measure,
)


TURN_FIELDS = [
    "turn_index",
    "valid_token_count",
    "logprob_valid",
    "insufficient_data",
    "micro_raw",
    "micro_health",
    "macro_health",
    "activity_raw",
    "activity_structural",
    "activity_effective",
    "activity",
    "micro_drop",
    "micro_excess",
    "acople",
    "acople_effective",
    "delta_instant",
    "xi_accumulated",
    "xi_norm",
    "lambda_remaining",
    "theta_dynamic",
    "viability_margin",
    "accumulated_viability_margin",
    "instant_viability_margin",
    "instant_threshold_crossed",
    "instant_recovered",
    "compression_gap",
    "viability",
    "delta",
    "xi",
    "lam",
    "theta",
    "distance_to_threshold",
    "xi_exceeds_theta",
    "threshold_crossed",
    "viability_status",
    "boundary_pressure",
    "boundary_side",
]


def write_turns_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TURN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in TURN_FIELDS})


def summary_payload(result: Dict[str, Any], delta_ref: float | None = None) -> Dict[str, Any]:
    return {
        "substrate_blind": result["substrate_blind"],
        "material_cost_measured": result["material_cost_measured"],
        "requires_exogenous_telemetry_for_material_cost": result["requires_exogenous_telemetry_for_material_cost"],
        "baseline_micro": result["baseline_micro"],
        "baseline_n_calib": result["baseline_n_calib"],
        "baseline_method": result["baseline_method"],
        "theta0": result["theta0"],
        "lambda0": result["lambda0"],
        "memory_beta": result["memory_beta"],
        "delta_ref": delta_ref,
        "delta_ref_note": "delta_ref is accepted for interface compatibility; it does not affect the current v0.2.1 core.",
        "collapse_threshold": result["collapse_threshold"],
        "critical_margin": result["critical_margin"],
        "valid_turns": result["valid_turns"],
        "invalid_turns": result["invalid_turns"],
        "final_viability": result["final_viability"],
        "min_viability": result["min_viability"],
        "final_viability_margin": result["final_viability"],
        "min_viability_margin": result["min_viability"],
        "final_accumulated_viability_margin": result["final_accumulated_viability_margin"],
        "final_instant_viability_margin": result["final_instant_viability_margin"],
        "compression_gap": None,
        "final_distance_to_threshold": result["final_distance_to_threshold"],
        "min_distance_to_threshold": result["min_distance_to_threshold"],
        "threshold_crossed": result["threshold_crossed"],
        "final_threshold_crossed": result["final_threshold_crossed"],
        "xi_exceeds_theta": result["xi_exceeds_theta"],
        "final_xi_exceeds_theta": result["final_xi_exceeds_theta"],
        "final_instant_threshold_crossed": result["final_instant_threshold_crossed"],
        "final_instant_recovered": result["final_instant_recovered"],
        "recovered_finally": result["recovered_finally"],
        "relapsed_after_recovery": result["relapsed_after_recovery"],
        "pulsation_subtype": result["pulsation_subtype"],
        "final_viability_status": result["viability_status"],
        "final_boundary_side": result["boundary_side"],
        "critical_turns": result["critical_turns"],
        "regime_label": result["regime_label"],
        "regime_description": result["regime_description"],
        "recovery_observed": result["recovery_observed"],
        "first_crossing_turn": result["first_crossing_turn"],
        "threshold_crossing_ratio": result["threshold_crossing_ratio"],
        "persistent_crossing_ratio": result["persistent_crossing_ratio"],
        "post_crossing_recovery_turns": result["post_crossing_recovery_turns"],
        "trajectory_assessment": result["trajectory_assessment"],
    }


def consensus(values: Sequence[Any]) -> Any:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def trajectory_assessment(crossing_stability: str, boundary_side: str | None, has_near_threshold: bool) -> str:
    side = boundary_side or "UNRESOLVED"
    if crossing_stability == "SENSITIVE":
        return f"BORDERLINE_CRITICAL_{side}"
    if crossing_stability == "STABLE_CROSSED":
        return f"THRESHOLD_CROSSED_{side}"
    if crossing_stability == "STABLE_NOT_CROSSED" and has_near_threshold:
        return f"NEAR_THRESHOLD_{side}"
    return f"VIABLE_{side}"


def clear_majority(values: Sequence[Any]) -> Any:
    if not values:
        return None
    value, count = Counter(values).most_common(1)[0]
    return value if count > (len(values) / 2) else None


def calibration_sensitivity_payload(
    turns: Sequence[Dict[str, Any]],
    windows: Sequence[int],
    theta0: float = DEFAULT_THETA0,
    lambda0: float = DEFAULT_LAMBDA0,
    memory_beta: float = DEFAULT_MEMORY_BETA,
) -> Dict[str, Any]:
    per_window = []
    for window in windows:
        result = measure(turns, calib_window=window, theta0=theta0, lambda0=lambda0, memory_beta=memory_beta)
        summary = summary_payload(result)
        per_window.append(
            {
                "calib_window": window,
                "baseline_micro": summary["baseline_micro"],
                "final_viability": summary["final_viability"],
                "final_distance_to_threshold": summary["final_distance_to_threshold"],
                "threshold_crossed": summary["threshold_crossed"],
                "final_viability_status": summary["final_viability_status"],
                "final_boundary_side": summary["final_boundary_side"],
                "critical_turns": summary["critical_turns"],
                "regime_label": summary["regime_label"],
                "trajectory_assessment": summary["trajectory_assessment"],
            }
        )

    final_viabilities = [row["final_viability"] for row in per_window if row["final_viability"] is not None]
    threshold_count = sum(1 for row in per_window if row["threshold_crossed"])
    threshold_rate = threshold_count / len(per_window) if per_window else 0.0
    boundary_counts = dict(Counter(row["final_boundary_side"] for row in per_window))
    boundary_consensus = consensus([row["final_boundary_side"] for row in per_window])
    critical_turn_values = [turn for row in per_window for turn in row["critical_turns"]]
    critical_turn_counts = dict(Counter(critical_turn_values))
    critical_turn_consensus = consensus(critical_turn_values)

    if threshold_rate == 1.0:
        crossing_stability = "STABLE_CROSSED"
    elif threshold_rate == 0.0:
        crossing_stability = "STABLE_NOT_CROSSED"
    else:
        crossing_stability = "SENSITIVE"

    has_near_threshold = any(row["final_viability_status"] == "NEAR_THRESHOLD" for row in per_window)
    regime_consensus = consensus([row["regime_label"] for row in per_window])
    trajectory_consensus = consensus([row["trajectory_assessment"] for row in per_window])
    return {
        "substrate_blind": True,
        "material_cost_measured": False,
        "requires_exogenous_telemetry_for_material_cost": True,
        "windows": list(windows),
        "theta0": theta0,
        "lambda0": lambda0,
        "memory_beta": memory_beta,
        "per_window": per_window,
        "final_viability_min": min(final_viabilities) if final_viabilities else None,
        "final_viability_max": max(final_viabilities) if final_viabilities else None,
        "final_viability_range": (max(final_viabilities) - min(final_viabilities)) if final_viabilities else None,
        "threshold_crossed_count": threshold_count,
        "threshold_crossed_rate": threshold_rate,
        "boundary_side_counts": boundary_counts,
        "boundary_side_consensus": boundary_consensus,
        "critical_turn_counts": critical_turn_counts,
        "critical_turn_consensus": critical_turn_consensus,
        "crossing_stability": crossing_stability,
        "regime_label_consensus": regime_consensus,
        "trajectory_assessment": trajectory_consensus
        or trajectory_assessment(crossing_stability, boundary_consensus, has_near_threshold),
    }


def calibration_sensitivity_report(payload: Dict[str, Any], raw_path: Path) -> str:
    per_window_lines = [
        (
            f"- window `{row['calib_window']}`: baseline_micro `{row['baseline_micro']}`, "
            f"final_viability `{row['final_viability']}`, "
            f"final_distance_to_threshold `{row['final_distance_to_threshold']}`, "
            f"threshold_crossed `{row['threshold_crossed']}`, "
            f"final_viability_status `{row['final_viability_status']}`, "
            f"final_boundary_side `{row['final_boundary_side']}`, "
            f"critical_turns `{row['critical_turns']}`"
        )
        for row in payload["per_window"]
    ]
    return "\n".join(
        [
            "# PRAMA Components Calibration Sensitivity",
            "",
            "## Scope",
            "",
            f"- source raw: `{raw_path}`",
            "- substrate_blind: `true`",
            "- material_cost_measured: `false`",
            "- requires_exogenous_telemetry_for_material_cost: `true`",
            "",
            "This sensitivity report reruns the same trajectory with multiple calibration windows. It does not change the underlying component calculations.",
            "",
            "Sensitivity does not change the detected exhaustion boundary when boundary_side_consensus is stable. It distinguishes qualitative robustness of the exhaustion boundary from binary sensitivity of the exact threshold crossing.",
            "",
            "## Aggregate",
            "",
            f"- windows: `{payload['windows']}`",
            f"- final_viability_min: `{payload['final_viability_min']}`",
            f"- final_viability_max: `{payload['final_viability_max']}`",
            f"- final_viability_range: `{payload['final_viability_range']}`",
            f"- threshold_crossed_count: `{payload['threshold_crossed_count']}`",
            f"- threshold_crossed_rate: `{payload['threshold_crossed_rate']}`",
            f"- boundary_side_counts: `{payload['boundary_side_counts']}`",
            f"- boundary_side_consensus: `{payload['boundary_side_consensus']}`",
            f"- critical_turn_counts: `{payload['critical_turn_counts']}`",
            f"- critical_turn_consensus: `{payload['critical_turn_consensus']}`",
            f"- crossing_stability: `{payload['crossing_stability']}`",
            f"- trajectory_assessment: `{payload['trajectory_assessment']}`",
            f"- regime_label_consensus: `{payload['regime_label_consensus']}`",
            "",
            "## Per Window",
            "",
            *per_window_lines,
            "",
            "## Methodological Note",
            "",
            "This report measures generative geometry derived from logprobs. It does not measure semantic truth, intention, or physical material cost.",
        ]
    )


def run_calibration_sensitivity(
    raw_path: Path,
    output_dir: Path,
    windows: Sequence[int],
    theta0: float = DEFAULT_THETA0,
    lambda0: float = DEFAULT_LAMBDA0,
    memory_beta: float = DEFAULT_MEMORY_BETA,
) -> Dict[str, Any]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    turns = raw.get("turns")
    if not isinstance(turns, list):
        raise ValueError(f"{raw_path} missing list field 'turns'")
    payload = calibration_sensitivity_payload(turns, windows, theta0=theta0, lambda0=lambda0, memory_beta=memory_beta)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "components_calibration_sensitivity.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "components_calibration_sensitivity.md").write_text(
        calibration_sensitivity_report(payload, raw_path),
        encoding="utf-8",
    )
    return payload


def parametric_sensitivity_payload(
    turns: Sequence[Dict[str, Any]],
    calib_window: int | None,
    theta0_grid: Sequence[float],
    lambda0_grid: Sequence[float],
    memory_beta_grid: Sequence[float],
) -> Dict[str, Any]:
    runs = []
    for theta0, lambda0, memory_beta in product(theta0_grid, lambda0_grid, memory_beta_grid):
        result = measure(
            turns,
            calib_window=calib_window,
            theta0=theta0,
            lambda0=lambda0,
            memory_beta=memory_beta,
        )
        runs.append(
            {
                "theta0": theta0,
                "lambda0": lambda0,
                "memory_beta": memory_beta,
                "final_viability": result["final_viability"],
                "min_viability": result["min_viability"],
                "threshold_crossed": result["threshold_crossed"],
                "first_crossing_turn": result["first_crossing_turn"],
                "recovery_observed": result["recovery_observed"],
                "regime_label": result["regime_label"],
                "trajectory_assessment": result["trajectory_assessment"],
                "final_boundary_side": result["boundary_side"],
                "threshold_crossing_ratio": result["threshold_crossing_ratio"],
                "persistent_crossing_ratio": result["persistent_crossing_ratio"],
            }
        )

    regime_labels = [row["regime_label"] for row in runs]
    trajectory_assessments = [row["trajectory_assessment"] for row in runs]
    first_crossing_turns = [
        row["first_crossing_turn"] for row in runs if row["first_crossing_turn"] is not None
    ]
    return {
        "substrate_blind": True,
        "material_cost_measured": False,
        "requires_exogenous_telemetry_for_material_cost": True,
        "theta0_grid": list(theta0_grid),
        "lambda0_grid": list(lambda0_grid),
        "memory_beta_grid": list(memory_beta_grid),
        "runs": runs,
        "regime_label_counts": dict(Counter(regime_labels)),
        "trajectory_assessment_counts": dict(Counter(trajectory_assessments)),
        "first_crossing_turns": first_crossing_turns,
        "robust_regime_label": clear_majority(regime_labels),
        "robust_trajectory_assessment": clear_majority(trajectory_assessments),
    }


def parametric_sensitivity_report(payload: Dict[str, Any], raw_path: Path) -> str:
    run_lines = [
        (
            f"- theta0 `{row['theta0']}`, lambda0 `{row['lambda0']}`, memory_beta `{row['memory_beta']}`: "
            f"regime `{row['regime_label']}`, assessment `{row['trajectory_assessment']}`, "
            f"threshold_crossed `{row['threshold_crossed']}`, first_crossing_turn `{row['first_crossing_turn']}`"
        )
        for row in payload["runs"]
    ]
    return "\n".join(
        [
            "# PRAMA Components Parametric Sensitivity",
            "",
            "## Scope",
            "",
            f"- source raw: `{raw_path}`",
            "- substrate_blind: `true`",
            "- material_cost_measured: `false`",
            "- requires_exogenous_telemetry_for_material_cost: `true`",
            "",
            "This report sweeps dynamic parameters over the same trajectory. It does not alter token parsing or logprob-derived component definitions.",
            "",
            "## Consensus",
            "",
            f"- regime_label_counts: `{payload['regime_label_counts']}`",
            f"- trajectory_assessment_counts: `{payload['trajectory_assessment_counts']}`",
            f"- first_crossing_turns: `{payload['first_crossing_turns']}`",
            f"- robust_regime_label: `{payload['robust_regime_label']}`",
            f"- robust_trajectory_assessment: `{payload['robust_trajectory_assessment']}`",
            "",
            "## Runs",
            "",
            *run_lines,
            "",
            "## Methodological Note",
            "",
            "Parametric sensitivity evaluates stability of the operational regime label under theta0, lambda0, and memory_beta choices.",
        ]
    )


def run_parametric_sensitivity(
    raw_path: Path,
    output_dir: Path,
    calib_window: int | None,
    theta0_grid: Sequence[float],
    lambda0_grid: Sequence[float],
    memory_beta_grid: Sequence[float],
) -> Dict[str, Any]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    turns = raw.get("turns")
    if not isinstance(turns, list):
        raise ValueError(f"{raw_path} missing list field 'turns'")
    payload = parametric_sensitivity_payload(
        turns,
        calib_window=calib_window,
        theta0_grid=theta0_grid,
        lambda0_grid=lambda0_grid,
        memory_beta_grid=memory_beta_grid,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "components_parametric_sensitivity.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "components_parametric_sensitivity.md").write_text(
        parametric_sensitivity_report(payload, raw_path),
        encoding="utf-8",
    )
    return payload


def report_text(
    result: Dict[str, Any],
    raw_path: Path,
    delta_ref: float | None = None,
    parametric_payload: Dict[str, Any] | None = None,
) -> str:
    summary = summary_payload(result, delta_ref=delta_ref)
    fallback_note = ""
    if result.get("baseline_warning"):
        fallback_note = f"\n- warning: {result['baseline_warning']}"
    if parametric_payload is None:
        parametric_lines = ["No parametric grid was requested."]
    else:
        parametric_lines = [
            "Parametric grid was requested.",
            "- generated: `components_parametric_sensitivity.json`",
            "- generated: `components_parametric_sensitivity.md`",
            f"- robust_regime_label: `{parametric_payload['robust_regime_label']}`",
            f"- robust_trajectory_assessment: `{parametric_payload['robust_trajectory_assessment']}`",
            f"- regime_label_counts: `{parametric_payload['regime_label_counts']}`",
            f"- trajectory_assessment_counts: `{parametric_payload['trajectory_assessment_counts']}`",
        ]
    return "\n".join(
        [
            "# PRAMA Components Report",
            "",
            "## Trajectory Viability",
            "",
            f"- final_viability: `{summary['final_viability']}`",
            f"- min_viability: `{summary['min_viability']}`",
            f"- final_accumulated_viability_margin: `{summary['final_accumulated_viability_margin']}`",
            f"- final_instant_viability_margin: `{summary['final_instant_viability_margin']}`",
            f"- final_distance_to_threshold: `{summary['final_distance_to_threshold']}`",
            f"- min_distance_to_threshold: `{summary['min_distance_to_threshold']}`",
            f"- threshold_crossed: `{summary['threshold_crossed']}`",
            f"- final_threshold_crossed: `{summary['final_threshold_crossed']}`",
            f"- final_viability_status: `{summary['final_viability_status']}`",
            f"- final_instant_threshold_crossed: `{summary['final_instant_threshold_crossed']}`",
            f"- final_instant_recovered: `{summary['final_instant_recovered']}`",
            f"- final_boundary_side: `{summary['final_boundary_side']}`",
            f"- critical_turns: `{summary['critical_turns']}`",
            f"- trajectory_assessment: `{summary['trajectory_assessment']}`",
            "",
            "## Dynamic Parameters",
            "",
            f"- theta0: `{summary['theta0']}`",
            f"- lambda0: `{summary['lambda0']}`",
            f"- memory_beta: `{summary['memory_beta']}`",
            f"- delta_ref: `{summary['delta_ref']}`",
            f"- delta_ref_note: {summary['delta_ref_note']}",
            "",
            "## Aptadynamic Regime",
            "",
            f"- regime_label: `{summary['regime_label']}`",
            f"- regime_description: `{summary['regime_description']}`",
            f"- recovery_observed: `{summary['recovery_observed']}`",
            f"- recovered_finally: `{summary['recovered_finally']}`",
            f"- relapsed_after_recovery: `{summary['relapsed_after_recovery']}`",
            f"- pulsation_subtype: `{summary['pulsation_subtype']}`",
            f"- first_crossing_turn: `{summary['first_crossing_turn']}`",
            f"- threshold_crossing_ratio: `{summary['threshold_crossing_ratio']}`",
            f"- persistent_crossing_ratio: `{summary['persistent_crossing_ratio']}`",
            f"- post_crossing_recovery_turns: `{summary['post_crossing_recovery_turns']}`",
            "",
            "Threshold crossing is interpreted as loss of point-regime viability; terminal collapse requires persistent instant_threshold_crossed without instant recovery.",
            "",
            "## Measurement Scope",
            "",
            "Layer 1 — Generative structure: logprobs, surprise, micro amplitude, macro continuity, acople, viability, structural debt.",
            "Layer 2 — Functional coupling: external judge, reference tracking, task fulfillment, semantic propagation.",
            "Layer 3 — Material cost: energy, GPU, memory, latency, heat, cooling, infrastructure telemetry.",
            "Layer 3 cannot be inferred from Layer 1.",
            "",
            "PRAMA Components is not a failure classifier. It is a trajectory dynamic-viability meter.",
            "Its object is acople between viable micro-amplitude and macro continuity.",
            "Viability expresses distance to the threshold where the trajectory stops sustaining continuity and amplitude at the same time.",
            "",
            "This canonical report measures generative structure over token logprobs. It does not use semantic proxy heuristics, lexical novelty, verbosity, forbidden-word counts, or textual self-similarity to decide viability.",
            "",
            "## Substrate-Blind Warning",
            "",
            SUBSTRATE_BLIND_WARNING,
            "This instrument measures generative geometry derived from logprobs. It does not measure semantic truth, intention, physical material cost, energy, GPU use, temperature, memory, or latency unless external telemetry is integrated.",
            "",
            "## Baseline",
            "",
            f"- source raw: `{raw_path}`",
            f"- baseline_micro: `{summary['baseline_micro']}`",
            f"- baseline_n_calib: `{summary['baseline_n_calib']}`",
            f"- baseline_method: `{summary['baseline_method']}`{fallback_note}",
            f"- valid turns: `{summary['valid_turns']}`",
            f"- invalid turns: `{summary['invalid_turns']}`",
            "- invalid logprob values, including sentinel-like values such as 9999, are ignored and never treated as collapse evidence.",
            "",
            "## Viability Definition",
            "",
            "acople = acople_raw = min(micro_health, macro_health)",
            "delta_instant = (1.0 - acople) * activity",
            "acople_effective = 1.0 - delta_instant",
            "viability_margin = theta_dynamic - xi_norm",
            "viability is retained as an output alias for viability_margin.",
            "distance_to_threshold is retained as the current viability_margin.",
            f"micro_scale_definition: {result['micro_scale_definition']}",
            COHERENCE_VIABILITY_NOTE,
            "",
            "## Threshold",
            "",
            "collapse_xi_norm(theta0, lambda0) = (theta0 * lambda0) / (1 + theta0 * lambda0)",
            "With lambda0 = 1, this reduces to theta0 / (1 + theta0).",
            f"- theta0: `{summary['theta0']}`",
            f"- lambda0: `{summary['lambda0']}`",
            f"- collapse_threshold: `{summary['collapse_threshold']}`",
            f"- critical_margin: `{summary['critical_margin']}`",
            f"- threshold_crossed: `{summary['threshold_crossed']}`",
            f"- xi_exceeds_theta: `{summary['xi_exceeds_theta']}`",
            f"- final_xi_exceeds_theta: `{summary['final_xi_exceeds_theta']}`",
            "",
            "## Parametric Sensitivity",
            "",
            *parametric_lines,
            "",
            "## Compression Gap",
            "",
            "compression_gap is reserved for the gap between task-required support and response-expressed support.",
            "It is currently `null` because no exogenous task-support channel exists in this component.",
            "It is not viability_margin.",
            "",
            "## Exhaustion Boundary",
            "",
            f"- final_boundary_side: `{summary['final_boundary_side']}`",
            "",
            "boundary_pressure is proximity to the viability threshold inside the critical band: 0.0 is far from the threshold; 1.0 is at or below the threshold.",
            "boundary_side is secondary. CONDENSATION is driven by micro_drop, DISSOLUTION by micro_excess, and DECOUPLING by macro_health loss.",
            "",
            "## Diagnostic Note",
            "",
            "Legacy diagnostic scores may still exist in memory for temporary compatibility, but the canonical report is governed by viability, distance_to_threshold, threshold_crossed, and boundary_side.",
            ECHO_NOTE,
        ]
    )


def run_from_raw(
    raw_path: Path,
    output_dir: Path,
    calib_window: int | None,
    theta0: float = DEFAULT_THETA0,
    lambda0: float = DEFAULT_LAMBDA0,
    memory_beta: float = DEFAULT_MEMORY_BETA,
    delta_ref: float | None = None,
    parametric_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    turns = raw.get("turns")
    if not isinstance(turns, list):
        raise ValueError(f"{raw_path} missing list field 'turns'")
    result = measure(turns, calib_window=calib_window, theta0=theta0, lambda0=lambda0, memory_beta=memory_beta)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_turns_csv(output_dir / "components_turns.csv", result["turns"])
    (output_dir / "components_summary.json").write_text(
        json.dumps(summary_payload(result, delta_ref=delta_ref), indent=2),
        encoding="utf-8",
    )
    (output_dir / "components_report.md").write_text(
        report_text(result, raw_path, delta_ref=delta_ref, parametric_payload=parametric_payload),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run canonical PRAMA generative-structure components.")
    parser.add_argument("--from-raw", required=True, help="PRAMA Monitor raw.json path.")
    parser.add_argument("--output-dir", required=True, help="Directory for components report artifacts.")
    parser.add_argument("--calib-window", type=int, default=None, help="Number of initial turns for frozen neutral baseline.")
    parser.add_argument("--windows", nargs="+", type=int, default=None, help="Calibration windows for sensitivity report.")
    parser.add_argument("--theta0", type=float, default=DEFAULT_THETA0, help="Base dynamic threshold parameter.")
    parser.add_argument("--lambda0", type=float, default=DEFAULT_LAMBDA0, help="Initial remanent permissivity.")
    parser.add_argument("--memory-beta", type=float, default=DEFAULT_MEMORY_BETA, help="Fractional memory accumulation factor.")
    parser.add_argument("--delta-ref", type=float, default=None, help="Reserved compatibility parameter; currently does not affect v0.2.1 core.")
    parser.add_argument("--theta0-grid", nargs="+", type=float, default=None, help="Optional theta0 values for parametric sensitivity.")
    parser.add_argument("--lambda0-grid", nargs="+", type=float, default=None, help="Optional lambda0 values for parametric sensitivity.")
    parser.add_argument("--memory-beta-grid", nargs="+", type=float, default=None, help="Optional memory_beta values for parametric sensitivity.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    windows = args.windows or [3, 5, 7]
    calib_window = args.calib_window if args.calib_window is not None else windows[0]
    raw_path = Path(args.from_raw)
    output_dir = Path(args.output_dir)
    parametric = None
    if args.theta0_grid or args.lambda0_grid or args.memory_beta_grid:
        parametric = run_parametric_sensitivity(
            raw_path,
            output_dir,
            calib_window=calib_window,
            theta0_grid=args.theta0_grid or [args.theta0],
            lambda0_grid=args.lambda0_grid or [args.lambda0],
            memory_beta_grid=args.memory_beta_grid or [args.memory_beta],
        )
    result = run_from_raw(
        raw_path,
        output_dir,
        calib_window,
        theta0=args.theta0,
        lambda0=args.lambda0,
        memory_beta=args.memory_beta,
        delta_ref=args.delta_ref,
        parametric_payload=parametric,
    )
    sensitivity = run_calibration_sensitivity(
        raw_path,
        output_dir,
        windows,
        theta0=args.theta0,
        lambda0=args.lambda0,
        memory_beta=args.memory_beta,
    )
    print(f"final_viability: {result['final_viability']}")
    print(f"threshold_crossed: {result['threshold_crossed']}")
    print(f"final_boundary_side: {result['boundary_side']}")
    print(f"final_viability_status: {result['viability_status']}")
    print(f"regime_label: {result['regime_label']}")
    print(f"trajectory_assessment: {result['trajectory_assessment']}")
    if parametric is not None:
        print(f"parametric_robust_regime_label: {parametric['robust_regime_label']}")
        print(f"parametric_robust_trajectory_assessment: {parametric['robust_trajectory_assessment']}")
    print(f"crossing_stability: {sensitivity['crossing_stability']}")
    print(f"sensitivity_trajectory_assessment: {sensitivity['trajectory_assessment']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
