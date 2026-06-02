from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence

from aptadynamik.observer.prama_components import (
    COHERENCE_VIABILITY_NOTE,
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
    "micro_drop",
    "micro_excess",
    "acople",
    "viability",
    "distance_to_threshold",
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


def summary_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "substrate_blind": result["substrate_blind"],
        "material_cost_measured": result["material_cost_measured"],
        "requires_exogenous_telemetry_for_material_cost": result["requires_exogenous_telemetry_for_material_cost"],
        "baseline_micro": result["baseline_micro"],
        "baseline_n_calib": result["baseline_n_calib"],
        "baseline_method": result["baseline_method"],
        "collapse_threshold": result["collapse_threshold"],
        "critical_margin": result["critical_margin"],
        "valid_turns": result["valid_turns"],
        "invalid_turns": result["invalid_turns"],
        "final_viability": result["final_viability"],
        "min_viability": result["min_viability"],
        "final_distance_to_threshold": result["final_distance_to_threshold"],
        "min_distance_to_threshold": result["min_distance_to_threshold"],
        "threshold_crossed": result["threshold_crossed"],
        "final_viability_status": result["viability_status"],
        "final_boundary_side": result["boundary_side"],
        "critical_turns": result["critical_turns"],
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


def calibration_sensitivity_payload(turns: Sequence[Dict[str, Any]], windows: Sequence[int]) -> Dict[str, Any]:
    per_window = []
    for window in windows:
        result = measure(turns, calib_window=window)
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
    return {
        "substrate_blind": True,
        "material_cost_measured": False,
        "requires_exogenous_telemetry_for_material_cost": True,
        "windows": list(windows),
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
        "trajectory_assessment": trajectory_assessment(crossing_stability, boundary_consensus, has_near_threshold),
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


def run_calibration_sensitivity(raw_path: Path, output_dir: Path, windows: Sequence[int]) -> Dict[str, Any]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    turns = raw.get("turns")
    if not isinstance(turns, list):
        raise ValueError(f"{raw_path} missing list field 'turns'")
    payload = calibration_sensitivity_payload(turns, windows)
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


def report_text(result: Dict[str, Any], raw_path: Path) -> str:
    summary = summary_payload(result)
    fallback_note = ""
    if result.get("baseline_warning"):
        fallback_note = f"\n- warning: {result['baseline_warning']}"
    return "\n".join(
        [
            "# PRAMA Components Report",
            "",
            "## Trajectory Viability",
            "",
            f"- final_viability: `{summary['final_viability']}`",
            f"- min_viability: `{summary['min_viability']}`",
            f"- final_distance_to_threshold: `{summary['final_distance_to_threshold']}`",
            f"- min_distance_to_threshold: `{summary['min_distance_to_threshold']}`",
            f"- threshold_crossed: `{summary['threshold_crossed']}`",
            f"- final_viability_status: `{summary['final_viability_status']}`",
            f"- final_boundary_side: `{summary['final_boundary_side']}`",
            f"- critical_turns: `{summary['critical_turns']}`",
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
            "acople = min(micro_health, macro_health)",
            "viability = acople",
            "distance_to_threshold = viability - collapse_threshold",
            f"micro_scale_definition: {result['micro_scale_definition']}",
            COHERENCE_VIABILITY_NOTE,
            "",
            "## Threshold",
            "",
            f"- collapse_threshold: `{summary['collapse_threshold']}`",
            f"- critical_margin: `{summary['critical_margin']}`",
            f"- threshold_crossed: `{summary['threshold_crossed']}`",
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


def run_from_raw(raw_path: Path, output_dir: Path, calib_window: int | None) -> Dict[str, Any]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    turns = raw.get("turns")
    if not isinstance(turns, list):
        raise ValueError(f"{raw_path} missing list field 'turns'")
    result = measure(turns, calib_window=calib_window)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_turns_csv(output_dir / "components_turns.csv", result["turns"])
    (output_dir / "components_summary.json").write_text(
        json.dumps(summary_payload(result), indent=2),
        encoding="utf-8",
    )
    (output_dir / "components_report.md").write_text(report_text(result, raw_path), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run canonical PRAMA generative-structure components.")
    parser.add_argument("--from-raw", required=True, help="PRAMA Monitor raw.json path.")
    parser.add_argument("--output-dir", required=True, help="Directory for components report artifacts.")
    parser.add_argument("--calib-window", type=int, default=None, help="Number of initial turns for frozen neutral baseline.")
    parser.add_argument("--windows", nargs="+", type=int, default=None, help="Calibration windows for sensitivity report.")
    args = parser.parse_args()
    windows = args.windows or [3, 5, 7]
    calib_window = args.calib_window if args.calib_window is not None else windows[0]
    result = run_from_raw(Path(args.from_raw), Path(args.output_dir), calib_window)
    sensitivity = run_calibration_sensitivity(Path(args.from_raw), Path(args.output_dir), windows)
    print(f"final_viability: {result['final_viability']}")
    print(f"threshold_crossed: {result['threshold_crossed']}")
    print(f"final_boundary_side: {result['boundary_side']}")
    print(f"final_viability_status: {result['viability_status']}")
    print(f"crossing_stability: {sensitivity['crossing_stability']}")
    print(f"trajectory_assessment: {sensitivity['trajectory_assessment']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
