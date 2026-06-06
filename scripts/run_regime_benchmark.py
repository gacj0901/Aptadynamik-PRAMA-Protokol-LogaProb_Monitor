from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from aptadynamik.prama_components import measure


METHODOLOGICAL_NOTE = (
    "Threshold crossing is a local viability event; regime classification "
    "requires sufficient trajectory history."
)


def synthetic_turn(index: int, logprobs: List[float]) -> Dict[str, Any]:
    return {
        "turn_index": index,
        "tokens": [{"token": f"t{index}_{token_index}", "top1_logprob": value} for token_index, value in enumerate(logprobs)],
    }


def short_calibrating_local_crossings() -> List[Dict[str, Any]]:
    return [synthetic_turn(0, [-0.8, -1.2])] + [
        synthetic_turn(index, [-0.1, -2.1]) for index in range(1, 5)
    ]


def organized_viability() -> List[Dict[str, Any]]:
    return [synthetic_turn(index, [-0.8, -1.2]) for index in range(12)]


def structural_pulsation() -> List[Dict[str, Any]]:
    return (
        [synthetic_turn(0, [-0.8, -1.2])]
        + [synthetic_turn(index, [-0.1, -2.1]) for index in range(1, 5)]
        + [synthetic_turn(index, [-0.8, -1.2]) for index in range(5, 14)]
    )


def entropic_collapse() -> List[Dict[str, Any]]:
    return [synthetic_turn(0, [-0.8, -1.2])] + [
        synthetic_turn(index, [-0.1, -2.1]) for index in range(1, 13)
    ]


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "short_calibrating_local_crossings": {
        "factory": short_calibrating_local_crossings,
        "expected_regime_label": "CALIBRATING",
        "expected_trajectory_assessment": "INSUFFICIENT_HISTORY",
        "interpretation": (
            "Local threshold crossings can occur in a short token-window sequence, "
            "but the classifier remains in calibration until enough trajectory history exists."
        ),
    },
    "organized_viability": {
        "factory": organized_viability,
        "expected_regime_label": "II_ORGANIZED_EQUILIBRIUM",
        "expected_trajectory_assessment": "VIABLE_ORGANIZED_EQUILIBRIUM",
        "interpretation": (
            "The trajectory has sufficient history and does not cross the dynamic threshold, "
            "so it remains in organized equilibrium."
        ),
    },
    "structural_pulsation": {
        "factory": structural_pulsation,
        "expected_regime_label": "III_STRUCTURAL_PULSATION",
        "expected_trajectory_assessment": "THRESHOLD_CROSSED_STRUCTURAL_PULSATION",
        "interpretation": (
            "The trajectory crosses the threshold and later recovers, indicating bounded "
            "structural pulsation rather than terminal collapse."
        ),
    },
    "entropic_collapse": {
        "factory": entropic_collapse,
        "expected_regime_label": "IV_ENTROPIC_COLLAPSE",
        "expected_trajectory_assessment": "ENTROPIC_COLLAPSE",
        "interpretation": (
            "The trajectory crosses the threshold persistently without recovery, matching "
            "the operational entropic-collapse criterion."
        ),
    },
}


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%Mh")
    return Path("results") / f"regime_benchmark_{stamp}"


def scenario_summary(name: str, result: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scenario": name,
        "expected_regime_label": expected["expected_regime_label"],
        "expected_trajectory_assessment": expected["expected_trajectory_assessment"],
        "regime_label": result.get("regime_label"),
        "trajectory_assessment": result.get("trajectory_assessment"),
        "threshold_crossing_ratio": result.get("threshold_crossing_ratio"),
        "persistent_crossing_ratio": result.get("persistent_crossing_ratio"),
        "recovery_observed": result.get("recovery_observed"),
        "first_crossing_turn": result.get("first_crossing_turn"),
        "first_crossing_window": result.get("first_crossing_window"),
        "threshold_crossed": result.get("threshold_crossed"),
        "xi_exceeds_theta": result.get("xi_exceeds_theta"),
        "valid_turns": result.get("valid_turns"),
        "critical_turns": result.get("critical_turns"),
        "passed": (
            result.get("regime_label") == expected["expected_regime_label"]
            and result.get("trajectory_assessment") == expected["expected_trajectory_assessment"]
        ),
    }


def write_scenario_report(path: Path, summary: Dict[str, Any], result: Dict[str, Any], interpretation: str) -> None:
    lines = [
        f"# Regime Benchmark Scenario: {summary['scenario']}",
        "",
        "## Parameters",
        "",
        f"- theta0: `{result.get('theta0')}`",
        f"- lambda0: `{result.get('lambda0')}`",
        f"- memory_beta: `{result.get('memory_beta')}`",
        f"- crossing_index_scope: `{result.get('crossing_index_scope')}`",
        f"- baseline_n_calib: `{result.get('baseline_n_calib')}`",
        "",
        "## Result",
        "",
        f"- expected_regime_label: `{summary['expected_regime_label']}`",
        f"- expected_trajectory_assessment: `{summary['expected_trajectory_assessment']}`",
        f"- regime_label: `{summary['regime_label']}`",
        f"- trajectory_assessment: `{summary['trajectory_assessment']}`",
        f"- threshold_crossing_ratio: `{summary['threshold_crossing_ratio']}`",
        f"- persistent_crossing_ratio: `{summary['persistent_crossing_ratio']}`",
        f"- recovery_observed: `{summary['recovery_observed']}`",
        f"- first_crossing_turn: `{summary['first_crossing_turn']}`",
        f"- first_crossing_window: `{summary['first_crossing_window']}`",
        "",
        "## Technical Interpretation",
        "",
        interpretation,
        "",
        "## Methodological Note",
        "",
        METHODOLOGICAL_NOTE,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_scenario(name: str, definition: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    turns = definition["factory"]()
    result = measure(turns, calib_window=1, crossing_index_scope="token_window")
    summary = scenario_summary(name, result, definition)
    scenario_dir = output_dir / name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "raw.json").write_text(
        json.dumps({"scenario": name, "synthetic_turns": turns, "result": result}, indent=2),
        encoding="utf-8",
    )
    (scenario_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_scenario_report(scenario_dir / "report.md", summary, result, definition["interpretation"])
    return summary


def run_benchmark(output_dir: Path | None = None) -> Dict[str, Any]:
    output = output_dir or default_output_dir()
    output.mkdir(parents=True, exist_ok=True)
    summaries = [run_scenario(name, definition, output) for name, definition in SCENARIOS.items()]
    aggregate = {
        "output_dir": str(output),
        "scenario_count": len(summaries),
        "passed": all(item["passed"] for item in summaries),
        "scenarios": summaries,
        "methodological_note": METHODOLOGICAL_NOTE,
    }
    (output / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic PRAMA regime benchmark scenarios.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to results/regime_benchmark_<timestamp>/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
