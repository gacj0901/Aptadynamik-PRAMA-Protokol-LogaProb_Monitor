from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from aptadynamik.observer.friction_absorption_metrics import (
    classify_absorption_or_friction,
    compute_commitment_shift,
    compute_elaboration,
    compute_recombination,
    compute_surprise_from_prama_turn,
)
from aptadynamik.observer.perturbation_taxonomy import iter_trials, load_protocol


TRIAL_FIELDS = [
    "trial_id",
    "model",
    "item_id",
    "topic",
    "perturbation_type",
    "baseline_prompt",
    "tracked_commitment",
    "perturbation_text",
    "perturbation_rule",
]

METRIC_FIELDS = [
    "source_file",
    "session_id",
    "model",
    "turn_index",
    "perturbation_type",
    "C_commitment_shift",
    "R_recombination",
    "S_surprise",
    "E_elaboration",
    "classification",
]


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def study_design_markdown(protocol: Dict[str, Any], trial_count: int) -> str:
    arms = "\n".join(f"- {arm}" for arm in protocol["arms"])
    variables = "\n".join(
        [
            "- C_commitment_shift: measured shift from tracked commitment to response.",
            "- R_recombination: reuse/recombination of prior context.",
            "- S_surprise: PRAMA Monitor entropy proxy where available.",
            "- E_elaboration: response length proxy.",
        ]
    )
    return "\n".join(
        [
            "# PRAMA Minimal-Structural Perturbation Study",
            "",
            "## Hypotheses",
            "",
            "H_A:",
            "Greater absorbable narrative content in the perturbation produces less real friction.",
            "",
            "H_B:",
            "Minimal-structural perturbation produces more real friction than abstract-content perturbation.",
            "",
            "H0:",
            "Perturbation type explains no meaningful variance in C, R, or S after controls.",
            "",
            "## Perturbation Types",
            "",
            arms,
            "",
            "## Dependent Variables",
            "",
            variables,
            "",
            "## Absorption vs Real Friction",
            "",
            "Absorption is initially operationalized as delta_C approximately 0, R high, S low, and E increased.",
            "Real friction is initially operationalized as delta_C nonzero, R decreases, and S increases.",
            "",
            "## Falsification Criteria",
            "",
            "- reject the friction thesis if perturbation type explains approximately zero variance in C/R/S",
            "- reject if abstract-content perturbations equal or exceed minimal-structural perturbations in friction",
            "- reject if no perturbation exceeds the control arm drift",
            "",
            "## Study Metadata",
            "",
            f"- study_id: {protocol['study_id']}",
            f"- models: {', '.join(protocol['model_list'])}",
            f"- trials: {trial_count}",
            f"- temperature: {protocol['temperature']}",
            f"- max_tokens: {protocol['max_tokens']}",
            f"- top_logprobs: {protocol['top_logprobs']}",
            f"- window_size: {protocol['window_size']}",
            "",
            "## Methodological Note",
            "",
            "This scaffold does not yet replace blinded human judges, preregistration, or the future ProbLog commitment tracker.",
            "",
        ]
    )


def run_dry_run(protocol_path: Path, output_dir: Path = Path("results/perturbation_study_dry_run")) -> List[Dict[str, Any]]:
    protocol = load_protocol(protocol_path)
    trials = list(iter_trials(protocol))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "trials.csv", trials, TRIAL_FIELDS)
    (output_dir / "study_design.md").write_text(study_design_markdown(protocol, len(trials)), encoding="utf-8")
    return trials


def _load_raw_files(results_dir: Path) -> List[Path]:
    files = sorted({*results_dir.rglob("raw.json"), *results_dir.rglob("*_raw.json")})
    return [path for path in files if path.is_file()]


def _infer_perturbation_type(path: Path) -> str:
    lower = str(path).lower()
    for name in ("minimal_structural", "abstract_content", "concrete_content", "control_neutral"):
        if name in lower:
            return name
    return "unknown"


def _turn_response(turn: Dict[str, Any]) -> str:
    return str(turn.get("assistant_message") or turn.get("response") or "")


def _history_before(turns: List[Dict[str, Any]], index: int) -> List[str]:
    history: List[str] = []
    for prior in turns[:index]:
        if prior.get("user_message"):
            history.append(str(prior["user_message"]))
        if prior.get("assistant_message"):
            history.append(str(prior["assistant_message"]))
    return history


def run_from_results(results_dir: Path, output_dir: Path | None = None) -> List[Dict[str, Any]]:
    target_dir = output_dir or results_dir
    rows: List[Dict[str, Any]] = []
    for raw_path in _load_raw_files(results_dir):
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        turns = raw.get("turns", [])
        if not isinstance(turns, list):
            continue
        commitment = str(raw.get("tracked_commitment") or raw.get("commitment") or "")
        perturbation_type = _infer_perturbation_type(raw_path)
        for index, turn in enumerate(turns):
            response = _turn_response(turn)
            delta_c = compute_commitment_shift(commitment, response) if commitment and response else 0.0
            recombination = compute_recombination(response, _history_before(turns, index))
            surprise = compute_surprise_from_prama_turn(turn)
            elaboration = compute_elaboration(response)
            rows.append(
                {
                    "source_file": str(raw_path),
                    "session_id": raw.get("session_id", raw_path.parent.name),
                    "model": raw.get("model", ""),
                    "turn_index": turn.get("turn_index", index),
                    "perturbation_type": perturbation_type,
                    "C_commitment_shift": round(delta_c, 6),
                    "R_recombination": round(recombination, 6),
                    "S_surprise": round(surprise, 6),
                    "E_elaboration": round(elaboration, 6),
                    "classification": classify_absorption_or_friction(delta_c, recombination, surprise, elaboration),
                }
            )

    write_csv(target_dir / "perturbation_metrics.csv", rows, METRIC_FIELDS)
    (target_dir / "perturbation_report.md").write_text(metrics_report(rows), encoding="utf-8")
    return rows


def metrics_report(rows: List[Dict[str, Any]]) -> str:
    by_class: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    for row in rows:
        by_class[row["classification"]] = by_class.get(row["classification"], 0) + 1
        by_type[row["perturbation_type"]] = by_type.get(row["perturbation_type"], 0) + 1

    lines = [
        "# PRAMA Minimal-Structural Perturbation Metrics",
        "",
        "## Available Proxy Rows",
        "",
        f"- rows: {len(rows)}",
        "",
        "## Perturbation Types Observed",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(by_type.items()))
    lines.extend(["", "## Classifications", ""])
    lines.extend(f"- {key}: {value}" for key, value in sorted(by_class.items()))
    lines.extend(
        [
            "",
            "## Methodological Note",
            "",
            "These are scaffold proxies only. They do not yet replace blinded human judges, preregistration, or the future ProbLog commitment tracker.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the PRAMA minimal-structural perturbation study scaffold.")
    parser.add_argument("--protocol", required=True, help="Path to minimal_structural_perturbations.yaml.")
    parser.add_argument("--dry-run", action="store_true", help="Validate protocol and enumerate trials.")
    parser.add_argument("--from-results", help="Compute proxy metrics from PRAMA Monitor raw.json files.")
    parser.add_argument("--output-dir", help="Optional output directory.")
    args = parser.parse_args()

    if args.dry_run == bool(args.from_results):
        parser.error("Choose exactly one mode: --dry-run or --from-results PATH")

    protocol_path = Path(args.protocol)
    if args.dry_run:
        output_dir = Path(args.output_dir) if args.output_dir else Path("results/perturbation_study_dry_run")
        trials = run_dry_run(protocol_path, output_dir)
        print(f"Validated protocol and wrote {len(trials)} trials to {output_dir}")
        return 0

    load_protocol(protocol_path)
    output_dir = Path(args.output_dir) if args.output_dir else None
    rows = run_from_results(Path(args.from_results), output_dir)
    print(f"Wrote {len(rows)} metric rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
