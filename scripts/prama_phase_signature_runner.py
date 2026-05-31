from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aptadynamik.observer.qualitative_signatures import (
    load_raw_session,
    paramend_rayleigh_fields_from_turns,
    sig_critical_slowing,
    sig_discontinuity,
    sig_hysteresis,
    sig_structural_target,
    synthetic_fold_series,
    synthetic_smooth_series,
    synthetic_validation,
    viability_series_from_raw,
)


SIGNATURE_FIELDS = [
    "model",
    "source_file",
    "condition",
    "direction",
    "n_turns",
    "discontinuity_ratio",
    "strongest_transition_turn",
    "hysteresis_area",
    "critical_slowing_score",
    "structural_target_shift",
    "paramend_rayleigh_median",
    "paramend_rayleigh_max",
    "min_restitution_g",
    "early_warning_turn",
    "min_viability",
    "max_viability",
    "mean_viability",
]

COMPARATIVE_FIELDS = [
    "model",
    "peripheral_transition_turn",
    "constitutive_transition_turn",
    "structural_target_effect",
    "peripheral_hysteresis",
    "constitutive_hysteresis",
    "max_discontinuity",
    "max_critical_slowing",
]


def _round(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def _series_stats(series: Sequence[float]) -> Dict[str, Any]:
    if not series:
        return {"min_viability": None, "max_viability": None, "mean_viability": None}
    return {
        "min_viability": _round(min(series)),
        "max_viability": _round(max(series)),
        "mean_viability": _round(mean(series)),
    }


def analyze_viability_series(
    series: Sequence[float],
    *,
    model: str,
    source_file: str,
    condition: str,
    direction: str,
    hysteresis_area: Optional[float] = None,
    structural_target_shift: Optional[float] = None,
    rayleigh_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    discontinuity = sig_discontinuity(series)
    slowing = sig_critical_slowing(series)
    row = {
        "model": model,
        "source_file": source_file,
        "condition": condition,
        "direction": direction,
        "n_turns": len(series),
        "discontinuity_ratio": _round(discontinuity["discontinuity_ratio"]),
        "strongest_transition_turn": discontinuity["strongest_transition_turn"],
        "hysteresis_area": _round(hysteresis_area),
        "critical_slowing_score": _round(slowing["critical_slowing_score"]),
        "structural_target_shift": _round(structural_target_shift),
        "paramend_rayleigh_median": _round((rayleigh_fields or {}).get("paramend_rayleigh_median")),
        "paramend_rayleigh_max": _round((rayleigh_fields or {}).get("paramend_rayleigh_max")),
        "min_restitution_g": _round((rayleigh_fields or {}).get("min_restitution_g")),
        "early_warning_turn": (rayleigh_fields or {}).get("early_warning_turn"),
        **_series_stats(series),
    }
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _report_lines(metadata: Dict[str, Any], rows: Sequence[Dict[str, Any]], comparative: Optional[Dict[str, Any]] = None) -> List[str]:
    lines = [
        "# PRAMA Phase Signature Report",
        "",
        "## Session / Model Metadata",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Viability Definition",
            "",
            "viability = avg_rigidity - avg_uncertainty",
            "",
            "## Detected Signatures",
            "",
            "| model | source | condition | direction | turns | discontinuity ratio | strongest transition turn | hysteresis area | critical slowing score | structural target shift | Rayleigh median | Rayleigh max | min g | early warning turn |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.get('model')} | {row.get('source_file')} | {row.get('condition')} | {row.get('direction')} | "
            f"{row.get('n_turns')} | {row.get('discontinuity_ratio')} | {row.get('strongest_transition_turn')} | "
            f"{row.get('hysteresis_area')} | {row.get('critical_slowing_score')} | {row.get('structural_target_shift')} | "
            f"{row.get('paramend_rayleigh_median')} | {row.get('paramend_rayleigh_max')} | {row.get('min_restitution_g')} | "
            f"{row.get('early_warning_turn')} |"
        )
    if comparative:
        lines.extend(
            [
                "",
                "Structural target comparison:",
                f"- peripheral transition turn: {comparative.get('peripheral_transition_turn')}",
                f"- constitutive transition turn: {comparative.get('constitutive_transition_turn')}",
                f"- structural target effect: {comparative.get('structural_target_effect')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Paramend Rayleigh Fatigue",
            "",
            "mu is estimated from PRAMA Monitor turn summaries. With full-session context, mu = 1 - normalized_viability where viability = avg_rigidity - avg_uncertainty. Without normalization context, the proxy is avg_uncertainty + entropy_std + entropy_range_normalized - avg_rigidity.",
            "mu_star is estimated from baseline/control/R0/R1 turns when labels exist; otherwise it uses the first 20% of turns.",
            "",
            "| model | median fatigue | max fatigue | minimum restitution g | early warning turn |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.get('model')} | {row.get('paramend_rayleigh_median')} | {row.get('paramend_rayleigh_max')} | "
            f"{row.get('min_restitution_g')} | {row.get('early_warning_turn')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Synthetic validation only validates the detector. Real raw.json analysis is empirical observation.",
            "This does not yet prove predictive-surprise Delta over exogenous user signs.",
            "It detects qualitative shifts in assistant-output geometry.",
            "",
            "## Methodological Note",
            "",
            "This module measures qualitative transition signatures over PRAMA Monitor output geometry. It does not replace the future predictive-surprise module based on incoming user signs.",
            "The corrected Paramend operator measures structural fatigue as displacement from a viable reference divided by restitution capacity. It does not treat low velocity as fatigue by itself.",
            "",
        ]
    )
    return lines


def write_report(path: Path, metadata: Dict[str, Any], rows: Sequence[Dict[str, Any]], comparative: Optional[Dict[str, Any]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_report_lines(metadata, rows, comparative)), encoding="utf-8")


def _synthetic_rows() -> List[Dict[str, Any]]:
    fold = synthetic_fold_series()
    smooth = synthetic_smooth_series()
    validation = synthetic_validation()
    rows = [
        analyze_viability_series(
            fold["up"],
            model="synthetic_fold",
            source_file="synthetic_fold",
            condition="fold",
            direction="up",
            hysteresis_area=sig_hysteresis(fold["up"], fold["down"])["hysteresis_area"],
        ),
        analyze_viability_series(
            smooth["up"],
            model="synthetic_smooth",
            source_file="synthetic_smooth",
            condition="smooth",
            direction="up",
            hysteresis_area=sig_hysteresis(smooth["up"], smooth["down"])["hysteresis_area"],
        ),
    ]
    rows[0]["critical_slowing_score"] = _round(validation["fold"]["critical_slowing"]["critical_slowing_score"])
    rows[1]["critical_slowing_score"] = _round(validation["smooth"]["critical_slowing"]["critical_slowing_score"])
    return rows


def run_synthetic(output_dir: Path = Path("results/phase_synthetic_validation")) -> List[Dict[str, Any]]:
    validation = synthetic_validation()
    fold_ok = all(validation["fold"][name]["triggered"] for name in ("discontinuity", "hysteresis", "critical_slowing"))
    smooth_ok = not any(validation["smooth"][name]["triggered"] for name in ("discontinuity", "hysteresis", "critical_slowing"))
    rows = _synthetic_rows()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "phase_signatures.csv", rows, SIGNATURE_FIELDS)
    write_report(
        output_dir / "phase_report.md",
        {
            "mode": "synthetic validation",
            "fold system triggers required signatures": fold_ok,
            "smooth system remains quiet": smooth_ok,
        },
        rows,
    )
    print("Synthetic validation")
    print(f"- fold triggers discontinuity/hysteresis/critical slowing: {fold_ok}")
    print(f"- smooth avoids those signatures: {smooth_ok}")
    return rows


def _condition_direction_from_path(path: Path) -> Tuple[str, str]:
    name = path.stem
    if name.endswith("_raw"):
        name = name[:-4]
    parts = name.split("_")
    direction = "unknown"
    if parts and parts[-1] in {"up", "down"}:
        direction = parts[-1]
        condition = "_".join(parts[:-1]) or "session"
    else:
        condition = name or "session"
    return condition, direction


def _load_series(path: Path) -> Tuple[Dict[str, Any], List[float], Dict[str, Any]]:
    raw = load_raw_session(path)
    rows = viability_series_from_raw(raw)
    turns = raw.get("turns", [])
    rayleigh = paramend_rayleigh_fields_from_turns(turns if isinstance(turns, list) else [])
    return raw, [row["viability"] for row in rows], rayleigh


def run_from_raw(raw_path: Path, output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    raw, series, rayleigh = _load_series(raw_path)
    session_id = raw.get("session_id", raw_path.stem)
    model = raw.get("model", "unknown")
    condition, direction = _condition_direction_from_path(raw_path)
    output = output_dir or Path("results") / f"phase_analysis_{session_id}"
    row = analyze_viability_series(
        series,
        model=model,
        source_file=str(raw_path),
        condition=condition,
        direction=direction,
        rayleigh_fields=rayleigh,
    )
    rows = [row]
    write_csv(output / "phase_signatures.csv", rows, SIGNATURE_FIELDS)
    write_report(
        output / "phase_report.md",
        {
            "mode": "real raw.json",
            "session_id": session_id,
            "model": model,
            "source_file": raw_path,
            "local drops": sig_discontinuity(series)["local_drops"],
        },
        rows,
    )
    return rows


def _find_raw_file(model_dir: Path, condition: str, direction: str) -> Optional[Path]:
    candidates = [
        model_dir / f"{condition}_{direction}_raw.json",
        model_dir / f"session_{condition}_{direction}_raw.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(model_dir.glob(f"*{condition}*{direction}*raw.json"))
    return matches[0] if matches else None


def _paired_hysteresis(series_by_key: Dict[Tuple[str, str], List[float]], condition: str) -> Optional[float]:
    up = series_by_key.get((condition, "up"))
    down = series_by_key.get((condition, "down"))
    if not up or not down:
        return None
    return sig_hysteresis(up, down)["hysteresis_area"]


def _comparative_row(model: str, series_by_key: Dict[Tuple[str, str], List[float]], rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    peripheral = series_by_key.get(("peripheral", "up"), [])
    constitutive = series_by_key.get(("constitutive", "up"), [])
    structural = sig_structural_target(peripheral, constitutive)
    return {
        "model": model,
        "peripheral_transition_turn": structural["peripheral_transition_turn"],
        "constitutive_transition_turn": structural["constitutive_transition_turn"],
        "structural_target_effect": _round(structural["structural_target_shift"]),
        "peripheral_hysteresis": _round(_paired_hysteresis(series_by_key, "peripheral")),
        "constitutive_hysteresis": _round(_paired_hysteresis(series_by_key, "constitutive")),
        "max_discontinuity": _round(max((float(row.get("discontinuity_ratio") or 0.0) for row in rows), default=0.0)),
        "max_critical_slowing": _round(max((float(row.get("critical_slowing_score") or 0.0) for row in rows), default=0.0)),
    }


def run_from_results(results_dir: Path) -> List[Dict[str, Any]]:
    comparative_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    model_dirs = [path for path in sorted(results_dir.iterdir()) if path.is_dir()]
    if not model_dirs:
        raise ValueError(f"no model directories found in {results_dir}")

    for model_dir in model_dirs:
        model = model_dir.name
        series_by_key: Dict[Tuple[str, str], List[float]] = {}
        rayleigh_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        raw_by_key: Dict[Tuple[str, str], Path] = {}
        for condition in ("peripheral", "constitutive"):
            for direction in ("up", "down"):
                raw_path = _find_raw_file(model_dir, condition, direction)
                if raw_path is None:
                    continue
                raw, series, rayleigh = _load_series(raw_path)
                series_by_key[(condition, direction)] = series
                rayleigh_by_key[(condition, direction)] = rayleigh
                raw_by_key[(condition, direction)] = raw_path
                model = raw.get("model", model)

        structural = sig_structural_target(
            series_by_key.get(("peripheral", "up"), []),
            series_by_key.get(("constitutive", "up"), []),
        )
        model_rows: List[Dict[str, Any]] = []
        for (condition, direction), series in sorted(series_by_key.items()):
            model_rows.append(
                analyze_viability_series(
                    series,
                    model=model,
                    source_file=str(raw_by_key[(condition, direction)]),
                    condition=condition,
                    direction=direction,
                    hysteresis_area=_paired_hysteresis(series_by_key, condition) if direction == "up" else None,
                    structural_target_shift=structural["structural_target_shift"] if condition == "constitutive" and direction == "up" else None,
                    rayleigh_fields=rayleigh_by_key.get((condition, direction)),
                )
            )

        comparative = _comparative_row(model, series_by_key, model_rows)
        comparative_rows.append(comparative)
        all_rows.extend(model_rows)
        write_csv(model_dir / "phase_signatures.csv", model_rows, SIGNATURE_FIELDS)
        write_report(
            model_dir / "phase_report.md",
            {"mode": "diagnostic directory", "model": model, "source_dir": model_dir},
            model_rows,
            comparative,
        )

    write_csv(results_dir / "comparative_phase_summary.csv", comparative_rows, COMPARATIVE_FIELDS)
    comparative_lines = [
        "# PRAMA Comparative Phase Signature Report",
        "",
        "Synthetic validation only validates the detector. Real raw.json analysis is empirical observation.",
        "",
        "| model | peripheral transition | constitutive transition | structural target effect | peripheral hysteresis | constitutive hysteresis | max discontinuity | max critical slowing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparative_rows:
        comparative_lines.append(
            f"| {row['model']} | {row['peripheral_transition_turn']} | {row['constitutive_transition_turn']} | "
            f"{row['structural_target_effect']} | {row['peripheral_hysteresis']} | {row['constitutive_hysteresis']} | "
            f"{row['max_discontinuity']} | {row['max_critical_slowing']} |"
        )
    comparative_lines.extend(
        [
            "",
            "## Methodological Note",
            "",
            "This module measures qualitative transition signatures over PRAMA Monitor output geometry. It does not replace the future predictive-surprise module based on incoming user signs.",
            "",
        ]
    )
    (results_dir / "comparative_phase_report.md").write_text("\n".join(comparative_lines), encoding="utf-8")
    return all_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PRAMA phase-transition signature analyses.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true", help="Run synthetic detector validation.")
    group.add_argument("--from-raw", help="Analyze one PRAMA Monitor raw.json file.")
    group.add_argument("--from-results", help="Analyze a structured diagnostic results directory.")
    parser.add_argument("--output-dir", help="Optional output directory for --synthetic or --from-raw.")
    args = parser.parse_args()

    if args.synthetic:
        output_dir = Path(args.output_dir) if args.output_dir else Path("results/phase_synthetic_validation")
        rows = run_synthetic(output_dir)
        print(f"Wrote {len(rows)} rows to {output_dir}")
        return 0

    if args.from_raw:
        output_dir = Path(args.output_dir) if args.output_dir else None
        rows = run_from_raw(Path(args.from_raw), output_dir)
        target = output_dir or Path("results") / f"phase_analysis_{load_raw_session(args.from_raw).get('session_id', Path(args.from_raw).stem)}"
        print(f"Wrote {len(rows)} rows to {target}")
        return 0

    rows = run_from_results(Path(args.from_results))
    print(f"Wrote {len(rows)} rows under {args.from_results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
