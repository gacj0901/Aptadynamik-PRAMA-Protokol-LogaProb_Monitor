from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple


ROW_FIELDS = [
    "raw_path",
    "session_id",
    "restriction_level",
    "restriction_source",
    "iota_proxy",
    "rho_proxy",
    "n_turns",
    "proxy_mode",
]


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _assistant_text(turn: Dict[str, Any]) -> str:
    return str(turn.get("assistant_message") or turn.get("response") or "")


def iota_proxy_from_raw(raw: Dict[str, Any]) -> float:
    texts = [_assistant_text(turn) for turn in raw.get("turns", []) if _assistant_text(turn)]
    if not texts:
        return 0.0
    joined = " ".join(texts)
    tokens = _tokens(joined)
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    avg_len = mean([len(_tokens(text)) for text in texts]) / 200.0
    return max(0.0, min(1.0, 0.7 * unique_ratio + 0.3 * min(avg_len, 1.0)))


def rho_proxy_from_raw(raw: Dict[str, Any]) -> float:
    markers = {
        "retract",
        "revise",
        "correction",
        "instead",
        "cannot",
        "constraint",
        "limit",
        "strict",
        "concise",
        "closure",
    }
    texts = [_assistant_text(turn) for turn in raw.get("turns", []) if _assistant_text(turn)]
    if not texts:
        return 0.0
    marker_hits = 0
    token_count = 0
    for text in texts:
        tokens = _tokens(text)
        token_count += len(tokens)
        marker_hits += sum(1 for token in tokens if token in markers)
    marker_rate = marker_hits / max(token_count, 1)
    finish_penalty = sum(1 for turn in raw.get("turns", []) if turn.get("finish_reason") == "length") / max(len(texts), 1)
    return max(0.0, min(1.0, marker_rate * 20.0 + finish_penalty))


def _restriction_from_metadata(raw: Dict[str, Any]) -> float | None:
    for key in ("restriction_level", "closure_level", "strict_closure_level"):
        if key in raw:
            return float(raw[key])
    metadata = raw.get("metadata")
    if isinstance(metadata, dict):
        for key in ("restriction_level", "closure_level", "strict_closure_level"):
            if key in metadata:
                return float(metadata[key])
    return None


def _restriction_from_filename(path: Path) -> float | None:
    match = re.search(r"(?:level|restriction|closure)[_-]?(\d+(?:\.\d+)?)", path.stem.lower())
    if match:
        return float(match.group(1))
    return None


def load_session_arg(value: str) -> Tuple[Path, float | None, str]:
    if ":" not in value:
        return Path(value), None, "pending_fallback"
    path_text, level_text = value.rsplit(":", 1)
    try:
        level = float(level_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid restriction level: {level_text}") from exc
    return Path(path_text), level, "explicit_cli"


def row_from_session(path: Path, restriction_level: float | None, restriction_source: str) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    turns = raw.get("turns", [])
    if not isinstance(turns, list):
        raise ValueError(f"{path} missing list field 'turns'")
    if restriction_level is None:
        metadata_level = _restriction_from_metadata(raw)
        if metadata_level is not None:
            restriction_level = metadata_level
            restriction_source = "metadata"
        else:
            filename_level = _restriction_from_filename(path)
            if filename_level is None:
                raise ValueError(f"{path} has no explicit restriction level, metadata level, or filename fallback")
            restriction_level = filename_level
            restriction_source = "filename_fallback"
    return {
        "raw_path": str(path),
        "session_id": raw.get("session_id", path.stem),
        "restriction_level": restriction_level,
        "restriction_source": restriction_source,
        "iota_proxy": iota_proxy_from_raw(raw),
        "rho_proxy": rho_proxy_from_raw(raw),
        "n_turns": len(turns),
        "proxy_mode": True,
    }


def linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def bootstrap_slope_ci(
    xs: Sequence[float],
    ys: Sequence[float],
    n: int = 2000,
) -> Dict[str, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    rng = random.Random(0)
    indices = list(range(len(xs)))
    slopes = []
    for _ in range(n):
        draw = [indices[rng.randrange(len(indices))] for _ in indices]
        slopes.append(linear_slope([xs[idx] for idx in draw], [ys[idx] for idx in draw]))
    slopes.sort()
    return {
        "mean": mean(slopes),
        "lo": slopes[int(0.025 * (len(slopes) - 1))],
        "hi": slopes[int(0.975 * (len(slopes) - 1))],
    }


def classify_sweep(iota_ci: Dict[str, float], rho_ci: Dict[str, float]) -> str:
    iota_positive = iota_ci["lo"] > 0 and iota_ci["hi"] > 0
    rho_negative = rho_ci["lo"] < 0 and rho_ci["hi"] < 0
    iota_crosses = iota_ci["lo"] <= 0 <= iota_ci["hi"]
    rho_crosses = rho_ci["lo"] <= 0 <= rho_ci["hi"]
    if iota_positive and rho_negative:
        return "compatible_with_compensation_under_constraint"
    if iota_crosses and rho_crosses:
        return "compatible_with_baseline_verbosity"
    if iota_positive or rho_negative:
        return "inconclusive_partial_signal"
    return "inconclusive_partial_signal"


def analyze_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    xs = [float(row["restriction_level"]) for row in rows]
    iotas = [float(row["iota_proxy"]) for row in rows]
    rhos = [float(row["rho_proxy"]) for row in rows]
    iota_ci = bootstrap_slope_ci(xs, iotas)
    rho_ci = bootstrap_slope_ci(xs, rhos)
    return {
        "proxy_mode": True,
        "n_sessions": len(rows),
        "slope_iota_vs_restriction": linear_slope(xs, iotas),
        "slope_rho_vs_restriction": linear_slope(xs, rhos),
        "iota_slope_ci95": iota_ci,
        "rho_slope_ci95": rho_ci,
        "classification": classify_sweep(iota_ci, rho_ci),
        "warning": "Proxy mode is not real Axiom 0 trigon measurement. It uses available text/logprob geometry traces only.",
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in ROW_FIELDS})


def write_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# PRAMA Closure Sweep Report",
        "",
        "LEGACY WARNING:",
        "This report uses historical proxy metrics. It is retained for comparison and calibration history only. It is not the canonical PRAMA structural measurement.",
        "",
        "## Purpose",
        "",
        "This sweep distinguishes compensation by expansion under constraint from baseline model verbosity.",
        "",
        "## Proxy Mode",
        "",
        f"- proxy_mode: {summary['proxy_mode']}",
        "iota_proxy estimates semantic expansion/novelty from currently available text traces.",
        "rho_proxy estimates retraction/correction/inhibition/constraint-compliance markers from currently available text traces.",
        "This is not real Axiom 0 trigon measurement.",
        "",
        "## Slopes",
        "",
        f"- slope_iota_vs_restriction: {summary['slope_iota_vs_restriction']}",
        f"- slope_rho_vs_restriction: {summary['slope_rho_vs_restriction']}",
        f"- iota CI95: {summary['iota_slope_ci95']}",
        f"- rho CI95: {summary['rho_slope_ci95']}",
        "",
        "## Session Inputs",
        "",
        "Each row records raw_path, restriction_level, and restriction_source.",
        "When --session path:level is provided, restriction_source is explicit_cli and overrides any metadata in raw.json.",
        "",
        "## Classification",
        "",
        summary["classification"],
        "",
        "## Methodological Warning",
        "",
        summary["warning"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_sweep(session_specs: Sequence[Tuple[Path, float | None, str]], output_dir: Path) -> Dict[str, Any]:
    rows = [row_from_session(path, level, source) for path, level, source in session_specs]
    summary = analyze_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "closure_sweep_rows.csv", rows)
    (output_dir / "closure_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output_dir / "closure_sweep_report.md", summary)
    return summary


def synthetic_compensation_rows() -> List[Dict[str, Any]]:
    return [
        {"restriction_level": 0, "iota_proxy": 0.20, "rho_proxy": 0.80},
        {"restriction_level": 1, "iota_proxy": 0.36, "rho_proxy": 0.63},
        {"restriction_level": 2, "iota_proxy": 0.52, "rho_proxy": 0.45},
        {"restriction_level": 3, "iota_proxy": 0.70, "rho_proxy": 0.27},
    ]


def synthetic_verbose_constant_rows() -> List[Dict[str, Any]]:
    return [
        {"restriction_level": 0, "iota_proxy": 0.50, "rho_proxy": 0.50},
        {"restriction_level": 1, "iota_proxy": 0.51, "rho_proxy": 0.49},
        {"restriction_level": 2, "iota_proxy": 0.50, "rho_proxy": 0.51},
        {"restriction_level": 3, "iota_proxy": 0.51, "rho_proxy": 0.50},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PRAMA strict-closure sweep analysis.")
    parser.add_argument("--session", action="append", type=load_session_arg, required=True, help="PATH:RESTRICTION_LEVEL")
    parser.add_argument("--output-dir", required=True, help="Directory for closure sweep outputs.")
    args = parser.parse_args()
    summary = run_sweep(args.session, Path(args.output_dir))
    print(f"classification: {summary['classification']}")
    print(f"wrote closure sweep outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
