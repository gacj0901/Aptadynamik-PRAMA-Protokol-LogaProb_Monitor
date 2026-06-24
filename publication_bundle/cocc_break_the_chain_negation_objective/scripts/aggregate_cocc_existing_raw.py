#!/usr/bin/env python
"""Aggregate logprob metrics from existing CoCC PRAMA raw.json files.

This script is offline-only: it reads existing raw.json files, computes
descriptive logprob-derived metrics, and writes aggregate artifacts. It does
not call APIs, run models, or modify raw.json inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CSV_FIELDS = [
    "session_id",
    "provider",
    "requested_model",
    "resolved_model",
    "model",
    "benchmark_name",
    "benchmark_alias",
    "perturbation_type",
    "item_id",
    "problem_id",
    "question_id",
    "difficulty",
    "finish_reason",
    "token_count",
    "assistant_chars",
    "mean_entropy",
    "max_entropy",
    "entropy_stdev",
    "p90_entropy",
    "mean_gap",
    "min_gap",
    "p10_gap",
    "mean_top1_logprob",
    "min_top1_logprob",
    "perplexity",
    "raw_path",
]


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.stdev(values)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def first_turn(raw: dict[str, Any]) -> dict[str, Any]:
    turns = raw.get("turns") or []
    if not turns:
        return {}
    turn = turns[0]
    return turn if isinstance(turn, dict) else {}


def extract_token_values(tokens: Iterable[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        number = finite_float(token.get(field))
        if number is not None:
            values.append(number)
    return values


def row_from_raw(raw_path: Path) -> dict[str, Any]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    turn = first_turn(raw)
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    turn_tokens = turn.get("tokens") or []
    tokens = [token for token in turn_tokens if isinstance(token, dict)]

    entropy_values = extract_token_values(tokens, "entropy")
    gap_values = extract_token_values(tokens, "gap")
    top1_values = extract_token_values(tokens, "top1_logprob")

    mean_top1 = mean(top1_values)
    perplexity = math.exp(-mean_top1) if mean_top1 is not None else None

    token_count = finite_float(turn.get("token_count"))
    if token_count is None:
        token_count = float(len(tokens))

    assistant_message = str(turn.get("assistant_message") or "")

    return {
        "session_id": raw.get("session_id") or raw_path.parent.name,
        "provider": raw.get("provider"),
        "requested_model": raw.get("requested_model"),
        "resolved_model": raw.get("resolved_model"),
        "model": raw.get("model"),
        "benchmark_name": raw.get("benchmark_name") or metadata.get("benchmark_name"),
        "benchmark_alias": raw.get("benchmark_alias") or metadata.get("benchmark_alias"),
        "perturbation_type": raw.get("perturbation_type") or metadata.get("perturbation_type"),
        "item_id": raw.get("item_id") or metadata.get("item_id"),
        "problem_id": raw.get("problem_id") or metadata.get("problem_id"),
        "question_id": metadata.get("question_id"),
        "difficulty": metadata.get("difficulty"),
        "finish_reason": turn.get("finish_reason"),
        "token_count": token_count,
        "assistant_chars": len(assistant_message),
        "mean_entropy": mean(entropy_values),
        "max_entropy": max(entropy_values) if entropy_values else None,
        "entropy_stdev": stdev(entropy_values),
        "p90_entropy": percentile(entropy_values, 0.90),
        "mean_gap": mean(gap_values),
        "min_gap": min(gap_values) if gap_values else None,
        "p10_gap": percentile(gap_values, 0.10),
        "mean_top1_logprob": mean_top1,
        "min_top1_logprob": min(top1_values) if top1_values else None,
        "perplexity": perplexity,
        "raw_path": str(raw_path),
    }


def numeric_summary(rows: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    values = [finite_float(row.get(field)) for row in rows]
    clean = [value for value in values if value is not None]
    return {
        "mean": mean(clean),
        "median": statistics.median(clean) if clean else None,
        "p90": percentile(clean, 0.90),
        "p10": percentile(clean, 0.10),
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
    }


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field) or "UNKNOWN") for row in rows)
    return dict(sorted(counts.items()))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in CSV_FIELDS})


def render_distribution(title: str, counts: dict[str, int]) -> list[str]:
    lines = [f"### {title}", "", "| value | count |", "|---|---:|"]
    for value, count in counts.items():
        lines.append(f"| {value} | {count} |")
    lines.append("")
    return lines


def render_metric_table(title: str, summaries: dict[str, dict[str, float | None]], fields: list[str]) -> list[str]:
    lines = [f"### {title}", "", "| metric | mean | median | p90 | p10 | min | max |", "|---|---:|---:|---:|---:|---:|---:|"]
    for field in fields:
        summary = summaries[field]
        lines.append(
            "| "
            + field
            + " | "
            + " | ".join(
                fmt(summary.get(name))
                for name in ("mean", "median", "p90", "p10", "min", "max")
            )
            + " |"
        )
    lines.append("")
    return lines


def write_report(rows: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# CoCC Existing Raw Logprob Aggregate Report",
        "",
        f"- raw_json_analyzed: `{summary['raw_json_analyzed']}`",
        "",
    ]
    lines.extend(render_distribution("benchmark_name", summary["distributions"]["benchmark_name"]))
    lines.extend(render_distribution("benchmark_alias", summary["distributions"]["benchmark_alias"]))
    lines.extend(render_distribution("perturbation_type", summary["distributions"]["perturbation_type"]))
    lines.extend(render_distribution("difficulty", summary["distributions"]["difficulty"]))
    lines.extend(render_distribution("finish_reason", summary["distributions"]["finish_reason"]))
    lines.extend(
        render_metric_table(
            "Length Metrics",
            summary["numeric"],
            ["token_count", "assistant_chars"],
        )
    )
    lines.extend(
        render_metric_table(
            "Logprob Metrics",
            summary["numeric"],
            [
                "mean_entropy",
                "p90_entropy",
                "mean_gap",
                "mean_top1_logprob",
                "perplexity",
            ],
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate(sessions_dir: Path, output_dir: Path) -> dict[str, Any]:
    raw_paths = sorted(sessions_dir.glob("*/raw.json"))
    if not raw_paths:
        raise SystemExit(f"no raw.json files found under {sessions_dir}")
    rows = [row_from_raw(path) for path in raw_paths]
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_fields = [
        "token_count",
        "assistant_chars",
        "mean_entropy",
        "max_entropy",
        "entropy_stdev",
        "p90_entropy",
        "mean_gap",
        "min_gap",
        "p10_gap",
        "mean_top1_logprob",
        "min_top1_logprob",
        "perplexity",
    ]
    summary = {
        "raw_json_analyzed": len(rows),
        "sessions_dir": str(sessions_dir),
        "output_dir": str(output_dir),
        "distributions": {
            "benchmark_name": distribution(rows, "benchmark_name"),
            "benchmark_alias": distribution(rows, "benchmark_alias"),
            "perturbation_type": distribution(rows, "perturbation_type"),
            "difficulty": distribution(rows, "difficulty"),
            "finish_reason": distribution(rows, "finish_reason"),
        },
        "numeric": {field: numeric_summary(rows, field) for field in numeric_fields},
    }

    write_csv(rows, output_dir / "aggregate_logprob_metrics.csv")
    (output_dir / "aggregate_logprob_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(rows, summary, output_dir / "aggregate_logprob_report.md")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = aggregate(Path(args.sessions_dir), Path(args.output_dir))
    print(f"raw_json_analyzed={summary['raw_json_analyzed']}")
    print(f"wrote {Path(args.output_dir) / 'aggregate_logprob_metrics.csv'}")
    print(f"wrote {Path(args.output_dir) / 'aggregate_logprob_report.md'}")
    print(f"wrote {Path(args.output_dir) / 'aggregate_logprob_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
