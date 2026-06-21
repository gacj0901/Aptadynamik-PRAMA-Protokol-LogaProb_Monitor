#!/usr/bin/env python
"""Offline PRAMA window analysis for existing CoCC raw.json sessions.

This script consumes already-generated raw.json files. It never calls APIs,
runs models, or modifies raw inputs. Tokens from turns[0].tokens are converted
into sliding PRAMA windows and passed through the existing PRAMA ProbLog
Components measurement layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from aptadynamik.prama_problog_components import measure  # noqa: E402


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
    "window_count",
    "mean_entropy",
    "entropy_variance",
    "mean_gap",
    "mean_top1_logprob",
    "perplexity",
    "threshold_crossing_ratio",
    "persistent_crossing_ratio",
    "recovery_observed",
    "regime",
    "trajectory_assessment",
    "truncation_flag",
    "raw_path",
]


def finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.variance(values)


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


def first_turn(raw: dict[str, Any]) -> dict[str, Any]:
    turns = raw.get("turns") or []
    if not turns:
        return {}
    turn = turns[0]
    return turn if isinstance(turn, dict) else {}


def token_values(tokens: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        value = finite_float(token.get(field))
        if value is not None:
            values.append(value)
    return values


def sliding_window_turns(tokens: list[dict[str, Any]], window_size: int, stride: int) -> list[dict[str, Any]]:
    if window_size <= 0 or stride <= 0:
        raise SystemExit("--window-size and --stride must be positive")
    windows: list[dict[str, Any]] = []
    if not tokens:
        return windows
    if len(tokens) <= window_size:
        chunks = [(0, tokens)]
    else:
        chunks = [(start, tokens[start : start + window_size]) for start in range(0, len(tokens) - window_size + 1, stride)]
        last_start = len(tokens) - window_size
        if chunks and chunks[-1][0] != last_start:
            chunks.append((last_start, tokens[last_start:]))
    for index, (_start, chunk) in enumerate(chunks):
        windows.append({"turn_index": index, "tokens": chunk})
    return windows


def truncation_flag(finish_reason: Any) -> bool:
    text = str(finish_reason or "").strip().casefold()
    return text in {"length", "max_tokens", "token_limit", "truncated"}


def row_from_raw(raw_path: Path, window_size: int, stride: int) -> dict[str, Any]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    turn = first_turn(raw)
    raw_tokens = turn.get("tokens") or []
    tokens = [token for token in raw_tokens if isinstance(token, dict)]

    entropy_values = token_values(tokens, "entropy")
    gap_values = token_values(tokens, "gap")
    logprob_values = token_values(tokens, "top1_logprob")
    mean_logprob = mean(logprob_values)
    perplexity = math.exp(-mean_logprob) if mean_logprob is not None else None

    window_turns = sliding_window_turns(tokens, window_size=window_size, stride=stride)
    prama = measure(
        window_turns,
        calib_window=1,
        min_turns_for_regime=3,
        min_windows_for_regime=12,
        crossing_index_scope="token_window",
    ) if window_turns else {}

    token_count = finite_float(turn.get("token_count"))
    if token_count is None:
        token_count = float(len(tokens))
    finish_reason = turn.get("finish_reason")

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
        "finish_reason": finish_reason,
        "token_count": token_count,
        "assistant_chars": len(str(turn.get("assistant_message") or "")),
        "window_count": len(window_turns),
        "mean_entropy": mean(entropy_values),
        "entropy_variance": variance(entropy_values),
        "mean_gap": mean(gap_values),
        "mean_top1_logprob": mean_logprob,
        "perplexity": perplexity,
        "threshold_crossing_ratio": prama.get("threshold_crossing_ratio"),
        "persistent_crossing_ratio": prama.get("persistent_crossing_ratio"),
        "recovery_observed": prama.get("recovery_observed"),
        "regime": prama.get("regime_label") or "UNAVAILABLE",
        "trajectory_assessment": prama.get("trajectory_assessment"),
        "truncation_flag": truncation_flag(finish_reason),
        "raw_path": str(raw_path),
    }


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "UNKNOWN") for row in rows).items()))


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


def nested_regime_distribution(rows: list[dict[str, Any]], by_field: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        group = str(row.get(by_field) or "UNKNOWN")
        regime = str(row.get("regime") or "UNKNOWN")
        grouped[group][regime] += 1
    return {group: dict(sorted(counter.items())) for group, counter in sorted(grouped.items())}


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


def render_nested_distribution(title: str, nested: dict[str, dict[str, int]]) -> list[str]:
    lines = [f"### {title}", "", "| group | regime | count |", "|---|---|---:|"]
    for group, counts in nested.items():
        for regime, count in counts.items():
            lines.append(f"| {group} | {regime} | {count} |")
    lines.append("")
    return lines


def render_summary_table(title: str, numeric: dict[str, dict[str, float | None]], fields: list[str]) -> list[str]:
    lines = [f"### {title}", "", "| metric | mean | median | p90 | p10 | min | max |", "|---|---:|---:|---:|---:|---:|---:|"]
    for field in fields:
        row = numeric[field]
        lines.append(
            f"| {field} | {fmt(row['mean'])} | {fmt(row['median'])} | {fmt(row['p90'])} | "
            f"{fmt(row['p10'])} | {fmt(row['min'])} | {fmt(row['max'])} |"
        )
    lines.append("")
    return lines


def render_top_table(title: str, rows: list[dict[str, Any]], sort_field: str) -> list[str]:
    ordered = sorted(rows, key=lambda row: finite_float(row.get(sort_field)) if finite_float(row.get(sort_field)) is not None else float("-inf"), reverse=True)[:10]
    lines = [f"### {title}", "", "| session_id | value | finish_reason | difficulty | regime |", "|---|---:|---|---|---|"]
    for row in ordered:
        lines.append(
            f"| {row.get('session_id')} | {fmt(row.get(sort_field))} | {row.get('finish_reason') or ''} | "
            f"{row.get('difficulty') or ''} | {row.get('regime') or ''} |"
        )
    lines.append("")
    return lines


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# CoCC PRAMA Window Analysis Report",
        "",
        f"- total raw_json analizados: `{summary['raw_json_analyzed']}`",
        f"- window_size: `{summary['window_size']}`",
        f"- stride: `{summary['stride']}`",
        "",
        "Cases with `finish_reason=length` are truncated by an external token limit and must be interpreted separately.",
        "",
    ]
    lines.extend(render_distribution("finish_reason", summary["distributions"]["finish_reason"]))
    lines.extend(render_distribution("regime", summary["distributions"]["regime"]))
    lines.extend(render_nested_distribution("regime by finish_reason", summary["regime_by_finish_reason"]))
    lines.extend(render_nested_distribution("regime by difficulty", summary["regime_by_difficulty"]))
    lines.extend(
        render_summary_table(
            "PRAMA ratios and token metrics",
            summary["numeric"],
            [
                "token_count",
                "assistant_chars",
                "mean_entropy",
                "entropy_variance",
                "mean_gap",
                "mean_top1_logprob",
                "perplexity",
                "threshold_crossing_ratio",
                "persistent_crossing_ratio",
            ],
        )
    )
    lines.extend(render_top_table("Top 10 by token_count", rows, "token_count"))
    lines.extend(render_top_table("Top 10 by mean_entropy", rows, "mean_entropy"))
    lines.extend(render_top_table("Top 10 by persistent_crossing_ratio", rows, "persistent_crossing_ratio"))
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(sessions_dir: Path, output_dir: Path, window_size: int, stride: int) -> dict[str, Any]:
    raw_paths = sorted(sessions_dir.glob("*/raw.json"))
    if not raw_paths:
        raise SystemExit(f"no raw.json files found under {sessions_dir}")
    rows = [row_from_raw(raw_path, window_size=window_size, stride=stride) for raw_path in raw_paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_fields = [
        "token_count",
        "assistant_chars",
        "window_count",
        "mean_entropy",
        "entropy_variance",
        "mean_gap",
        "mean_top1_logprob",
        "perplexity",
        "threshold_crossing_ratio",
        "persistent_crossing_ratio",
    ]
    summary = {
        "raw_json_analyzed": len(rows),
        "sessions_dir": str(sessions_dir),
        "output_dir": str(output_dir),
        "window_size": window_size,
        "stride": stride,
        "distributions": {
            "finish_reason": distribution(rows, "finish_reason"),
            "regime": distribution(rows, "regime"),
            "benchmark_name": distribution(rows, "benchmark_name"),
            "benchmark_alias": distribution(rows, "benchmark_alias"),
            "perturbation_type": distribution(rows, "perturbation_type"),
            "difficulty": distribution(rows, "difficulty"),
        },
        "regime_by_finish_reason": nested_regime_distribution(rows, "finish_reason"),
        "regime_by_difficulty": nested_regime_distribution(rows, "difficulty"),
        "numeric": {field: numeric_summary(rows, field) for field in numeric_fields},
    }
    write_csv(rows, output_dir / "cocc_prama_window_metrics.csv")
    (output_dir / "cocc_prama_window_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(summary, rows, output_dir / "cocc_prama_window_report.md")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = analyze(
        Path(args.sessions_dir),
        Path(args.output_dir),
        window_size=args.window_size,
        stride=args.stride,
    )
    print(f"raw_json_analyzed={summary['raw_json_analyzed']}")
    print(f"wrote {Path(args.output_dir) / 'cocc_prama_window_metrics.csv'}")
    print(f"wrote {Path(args.output_dir) / 'cocc_prama_window_report.md'}")
    print(f"wrote {Path(args.output_dir) / 'cocc_prama_window_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
