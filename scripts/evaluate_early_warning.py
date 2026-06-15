#!/usr/bin/env python
"""Evaluate external early-warning signal against logprob-derived baselines."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from aptadynamik.evaluation.early_warning_metrics import (
    BASELINE_SCORE_FIELDS,
    PRIMARY_PRAMA_SCORE,
    PRAMA_SCORE_FIELDS,
    auroc,
    baseline_score_from_tokens,
    confusion_at_threshold,
    matched_fpr_comparison,
    prama_score_from_row,
    precision_recall_at_threshold,
    safe_float,
    threshold_at_fpr,
)

REQUIRED_LABEL_FIELDS = {"session_id", "label", "event_token", "event_turn", "event_type"}


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"labels file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("labels.csv is empty or missing a header")
        missing = REQUIRED_LABEL_FIELDS.difference(reader.fieldnames)
        if missing:
            raise SystemExit(f"labels.csv missing required fields: {', '.join(sorted(missing))}")
        rows = {row["session_id"]: row for row in reader if row.get("session_id")}
    if not rows:
        raise SystemExit("labels.csv contains no labeled sessions")
    return rows


def _session_id_from_result(raw: dict[str, Any], path: Path) -> str:
    return str(raw.get("session_id") or path.stem)


def _flatten_turns(raw: dict[str, Any]) -> list[dict[str, Any]]:
    turns = raw.get("turns") or []
    out: list[dict[str, Any]] = []
    cumulative_tokens: list[dict[str, Any]] = []
    for turn in turns:
        tokens = list(turn.get("tokens") or [])
        cumulative_tokens.extend(tokens)
        token_count = int(turn.get("token_count") or len(tokens) or 0)
        cumulative_position = len(cumulative_tokens) if cumulative_tokens else sum(int(row.get("token_count", 0)) for row in out) + token_count
        summary = turn.get("metrics_summary") or turn.get("summary") or turn
        row = dict(summary)
        row["turn_index"] = int(turn.get("turn_index") or len(out))
        row["token_position"] = cumulative_position
        row["token_count"] = token_count
        row["tokens_causal_prefix"] = list(cumulative_tokens)
        for field in BASELINE_SCORE_FIELDS:
            row.setdefault(field, baseline_score_from_tokens(field, cumulative_tokens))
        out.append(row)
    return out


def _derive_score(row: dict[str, Any], field: str) -> float:
    if field in PRAMA_SCORE_FIELDS:
        return prama_score_from_row(field, row)
    return safe_float(row.get(field))


def load_examples(paths: list[Path], labels: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        session_id = _session_id_from_result(raw, path)
        label = labels.get(session_id)
        if not label:
            raise SystemExit(f"missing label for session_id {session_id} from {path}")
        turns = _flatten_turns(raw)
        if not turns:
            raise SystemExit(f"input has no turns: {path}")
        event_token = int(float(label["event_token"]))
        final_token = int(turns[-1].get("token_position", event_token))
        final_outcome_proxy = event_token >= final_token
        # Causal choice: latest available row at or before event_token.
        chosen = turns[0]
        for turn in turns:
            if int(turn.get("token_position", 0)) <= event_token:
                chosen = turn
        example = {
            "session_id": session_id,
            "label": label["label"],
            "event_token": event_token,
            "event_turn": int(float(label["event_turn"])),
            "event_type": label["event_type"],
            "token_position": int(chosen.get("token_position", event_token)),
            "turn_index": int(chosen.get("turn_index", 0)),
            "split": (label.get("split") or "test").strip().lower(),
            "final_outcome_proxy": final_outcome_proxy,
            "scores": {},
        }
        for field in PRAMA_SCORE_FIELDS + BASELINE_SCORE_FIELDS:
            example["scores"][field] = _derive_score(chosen, field)
        examples.append(example)
    return examples


def _safe_auroc(scores: list[float], labels: list[Any]) -> float | None:
    try:
        return auroc(scores, labels)
    except ValueError:
        return None


def _metric_block(train: list[dict[str, Any]], test: list[dict[str, Any]], field: str, target_fpr: float) -> dict[str, Any]:
    labels_train = [row["label"] for row in train]
    labels_test = [row["label"] for row in test]
    train_scores = [row["scores"][field] for row in train]
    test_scores = [row["scores"][field] for row in test]
    try:
        threshold = threshold_at_fpr(train_scores, labels_train, target_fpr)
        confusion = confusion_at_threshold(test_scores, labels_test, threshold)
        precision_recall = precision_recall_at_threshold(test_scores, labels_test, threshold)
        error = None
    except ValueError as exc:
        threshold = None
        confusion = None
        precision_recall = None
        error = str(exc)
    return {
        "threshold": threshold,
        "auroc": _safe_auroc(test_scores, labels_test),
        "confusion": confusion,
        "precision_recall": precision_recall,
        "error": error,
    }


def _status_from_comparison(best_prama: tuple[str, float] | None, best_baseline: tuple[str, float] | None, best_comparison: dict[str, Any] | None) -> str:
    if best_prama is None or best_baseline is None:
        return "inconclusive"
    auroc_positive = best_prama[1] > best_baseline[1]
    lead_delta = None if best_comparison is None else best_comparison.get("lead_token_delta")
    lead_positive = lead_delta is not None and lead_delta > 0
    if auroc_positive or lead_positive:
        return "positive"
    return "negative"


def evaluate_split(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    target_fpr: float,
    primary_score: str = PRIMARY_PRAMA_SCORE,
) -> dict[str, Any]:
    if primary_score not in PRAMA_SCORE_FIELDS:
        raise ValueError(f"unknown primary PRAMA score: {primary_score}")
    if not train:
        train = test
    output: dict[str, Any] = {"target_fpr": target_fpr, "primary_score": primary_score, "prama": {}, "baselines": {}, "matched_fpr": {}}

    for field in PRAMA_SCORE_FIELDS:
        output["prama"][field] = _metric_block(train, test, field, target_fpr)

    for field in BASELINE_SCORE_FIELDS:
        output["baselines"][field] = _metric_block(train, test, field, target_fpr)

    labels_test = [row["label"] for row in test]
    token_positions = [row["token_position"] for row in test]
    event_tokens = [row["event_token"] for row in test]
    primary_scores = [row["scores"][primary_score] for row in test]
    for field in BASELINE_SCORE_FIELDS:
        try:
            output["matched_fpr"][field] = matched_fpr_comparison(
                primary_scores,
                [row["scores"][field] for row in test],
                labels_test,
                token_positions,
                event_tokens,
                target_fpr,
            )
        except ValueError as exc:
            output["matched_fpr"][field] = {"error": str(exc)}

    comparable_prama = [(name, data["auroc"]) for name, data in output["prama"].items() if data["auroc"] is not None]
    comparable_base = [(name, data["auroc"]) for name, data in output["baselines"].items() if data["auroc"] is not None]
    output["best_prama_by_auroc"] = max(comparable_prama, key=lambda item: item[1], default=None)
    output["best_baseline_by_auroc"] = max(comparable_base, key=lambda item: item[1], default=None)
    best_baseline_name = output["best_baseline_by_auroc"][0] if output["best_baseline_by_auroc"] else None
    best_comparison = output["matched_fpr"].get(best_baseline_name) if best_baseline_name else None
    output["status"] = _status_from_comparison(output["best_prama_by_auroc"], output["best_baseline_by_auroc"], best_comparison)
    return output


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Empirical Result 008 Early-Warning Evaluation",
        "",
        "PRAMA is evaluated as a structural early-warning signal, not as a semantic hallucination detector.",
        "",
        "## Null Hypothesis",
        "",
        "PRAMA adds no predictive signal beyond raw entropy/logprob/perplexity baselines.",
        "",
        "## Summary",
        "",
        f"- target_fpr: `{metrics['target_fpr']}`",
        f"- primary_score: `{metrics['primary_score']}`",
        f"- n_examples: `{metrics['n_examples']}`",
        f"- split_aware: `{metrics['split_aware']}`",
        f"- final_outcome_proxy_count: `{metrics['final_outcome_proxy_count']}`",
    ]
    for split_name, split_metrics in metrics["splits"].items():
        lines.extend([
            "",
            f"## Split: {split_name}",
            "",
            f"- status: `{split_metrics.get('status')}`",
            f"- best_prama_by_auroc: `{split_metrics.get('best_prama_by_auroc')}`",
            f"- best_baseline_by_auroc: `{split_metrics.get('best_baseline_by_auroc')}`",
            "",
            "### PRAMA Scores",
            "",
        ])
        for name, data in split_metrics["prama"].items():
            lines.append(f"- {name}: AUROC={data['auroc']} threshold={data['threshold']} confusion={data['confusion']}")
        lines.extend(["", "### Baselines", ""])
        for name, data in split_metrics["baselines"].items():
            lines.append(f"- {name}: AUROC={data['auroc']} threshold={data['threshold']} confusion={data['confusion']}")
    lines.extend([
        "",
        "## Methodological Note",
        "",
        "Ground truth must come from automatic verification, not from PRAMA and not from the author. A null result is admissible and must be reported.",
        "If event_token equals the final token, lead time is a final-outcome proxy rather than a localized failure lead.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--primary-score", default=PRIMARY_PRAMA_SCORE)
    parser.add_argument("--output-dir", type=Path, default=Path("results/early_warning_eval"))
    args = parser.parse_args()

    labels = load_labels(args.labels)
    examples = load_examples(args.inputs, labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_values = {row["split"] for row in examples}
    split_aware = bool(split_values.difference({"", "test"}))
    metrics: dict[str, Any] = {
        "target_fpr": args.target_fpr,
        "primary_score": args.primary_score,
        "n_examples": len(examples),
        "split_aware": split_aware,
        "final_outcome_proxy_count": sum(1 for row in examples if row["final_outcome_proxy"]),
        "label_counts": dict(Counter(str(row["label"]) for row in examples)),
        "splits": {},
    }

    if split_aware:
        train = [row for row in examples if row["split"] in {"train", "calibration", "calib"}]
        test = [row for row in examples if row["split"] == "test"]
        if not test:
            raise SystemExit("split column exists but no test examples were found")
        metrics["splits"]["test"] = evaluate_split(train, test, args.target_fpr, args.primary_score)
    else:
        metrics["splits"]["all"] = evaluate_split(examples, examples, args.target_fpr, args.primary_score)

    (args.output_dir / "early_warning_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(args.output_dir / "early_warning_report.md", metrics)
    print(f"wrote {args.output_dir / 'early_warning_metrics.json'}")
    print(f"wrote {args.output_dir / 'early_warning_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
