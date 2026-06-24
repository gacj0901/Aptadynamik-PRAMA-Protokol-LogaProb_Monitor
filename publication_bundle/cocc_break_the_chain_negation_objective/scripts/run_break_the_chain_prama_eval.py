#!/usr/bin/env python
"""Strict CoCC / Break-The-Chain code-generation PRAMA runner.

This runner consumes normalized Chain-of-Code Collapse perturbation data. It
refuses clean LiveCodeBench datasets, ShortcutQA schemas, and native harness
execution. PRAMA uses only perturbed prompts, metadata, and verifier mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    text_path = str(path)
    if text_path not in sys.path:
        sys.path.insert(0, text_path)

LABEL_FIELDS = [
    "session_id",
    "label",
    "event_token",
    "event_turn",
    "event_type",
    "prompt_id",
    "expected_answer",
    "observed_answer",
    "verifier_name",
    "backend",
    "split",
    "benchmark_name",
    "perturbation_type",
]

CANONICAL_BENCHMARK = "chain_of_code_collapse"
BENCHMARK_ALIAS = "break_the_chain_code_generation"
LIVECODEBENCH_FIELDS = {"question_content", "public_test_cases", "question_id"}
SHORTCUTQA_FIELDS = {"shortcut", "shortcut_answer", "shortcut_reasoning", "question_type"}
PERTURBED_PROMPT_FIELDS = ("perturbed_prompt", "prompt_perturbed", "btc_prompt", "cocc_prompt")
ID_FIELDS = ("problem_id", "item_id", "id", "question_id")
VERIFIER_FIELDS = ("verifier_ref", "test_ref", "lcb_problem_ref", "public_test_cases", "test_cases", "expected_output")
NATIVE_HARNESS_FIELDS = {
    "anthropic_model",
    "anthropic_response",
    "gemini_model",
    "gemini_response",
    "native_harness",
    "native_harness_used",
}


@dataclass(frozen=True)
class CoccItem:
    problem_id: str
    item_id: str
    perturbed_prompt: str
    perturbation_type: str
    split: str
    verifier_name: str
    verifier_ref: str
    label: int | None
    expected_answer: str
    source: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or uuid4().hex


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"dataset file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        for key in ("items", "data", "examples", "rows"):
            value = data.get(key) if isinstance(data, dict) else None
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        raise SystemExit("JSON dataset must be a list or contain items/data/examples/rows")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise SystemExit(f"unsupported dataset suffix: {path.suffix}")


def first_present(row: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return None


def normalize_label(value: Any) -> int | None:
    if value is None or value == "":
        return None
    text = str(value).strip().casefold()
    if text in {"1", "fail", "failed", "false", "incorrect", "wrong", "no"}:
        return 1
    if text in {"0", "pass", "passed", "true", "correct", "yes"}:
        return 0
    return None


def validate_cocc_dataset(rows: list[dict[str, Any]]) -> list[CoccItem]:
    if not rows:
        raise SystemExit("dataset contains no rows")
    keys = set().union(*(row.keys() for row in rows[:20]))
    benchmark_names = {str(row.get("benchmark_name") or "").strip() for row in rows[:20] if row.get("benchmark_name")}
    benchmark_aliases = {str(row.get("benchmark_alias") or "").strip() for row in rows[:20] if row.get("benchmark_alias")}
    if keys.intersection(SHORTCUTQA_FIELDS) or "shortcutqa" in {name.casefold() for name in benchmark_names}:
        raise SystemExit("ShortcutQA schema detected; this is a different benchmark and is not used here.")
    if keys.intersection(NATIVE_HARNESS_FIELDS):
        raise SystemExit("Native Anthropic/Gemini harness fields detected. PRAMA must use normalized CoCC inputs only.")
    if LIVECODEBENCH_FIELDS.issubset(keys) and "perturbation_type" not in keys:
        raise SystemExit("LiveCodeBench-clean detected. Refusing to run as Chain-of-Code Collapse.")
    if "perturbation_type" not in keys:
        raise SystemExit("This is not a Chain-of-Code Collapse dataset: perturbation_type missing.")
    if not any(field in keys for field in PERTURBED_PROMPT_FIELDS):
        raise SystemExit("This is not a Chain-of-Code Collapse dataset: perturbed_prompt missing.")
    if not benchmark_names or benchmark_names != {CANONICAL_BENCHMARK}:
        raise SystemExit("benchmark_name must be chain_of_code_collapse")
    if not benchmark_aliases or benchmark_aliases != {BENCHMARK_ALIAS}:
        raise SystemExit("benchmark_alias must be break_the_chain_code_generation")
    items: list[CoccItem] = []
    for index, row in enumerate(rows):
        perturbation_type = str(row.get("perturbation_type") or "").strip()
        prompt = first_present(row, PERTURBED_PROMPT_FIELDS)
        verifier_ref = first_present(row, VERIFIER_FIELDS)
        benchmark_name = str(row.get("benchmark_name") or "").strip()
        benchmark_alias = str(row.get("benchmark_alias") or "").strip()
        if benchmark_name != CANONICAL_BENCHMARK or benchmark_alias != BENCHMARK_ALIAS:
            raise SystemExit("all rows must use canonical Chain-of-Code Collapse benchmark metadata")
        if not perturbation_type or not prompt:
            continue
        if not verifier_ref:
            raise SystemExit("dataset has no verifier/mapping for CoCC item")
        problem_id = str(first_present(row, ID_FIELDS) or f"item-{index}")
        item_id = str(row.get("item_id") or problem_id)
        label = normalize_label(first_present(row, ("label", "passed", "pass", "is_correct")))
        items.append(
            CoccItem(
                problem_id=problem_id,
                item_id=item_id,
                perturbed_prompt=str(prompt),
                perturbation_type=perturbation_type,
                split=str(row.get("split") or "test"),
                verifier_name="cocc_external_verifier_mapping",
                verifier_ref=str(verifier_ref),
                label=label,
                expected_answer=str(row.get("expected_answer") or row.get("expected_output") or ""),
                source=dict(row),
            )
        )
    if not items:
        raise SystemExit("dataset has no usable CoCC rows")
    return items


def synthetic_raw_for_dry_run(session_id: str, item: CoccItem, provider: str, model: str) -> dict[str, Any]:
    tokens = []
    words = ("dry run cocc perturbation " + item.perturbation_type + " " + item.perturbed_prompt).split()[:32]
    for index, word in enumerate(words or ["dry"]):
        lp = -0.2 - (index % 5) * 0.05
        tokens.append({"token": word, "top1_logprob": lp, "top_logprobs": [lp, lp - 1.0], "gap": 1.0, "entropy": 0.58})
    return {
        "session_id": session_id,
        "provider": provider,
        "requested_model": model,
        "resolved_model": model,
        "model": model,
        "benchmark_name": CANONICAL_BENCHMARK,
        "benchmark_alias": BENCHMARK_ALIAS,
        "perturbation_type": item.perturbation_type,
        "verifier_ref": item.verifier_ref,
        "turns": [
            {
                "turn_index": 0,
                "user_message": item.perturbed_prompt,
                "assistant_message": "DRY RUN: no API call executed.",
                "finish_reason": "dry_run",
                "token_count": len(tokens),
                "tokens": tokens,
            }
        ],
    }


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def call_prama_backend(item: CoccItem, args: argparse.Namespace) -> Any:
    messages = [{"role": "user", "content": item.perturbed_prompt}]
    if args.provider == "deepseek":
        from aptadynamik.pipelines.deepseek import DeepSeekConfig, deepseek_chat_completion

        return deepseek_chat_completion(
            messages,
            DeepSeekConfig(
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_logprobs=args.top_logprobs,
            ),
        )
    if args.provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the OpenAI SDK to run OpenAI CoCC sessions.") from exc
        client = OpenAI()
        return client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            logprobs=True,
            top_logprobs=args.top_logprobs,
        )
    raise SystemExit(f"unsupported provider: {args.provider}")


def raw_from_response(session_id: str, item: CoccItem, args: argparse.Namespace, response: Any) -> dict[str, Any]:
    from aptadynamik.pipelines.deepseek import deepseek_response_to_raw_turn

    turn = deepseek_response_to_raw_turn(response, turn_index=0, user_message=item.perturbed_prompt)
    resolved_model = _get_attr(response, "model", args.model) or args.model
    return {
        "session_id": session_id,
        "provider": args.provider,
        "requested_model": args.model,
        "resolved_model": resolved_model,
        "model": resolved_model,
        "benchmark_name": CANONICAL_BENCHMARK,
        "benchmark_alias": BENCHMARK_ALIAS,
        "perturbation_type": item.perturbation_type,
        "item_id": item.item_id,
        "problem_id": item.problem_id,
        "source_file": str(item.source.get("source_file") or ""),
        "verifier_ref": item.verifier_ref,
        "metadata": {
            "benchmark_name": CANONICAL_BENCHMARK,
            "benchmark_alias": BENCHMARK_ALIAS,
            "perturbation_type": item.perturbation_type,
            "item_id": item.item_id,
            "problem_id": item.problem_id,
            "source_file": str(item.source.get("source_file") or ""),
            "clean_prompt": str(item.source.get("clean_prompt") or item.source.get("question_content") or ""),
            "verifier_ref": item.verifier_ref,
            "difficulty": item.source.get("difficulty"),
            "platform": item.source.get("platform"),
            "contest_id": item.source.get("contest_id"),
        },
        "turns": [turn],
}


def call_backend_with_retries(
    item: CoccItem,
    args: argparse.Namespace,
    call_fn: Callable[[CoccItem, argparse.Namespace], Any],
) -> Any:
    attempts = max(1, int(getattr(args, "max_attempts", 1) or 1))
    sleep_seconds = max(0.0, float(getattr(args, "retry_sleep_seconds", 0.0) or 0.0))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call_fn(item, args)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            print(
                f"backend call failed for item_id={item.item_id} "
                f"(attempt {attempt}/{attempts}): {exc}; retrying in {sleep_seconds:g}s",
                file=sys.stderr,
            )
            if sleep_seconds > 0.0:
                time.sleep(sleep_seconds)
    raise RuntimeError(
        f"backend call failed after {attempts} attempt(s) for item_id={item.item_id}: {last_error}"
    ) from last_error


def write_outputs(
    items: list[CoccItem],
    args: argparse.Namespace,
    call_fn: Callable[[CoccItem, argparse.Namespace], Any] | None = None,
) -> dict[str, Any]:
    out = Path(args.output_dir)
    sessions_dir = out / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    index_rows = []
    allowed_types = set(args.perturbation_type or [])
    selected = items
    if allowed_types:
        selected = [item for item in selected if item.perturbation_type in allowed_types]
    selected = selected[: args.n] if args.n else selected
    for item in selected:
        session_id = sanitize_id(f"cocc-{item.item_id}-{uuid4().hex[:8]}")
        supported = item.label in {0, 1}
        raw_path = sessions_dir / session_id / "raw.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            raw = synthetic_raw_for_dry_run(session_id, item, args.provider, args.model)
        else:
            response = call_backend_with_retries(item, args, call_fn or call_prama_backend)
            raw = raw_from_response(session_id, item, args, response)
        raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
        event_token = raw["turns"][-1]["token_count"]
        assistant_message = raw["turns"][-1].get("assistant_message", "")
        if supported:
            rows.append(
                {
                    "session_id": session_id,
                    "label": item.label,
                    "event_token": event_token,
                    "event_turn": 0,
                    "event_type": "verification_failure" if item.label == 1 else "final_answer",
                    "prompt_id": item.problem_id,
                    "expected_answer": item.expected_answer,
                    "observed_answer": assistant_message,
                    "verifier_name": item.verifier_name,
                    "backend": args.provider,
                    "split": item.split,
                    "benchmark_name": CANONICAL_BENCHMARK,
                    "perturbation_type": item.perturbation_type,
                }
            )
        index_rows.append(
            {
                "session_id": session_id,
                "problem_id": item.problem_id,
                "item_id": item.item_id,
                "perturbation_type": item.perturbation_type,
                "supported": supported,
                "unsupported_reason": "" if supported else "missing_external_label_or_verifier_result",
                "raw_path": str(raw_path),
                "benchmark_name": CANONICAL_BENCHMARK,
                "benchmark_alias": BENCHMARK_ALIAS,
                "source_file": str(item.source.get("source_file") or ""),
            }
        )
    labels_path = out / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    index = {
        "generated_at": utc_now(),
        "benchmark_name": CANONICAL_BENCHMARK,
        "benchmark_alias": BENCHMARK_ALIAS,
        "dry_run": bool(args.dry_run),
        "provider": args.provider,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_logprobs": args.top_logprobs,
        "native_harness_used": False,
        "native_harness_note": "Anthropic/Gemini harness is not used by PRAMA.",
        "session_count": len(index_rows),
        "labeled_count": len(rows),
        "sessions": index_rows,
    }
    (out / "session_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [
        "# Chain-of-Code Collapse PRAMA Session Index",
        "",
        "CoCC/BTC code-generation perturbation evaluation uses normalized perturbed prompts only.",
        "The native Anthropic/Gemini harness is not used by PRAMA.",
        "",
        f"- benchmark_name: `{CANONICAL_BENCHMARK}`",
        f"- benchmark_alias: `{BENCHMARK_ALIAS}`",
        f"- dry_run: `{bool(args.dry_run)}`",
        f"- provider: `{args.provider}`",
        f"- model: `{args.model}`",
        f"- native_harness_used: `False`",
        f"- session_count: `{len(index_rows)}`",
        f"- labeled_count: `{len(rows)}`",
        "",
        "| session_id | problem_id | perturbation_type | supported | source_file | raw_path |",
        "|---|---|---|---|---|---|",
    ]
    for row in index_rows:
        md.append(
            f"| {row['session_id']} | {row['problem_id']} | {row['perturbation_type']} | "
            f"{row['supported']} | {row['source_file']} | {row['raw_path']} |"
        )
    (out / "session_index.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "validation").mkdir(exist_ok=True)
    (out / "evaluation").mkdir(exist_ok=True)
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-dataset-file", required=True)
    parser.add_argument("--provider", choices=["deepseek", "openai"], default="deepseek")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=30.0)
    parser.add_argument("--output-dir", default="results/break_the_chain_prama_deepseek")
    parser.add_argument("--perturbation-type", action="append")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None, call_fn: Callable[[CoccItem, argparse.Namespace], Any] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = load_dataset(Path(args.btc_dataset_file))
    items = validate_cocc_dataset(rows)
    index = write_outputs(items, args, call_fn=call_fn)
    print(f"benchmark_name={CANONICAL_BENCHMARK}")
    print(f"session_count={index['session_count']}")
    print(f"labeled_count={index['labeled_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
