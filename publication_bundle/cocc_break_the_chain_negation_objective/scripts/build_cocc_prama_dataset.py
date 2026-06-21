#!/usr/bin/env python
"""Build a normalized PRAMA JSONL dataset from official CoCC pickle files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inspect_cocc_dataset import (
    LIVECODEBENCH_FIELDS,
    SHORTCUTQA_FIELDS,
    load_rows,
    modified_columns,
    perturbation_type_from_path_or_column,
)

CANONICAL_BENCHMARK = "chain_of_code_collapse"
BENCHMARK_ALIAS = "break_the_chain_code_generation"
REQUIRED_REAL_FIELDS = {
    "question_title",
    "question_content",
    "platform",
    "question_id",
    "contest_id",
    "contest_date",
    "starter_code",
    "difficulty",
    "public_test_cases",
    "private_test_cases",
    "metadata",
}


def discover_official_pickle_files(repo_dir: Path) -> list[Path]:
    data_modified = repo_dir / "data_modified"
    search_root = data_modified if data_modified.exists() else repo_dir
    files = sorted(search_root.glob("modified_problems_*.pkl"))
    if files:
        return files
    return sorted(path for path in search_root.rglob("*.pkl") if "data_modified" in path.parts and path.name.startswith("modified_problems_"))


def reject_if_clean_or_shortcut(rows: list[dict[str, Any]]) -> None:
    keys = set().union(*(row.keys() for row in rows[:20])) if rows else set()
    if LIVECODEBENCH_FIELDS.issubset(keys) and not modified_columns(keys):
        raise SystemExit("LiveCodeBench-clean detected; this is not CoCC perturbed data.")
    if keys.intersection(SHORTCUTQA_FIELDS):
        raise SystemExit("ShortcutQA-like schema detected; this is a different benchmark and is not used here.")


def normalize_value(value: Any) -> Any:
    if value is None:
        return None
    # Keep JSON-compatible structures when pandas stores lists/dicts in object columns.
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def normalize_official_file(path: Path, limit_per_perturbation: int | None = None) -> list[dict[str, Any]]:
    rows = load_rows(path)
    if not rows:
        return []
    reject_if_clean_or_shortcut(rows)
    keys = set().union(*(row.keys() for row in rows[:20]))
    missing = REQUIRED_REAL_FIELDS.difference(keys)
    mods = modified_columns(keys)
    if missing:
        raise SystemExit(f"{path} missing required CoCC fields: {', '.join(sorted(missing))}")
    if len(mods) != 1:
        raise SystemExit(f"{path} must contain exactly one *_modified column")
    modified_column = mods[0]
    perturbation_type = perturbation_type_from_path_or_column(path, keys)
    if not perturbation_type:
        raise SystemExit(f"{path} missing perturbation_type")

    out: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        perturbed_prompt = row.get(modified_column)
        if perturbed_prompt in (None, ""):
            continue
        question_id = normalize_value(row.get("question_id"))
        item_id = f"{question_id}_{perturbation_type}_{row_index}"
        out.append(
            {
                "benchmark_name": CANONICAL_BENCHMARK,
                "benchmark_alias": BENCHMARK_ALIAS,
                "source_file": str(path),
                "perturbation_type": perturbation_type,
                "item_id": str(item_id),
                "problem_id": str(question_id),
                "question_title": normalize_value(row.get("question_title")),
                "question_id": question_id,
                "platform": normalize_value(row.get("platform")),
                "contest_id": normalize_value(row.get("contest_id")),
                "contest_date": normalize_value(row.get("contest_date")),
                "difficulty": normalize_value(row.get("difficulty")),
                "clean_prompt": normalize_value(row.get("question_content")),
                "perturbed_prompt": normalize_value(perturbed_prompt),
                "starter_code": normalize_value(row.get("starter_code")),
                "public_test_cases": normalize_value(row.get("public_test_cases")),
                "private_test_cases": normalize_value(row.get("private_test_cases")),
                "metadata": normalize_value(row.get("metadata")),
                "verifier_ref": f"question_id:{question_id}",
                "test_ref": f"public_test_cases:{question_id}",
                "split": "test",
            }
        )
        if limit_per_perturbation is not None and len(out) >= limit_per_perturbation:
            break
    return out


def normalize_rows(repo_dir: Path, limit_per_perturbation: int | None = None) -> list[dict[str, Any]]:
    files = discover_official_pickle_files(repo_dir)
    if not files:
        discovered = []
        for path in repo_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv", ".pkl", ".pickle"}:
                discovered.extend(load_rows(path))
        reject_if_clean_or_shortcut(discovered)
        raise SystemExit("no official CoCC modified_problems_*.pkl files found")

    out: list[dict[str, Any]] = []
    for path in files:
        # Paper-native result logs under results/attacked are outputs, not PRAMA inputs.
        if "results" in path.parts and "attacked" in path.parts:
            continue
        out.extend(normalize_official_file(path, limit_per_perturbation))
    if not out:
        raise SystemExit("no normalizable CoCC rows with verifier mapping found")
    return out


def write_jsonl(rows: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--limit-per-perturbation", type=int, default=None)
    args = parser.parse_args(argv)
    rows = normalize_rows(Path(args.repo_dir), args.limit_per_perturbation)
    write_jsonl(rows, Path(args.output_file))
    print(f"wrote {len(rows)} normalized CoCC items to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
