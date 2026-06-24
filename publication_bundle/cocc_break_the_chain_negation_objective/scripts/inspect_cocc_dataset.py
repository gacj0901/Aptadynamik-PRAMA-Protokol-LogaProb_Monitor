#!/usr/bin/env python
"""Inspect a local Chain-of-Code Collapse repository for dataset schema hints."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised only when pandas is absent.
    pd = None

INTERESTING_FILES = {
    "main_perturbation.py",
    "run_script_main_perturbation.py",
    "maps.py",
    "modify_problem.py",
}
DATA_DIR_HINTS = {"data_modified", "data"}
REAL_COCC_REQUIRED_FIELDS = {"question_content", "public_test_cases", "metadata"}
PROMPT_FIELDS = ("perturbed_prompt", "prompt_perturbed", "modified_prompt", "prompt", "question_content")
CLEAN_FIELDS = ("clean_prompt", "original_prompt", "question_content", "clean_problem_ref", "source_problem")
ID_FIELDS = ("problem_id", "question_id", "item_id", "id")
VERIFIER_FIELDS = ("verifier_ref", "test_ref", "lcb_problem_ref", "public_test_cases", "test_cases", "expected_output")
SHORTCUTQA_FIELDS = {"shortcut", "shortcut_answer", "shortcut_reasoning", "question_type"}
LIVECODEBENCH_FIELDS = {"question_content", "public_test_cases", "question_id"}


def modified_columns(fields: set[str]) -> list[str]:
    return sorted(field for field in fields if field.endswith("_modified"))


def perturbation_type_from_path_or_column(path: Path | None, fields: set[str]) -> str:
    columns = modified_columns(fields)
    if len(columns) == 1:
        return columns[0].removesuffix("_modified")
    if path:
        stem = path.stem
        if stem.startswith("modified_problems_"):
            return stem.removeprefix("modified_problems_")
    return ""


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            for key in ("items", "data", "examples", "rows"):
                value = data.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
            return [data]
    return []


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_pickle_rows(path: Path) -> list[dict[str, Any]]:
    if pd is not None:
        try:
            data = pd.read_pickle(path)
        except Exception:
            data = None
        if data is not None:
            if hasattr(data, "to_dict") and hasattr(data, "columns"):
                return list(data.to_dict(orient="records"))
            if isinstance(data, list):
                return [row for row in data if isinstance(row, dict)]
            if isinstance(data, dict):
                return [data]
    try:
        with path.open("rb") as handle:
            data = pickle.load(handle)
    except Exception:
        return []
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                rows = [row for row in value if isinstance(row, dict)]
                if rows:
                    return rows
        return [data]
    return []


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return read_json_rows(path)
    if suffix == ".csv":
        return read_csv_rows(path)
    if suffix in {".pkl", ".pickle"}:
        return read_pickle_rows(path)
    return []


def classify_rows(rows: list[dict[str, Any]], path: Path | None = None) -> dict[str, Any]:
    keys = set().union(*(row.keys() for row in rows[:20])) if rows else set()
    mods = modified_columns(keys)
    perturbation_type = perturbation_type_from_path_or_column(path, keys)
    has_perturbation_type = "perturbation_type" in keys
    has_perturbed_prompt = any(field in keys for field in PROMPT_FIELDS)
    has_clean_ref = any(field in keys for field in CLEAN_FIELDS)
    has_id = any(field in keys for field in ID_FIELDS)
    has_verifier = any(field in keys for field in VERIFIER_FIELDS)
    shortcutqa_like = bool(keys.intersection(SHORTCUTQA_FIELDS))
    livecodebench_clean = LIVECODEBENCH_FIELDS.issubset(keys) and not has_perturbation_type
    real_cocc_dataframe = (
        REAL_COCC_REQUIRED_FIELDS.issubset(keys)
        and "question_id" in keys
        and len(mods) == 1
        and bool(perturbation_type)
    )
    usable_cocc_candidate = (has_perturbation_type and has_perturbed_prompt and has_id) or real_cocc_dataframe
    return {
        "fields": sorted(keys),
        "row_count": len(rows),
        "modified_columns": mods,
        "perturbation_type": perturbation_type,
        "real_cocc_dataframe": real_cocc_dataframe,
        "has_perturbation_type": has_perturbation_type,
        "has_perturbed_prompt": has_perturbed_prompt,
        "has_clean_ref": has_clean_ref,
        "has_verifier_mapping": has_verifier,
        "has_problem_id": has_id,
        "shortcutqa_like": shortcutqa_like,
        "livecodebench_clean": livecodebench_clean,
        "usable_cocc_candidate": usable_cocc_candidate,
    }


def inspect_repo(repo_dir: Path) -> dict[str, Any]:
    if not repo_dir.exists():
        raise SystemExit(f"repo-dir not found: {repo_dir}")
    data_files = []
    script_files = []
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_dir).as_posix()
        if path.name in INTERESTING_FILES or path.parent.name in DATA_DIR_HINTS:
            script_files.append(rel) if path.suffix.lower() == ".py" else None
        rel_parts = set(Path(rel).parts)
        in_attacked_results = "results" in rel_parts and "attacked" in rel_parts
        if path.suffix.lower() in {".json", ".jsonl", ".csv", ".pkl", ".pickle", ".parquet"}:
            rows = [] if path.suffix.lower() == ".parquet" else load_rows(path)
            classified = classify_rows(rows, path)
            if in_attacked_results:
                classified["usable_cocc_candidate"] = False
                classified["native_result_log"] = True
            else:
                classified["native_result_log"] = False
            data_files.append({"path": rel, **classified, "parquet_unread": path.suffix.lower() == ".parquet"})
    usable = [row for row in data_files if row.get("usable_cocc_candidate")]
    clean_only = [row for row in data_files if row.get("livecodebench_clean")]
    shortcutqa = [row for row in data_files if row.get("shortcutqa_like")]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_dir": str(repo_dir),
        "status": "usable CoCC candidate located" if usable else "no usable CoCC schema located",
        "script_files": sorted(set(script_files)),
        "data_files": data_files,
        "usable_candidate_count": len(usable),
        "livecodebench_clean_count": len(clean_only),
        "shortcutqa_like_count": len(shortcutqa),
        "notes": [
            "No APIs were called.",
            "Native Anthropic/Gemini harness is not used by PRAMA.",
            "Use build_cocc_prama_dataset.py only after local schema inspection succeeds.",
        ],
    }


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cocc_schema_report.json"
    md_path = output_dir / "cocc_schema_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# CoCC Schema Inspection Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- repo_dir: `{report['repo_dir']}`",
        f"- status: `{report['status']}`",
        f"- usable_candidate_count: `{report['usable_candidate_count']}`",
        f"- livecodebench_clean_count: `{report['livecodebench_clean_count']}`",
        f"- shortcutqa_like_count: `{report['shortcutqa_like_count']}`",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in report["notes"])
    lines.extend(["", "## Data Files", ""])
    for row in report["data_files"]:
        lines.append(f"- `{row['path']}` rows={row['row_count']} usable_cocc_candidate={row['usable_cocc_candidate']} fields={', '.join(row['fields'][:20])}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--output-dir", default="results/local_asset_search")
    args = parser.parse_args(argv)
    report = inspect_repo(Path(args.repo_dir))
    json_path, md_path = write_report(report, Path(args.output_dir))
    print(report["status"])
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
