#!/usr/bin/env python
"""Run Empirical Result 008 on a local external benchmark dataset.

This adapter consumes externally supplied benchmark files and verifier labels or
answer keys. It does not download data, fabricate labels, or rebalance classes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    text_path = str(path)
    if text_path not in sys.path:
        sys.path.insert(0, text_path)

from aptadynamik.pipelines.deepseek import (
    DeepSeekConfig,
    _get_attr,
    deepseek_chat_completion,
    deepseek_response_to_raw_turn,
)
from aptadynamik.prama_problog_components import measure

PROMPT_FIELDS = ("prompt", "question", "problem", "input", "instruction")
ANSWER_FIELDS = ("answer", "expected_answer", "target")
EXTERNAL_LABEL_FIELDS = ("label", "is_correct", "pass", "passed", "verdict")
METADATA_FIELDS = ("id", "problem_id", "perturbation_type", "category", "split", "source", "difficulty")
LIVECODEBENCH_FIELDS = {"question_content", "public_test_cases", "question_id"}
SUPPORTED_SUFFIXES = {".jsonl", ".json", ".csv"}
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


@dataclass(frozen=True)
class SchemaInfo:
    fields: list[str]
    prompt_field: str | None
    answer_field: str | None
    label_field: str | None
    metadata_fields: list[str]
    recognized: bool
    reason: str


@dataclass(frozen=True)
class BenchmarkItem:
    prompt_id: str
    prompt: str
    expected_answer: str | None
    external_label: int | None
    perturbation_type: str
    split: str
    benchmark_name: str
    verifier_name: str
    source_metadata: dict[str, Any]
    item_type: str = "textual"
    question_title: str = ""
    difficulty: str = ""
    platform: str = ""
    starter_code: str = ""
    public_test_cases: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class Verification:
    observed_answer: str
    label: int | None
    event_type: str
    verifier_name: str
    extracted_code: str | None = None
    verifier_result: dict[str, Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or uuid4().hex


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def extract_final_answer(text: str) -> str | None:
    final_lines = [line.strip() for line in str(text or "").splitlines() if line.strip().upper().startswith("FINAL:")]
    if not final_lines:
        return None
    return final_lines[-1].split(":", 1)[1].strip()


def extract_python_code(text: str) -> str:
    """Extract Python code from a model response.

    This is only a formatting extraction helper. It does not judge correctness.
    """

    body = str(text or "").strip()
    fenced = re.search(r"```python\s*(.*?)```", body, flags=re.IGNORECASE | re.DOTALL)
    if not fenced:
        fenced = re.search(r"```\s*(.*?)```", body, flags=re.DOTALL)
    if fenced:
        body = fenced.group(1)
    cleaned: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        if stripped.lower() in {"python", "py"}:
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def normalize_stdout(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.rstrip().split("\n"))


def parse_public_test_cases(value: Any) -> list[dict[str, Any]] | None:
    if value is None or value == "":
        return None
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, dict):
        for key in ("public_test_cases", "tests", "test_cases", "cases"):
            cases = parsed.get(key)
            if isinstance(cases, list):
                parsed = cases
                break
    if not isinstance(parsed, list):
        return None
    cases = [case for case in parsed if isinstance(case, dict)]
    return cases or None


def run_python_public_tests(code: str, public_test_cases: Sequence[dict[str, Any]], timeout_seconds: int = 3) -> dict[str, Any]:
    """Run LiveCodeBench public stdin tests with lightweight subprocess isolation.

    This is not a strong sandbox. It uses a temporary directory, no shell, a
    small environment, and a timeout per test. It should not be treated as
    infrastructure-level isolation.
    """

    if not code.strip():
        return {"passed": False, "failure_reason": "no_code", "passed_count": 0, "failed_count": 1, "total_tests": len(public_test_cases)}
    if any(str(case.get("testtype") or "stdin").lower() != "stdin" for case in public_test_cases):
        return {
            "passed": False,
            "failure_reason": "unsupported_testtype",
            "passed_count": 0,
            "failed_count": len(public_test_cases),
            "total_tests": len(public_test_cases),
        }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        solution = root / "solution.py"
        solution.write_text(code, encoding="utf-8")
        env = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        compile_result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(solution)],
            cwd=str(root),
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds,
            check=False,
        )
        if compile_result.returncode != 0:
            return {
                "passed": False,
                "failure_reason": "compile_error",
                "passed_count": 0,
                "failed_count": len(public_test_cases),
                "total_tests": len(public_test_cases),
                "stderr": compile_result.stderr[-500:],
            }
        passed_count = 0
        for case in public_test_cases:
            expected = normalize_stdout(case.get("output", ""))
            try:
                completed = subprocess.run(
                    [sys.executable, str(solution)],
                    input=str(case.get("input", "")),
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return {
                    "passed": False,
                    "failure_reason": "timeout",
                    "passed_count": passed_count,
                    "failed_count": len(public_test_cases) - passed_count,
                    "total_tests": len(public_test_cases),
                }
            if completed.returncode != 0:
                return {
                    "passed": False,
                    "failure_reason": "runtime_error",
                    "passed_count": passed_count,
                    "failed_count": len(public_test_cases) - passed_count,
                    "total_tests": len(public_test_cases),
                    "stderr": completed.stderr[-500:],
                }
            if normalize_stdout(completed.stdout) != expected:
                return {
                    "passed": False,
                    "failure_reason": "wrong_answer",
                    "passed_count": passed_count,
                    "failed_count": len(public_test_cases) - passed_count,
                    "total_tests": len(public_test_cases),
                    "expected": expected,
                    "actual": normalize_stdout(completed.stdout),
                }
            passed_count += 1
    return {
        "passed": True,
        "failure_reason": "public_tests_pass",
        "passed_count": passed_count,
        "failed_count": 0,
        "total_tests": len(public_test_cases),
    }


def label_value_to_failure(value: Any) -> int | None:
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    if isinstance(value, bool):
        return 0 if value else 1
    if text in {"0", "false", "fail", "failed", "incorrect", "wrong", "negative", "pathological", "failure"}:
        # Numeric 0 in this adapter follows ER008 labels: 0=success. Named failure values are positive.
        return 0 if text == "0" else 1
    if text in {"1", "true", "pass", "passed", "correct", "success", "healthy", "negative_ok"}:
        # Numeric 1 in external datasets often means pass; named pass values are success.
        return 1 if text == "1" else 0
    return None


def load_json_any(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("data", "examples", "items", "records", "rows"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        return [data]
    return []


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_dataset_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"dataset file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix == ".json":
        return load_json_any(path)
    if suffix == ".csv":
        return load_csv(path)
    raise SystemExit(f"unsupported dataset file type: {path.suffix}")


def dataset_files_from_dir(path: Path) -> list[Path]:
    if not path.exists():
        raise SystemExit(f"benchmark dir not found: {path}")
    if not path.is_dir():
        raise SystemExit(f"benchmark dir is not a directory: {path}")
    return sorted(file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES)


def load_dataset(dataset_file: Path | None, benchmark_dir: Path | None) -> tuple[list[dict[str, Any]], list[Path]]:
    if not dataset_file and not benchmark_dir:
        raise SystemExit("provide --dataset-file or --benchmark-dir; this adapter does not download benchmark data")
    files = [dataset_file] if dataset_file else dataset_files_from_dir(benchmark_dir or Path())
    rows: list[dict[str, Any]] = []
    for file in files:
        if file is None:
            continue
        for row in load_dataset_file(file):
            cloned = dict(row)
            cloned.setdefault("source_file", str(file))
            rows.append(cloned)
    if not rows:
        raise SystemExit("dataset contains no rows")
    return rows, [file for file in files if file is not None]


def detect_schema(rows: Sequence[dict[str, Any]]) -> SchemaInfo:
    fields = sorted({key for row in rows for key in row.keys()})
    field_set = set(fields)
    if LIVECODEBENCH_FIELDS.issubset(field_set):
        metadata_fields = [field for field in ("question_title", "platform", "difficulty", "contest_id", "contest_date", "metadata", "_source_dataset", "_source_file", "_sample_index") if field in field_set]
        return SchemaInfo(
            fields,
            "question_content",
            None,
            None,
            metadata_fields,
            True,
            "recognized LiveCodeBench code_generation_lite schema",
        )
    prompt_field = next((field for field in PROMPT_FIELDS if field in field_set), None)
    answer_field = next((field for field in ANSWER_FIELDS if field in field_set), None)
    label_field = next((field for field in EXTERNAL_LABEL_FIELDS if field in field_set), None)
    metadata_fields = [field for field in METADATA_FIELDS if field in field_set]
    if not prompt_field:
        return SchemaInfo(fields, None, answer_field, label_field, metadata_fields, False, "missing prompt/question/problem/input/instruction field")
    if not answer_field and not label_field:
        return SchemaInfo(fields, prompt_field, None, None, metadata_fields, True, "prompt recognized, but verifier unsupported without answer key or external label")
    return SchemaInfo(fields, prompt_field, answer_field, label_field, metadata_fields, True, "recognized textual benchmark schema")


def write_schema_report(path: Path, schema: SchemaInfo, rows: Sequence[dict[str, Any]], files: Sequence[Path]) -> None:
    lines = [
        "# External Benchmark Dataset Schema Report",
        "",
        f"- recognized: `{schema.recognized}`",
        f"- reason: `{schema.reason}`",
        f"- files: `{[str(file) for file in files]}`",
        f"- row_count: `{len(rows)}`",
        f"- prompt_field: `{schema.prompt_field}`",
        f"- answer_field: `{schema.answer_field}`",
        f"- label_field: `{schema.label_field}`",
        f"- metadata_fields: `{schema.metadata_fields}`",
        "",
        "## Fields",
        "",
    ]
    lines.extend(f"- `{field}`" for field in schema.fields)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def row_to_item(row: dict[str, Any], schema: SchemaInfo, benchmark_name: str, index: int) -> BenchmarkItem:
    assert schema.prompt_field is not None
    if benchmark_name == "livecodebench" or LIVECODEBENCH_FIELDS.issubset(set(row.keys())):
        prompt_id = sanitize_id(str(row.get("question_id") or f"item_{index:04d}"))
        public_tests = parse_public_test_cases(row.get("public_test_cases"))
        difficulty = str(row.get("difficulty") or "").strip()
        verifier_name = "livecodebench_public_tests_v1" if public_tests else "unsupported_verifier"
        return BenchmarkItem(
            prompt_id=prompt_id,
            prompt=str(row.get("question_content") or ""),
            expected_answer="public_tests_pass" if public_tests else None,
            external_label=None,
            perturbation_type=difficulty or "livecodebench",
            split=str(row.get("split") or "").strip().lower(),
            benchmark_name="livecodebench",
            verifier_name=verifier_name,
            source_metadata={field: row.get(field) for field in sorted(row.keys()) if field not in {"question_content", "public_test_cases"}},
            item_type="livecodebench",
            question_title=str(row.get("question_title") or ""),
            difficulty=difficulty,
            platform=str(row.get("platform") or ""),
            starter_code=str(row.get("starter_code") or ""),
            public_test_cases=public_tests,
        )
    raw_id = row.get("id") or row.get("problem_id") or f"item_{index:04d}"
    expected_answer = str(row.get(schema.answer_field, "")).strip() if schema.answer_field else None
    if expected_answer == "":
        expected_answer = None
    external_label = label_value_to_failure(row.get(schema.label_field)) if schema.label_field else None
    if schema.label_field and external_label is None and expected_answer is None:
        verifier_name = "unsupported_verifier"
    elif external_label is not None:
        verifier_name = f"external_{schema.label_field}_label"
    elif expected_answer is not None:
        verifier_name = "final_exact_match_normalized_v1"
    else:
        verifier_name = "unsupported_verifier"
    return BenchmarkItem(
        prompt_id=sanitize_id(str(raw_id)),
        prompt=str(row.get(schema.prompt_field) or ""),
        expected_answer=expected_answer,
        external_label=external_label,
        perturbation_type=str(row.get("perturbation_type") or row.get("category") or "").strip(),
        split=str(row.get("split") or "").strip().lower(),
        benchmark_name=benchmark_name,
        verifier_name=verifier_name,
        source_metadata={field: row.get(field) for field in sorted(row.keys()) if field not in {schema.prompt_field, schema.answer_field, schema.label_field}},
    )


def wrap_prompt(prompt: str) -> str:
    return (
        "Answer the following benchmark item. End with one final line exactly:\n"
        "FINAL: <answer>\n\n"
        f"{prompt}"
    )


def wrap_prompt_for_item(item: BenchmarkItem) -> str:
    if item.item_type == "livecodebench":
        return (
            "You are solving a programming benchmark item.\n"
            "Write a complete Python 3 program that reads from standard input and writes to standard output.\n"
            "Return only code. Do not use markdown.\n\n"
            f"Problem:\n{item.prompt}\n\n"
            f"Starter code, if useful:\n{item.starter_code}\n"
        )
    return wrap_prompt(item.prompt)


def verify_item(item: BenchmarkItem, assistant_message: str) -> Verification:
    if item.item_type == "livecodebench":
        code = extract_python_code(assistant_message)
        if not item.public_test_cases:
            return Verification(
                observed_answer=assistant_message.strip(),
                label=None,
                event_type="unsupported_verifier",
                verifier_name="unsupported_verifier",
                extracted_code=code,
                verifier_result={"passed": False, "failure_reason": "unsupported_verifier", "passed_count": 0, "failed_count": 0, "total_tests": 0},
            )
        result = run_python_public_tests(code, item.public_test_cases)
        if result["failure_reason"] == "unsupported_testtype":
            return Verification(
                observed_answer=result["failure_reason"],
                label=None,
                event_type="unsupported_verifier",
                verifier_name="unsupported_verifier",
                extracted_code=code,
                verifier_result=result,
            )
        if result["passed"]:
            event_type = "final_answer"
            label = 0
        elif result["failure_reason"] == "wrong_answer":
            event_type = "unit_test_failure"
            label = 1
        elif result["failure_reason"] == "no_code":
            event_type = "verification_failure"
            label = 1
        else:
            event_type = str(result["failure_reason"])
            label = 1
        return Verification(
            observed_answer="public_tests_pass" if result["passed"] else str(result["failure_reason"]),
            label=label,
            event_type=event_type,
            verifier_name="livecodebench_public_tests_v1",
            extracted_code=code,
            verifier_result=result,
        )
    if item.external_label is not None:
        observed = extract_final_answer(assistant_message) or assistant_message.strip()
        event_type = "external_failure" if item.external_label == 1 else "final_answer"
        return Verification(observed, item.external_label, event_type, item.verifier_name)
    if item.expected_answer is None:
        return Verification(extract_final_answer(assistant_message) or assistant_message.strip(), None, "unsupported_verifier", "unsupported_verifier")
    observed = extract_final_answer(assistant_message)
    if observed is None:
        return Verification(assistant_message.strip(), 1, "verification_failure", item.verifier_name)
    correct = normalize_text(observed) == normalize_text(item.expected_answer)
    return Verification(observed, 0 if correct else 1, "final_answer" if correct else "verification_failure", item.verifier_name)


def openai_chat_completion(messages: Sequence[dict[str, str]], model: str, temperature: float, max_tokens: int, top_logprobs: int):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not defined.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI SDK to use OpenAI: python -m pip install openai") from exc
    client = OpenAI(api_key=api_key)
    return client.chat.completions.create(
        model=model,
        messages=list(messages),
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=top_logprobs,
    )


def call_backend(provider: str, model: str, prompt: str, temperature: float, max_tokens: int, top_logprobs: int):
    messages = [{"role": "user", "content": prompt}]
    if provider == "deepseek":
        config = DeepSeekConfig(model=model, temperature=temperature, max_tokens=max_tokens, top_logprobs=top_logprobs)
        return deepseek_chat_completion(messages, config)
    if provider == "openai":
        return openai_chat_completion(messages, model, temperature, max_tokens, top_logprobs)
    raise ValueError(f"unsupported provider: {provider}")


def variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    return sum((value - mean_value) ** 2 for value in values) / len(values)


def build_token_windows(
    tokens: Sequence[dict[str, Any]],
    window_size: int = 16,
    min_window_tokens: int = 4,
) -> list[dict[str, Any]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if min_window_tokens <= 0:
        raise ValueError("min_window_tokens must be positive")
    token_list = list(tokens)
    chunks = [token_list[start : start + window_size] for start in range(0, len(token_list), window_size)]
    if len(chunks) > 1 and len(chunks[-1]) < min_window_tokens:
        chunks.pop()
    return [
        {
            "turn_index": index,
            "token_count": len(chunk),
            "tokens": chunk,
        }
        for index, chunk in enumerate(chunks)
        if chunk
    ]


def attach_prama_summaries(windows: list[dict[str, Any]]) -> dict[str, Any]:
    calib_window = min(3, max(1, len(windows) // 4))
    result = measure(
        windows,
        calib_window=calib_window,
        micro_health_mode="log_ratio",
        micro_health_scale=1.0,
        baseline_stat="median",
        min_calib_tokens=4,
        crossing_index_scope="turn",
    )
    rows = result.get("turns") or []
    for window, summary in zip(windows, rows):
        window["summary"] = summary
        window["metrics_summary"] = summary
    return {key: value for key, value in result.items() if key != "turns"}


def completion_status_from_finish_reason(generated_token_count: int, max_tokens_requested: int, finish_reason: Any) -> tuple[bool, str]:
    reason = str(finish_reason or "").strip().lower()
    truncated = generated_token_count >= max_tokens_requested or reason in {"length", "max_tokens", "max_tokens_reached"}
    if truncated:
        return True, "truncated"
    if reason in {"stop", "completed"}:
        return False, "completed"
    return False, "unknown"


def run_item(
    item: BenchmarkItem,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_logprobs: int,
    output_dir: Path,
    call_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    prompt = wrap_prompt_for_item(item)
    response = call_fn(provider, model, prompt, temperature, max_tokens, top_logprobs) if call_fn else call_backend(
        provider, model, prompt, temperature, max_tokens, top_logprobs
    )
    turn = deepseek_response_to_raw_turn(response, turn_index=0, user_message=prompt)
    resolved_model = str(_get_attr(response, "model", model) or model)
    verification = verify_item(item, turn["assistant_message"])
    windows = build_token_windows(turn["tokens"], window_size=16, min_window_tokens=4)
    if not windows:
        raise RuntimeError("response produced no token windows")
    prama_measure_summary = attach_prama_summaries(windows)
    valid_prama_windows = sum(1 for window in windows if (window.get("summary") or {}).get("logprob_valid"))
    generated_token_count = len(turn.get("tokens") or [])
    represented_token_count = sum(int(window.get("token_count") or 0) for window in windows)
    event_turn = int(windows[-1].get("turn_index", len(windows) - 1)) if windows else 0
    supported_prama = valid_prama_windows >= 3 and represented_token_count > 0
    final_summary = windows[-1].get("summary") or {}
    boundary_values = [
        float((window.get("summary") or {}).get("boundary_pressure"))
        for window in windows
        if (window.get("summary") or {}).get("boundary_pressure") is not None
    ]
    xi_values = [
        float((window.get("summary") or {}).get("xi_norm"))
        for window in windows
        if (window.get("summary") or {}).get("xi_norm") is not None
    ]
    session_id = sanitize_id(f"{item.benchmark_name}_{item.prompt_id}_{uuid4().hex[:8]}")
    session_dir = output_dir / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    finish_reason = turn.get("finish_reason")
    truncated_by_max_tokens, completion_status = completion_status_from_finish_reason(
        generated_token_count,
        max_tokens,
        finish_reason,
    )
    prama_summary_without_turns = {key: value for key, value in prama_measure_summary.items() if key != "turns"}
    raw = {
        "session_id": session_id,
        "benchmark_name": item.benchmark_name,
        "benchmark_item_id": item.prompt_id,
        "provider": provider,
        "requested_model": model,
        "resolved_model": resolved_model,
        "model": resolved_model,
        "created_at": utc_now(),
        "prompt_id": item.prompt_id,
        "prompt": prompt if item.item_type == "livecodebench" else item.prompt,
        "benchmark_prompt": item.prompt,
        "wrapped_prompt": prompt,
        "question_title": item.question_title,
        "difficulty": item.difficulty,
        "platform": item.platform,
        "starter_code_present": bool(item.starter_code.strip()),
        "expected_answer": item.expected_answer,
        "observed_answer": turn["assistant_message"] if item.item_type == "livecodebench" else verification.observed_answer,
        "extracted_code": verification.extracted_code,
        "verifier_name": verification.verifier_name,
        "verifier_result": verification.verifier_result,
        "perturbation_type": item.perturbation_type,
        "source_metadata": item.source_metadata,
        "assistant_message": turn["assistant_message"],
        "finish_reason": finish_reason,
        "final_token_count": generated_token_count,
        "generated_token_count": generated_token_count,
        "represented_token_count": represented_token_count,
        "max_tokens_requested": max_tokens,
        "truncated_by_max_tokens": truncated_by_max_tokens,
        "completion_status": completion_status,
        "final_outcome_proxy": True,
        "window_count": len(windows),
        "valid_prama_windows": valid_prama_windows,
        "unsupported_prama_windows": not supported_prama,
        "prama_primary_score_variance": variance(boundary_values),
        "primary_score_min": min(boundary_values) if boundary_values else None,
        "primary_score_max": max(boundary_values) if boundary_values else None,
        "prama_measure_summary": prama_summary_without_turns,
        "turns": windows,
    }
    raw_path = session_dir / "raw.json"
    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "session_id": session_id,
        "label": "" if verification.label is None or not supported_prama else str(verification.label),
        "event_token": str(represented_token_count),
        "event_turn": str(event_turn),
        "event_type": verification.event_type if supported_prama else "unsupported_prama_windows",
        "prompt_id": item.prompt_id,
        "expected_answer": item.expected_answer or "",
        "observed_answer": verification.observed_answer,
        "verifier_name": verification.verifier_name,
        "backend": provider,
        "split": item.split,
        "benchmark_name": item.benchmark_name,
        "perturbation_type": item.perturbation_type,
        "difficulty": item.difficulty,
        "platform": item.platform,
        "raw_path": str(raw_path),
        "supported": verification.label is not None and supported_prama,
        "unsupported_verifier": verification.label is None,
        "unsupported_prama_windows": not supported_prama,
        "final_token_count": generated_token_count,
        "generated_token_count": generated_token_count,
        "represented_token_count": represented_token_count,
        "max_tokens_requested": max_tokens,
        "finish_reason": finish_reason,
        "truncated_by_max_tokens": truncated_by_max_tokens,
        "completion_status": completion_status,
        "window_count": len(windows),
        "valid_prama_windows": valid_prama_windows,
        "verifier_result": verification.verifier_result,
        "boundary_pressure_final": final_summary.get("boundary_pressure"),
        "xi_norm_final": final_summary.get("xi_norm"),
        "instant_viability_margin_final": final_summary.get("instant_viability_margin"),
        "boundary_pressure_values": boundary_values,
        "xi_norm_values": xi_values,
    }


def write_labels(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for row in rows:
            try:
                event_token = int(float(row.get("event_token", "")))
                represented_token_count = int(float(row.get("represented_token_count", event_token)))
            except (TypeError, ValueError):
                continue
            if (
                row.get("supported")
                and str(row.get("label")) in {"0", "1"}
                and represented_token_count > 0
                and event_token <= represented_token_count
            ):
                writer.writerow({field: row.get(field, "") for field in LABEL_FIELDS})


def write_session_index(output_dir: Path, rows: Sequence[dict[str, Any]], args: argparse.Namespace, schema: SchemaInfo, files: Sequence[Path]) -> None:
    supported = [row for row in rows if row.get("supported")]
    supported_failures = [row for row in supported if str(row.get("label")) == "1"]
    truncated_failures = [row for row in supported_failures if row.get("truncated_by_max_tokens")]
    truncation_warning = ""
    if supported_failures and len(truncated_failures) / len(supported_failures) > 0.5:
        truncation_warning = (
            "methodological warning: failures are strongly associated with max_tokens truncation; "
            "AUROC may reflect truncation/length rather than structural failure."
        )
        print(truncation_warning)
    labels_path = output_dir / "labels.csv"
    payload = {
        "generated_at": utc_now(),
        "benchmark_name": args.benchmark_name,
        "provider": args.provider,
        "requested_model": args.model,
        "dataset_files": [str(file) for file in files],
        "schema": schema.__dict__,
        "total_items_seen": args.total_items_seen,
        "total_items_used": len(supported),
        "total_unsupported": sum(1 for row in rows if not row.get("supported")),
        "total_unsupported_verifier": sum(1 for row in rows if row.get("unsupported_verifier") or row.get("event_type") == "unsupported_verifier"),
        "total_unsupported_prama_windows": sum(1 for row in rows if row.get("unsupported_prama_windows") or row.get("event_type") == "unsupported_prama_windows"),
        "positive_count": sum(1 for row in supported if str(row.get("label")) == "1"),
        "negative_count": sum(1 for row in supported if str(row.get("label")) == "0"),
        "pass_count": sum(1 for row in supported if str(row.get("label")) == "0"),
        "fail_count": sum(1 for row in supported if str(row.get("label")) == "1"),
        "truncated_count": sum(1 for row in rows if row.get("truncated_by_max_tokens")),
        "truncated_pass_count": sum(1 for row in supported if str(row.get("label")) == "0" and row.get("truncated_by_max_tokens")),
        "truncated_fail_count": len(truncated_failures),
        "truncation_warning": truncation_warning,
        "split_counts": dict(sorted(Counter(row.get("split") or "unspecified" for row in supported).items())),
        "perturbation_type_counts": dict(sorted(Counter(row.get("perturbation_type") or "unspecified" for row in supported).items())),
        "difficulty_counts": dict(sorted(Counter(row.get("difficulty") or row.get("perturbation_type") or "unspecified" for row in rows).items())),
        "platform_counts": dict(sorted(Counter(row.get("platform") or "unspecified" for row in rows).items())),
        "labels_path": str(labels_path),
        "validation_path": str(output_dir / "validation"),
        "evaluation_path": str(output_dir / "evaluation"),
        "sessions": list(rows),
    }
    (output_dir / "session_index.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# External Benchmark PRAMA Session Index",
        "",
        f"- benchmark_name: `{args.benchmark_name}`",
        f"- provider: `{args.provider}`",
        f"- requested_model: `{args.model}`",
        f"- total_items_seen: `{payload['total_items_seen']}`",
        f"- total_items_used: `{payload['total_items_used']}`",
        f"- total_unsupported: `{payload['total_unsupported']}`",
        f"- total_unsupported_verifier: `{payload['total_unsupported_verifier']}`",
        f"- total_unsupported_prama_windows: `{payload['total_unsupported_prama_windows']}`",
        f"- positive_count: `{payload['positive_count']}`",
        f"- negative_count: `{payload['negative_count']}`",
        f"- pass_count: `{payload['pass_count']}`",
        f"- fail_count: `{payload['fail_count']}`",
        f"- truncated_count: `{payload['truncated_count']}`",
        f"- truncated_pass_count: `{payload['truncated_pass_count']}`",
        f"- truncated_fail_count: `{payload['truncated_fail_count']}`",
        f"- labels_path: `{labels_path}`",
        "",
        "## Truncation Warning",
        "",
        truncation_warning or "None",
        "",
        "| session_id | supported | label | split | prompt_id | perturbation | generated_tokens | represented_tokens | truncated | windows | valid_prama | raw |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('session_id', '')} | {row.get('supported')} | {row.get('label', '')} | {row.get('split', '')} | "
            f"{row.get('prompt_id', '')} | {row.get('perturbation_type', '')} | {row.get('generated_token_count', row.get('final_token_count', ''))} | "
            f"{row.get('represented_token_count', '')} | {row.get('truncated_by_max_tokens', '')} | "
            f"{row.get('window_count', '')} | {row.get('valid_prama_windows', '')} | {row.get('raw_path', '')} |"
        )
    (output_dir / "session_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_internal_commands(
    labels_path: Path,
    raw_paths: Sequence[str],
    validation_dir: Path,
    evaluation_dir: Path,
    target_fpr: float,
    primary_score: str,
) -> tuple[list[str], list[str]]:
    validate_cmd = [
        sys.executable,
        "-m",
        "scripts.validate_early_warning_inputs",
        "--labels",
        str(labels_path),
        "--inputs",
        *[str(path) for path in raw_paths],
        "--primary-score",
        primary_score,
        "--output-dir",
        str(validation_dir),
    ]
    evaluate_cmd = [
        sys.executable,
        "-m",
        "scripts.evaluate_early_warning",
        "--labels",
        str(labels_path),
        "--inputs",
        *[str(path) for path in raw_paths],
        "--target-fpr",
        str(target_fpr),
        "--primary-score",
        primary_score,
        "--output-dir",
        str(evaluation_dir),
    ]
    return validate_cmd, evaluate_cmd


def run_subprocess(command: list[str], cwd: Path) -> int:
    print(" ".join(command))
    return int(subprocess.run(command, cwd=str(cwd), check=False).returncode)


def has_two_classes(rows: Sequence[dict[str, Any]]) -> bool:
    labels = {str(row.get("label")) for row in rows if row.get("supported")}
    return "0" in labels and "1" in labels


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-name", default="break_the_chain")
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--provider", choices=["deepseek", "openai"], required=True)
    parser.add_argument("--model")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/external_break_the_chain_prama"))
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--primary-score", default="boundary_pressure")
    parser.add_argument("--split-field", default="split")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-api", action="store_true")
    return parser.parse_args(argv)


def _existing_rows_from_output(output_dir: Path, items: Sequence[BenchmarkItem], provider: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_prompt = {item.prompt_id: item for item in items}
    for raw_path in sorted((output_dir / "sessions").rglob("raw.json")):
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        prompt_id = str(raw.get("prompt_id") or raw.get("benchmark_item_id") or "")
        item = by_prompt.get(prompt_id)
        if not item:
            continue
        verification = verify_item(item, str(raw.get("assistant_message") or ""))
        turns = raw.get("turns") or []
        generated_token_count = int(raw.get("generated_token_count") or raw.get("final_token_count") or sum(int(turn.get("token_count") or 0) for turn in turns))
        represented_token_count = int(raw.get("represented_token_count") or sum(int(turn.get("token_count") or 0) for turn in turns))
        valid_prama_windows = sum(1 for turn in turns if (turn.get("metrics_summary") or turn.get("summary") or {}).get("logprob_valid"))
        supported_prama = represented_token_count > 0 and valid_prama_windows >= 3
        finish_reason = raw.get("finish_reason")
        max_tokens_requested = int(raw.get("max_tokens_requested") or 0)
        truncated_by_max_tokens = bool(raw.get("truncated_by_max_tokens")) if "truncated_by_max_tokens" in raw else False
        rows.append(
            {
                "session_id": raw.get("session_id") or raw_path.parent.name,
                "label": "" if verification.label is None or not supported_prama else str(verification.label),
                "event_token": str(represented_token_count),
                "event_turn": str(max(0, len(turns) - 1)),
                "event_type": verification.event_type if supported_prama else "unsupported_prama_windows",
                "prompt_id": item.prompt_id,
                "expected_answer": item.expected_answer or "",
                "observed_answer": verification.observed_answer,
                "verifier_name": verification.verifier_name,
                "backend": provider,
                "split": item.split,
                "benchmark_name": item.benchmark_name,
                "perturbation_type": item.perturbation_type,
                "raw_path": str(raw_path),
                "supported": verification.label is not None and supported_prama,
                "unsupported_verifier": verification.label is None,
                "unsupported_prama_windows": not supported_prama,
                "final_token_count": generated_token_count,
                "generated_token_count": generated_token_count,
                "represented_token_count": represented_token_count,
                "max_tokens_requested": max_tokens_requested,
                "finish_reason": finish_reason,
                "truncated_by_max_tokens": truncated_by_max_tokens,
                "completion_status": raw.get("completion_status") or ("truncated" if truncated_by_max_tokens else "unknown"),
                "window_count": len(turns),
                "valid_prama_windows": valid_prama_windows,
            }
        )
    return rows


def main(argv: Sequence[str] | None = None, call_fn: Callable[..., Any] | None = None, run_eval: bool = True) -> int:
    args = parse_args(argv)
    if args.n <= 0:
        raise SystemExit("--n must be positive")
    args.model = args.model or ("deepseek-chat" if args.provider == "deepseek" else "gpt-4o-mini")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_raw, files = load_dataset(args.dataset_file, args.benchmark_dir)
    schema = detect_schema(rows_raw)
    write_schema_report(args.output_dir / "dataset_schema_report.md", schema, rows_raw, files)
    if not schema.recognized:
        raise SystemExit(f"dataset schema not recognized: {schema.reason}")

    items = [row_to_item(row, schema, args.benchmark_name, index) for index, row in enumerate(rows_raw)]
    args.total_items_seen = len(items)
    rng = random.Random(args.seed)
    rng.shuffle(items)
    selected = items[: args.n]

    if args.dry_run:
        write_session_index(args.output_dir, [], args, schema, files)
        print(f"schema report: {args.output_dir / 'dataset_schema_report.md'}")
        return 0

    if args.no_api:
        rows = _existing_rows_from_output(args.output_dir, selected, args.provider)
    else:
        env_name = "DEEPSEEK_API_KEY" if args.provider == "deepseek" else "OPENAI_API_KEY"
        if call_fn is None and not os.environ.get(env_name):
            raise SystemExit(f"{env_name} is not defined.")
        rows = []
        for item in selected:
            if item.verifier_name == "unsupported_verifier":
                rows.append(
                    {
                        "session_id": "",
                        "label": "",
                        "event_token": "",
                        "event_turn": "",
                        "event_type": "unsupported_verifier",
                        "prompt_id": item.prompt_id,
                        "expected_answer": item.expected_answer or "",
                        "observed_answer": "",
                        "verifier_name": "unsupported_verifier",
                        "backend": args.provider,
                        "split": item.split,
                        "benchmark_name": item.benchmark_name,
                        "perturbation_type": item.perturbation_type,
                        "difficulty": item.difficulty,
                        "platform": item.platform,
                        "raw_path": "",
                        "supported": False,
                        "unsupported_verifier": True,
                        "unsupported_prama_windows": False,
                        "verifier_result": {"passed": False, "failure_reason": "unsupported_verifier", "passed_count": 0, "failed_count": 0, "total_tests": 0},
                    }
                )
                continue
            try:
                row = run_item(
                    item,
                    args.provider,
                    args.model,
                    args.temperature,
                    args.max_tokens,
                    args.top_logprobs,
                    args.output_dir,
                    call_fn=call_fn,
                )
                rows.append(row)
                print(f"[{len(rows)}] {row['session_id']} label={row['label']} prompt_id={row['prompt_id']}")
            except Exception as exc:  # noqa: BLE001 - CLI should continue across benchmark items.
                print(f"item {item.prompt_id} failed: {exc}")

    labels_path = args.output_dir / "labels.csv"
    write_labels(labels_path, rows)
    write_session_index(args.output_dir, rows, args, schema, files)
    supported_rows = [row for row in rows if row.get("supported")]
    if not has_two_classes(supported_rows):
        if args.benchmark_name == "livecodebench":
            print("external benchmark inconclusive: only one class observed under public tests")
        else:
            print("external benchmark evaluation inconclusive: only one class observed; validation/evaluation skipped")
        return 2
    if not supported_rows:
        print("external benchmark evaluation inconclusive: no supported verifier rows")
        return 2
    raw_paths = [row["raw_path"] for row in supported_rows]
    validation_dir = args.output_dir / "validation"
    evaluation_dir = args.output_dir / "evaluation"
    validate_cmd, evaluate_cmd = build_internal_commands(labels_path, raw_paths, validation_dir, evaluation_dir, args.target_fpr, args.primary_score)
    repo_root = Path(__file__).resolve().parents[1]
    if run_eval:
        if run_subprocess(validate_cmd, repo_root) != 0:
            raise SystemExit("input validation failed; evaluation skipped")
        if run_subprocess(evaluate_cmd, repo_root) != 0:
            raise SystemExit("early-warning evaluation failed")
    else:
        print("Validation command:")
        print(" ".join(validate_cmd))
        print("Evaluation command:")
        print(" ".join(evaluate_cmd))
    print(f"labels: {labels_path}")
    print(f"validation: {validation_dir}")
    print(f"evaluation: {evaluation_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
