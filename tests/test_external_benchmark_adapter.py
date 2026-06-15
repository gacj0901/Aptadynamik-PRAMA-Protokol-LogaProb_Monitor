import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_external_benchmark_prama_eval import (
    BenchmarkItem,
    build_internal_commands,
    detect_schema,
    extract_final_answer,
    extract_python_code,
    load_dataset_file,
    main as external_main,
    parse_public_test_cases,
    run_item,
    run_python_public_tests,
    verify_item,
)


def fake_response(text):
    words = text.split() or [text]
    content = []
    for index, token in enumerate(words):
        logprob = -0.12 - (index % 5) * 0.04 - (0.03 if "wrong" in text and index > len(words) // 2 else 0.0)
        content.append(
            {
                "token": token,
                "logprob": logprob,
                "top_logprobs": [{"logprob": logprob}, {"logprob": logprob - 1.0}],
            }
        )
    return {
        "model": "fake-model",
        "choices": [
            {
                "message": {"content": text},
                "finish_reason": "stop",
                "logprobs": {"content": content},
            }
        ],
    }


class AlternatingFakeProvider:
    def __init__(self):
        self.calls = 0

    def __call__(self, provider, model, prompt, temperature, max_tokens, top_logprobs):
        self.calls += 1
        final = "FINAL: alpha" if self.calls % 2 else "FINAL: wrong"
        return fake_response(
            "Step 1 read the benchmark item carefully. "
            "Step 2 identify the requested answer from the supplied item. "
            "Step 3 preserve the benchmark content without adding a new task. "
            "Step 4 provide the final answer in the required format. "
            f"\n{final}"
        )


class LiveCodeBenchFakeProvider:
    def __init__(self):
        self.calls = 0

    def __call__(self, provider, model, prompt, temperature, max_tokens, top_logprobs):
        self.calls += 1
        if self.calls % 2:
            code = (
                "import sys\n"
                "# The following comments make the response long enough for several PRAMA windows.\n"
                "# Read all input tokens, convert them into integers, and sum the complete list.\n"
                "# This benchmark smoke program intentionally keeps the algorithm simple and deterministic.\n"
                "data = sys.stdin.read().split()\n"
                "nums = [int(x) for x in data]\n"
                "print(sum(nums))\n"
            )
        else:
            code = (
                "import sys\n"
                "# Wrong but syntactically valid program with enough lexical material for windowing.\n"
                "# It ignores the input and prints a constant value, causing a public test failure.\n"
                "# The adapter should mark this as a functional external failure.\n"
                "print(0)\n"
            )
        return fake_response(code)


class ShortLiveCodeBenchFakeProvider:
    def __call__(self, provider, model, prompt, temperature, max_tokens, top_logprobs):
        return fake_response("print(3)")


class CountTokenProvider:
    def __init__(self, token_count, message="FINAL: alpha"):
        self.token_count = token_count
        self.message = message

    def __call__(self, provider, model, prompt, temperature, max_tokens, top_logprobs):
        content = [
            {
                "token": f"tok{i}",
                "logprob": -0.1 - (i % 7) * 0.03,
                "top_logprobs": [{"logprob": -0.1 - (i % 7) * 0.03}, {"logprob": -1.4}],
            }
            for i in range(self.token_count)
        ]
        return {
            "model": "fake-model",
            "choices": [
                {
                    "message": {"content": self.message},
                    "finish_reason": "length" if self.token_count >= max_tokens else "stop",
                    "logprobs": {"content": content},
                }
            ],
        }


class ExternalBenchmarkAdapterTests(unittest.TestCase):
    def test_script_runs_as_file_help(self):
        completed = subprocess.run(
            [sys.executable, "scripts/run_external_benchmark_prama_eval.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--dataset-file", completed.stdout)

    def test_load_jsonl_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "data.jsonl"
            jsonl.write_text('{"question":"q1","answer":"a1"}\n{"question":"q2","answer":"a2"}\n', encoding="utf-8")
            csv_path = root / "data.csv"
            csv_path.write_text("question,answer\nq1,a1\n", encoding="utf-8")
            self.assertEqual(len(load_dataset_file(jsonl)), 2)
            self.assertEqual(len(load_dataset_file(csv_path)), 1)

    def test_detect_schema_and_unknown_schema(self):
        schema = detect_schema([{"question": "What?", "answer": "A"}])
        self.assertTrue(schema.recognized)
        self.assertEqual(schema.prompt_field, "question")
        self.assertEqual(schema.answer_field, "answer")

        unknown = detect_schema([{"foo": "bar"}])
        self.assertFalse(unknown.recognized)
        self.assertIn("missing prompt", unknown.reason)

    def test_adapter_does_not_fabricate_labels(self):
        schema = detect_schema([{"question": "What?"}])
        self.assertTrue(schema.recognized)
        self.assertIsNone(schema.answer_field)
        self.assertIsNone(schema.label_field)
        self.assertIn("unsupported", schema.reason)

    def test_livecodebench_schema_recognized(self):
        schema = detect_schema(
            [
                {
                    "question_id": "q1",
                    "question_content": "Read two ints and print their sum.",
                    "public_test_cases": [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}],
                }
            ]
        )
        self.assertTrue(schema.recognized)
        self.assertEqual(schema.prompt_field, "question_content")
        self.assertIn("LiveCodeBench", schema.reason)

    def test_public_test_cases_parse_string_and_list(self):
        cases = [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]
        self.assertEqual(parse_public_test_cases(json.dumps(cases)), cases)
        self.assertEqual(parse_public_test_cases(cases), cases)

    def test_extract_python_code_fenced_and_plain(self):
        self.assertEqual(extract_python_code("```python\nprint(1)\n```"), "print(1)")
        self.assertEqual(extract_python_code("```\nprint(2)\n```"), "print(2)")
        self.assertEqual(extract_python_code("print(3)"), "print(3)")

    def test_run_python_public_tests_pass_wrong_timeout_and_unsupported(self):
        cases = [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]
        passed = run_python_public_tests("import sys\nprint(sum(map(int, sys.stdin.read().split())))", cases)
        self.assertTrue(passed["passed"])
        wrong = run_python_public_tests("print(0)", cases)
        self.assertFalse(wrong["passed"])
        self.assertEqual(wrong["failure_reason"], "wrong_answer")
        timeout = run_python_public_tests("while True:\n    pass\n", cases, timeout_seconds=1)
        self.assertFalse(timeout["passed"])
        self.assertEqual(timeout["failure_reason"], "timeout")
        unsupported = run_python_public_tests("print(1)", [{"input": "", "output": "", "testtype": "functional"}])
        self.assertEqual(unsupported["failure_reason"], "unsupported_testtype")

    def test_final_exact_match_verifier(self):
        item = BenchmarkItem(
            prompt_id="p1",
            prompt="Question",
            expected_answer="Alpha Beta",
            external_label=None,
            perturbation_type="",
            split="",
            benchmark_name="test",
            verifier_name="final_exact_match_normalized_v1",
            source_metadata={},
        )
        self.assertEqual(extract_final_answer("Reasoning\nFINAL: Alpha Beta"), "Alpha Beta")
        self.assertEqual(verify_item(item, "Reasoning\nFINAL: alpha   beta").label, 0)
        self.assertEqual(verify_item(item, "Reasoning\nFINAL: gamma").label, 1)
        self.assertEqual(verify_item(item, "No final line").event_type, "verification_failure")

    def test_unsupported_verifier_excluded_and_index_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            data = Path(tmp) / "data.jsonl"
            data.write_text('{"question":"q1"}\n{"question":"q2"}\n', encoding="utf-8")
            provider = AlternatingFakeProvider()
            code = external_main(
                [
                    "--benchmark-name",
                    "break_the_chain",
                    "--dataset-file",
                    str(data),
                    "--provider",
                    "deepseek",
                    "--model",
                    "fake-model",
                    "--n",
                    "2",
                    "--output-dir",
                    str(out),
                ],
                call_fn=provider,
                run_eval=False,
            )
            self.assertEqual(code, 2)
            self.assertEqual(provider.calls, 0)
            labels = (out / "labels.csv").read_text(encoding="utf-8")
            self.assertEqual(len(labels.strip().splitlines()), 1)
            index = json.loads((out / "session_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["total_unsupported"], 2)
            self.assertEqual(index["total_items_used"], 0)

    def test_labels_index_and_fake_provider_raw_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            data = Path(tmp) / "data.jsonl"
            rows = [
                {"id": "a", "question": "External item A", "answer": "alpha", "split": "train", "perturbation_type": "control"},
                {"id": "b", "question": "External item B", "answer": "alpha", "split": "train", "perturbation_type": "control"},
                {"id": "c", "question": "External item C", "answer": "alpha", "split": "test", "perturbation_type": "perturbed"},
                {"id": "d", "question": "External item D", "answer": "alpha", "split": "test", "perturbation_type": "perturbed"},
            ]
            data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            code = external_main(
                [
                    "--benchmark-name",
                    "break_the_chain",
                    "--dataset-file",
                    str(data),
                    "--provider",
                    "deepseek",
                    "--model",
                    "fake-model",
                    "--n",
                    "4",
                    "--output-dir",
                    str(out),
                    "--seed",
                    "1",
                ],
                call_fn=AlternatingFakeProvider(),
                run_eval=False,
            )
            self.assertEqual(code, 0)
            self.assertTrue((out / "labels.csv").exists())
            self.assertTrue((out / "session_index.json").exists())
            self.assertTrue((out / "session_index.md").exists())
            with (out / "labels.csv").open(newline="", encoding="utf-8") as handle:
                label_rows = list(csv.DictReader(handle))
            self.assertEqual(len(label_rows), 4)
            self.assertEqual({row["label"] for row in label_rows}, {"0", "1"})
            self.assertIn("benchmark_name", label_rows[0])
            self.assertIn("perturbation_type", label_rows[0])
            index = json.loads((out / "session_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["positive_count"], 2)
            self.assertEqual(index["negative_count"], 2)
            raw_files = list((out / "sessions").rglob("raw.json"))
            self.assertEqual(len(raw_files), 4)
            raw = json.loads(raw_files[0].read_text(encoding="utf-8"))
            self.assertEqual(raw["benchmark_name"], "break_the_chain")
            self.assertIn("source_metadata", raw)
            self.assertGreaterEqual(len(raw["turns"]), 2)
            self.assertIn("metrics_summary", raw["turns"][0])
            self.assertIn("boundary_pressure", raw["turns"][0]["metrics_summary"])

    def test_livecodebench_fake_provider_raw_labels_and_verifier_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            data = Path(tmp) / "lcb.jsonl"
            rows = [
                {
                    "question_id": "a",
                    "question_title": "sum",
                    "question_content": "Read integers and print their sum.",
                    "difficulty": "easy",
                    "platform": "leetcode",
                    "starter_code": "",
                    "public_test_cases": [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}],
                },
                {
                    "question_id": "b",
                    "question_title": "sum",
                    "question_content": "Read integers and print their sum.",
                    "difficulty": "medium",
                    "platform": "leetcode",
                    "starter_code": "",
                    "public_test_cases": [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}],
                },
            ]
            data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            code = external_main(
                [
                    "--benchmark-name",
                    "livecodebench",
                    "--dataset-file",
                    str(data),
                    "--provider",
                    "deepseek",
                    "--model",
                    "fake-model",
                    "--n",
                    "2",
                    "--output-dir",
                    str(out),
                    "--seed",
                    "2",
                ],
                call_fn=LiveCodeBenchFakeProvider(),
                run_eval=False,
            )
            self.assertEqual(code, 0)
            with (out / "labels.csv").open(newline="", encoding="utf-8") as handle:
                label_rows = list(csv.DictReader(handle))
            self.assertEqual({row["label"] for row in label_rows}, {"0", "1"})
            self.assertEqual({row["verifier_name"] for row in label_rows}, {"livecodebench_public_tests_v1"})
            self.assertEqual({row["expected_answer"] for row in label_rows}, {"public_tests_pass"})
            raw_files = list((out / "sessions").rglob("raw.json"))
            self.assertEqual(len(raw_files), 2)
            raw = json.loads(raw_files[0].read_text(encoding="utf-8"))
            self.assertIn("verifier_result", raw)
            self.assertIn("extracted_code", raw)
            self.assertEqual(raw["benchmark_name"], "livecodebench")
            index = json.loads((out / "session_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["pass_count"], 1)
            self.assertEqual(index["fail_count"], 1)
            self.assertIn("easy", index["difficulty_counts"])

    def test_event_token_uses_represented_tokens_not_generated_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = BenchmarkItem(
                prompt_id="p1",
                prompt="External item",
                expected_answer="alpha",
                external_label=None,
                perturbation_type="control",
                split="",
                benchmark_name="break_the_chain",
                verifier_name="final_exact_match_normalized_v1",
                source_metadata={},
            )
            row = run_item(
                item,
                "deepseek",
                "fake-model",
                0.2,
                162,
                5,
                Path(tmp),
                call_fn=CountTokenProvider(162),
            )
            self.assertEqual(row["generated_token_count"], 162)
            self.assertEqual(row["represented_token_count"], 160)
            self.assertEqual(row["event_token"], "160")
            raw = json.loads(Path(row["raw_path"]).read_text(encoding="utf-8"))
            self.assertEqual(raw["generated_token_count"], 162)
            self.assertEqual(raw["represented_token_count"], 160)
            self.assertTrue(raw["truncated_by_max_tokens"])
            self.assertEqual(raw["completion_status"], "truncated")

    def test_labels_csv_never_writes_event_token_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            data = Path(tmp) / "data.jsonl"
            rows = [
                {"id": "a", "question": "External item A", "answer": "alpha"},
                {"id": "b", "question": "External item B", "answer": "beta"},
            ]
            data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            external_main(
                [
                    "--benchmark-name",
                    "break_the_chain",
                    "--dataset-file",
                    str(data),
                    "--provider",
                    "deepseek",
                    "--model",
                    "fake-model",
                    "--n",
                    "2",
                    "--max-tokens",
                    "162",
                    "--output-dir",
                    str(out),
                    "--seed",
                    "1",
                ],
                call_fn=CountTokenProvider(162),
                run_eval=False,
            )
            with (out / "labels.csv").open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    raw = json.loads((out / "sessions" / row["session_id"] / "raw.json").read_text(encoding="utf-8"))
                    self.assertLessEqual(int(row["event_token"]), raw["represented_token_count"])

    def test_truncation_warning_when_failures_are_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            data = Path(tmp) / "lcb.jsonl"
            rows = [
                {
                    "question_id": "a",
                    "question_content": "Read integers and print their sum.",
                    "difficulty": "easy",
                    "platform": "leetcode",
                    "public_test_cases": [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}],
                },
                {
                    "question_id": "b",
                    "question_content": "Read integers and print their sum.",
                    "difficulty": "easy",
                    "platform": "leetcode",
                    "public_test_cases": [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}],
                },
            ]
            data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            external_main(
                [
                    "--benchmark-name",
                    "livecodebench",
                    "--dataset-file",
                    str(data),
                    "--provider",
                    "deepseek",
                    "--model",
                    "fake-model",
                    "--n",
                    "2",
                    "--max-tokens",
                    "64",
                    "--output-dir",
                    str(out),
                    "--seed",
                    "1",
                ],
                call_fn=CountTokenProvider(64, message="print(0)"),
                run_eval=False,
            )
            index = json.loads((out / "session_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["truncated_count"], 2)
            self.assertEqual(index["truncated_fail_count"], 2)
            self.assertIn("methodological warning", index["truncation_warning"])
            self.assertIn("methodological warning", (out / "session_index.md").read_text(encoding="utf-8"))

    def test_livecodebench_unsupported_testtype_and_prama_windows_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            data = Path(tmp) / "lcb.jsonl"
            unsupported = {
                "question_id": "unsupported",
                "question_content": "Return one.",
                "public_test_cases": [{"input": "", "output": "1\n", "testtype": "functional"}],
            }
            short = {
                "question_id": "short",
                "question_content": "Return three.",
                "public_test_cases": [{"input": "", "output": "3\n", "testtype": "stdin"}],
            }
            data.write_text(json.dumps(unsupported) + "\n" + json.dumps(short) + "\n", encoding="utf-8")
            code = external_main(
                [
                    "--benchmark-name",
                    "livecodebench",
                    "--dataset-file",
                    str(data),
                    "--provider",
                    "deepseek",
                    "--model",
                    "fake-model",
                    "--n",
                    "2",
                    "--output-dir",
                    str(out),
                    "--seed",
                    "1",
                ],
                call_fn=ShortLiveCodeBenchFakeProvider(),
                run_eval=False,
            )
            self.assertEqual(code, 2)
            labels = (out / "labels.csv").read_text(encoding="utf-8")
            self.assertEqual(len(labels.strip().splitlines()), 1)
            index = json.loads((out / "session_index.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(index["total_unsupported_verifier"], 1)
            self.assertGreaterEqual(index["total_unsupported_prama_windows"], 1)

    def test_unknown_schema_writes_report_and_aborts_before_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            data = Path(tmp) / "bad.json"
            data.write_text(json.dumps([{"foo": "bar"}]), encoding="utf-8")
            provider = AlternatingFakeProvider()
            with self.assertRaises(SystemExit):
                external_main(
                    [
                        "--benchmark-name",
                        "break_the_chain",
                        "--dataset-file",
                        str(data),
                        "--provider",
                        "deepseek",
                        "--model",
                        "fake-model",
                        "--output-dir",
                        str(out),
                    ],
                    call_fn=provider,
                    run_eval=False,
                )
            self.assertTrue((out / "dataset_schema_report.md").exists())
            self.assertEqual(provider.calls, 0)

    def test_internal_commands_use_module_execution(self):
        validate_cmd, evaluate_cmd = build_internal_commands(
            Path("labels.csv"),
            ["results/a/raw.json"],
            Path("validation"),
            Path("evaluation"),
            0.1,
            "boundary_pressure",
        )
        self.assertIn("-m", validate_cmd)
        self.assertIn("scripts.validate_early_warning_inputs", validate_cmd)
        self.assertIn("-m", evaluate_cmd)
        self.assertIn("scripts.evaluate_early_warning", evaluate_cmd)
        self.assertNotIn("PYTHONPATH", " ".join(validate_cmd + evaluate_cmd))


if __name__ == "__main__":
    unittest.main()

