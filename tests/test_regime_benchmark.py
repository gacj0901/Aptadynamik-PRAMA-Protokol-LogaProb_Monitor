import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_regime_benchmark.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_regime_benchmark", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


class TestRegimeBenchmark(unittest.TestCase):
    def test_benchmark_generates_four_expected_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "regime_benchmark"
            result = RUNNER.run_benchmark(output_dir)

            self.assertEqual(result["scenario_count"], 4)
            self.assertTrue(result["passed"])
            scenario_names = {item["scenario"] for item in result["scenarios"]}
            self.assertEqual(
                scenario_names,
                {
                    "short_calibrating_local_crossings",
                    "organized_viability",
                    "structural_pulsation",
                    "entropic_collapse",
                },
            )

    def test_each_scenario_produces_expected_regime_and_artifacts(self):
        expected = {
            "short_calibrating_local_crossings": ("CALIBRATING", "INSUFFICIENT_HISTORY"),
            "organized_viability": ("II_ORGANIZED_EQUILIBRIUM", "VIABLE_ORGANIZED_EQUILIBRIUM"),
            "structural_pulsation": ("III_STRUCTURAL_PULSATION", "THRESHOLD_CROSSED_STRUCTURAL_PULSATION"),
            "entropic_collapse": ("IV_ENTROPIC_COLLAPSE", "ENTROPIC_COLLAPSE"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "regime_benchmark"
            RUNNER.run_benchmark(output_dir)

            for scenario, (regime_label, trajectory_assessment) in expected.items():
                scenario_dir = output_dir / scenario
                self.assertTrue((scenario_dir / "raw.json").exists())
                self.assertTrue((scenario_dir / "report.md").exists())
                self.assertTrue((scenario_dir / "summary.json").exists())
                summary = json.loads((scenario_dir / "summary.json").read_text(encoding="utf-8"))
                report = (scenario_dir / "report.md").read_text(encoding="utf-8")

                self.assertEqual(summary["regime_label"], regime_label)
                self.assertEqual(summary["trajectory_assessment"], trajectory_assessment)
                self.assertIn("Threshold crossing is a local viability event", report)

    def test_short_calibrating_local_crossings_cannot_emit_entropic_collapse(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "regime_benchmark"
            RUNNER.run_benchmark(output_dir)
            summary = json.loads(
                (output_dir / "short_calibrating_local_crossings" / "summary.json").read_text(encoding="utf-8")
            )

            self.assertEqual(summary["regime_label"], "CALIBRATING")
            self.assertEqual(summary["trajectory_assessment"], "INSUFFICIENT_HISTORY")
            self.assertNotEqual(summary["regime_label"], "IV_ENTROPIC_COLLAPSE")

    def test_manifest_contains_reproducibility_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "regime_benchmark"
            RUNNER.run_benchmark(output_dir)
            manifest_path = output_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertTrue(manifest_path.exists())
            self.assertIn("generated_at", manifest)
            self.assertIn("git_commit_sha", manifest)
            self.assertIn("git_branch", manifest)
            self.assertIn("python_version", manifest)
            self.assertIn("platform", manifest)
            self.assertIn("benchmark_version", manifest)
            self.assertEqual(manifest["scenario_count"], 4)
            self.assertEqual(len(manifest["scenarios"]), 4)
            self.assertTrue(all(item["passed"] for item in manifest["scenarios"]))
            self.assertEqual(manifest["aggregate_report_path"], str(output_dir / "aggregate_report.md"))

    def test_manifest_scenario_hashes_are_stable_from_summary_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "regime_benchmark"
            RUNNER.run_benchmark(output_dir)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

            for item in manifest["scenarios"]:
                self.assertTrue(item["result_hash"])
                summary = json.loads(Path(item["summary_path"]).read_text(encoding="utf-8"))
                self.assertEqual(item["result_hash"], RUNNER.stable_summary_hash(summary))

    def test_benchmark_writes_only_inside_requested_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "regime_benchmark"
            RUNNER.run_benchmark(output_dir)

            self.assertEqual({path.name for path in root.iterdir()}, {"regime_benchmark"})

    def test_aggregate_report_contains_scenarios_note_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "regime_benchmark"
            RUNNER.run_benchmark(output_dir)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            aggregate_report = output_dir / "aggregate_report.md"
            text = aggregate_report.read_text(encoding="utf-8")

            self.assertTrue(aggregate_report.exists())
            self.assertIn("# PRAMA Regime Benchmark Aggregate Report", text)
            self.assertIn("Threshold crossing is a local viability event", text)
            for item in manifest["scenarios"]:
                self.assertIn(item["scenario_name"], text)
                self.assertIn(item["result_hash"], text)


if __name__ == "__main__":
    unittest.main()
