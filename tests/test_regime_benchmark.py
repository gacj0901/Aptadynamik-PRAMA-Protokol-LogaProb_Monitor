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


if __name__ == "__main__":
    unittest.main()
