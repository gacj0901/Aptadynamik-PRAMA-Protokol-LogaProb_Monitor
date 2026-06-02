import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from aptadynamik.observer.viability_metrics import viability_corrected, viability_legacy


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "prama_closure_sweep_runner.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("prama_closure_sweep_runner", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


class TestViabilityAndClosureSweep(unittest.TestCase):
    def test_defensive_extreme_rigidity_lowers_corrected_viability(self):
        basal = viability_corrected(0.55, 0.20, r0=0.55, u0=0.20)
        defensive = viability_corrected(0.95, 0.20, r0=0.55, u0=0.20)

        self.assertGreater(basal, 0.95)
        self.assertLess(defensive, basal)
        self.assertLess(defensive, 0.2)

    def test_erratic_low_rigidity_high_uncertainty_lowers_corrected_viability(self):
        basal = viability_corrected(0.55, 0.20, r0=0.55, u0=0.20)
        erratic = viability_corrected(0.15, 0.70, r0=0.55, u0=0.20)

        self.assertLess(erratic, basal)
        self.assertEqual(erratic, 0.0)

    def test_basal_case_has_high_corrected_viability(self):
        self.assertGreater(viability_corrected(0.56, 0.21, r0=0.55, u0=0.20), 0.9)

    def test_legacy_viability_is_preserved_for_comparison(self):
        self.assertAlmostEqual(viability_legacy(0.7, 0.2), 0.5)

    def test_synthetic_compensation_classifies_as_compensation(self):
        summary = RUNNER.analyze_rows(RUNNER.synthetic_compensation_rows())

        self.assertEqual(summary["classification"], "compatible_with_compensation_under_constraint")

    def test_synthetic_verbose_constant_classifies_as_baseline_verbosity(self):
        summary = RUNNER.analyze_rows(RUNNER.synthetic_verbose_constant_rows())

        self.assertEqual(summary["classification"], "compatible_with_baseline_verbosity")

    def test_closure_sweep_explicit_cli_level_overrides_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "raw_level0.json"
            raw_path.write_text(
                json.dumps(
                    {
                        "session_id": "override",
                        "restriction_level": 99,
                        "turns": [
                            {
                                "turn_index": 0,
                                "assistant_message": "novel strict closure response",
                                "finish_reason": "stop",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            row = RUNNER.row_from_session(raw_path, 2.0, "explicit_cli")

        self.assertEqual(row["raw_path"], str(raw_path))
        self.assertEqual(row["restriction_level"], 2.0)
        self.assertEqual(row["restriction_source"], "explicit_cli")


if __name__ == "__main__":
    unittest.main()
