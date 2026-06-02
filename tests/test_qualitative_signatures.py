import importlib.util
import csv
import json
import tempfile
import unittest
from pathlib import Path

from aptadynamik.observer.qualitative_signatures import (
    sig_discontinuity,
    sig_hysteresis,
    synthetic_smooth_series,
    synthetic_validation,
    viability_from_turn,
)


RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prama_phase_signature_runner.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("prama_phase_signature_runner", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)
run_from_raw = RUNNER.run_from_raw


class TestQualitativeSignatures(unittest.TestCase):
    def test_synthetic_fold_triggers_discontinuity(self):
        validation = synthetic_validation()

        self.assertTrue(validation["fold"]["discontinuity"]["triggered"])

    def test_synthetic_smooth_does_not_trigger_discontinuity(self):
        validation = synthetic_validation()

        self.assertFalse(validation["smooth"]["discontinuity"]["triggered"])

    def test_viability_from_turn_computes_rigidity_minus_uncertainty(self):
        turn = {
            "turn_index": 2,
            "token_count": 17,
            "summary": {
                "avg_rigidity": 0.72,
                "avg_uncertainty": 0.21,
                "avg_entropy_norm": 0.35,
                "max_entropy_range": 0.44,
                "max_entropy_std": 0.12,
            },
        }

        result = viability_from_turn(turn)

        self.assertEqual(result["turn_index"], 2)
        self.assertAlmostEqual(result["viability_legacy"], 0.51)
        self.assertAlmostEqual(result["viability"], 0.51)
        self.assertAlmostEqual(result["entropy_range"], 0.44)
        self.assertAlmostEqual(result["entropy_std"], 0.12)

    def test_viability_from_turn_fails_clearly_on_missing_summary_field(self):
        turn = {"turn_index": 0, "token_count": 1, "summary": {"avg_rigidity": 0.5}}

        with self.assertRaisesRegex(ValueError, "avg_uncertainty"):
            viability_from_turn(turn)

    def test_sig_discontinuity_detects_sharp_drop(self):
        result = sig_discontinuity([0.91, 0.88, 0.84, 0.22, 0.20])

        self.assertTrue(result["triggered"])
        self.assertEqual(result["strongest_transition_turn"], 3)

    def test_sig_hysteresis_returns_positive_area_for_separated_branches(self):
        result = sig_hysteresis([0.9, 0.82, 0.74, 0.35], [0.50, 0.55, 0.60, 0.65])

        self.assertTrue(result["triggered"])
        self.assertGreater(result["hysteresis_area"], 0)

    def test_loading_minimal_raw_json_produces_phase_signatures_csv(self):
        raw = {
            "session_id": "minimal",
            "model": "test-model",
            "turns": [
                {
                    "turn_index": 0,
                    "token_count": 10,
                    "summary": {
                        "avg_rigidity": 0.8,
                        "avg_uncertainty": 0.1,
                        "avg_entropy_norm": 0.2,
                        "max_entropy_range": 0.1,
                        "max_entropy_std": 0.02,
                    },
                },
                {
                    "turn_index": 1,
                    "token_count": 11,
                    "summary": {
                        "avg_rigidity": 0.76,
                        "avg_uncertainty": 0.12,
                        "avg_entropy_norm": 0.25,
                        "max_entropy_range": 0.14,
                        "max_entropy_std": 0.03,
                    },
                },
                {
                    "turn_index": 2,
                    "token_count": 12,
                    "summary": {
                        "avg_rigidity": 0.35,
                        "avg_uncertainty": 0.4,
                        "avg_entropy_norm": 0.55,
                        "max_entropy_range": 0.51,
                        "max_entropy_std": 0.18,
                    },
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "session_minimal_raw.json"
            raw_path.write_text(json.dumps(raw), encoding="utf-8")

            run_from_raw(raw_path, tmp_path / "phase_analysis_minimal")

            self.assertTrue((tmp_path / "phase_analysis_minimal" / "phase_signatures.csv").exists())
            turns_path = tmp_path / "phase_analysis_minimal" / "phase_turns_corrected.csv"
            self.assertTrue(turns_path.exists())
            with turns_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(
                    reader.fieldnames,
                    [
                        "turn_index",
                        "avg_rigidity",
                        "avg_uncertainty",
                        "avg_entropy_norm",
                        "viability_legacy",
                        "viability_corrected",
                        "corrected_fatigue",
                        "baseline_r0",
                        "baseline_u0",
                        "baseline_method",
                        "viability_scale",
                    ],
                )
                row = next(reader)
            self.assertEqual(row["turn_index"], "0")
            self.assertIn(row["baseline_method"], {"first_20_percent_fallback", "labeled_R0_R1_or_control"})

    def test_report_includes_required_language(self):
        raw = {
            "session_id": "report-test",
            "model": "test-model",
            "turns": [
                {
                    "turn_index": 0,
                    "token_count": 8,
                    "summary": {
                        "avg_rigidity": 0.7,
                        "avg_uncertainty": 0.2,
                        "avg_entropy_norm": 0.3,
                        "max_entropy_range": 0.2,
                        "max_entropy_std": 0.05,
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "session_report_raw.json"
            output_dir = tmp_path / "phase_report"
            raw_path.write_text(json.dumps(raw), encoding="utf-8")

            run_from_raw(raw_path, output_dir)
            report = (output_dir / "phase_report.md").read_text(encoding="utf-8")

        self.assertIn("PRAMA Phase Signature Report", report)
        self.assertIn("viability_legacy = avg_rigidity - avg_uncertainty", report)
        self.assertIn("corrected viability is a geometry proxy", report)
        self.assertIn("Methodological Note", report)

    def test_smooth_series_has_no_discontinuity(self):
        smooth = synthetic_smooth_series()

        self.assertFalse(sig_discontinuity(smooth["up"])["triggered"])


if __name__ == "__main__":
    unittest.main()
