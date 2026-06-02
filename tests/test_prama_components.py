import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from aptadynamik.prama_components import (
    SUBSTRATE_BLIND_WARNING,
    centered_health,
    collapse_xi_norm,
    measure,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "prama_components_runner.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("prama_components_runner", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


def turn(index, logprobs):
    return {
        "turn_index": index,
        "tokens": [{"top1_logprob": value} for value in logprobs],
    }


def mixed_turn(index, token_values):
    return {
        "turn_index": index,
        "tokens": token_values,
    }


def forbidden_names():
    return {"ri" + "ma", "mo" + "de", "coll" + "apsed"}


class TestPramaComponents(unittest.TestCase):
    def test_measure_returns_canonical_turn_fields(self):
        result = measure([turn(0, [-0.2, -0.4]), turn(1, [-0.25, -0.45])], calib_window=1)
        row = result["turns"][0]
        required = {
            "turn_index",
            "valid_token_count",
            "logprob_valid",
            "insufficient_data",
            "micro_raw",
            "micro_health",
            "macro_health",
            "activity",
            "micro_drop",
            "micro_excess",
            "acople",
            "acople_effective",
            "delta_instant",
            "xi_accumulated",
            "xi_norm",
            "lambda_remaining",
            "theta_dynamic",
            "viability_margin",
            "compression_gap",
            "viability",
            "delta",
            "xi",
            "lam",
            "theta",
            "distance_to_threshold",
            "threshold_crossed",
            "viability_status",
            "boundary_pressure",
            "boundary_side",
        }

        self.assertTrue(required.issubset(row.keys()))
        self.assertTrue(forbidden_names().isdisjoint(row.keys()))

    def test_acople_effective_and_alias_definitions(self):
        result = measure([turn(0, [-0.75, -1.25]), turn(1, [-0.4, -1.6])], calib_window=1)
        row = result["turns"][-1]

        self.assertAlmostEqual(row["acople"], min(row["micro_health"], row["macro_health"]))
        self.assertAlmostEqual(row["delta_instant"], (1.0 - row["acople"]) * row["activity"])
        self.assertAlmostEqual(row["acople_effective"], 1.0 - row["delta_instant"])
        self.assertAlmostEqual(row["viability"], row["viability_margin"])
        self.assertAlmostEqual(row["delta"], row["delta_instant"])
        self.assertAlmostEqual(row["xi"], row["xi_accumulated"])
        self.assertAlmostEqual(row["lam"], row["lambda_remaining"])
        self.assertAlmostEqual(row["theta"], row["theta_dynamic"])
        self.assertAlmostEqual(row["distance_to_threshold"], row["viability_margin"])
        self.assertEqual(row["threshold_crossed"], row["viability_margin"] <= 0.0)

    def test_viable_status_values(self):
        viable = measure(
            [turn(0, [-0.75, -1.25]), turn(1, [-0.72, -1.22])],
            calib_window=1,
            critical_margin=0.05,
        )["turns"][-1]
        near = measure(
            [turn(0, [-0.75, -1.25]), turn(1, [-0.75, -1.25]), turn(2, [-0.95, -1.05])],
            calib_window=1,
            theta0=0.20,
            critical_margin=0.10,
        )["turns"][-1]
        crossed = measure(
            [turn(0, [-0.75, -1.25]), turn(1, [-0.4, -1.6])],
            calib_window=1,
            theta0=0.05,
            critical_margin=0.05,
        )["turns"][-1]

        self.assertEqual(viable["viability_status"], "VIABLE")
        self.assertEqual(near["viability_status"], "NEAR_THRESHOLD")
        self.assertEqual(crossed["viability_status"], "THRESHOLD_CROSSED")

    def test_boundary_side_values_are_clean(self):
        result = measure([turn(0, [-0.75, -1.25]), turn(1, [-0.4, -1.6])], calib_window=1)
        row = result["turns"][-1]
        allowed = {"CENTERED", "CONDENSATION", "DISSOLUTION", "DECOUPLING", "UNRESOLVED"}

        self.assertIn(row["boundary_side"], allowed)
        self.assertEqual(row["boundary_side"], row["boundary_side"].strip())
        self.assertEqual(row["viability_status"], row["viability_status"].strip())

    def test_boundary_pressure_is_threshold_proximity(self):
        result = measure(
            [turn(0, [-0.75, -1.25]), turn(1, [-0.9, -1.1])],
            calib_window=1,
            theta0=0.20,
            critical_margin=0.10,
        )
        row = result["turns"][-1]
        expected = max(
            0.0,
            min(
                1.0,
                1.0 - (row["viability_margin"] / result["critical_margin"]),
            ),
        )

        self.assertAlmostEqual(row["boundary_pressure"], expected)

    def test_invalid_logprobs_do_not_create_sentinel_values(self):
        result = measure(
            [
                mixed_turn(
                    0,
                    [
                        {"top1_logprob": -0.2},
                        {"top1_logprob": 9999.0},
                        {"top1_logprob": float("nan")},
                        {"top1_logprob": -0.4},
                    ],
                )
            ],
            calib_window=1,
        )
        row = result["turns"][0]

        self.assertEqual(row["valid_token_count"], 2)
        self.assertEqual(row["invalid_token_count"], 2)
        self.assertLess(row["micro_raw"], 1.0)

    def test_insufficient_data_does_not_cross_threshold(self):
        result = measure([mixed_turn(0, [{"top1_logprob": 9999.0}, {"token": "x"}])], calib_window=1)
        row = result["turns"][0]

        self.assertFalse(row["logprob_valid"])
        self.assertTrue(row["insufficient_data"])
        self.assertIsNone(row["viability"])
        self.assertFalse(row["threshold_crossed"])

    def test_substrate_blind_flags(self):
        result = measure([turn(0, [-0.2, -0.4]), turn(1, [-0.2, -0.4])], calib_window=1)

        self.assertTrue(result["substrate_blind"])
        self.assertFalse(result["material_cost_measured"])
        self.assertTrue(result["requires_exogenous_telemetry_for_material_cost"])

    def test_ack_low_activity_uses_effective_acople(self):
        raw_turns = [
            {
                "turn_index": 0,
                "activity": 1.0,
                "tokens": [{"top1_logprob": -0.8}, {"top1_logprob": -1.2}],
            },
            {
                "turn_index": 1,
                "activity": 0.0,
                "tokens": [{"top1_logprob": -0.1}, {"top1_logprob": -2.0}],
            },
        ]
        row = measure(raw_turns, calib_window=1)["turns"][-1]

        self.assertAlmostEqual(row["delta_instant"], 0.0)
        self.assertAlmostEqual(row["xi_accumulated"], 0.0)
        self.assertAlmostEqual(row["acople_effective"], 1.0)

    def test_compression_gap_is_none(self):
        row = measure([turn(0, [-0.2, -0.4]), turn(1, [-0.25, -0.45])], calib_window=1)["turns"][0]

        self.assertIsNone(row["compression_gap"])

    def test_collapse_xi_norm_uses_general_formula(self):
        theta0 = 0.35
        lambda0 = 0.5
        expected = (theta0 * lambda0) / (1.0 + theta0 * lambda0)

        self.assertAlmostEqual(collapse_xi_norm(theta0, lambda0), expected)
        self.assertAlmostEqual(collapse_xi_norm(theta0, 1.0), theta0 / (1.0 + theta0))

    def test_centered_health_penalizes_low_and_high_micro(self):
        baseline = 1.0
        centered = centered_health(1.0, baseline)
        low = centered_health(0.2, baseline)
        high = centered_health(1.8, baseline)

        self.assertGreater(centered, low)
        self.assertGreater(centered, high)
        self.assertAlmostEqual(centered, 1.0)

    def test_runner_outputs_canonical_artifacts(self):
        raw = {"session_id": "components", "turns": [turn(0, [-0.75, -1.25]), turn(1, [-0.4, -1.6])]}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "raw.json"
            out_dir = tmp_path / "out"
            raw_path.write_text(json.dumps(raw), encoding="utf-8")

            RUNNER.run_from_raw(raw_path, out_dir, calib_window=1)
            summary = json.loads((out_dir / "components_summary.json").read_text(encoding="utf-8"))
            turns_text = (out_dir / "components_turns.csv").read_text(encoding="utf-8")
            report = (out_dir / "components_report.md").read_text(encoding="utf-8")

        rows = list(csv.DictReader(io.StringIO(turns_text)))
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(forbidden_names().isdisjoint(row.keys()))
        self.assertTrue(forbidden_names().isdisjoint(summary.keys()))
        self.assertIn("Substrate-Blind Warning", report)
        self.assertIn(SUBSTRATE_BLIND_WARNING, report)
        for forbidden in forbidden_names():
            self.assertNotIn(forbidden, report)
            self.assertNotIn(forbidden, turns_text)
            self.assertNotIn(forbidden, json.dumps(summary))

    def test_calibration_sensitivity_payload(self):
        raw_turns = [
            turn(0, [-0.75, -1.25]),
            turn(1, [-0.75, -1.25]),
            turn(2, [-0.75, -1.25]),
            turn(3, [-0.75, -1.25]),
            turn(4, [-0.4, -1.6]),
            turn(5, [-0.4, -1.6]),
            turn(6, [-0.4, -1.6]),
        ]
        payload = RUNNER.calibration_sensitivity_payload(raw_turns, [3, 5, 7])
        required = {
            "substrate_blind",
            "material_cost_measured",
            "requires_exogenous_telemetry_for_material_cost",
            "windows",
            "per_window",
            "final_viability_min",
            "final_viability_max",
            "final_viability_range",
            "threshold_crossed_count",
            "threshold_crossed_rate",
            "boundary_side_counts",
            "boundary_side_consensus",
            "critical_turn_counts",
            "critical_turn_consensus",
            "crossing_stability",
            "trajectory_assessment",
        }

        self.assertTrue(required.issubset(payload.keys()))
        self.assertEqual(payload["windows"], [3, 5, 7])
        self.assertEqual(len(payload["per_window"]), 3)
        self.assertTrue(forbidden_names().isdisjoint(payload.keys()))
        for row in payload["per_window"]:
            self.assertIn("calib_window", row)
            self.assertIn("baseline_micro", row)
            self.assertIn("final_boundary_side", row)
            self.assertTrue(forbidden_names().isdisjoint(row.keys()))

    def test_calibration_sensitivity_files_are_written(self):
        raw = {
            "session_id": "components",
            "turns": [
                turn(0, [-0.75, -1.25]),
                turn(1, [-0.75, -1.25]),
                turn(2, [-0.75, -1.25]),
                turn(3, [-0.4, -1.6]),
                turn(4, [-0.4, -1.6]),
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "raw.json"
            out_dir = tmp_path / "out"
            raw_path.write_text(json.dumps(raw), encoding="utf-8")

            payload = RUNNER.run_calibration_sensitivity(raw_path, out_dir, [3, 5, 7])
            sensitivity_json = json.loads((out_dir / "components_calibration_sensitivity.json").read_text(encoding="utf-8"))
            sensitivity_md = (out_dir / "components_calibration_sensitivity.md").read_text(encoding="utf-8")

        self.assertEqual(payload["windows"], [3, 5, 7])
        self.assertEqual(sensitivity_json["windows"], [3, 5, 7])
        self.assertIn("boundary_side_consensus", sensitivity_md)
        self.assertIn("crossing_stability", sensitivity_md)
        for forbidden in forbidden_names():
            self.assertNotIn(forbidden, sensitivity_md)
            self.assertNotIn(forbidden, json.dumps(sensitivity_json))


if __name__ == "__main__":
    unittest.main()
