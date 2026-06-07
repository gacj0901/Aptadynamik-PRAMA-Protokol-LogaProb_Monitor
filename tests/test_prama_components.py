import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from aptadynamik.prama_problog_components import (
    SUBSTRATE_BLIND_WARNING,
    centered_health,
    measure,
)
from aptadynamik.prama_protokol_core import (
    collapse_xi_norm,
    normalize_regime_label,
    normalize_trajectory_assessment,
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
            "activity_raw",
            "activity_structural",
            "activity_effective",
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
            "accumulated_viability_margin",
            "instant_viability_margin",
            "instant_threshold_crossed",
            "instant_recovered",
            "compression_gap",
            "viability",
            "delta",
            "xi",
            "lam",
            "theta",
            "distance_to_threshold",
            "xi_exceeds_theta",
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
        self.assertAlmostEqual(row["delta_instant"], (1.0 - row["acople"]) * row["activity_effective"])
        self.assertAlmostEqual(row["acople_effective"], 1.0 - row["delta_instant"])
        self.assertAlmostEqual(row["activity"], row["activity_effective"])
        self.assertLessEqual(row["activity"], row["activity_structural"])
        self.assertAlmostEqual(row["viability"], row["viability_margin"])
        self.assertAlmostEqual(row["delta"], row["delta_instant"])
        self.assertAlmostEqual(row["xi"], row["xi_accumulated"])
        self.assertAlmostEqual(row["lam"], row["lambda_remaining"])
        self.assertAlmostEqual(row["theta"], row["theta_dynamic"])
        self.assertAlmostEqual(row["distance_to_threshold"], row["viability_margin"])
        self.assertEqual(row["threshold_crossed"], row["xi_exceeds_theta"])
        self.assertAlmostEqual(row["accumulated_viability_margin"], row["viability_margin"])
        self.assertAlmostEqual(row["instant_viability_margin"], row["theta_dynamic"] - row["delta_instant"])

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

    def test_ack_raw_activity_does_not_inject_structural_delta(self):
        raw_turns = [
            {
                "turn_index": 0,
                "activity": 0.625,
                "tokens": [{"top1_logprob": -0.5}, {"top1_logprob": -0.5}],
            },
            {
                "turn_index": 1,
                "activity": 1.0,
                "tokens": [{"top1_logprob": -0.4}, {"top1_logprob": -0.8}],
            },
        ]
        result = measure(raw_turns, calib_window=2)
        row = result["turns"][0]

        self.assertAlmostEqual(row["micro_raw"], 0.0)
        self.assertAlmostEqual(row["activity_raw"], 0.625)
        self.assertAlmostEqual(row["activity_structural"], 0.0)
        self.assertAlmostEqual(row["activity_effective"], 0.0)
        self.assertAlmostEqual(row["activity"], row["activity_effective"])
        self.assertAlmostEqual(row["delta_instant"], 0.0)
        self.assertAlmostEqual(row["acople_effective"], 1.0)
        self.assertFalse(row["threshold_crossed"])
        self.assertNotIn(0, result["critical_turns"])
        self.assertNotEqual(result["first_crossing_turn"], 0)

    def test_activity_never_exceeds_structural_activity(self):
        row = measure(
            [
                {
                    "turn_index": 0,
                    "activity": 0.9,
                    "tokens": [{"top1_logprob": -0.2}, {"top1_logprob": -0.3}],
                }
            ],
            calib_window=1,
        )["turns"][0]

        self.assertLessEqual(row["activity"], row["activity_structural"])
        self.assertLessEqual(row["activity_effective"], row["activity_structural"])

    def test_critical_turns_exclude_zero(self):
        result = measure(
            [
                {
                    "turn_index": 0,
                    "activity": 1.0,
                    "tokens": [{"top1_logprob": -0.5}, {"top1_logprob": -0.5}],
                }
            ],
            calib_window=1,
            theta0=0.01,
        )

        self.assertFalse(result["turns"][0]["threshold_crossed"])
        self.assertNotIn(0, result["critical_turns"])
        self.assertIsNone(result["first_crossing_turn"])

    def test_compression_gap_is_none(self):
        row = measure([turn(0, [-0.2, -0.4]), turn(1, [-0.25, -0.45])], calib_window=1)["turns"][0]

        self.assertIsNone(row["compression_gap"])

    def test_organized_stability_without_crossing(self):
        result = measure(
            [
                turn(0, [-0.75, -1.25]),
                turn(1, [-0.76, -1.24]),
                turn(2, [-0.74, -1.26]),
            ],
            calib_window=1,
        )

        self.assertEqual(result["regime_label"], "II_ORGANIZED_STABILITY")
        self.assertEqual(result["trajectory_assessment"], "VIABLE_ORGANIZED_STABILITY")
        self.assertFalse(result["threshold_crossed"])
        self.assertFalse(result["recovery_observed"])
        self.assertIsNone(result["first_crossing_turn"])

    def test_organized_equilibrium_legacy_aliases_normalize_to_stability(self):
        self.assertEqual(normalize_regime_label("II_ORGANIZED_EQUILIBRIUM"), "II_ORGANIZED_STABILITY")
        self.assertEqual(
            normalize_trajectory_assessment("VIABLE_ORGANIZED_EQUILIBRIUM"),
            "VIABLE_ORGANIZED_STABILITY",
        )
        self.assertEqual(normalize_regime_label("II_ORGANIZED_STABILITY"), "II_ORGANIZED_STABILITY")

    def test_crossing_with_recovery_is_structural_pulsation(self):
        result = measure(
            [
                turn(0, [-0.8, -1.2]),
                turn(1, [-0.1, -2.1]),
                turn(2, [-0.8, -1.2]),
                turn(3, [-0.8, -1.2]),
                turn(4, [-0.8, -1.2]),
                turn(5, [-0.8, -1.2]),
            ],
            calib_window=1,
        )

        self.assertEqual(result["regime_label"], "III_STRUCTURAL_PULSATION")
        self.assertEqual(result["trajectory_assessment"], "THRESHOLD_CROSSED_STRUCTURAL_PULSATION")
        self.assertTrue(result["threshold_crossed"])
        self.assertTrue(result["recovery_observed"])
        self.assertIsNotNone(result["first_crossing_turn"])
        self.assertTrue(result["post_crossing_recovery_turns"])

    def test_persistent_crossing_is_entropic_collapse(self):
        result = measure(
            [
                turn(0, [-0.8, -1.2]),
                turn(1, [-0.1, -2.1]),
                turn(2, [-0.1, -2.1]),
                turn(3, [-0.1, -2.1]),
                turn(4, [-0.1, -2.1]),
                turn(5, [-0.1, -2.1]),
            ],
            calib_window=1,
        )

        self.assertEqual(result["regime_label"], "IV_ENTROPIC_COLLAPSE")
        self.assertEqual(result["trajectory_assessment"], "ENTROPIC_COLLAPSE")
        self.assertFalse(result["recovery_observed"])
        self.assertGreaterEqual(result["persistent_crossing_ratio"], 0.80)
        self.assertLess(result["final_viability"], 0.0)

    def test_token_window_short_response_with_local_crossings_is_calibrating(self):
        result = measure(
            [
                turn(0, [-0.8, -1.2]),
                turn(1, [-0.1, -2.1]),
                turn(2, [-0.1, -2.1]),
                turn(3, [-0.1, -2.1]),
                turn(4, [-0.1, -2.1]),
            ],
            calib_window=1,
            crossing_index_scope="token_window",
        )
        crossed_windows = [row for row in result["turns"] if row["threshold_crossed"]]

        self.assertEqual(len(result["turns"]), 5)
        self.assertGreaterEqual(len(crossed_windows), 4)
        self.assertTrue(result["threshold_crossed"])
        self.assertTrue(result["xi_exceeds_theta"])
        self.assertEqual(result["regime_label"], "CALIBRATING")
        self.assertEqual(result["trajectory_assessment"], "INSUFFICIENT_HISTORY")
        self.assertNotEqual(result["regime_label"], "IV_ENTROPIC_COLLAPSE")
        self.assertTrue(result["local_threshold_cascade"])
        self.assertEqual(result["crossing_index_scope"], "token_window")
        self.assertEqual(result["first_crossing_window"], 1)
        self.assertEqual(result["persistent_crossing_ratio"], 0.0)

    def test_token_window_sufficient_history_can_be_entropic_collapse(self):
        result = measure(
            [turn(0, [-0.8, -1.2])]
            + [turn(index, [-0.1, -2.1]) for index in range(1, 13)],
            calib_window=1,
            crossing_index_scope="token_window",
        )

        self.assertEqual(len(result["turns"]), 13)
        self.assertEqual(result["regime_label"], "IV_ENTROPIC_COLLAPSE")
        self.assertEqual(result["trajectory_assessment"], "ENTROPIC_COLLAPSE")
        self.assertGreaterEqual(result["persistent_crossing_ratio"], 0.80)

    def test_token_window_crossing_with_recovery_is_structural_pulsation_after_history(self):
        result = measure(
            [turn(0, [-0.8, -1.2])]
            + [turn(index, [-0.1, -2.1]) for index in range(1, 5)]
            + [turn(index, [-0.8, -1.2]) for index in range(5, 14)],
            calib_window=1,
            crossing_index_scope="token_window",
        )

        self.assertEqual(result["regime_label"], "III_STRUCTURAL_PULSATION")
        self.assertEqual(result["trajectory_assessment"], "THRESHOLD_CROSSED_STRUCTURAL_PULSATION")
        self.assertTrue(result["recovery_observed"])
        self.assertTrue(result["post_crossing_recovery_turns"])
        self.assertTrue(result["recovered_finally"])
        self.assertFalse(result["relapsed_after_recovery"])
        self.assertEqual(result["pulsation_subtype"], "RECOVERED_FINAL")

    def test_structural_pulsation_with_final_relapse_reports_subtype(self):
        result = measure(
            [turn(0, [-0.8, -1.2])]
            + [turn(index, [-0.1, -2.1]) for index in range(1, 3)]
            + [turn(index, [-0.8, -1.2]) for index in range(3, 10)]
            + [turn(index, [-0.1, -2.1]) for index in range(10, 14)],
            calib_window=1,
            crossing_index_scope="token_window",
        )

        self.assertEqual(result["regime_label"], "III_STRUCTURAL_PULSATION")
        self.assertTrue(result["recovery_observed"])
        self.assertFalse(result["recovered_finally"])
        self.assertTrue(result["relapsed_after_recovery"])
        self.assertEqual(result["pulsation_subtype"], "RELAPSED_AFTER_RECOVERY")

    def test_non_pulsation_has_no_pulsation_subtype(self):
        result = measure([turn(index, [-0.8, -1.2]) for index in range(12)], calib_window=1)

        self.assertNotEqual(result["regime_label"], "III_STRUCTURAL_PULSATION")
        self.assertIsNone(result["pulsation_subtype"])
        self.assertFalse(result["relapsed_after_recovery"])

    def test_low_activity_low_raw_acople_can_be_subcritical_dissolution(self):
        low_activity_turns = [
            {
                "turn_index": 0,
                "activity": 0.0,
                "tokens": [{"top1_logprob": -0.5}, {"top1_logprob": -1.5}],
            },
            {
                "turn_index": 1,
                "activity": 0.0,
                "tokens": [{"top1_logprob": -0.5}, {"top1_logprob": -0.5}],
            },
            {
                "turn_index": 2,
                "activity": 0.0,
                "tokens": [{"top1_logprob": -0.5}, {"top1_logprob": -0.5}],
            },
            {
                "turn_index": 3,
                "activity": 0.0,
                "tokens": [{"top1_logprob": -0.5}, {"top1_logprob": -0.5}],
            },
            {
                "turn_index": 4,
                "activity": 0.0,
                "tokens": [{"top1_logprob": -0.5}, {"top1_logprob": -0.5}],
            },
        ]
        result = measure(low_activity_turns, calib_window=1)

        self.assertEqual(result["regime_label"], "I_SUBCRITICAL_DISSOLUTION")
        self.assertEqual(result["trajectory_assessment"], "SUBCRITICAL_DISSOLUTION")
        self.assertFalse(result["threshold_crossed"])
        self.assertFalse(result["xi_exceeds_theta"])
        self.assertIsNone(result["first_crossing_turn"])

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

    def test_runner_accepts_dynamic_parameter_arguments(self):
        args = RUNNER.build_parser().parse_args(
            [
                "--from-raw",
                "raw.json",
                "--output-dir",
                "out",
                "--theta0",
                "0.5",
                "--lambda0",
                "0.8",
                "--memory-beta",
                "0.4",
                "--delta-ref",
                "1.0",
                "--theta0-grid",
                "0.35",
                "0.5",
                "--lambda0-grid",
                "1.0",
                "--memory-beta-grid",
                "0.3",
                "0.7",
            ]
        )

        self.assertAlmostEqual(args.theta0, 0.5)
        self.assertAlmostEqual(args.lambda0, 0.8)
        self.assertAlmostEqual(args.memory_beta, 0.4)
        self.assertAlmostEqual(args.delta_ref, 1.0)
        self.assertEqual(args.theta0_grid, [0.35, 0.5])
        self.assertEqual(args.lambda0_grid, [1.0])
        self.assertEqual(args.memory_beta_grid, [0.3, 0.7])

    def test_parametric_sensitivity_minimal_two_theta0(self):
        raw = {
            "session_id": "components",
            "turns": [
                turn(0, [-0.75, -1.25]),
                turn(1, [-0.4, -1.6]),
                turn(2, [-0.75, -1.25]),
                turn(3, [-0.75, -1.25]),
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "raw.json"
            out_dir = tmp_path / "out"
            raw_path.write_text(json.dumps(raw), encoding="utf-8")

            payload = RUNNER.run_parametric_sensitivity(
                raw_path,
                out_dir,
                calib_window=1,
                theta0_grid=[0.35, 0.5],
                lambda0_grid=[1.0],
                memory_beta_grid=[0.3],
            )
            sensitivity_json = json.loads((out_dir / "components_parametric_sensitivity.json").read_text(encoding="utf-8"))
            sensitivity_md = (out_dir / "components_parametric_sensitivity.md").read_text(encoding="utf-8")

        self.assertEqual(len(payload["runs"]), 2)
        self.assertEqual(len(sensitivity_json["runs"]), 2)
        for row in sensitivity_json["runs"]:
            self.assertIn("regime_label", row)
            self.assertIn("trajectory_assessment", row)
            self.assertIn("threshold_crossing_ratio", row)
            self.assertIn("persistent_crossing_ratio", row)
        self.assertIn("regime_label_counts", sensitivity_json)
        self.assertIn("trajectory_assessment_counts", sensitivity_json)
        self.assertIn("robust_regime_label", sensitivity_json)
        self.assertIn("robust_trajectory_assessment", sensitivity_json)
        self.assertIn("Parametric Sensitivity", sensitivity_md)

    def test_runner_report_mentions_parametric_sensitivity_when_requested(self):
        raw = {
            "session_id": "components",
            "turns": [turn(0, [-0.75, -1.25]), turn(1, [-0.4, -1.6]), turn(2, [-0.75, -1.25])],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "raw.json"
            out_dir = tmp_path / "out"
            raw_path.write_text(json.dumps(raw), encoding="utf-8")

            parametric = RUNNER.run_parametric_sensitivity(
                raw_path,
                out_dir,
                calib_window=1,
                theta0_grid=[0.35, 0.5],
                lambda0_grid=[1.0],
                memory_beta_grid=[0.3],
            )
            RUNNER.run_from_raw(
                raw_path,
                out_dir,
                calib_window=1,
                theta0=0.5,
                lambda0=1.0,
                memory_beta=0.3,
                delta_ref=2.0,
                parametric_payload=parametric,
            )
            summary = json.loads((out_dir / "components_summary.json").read_text(encoding="utf-8"))
            report = (out_dir / "components_report.md").read_text(encoding="utf-8")

        self.assertAlmostEqual(summary["theta0"], 0.5)
        self.assertAlmostEqual(summary["lambda0"], 1.0)
        self.assertAlmostEqual(summary["memory_beta"], 0.3)
        self.assertAlmostEqual(summary["delta_ref"], 2.0)
        self.assertIn("components_parametric_sensitivity.json", report)
        self.assertIn("robust_regime_label", report)


if __name__ == "__main__":
    unittest.main()
