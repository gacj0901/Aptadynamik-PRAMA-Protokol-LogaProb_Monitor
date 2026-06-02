import importlib.util
import unittest
from pathlib import Path

from aptadynamik.observer.axial_inherence_adapters import MockAxialInherenceAdapters
from aptadynamik.observer.axial_inherence_monitor import (
    bootstrap_ci,
    compute_iota,
    dominances,
    fatigue_series,
    precedence_lead,
    rotational_mobility,
    run_axial_inherence_session,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "prama_axial_inherence_runner.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("prama_axial_inherence_runner", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


class TestAxialInherenceMonitor(unittest.TestCase):
    def test_compute_iota_returns_one_with_no_history(self):
        self.assertEqual(compute_iota([1.0, 0.0], []), 1.0)

    def test_compute_iota_decreases_for_similar_history(self):
        iota = compute_iota([1.0, 0.0], [[0.99, 0.01]])

        self.assertLess(iota, 0.05)

    def test_dominances_compute_i_k_r(self):
        result = dominances(0.4, 0.25, 0.7)

        self.assertAlmostEqual(result[0], 0.4)
        self.assertAlmostEqual(result[1], 0.75)
        self.assertAlmostEqual(result[2], 0.7)

    def test_rotational_mobility_high_for_rotating_sequence(self):
        seq = ["i", "k", "r"] * 6

        self.assertGreater(rotational_mobility(seq)[-1], 0.8)

    def test_rotational_mobility_low_for_stagnant_sequence_after_warmup(self):
        seq = ["k"] * 18

        self.assertLess(rotational_mobility(seq)[-1], 0.2)

    def test_fatigue_series_rises_when_mobility_collapses(self):
        rotating = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] * 4
        stagnant = [(0, 1, 0)] * 12
        fatigue = fatigue_series(rotating + stagnant)

        self.assertGreater(fatigue[-1], fatigue[10])

    def test_precedence_lead_positive_when_fatigue_crosses_before_function_drop(self):
        ikr = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] * 3 + [(0, 1, 0)] * 16
        fn = [0.95] * 18 + [0.2] * 7

        self.assertGreater(precedence_lead(ikr, fn), 0)

    def test_bootstrap_ci_returns_mean_lo_hi_n(self):
        result = bootstrap_ci([1, 2, 3], n=200)

        self.assertEqual(set(result), {"mean", "lo", "hi", "n"})
        self.assertEqual(result["n"], 3)
        self.assertLessEqual(result["lo"], result["mean"])
        self.assertLessEqual(result["mean"], result["hi"])

    def test_axial_inherence_report_includes_required_language(self):
        report = RUNNER.axial_inherence_report_markdown(mode="test", sessions=[])

        self.assertIn("PRAMA Axial Inherence Monitor Report", report)
        self.assertIn("Axial inherence refers to the operational rotation", report)
        self.assertIn("La inherencia axial refiere a la rotación operacional", report)
        self.assertIn("Exogenous Judge Constraint", report)
        self.assertIn("lead = t(function drops) - t(rotation stagnates)", report)
        self.assertIn("Methodological Note", report)

    def test_mock_axial_inherence_adapters_selftest_has_higher_echo_fatigue(self):
        task = {"task_id": "mock", "prompt": "continue"}
        normal = run_axial_inherence_session(MockAxialInherenceAdapters(), task, condition="rotating", max_turns=36)
        echo = run_axial_inherence_session(
            MockAxialInherenceAdapters(),
            task,
            condition="stagnated_echo_decline",
            max_turns=36,
        )

        self.assertGreater(echo["fatigue"][-1], normal["fatigue"][-1])


if __name__ == "__main__":
    unittest.main()
