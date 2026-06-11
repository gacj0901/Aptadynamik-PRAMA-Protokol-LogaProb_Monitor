"""Tests for micro_health modes, baseline guards, and sensitivity-window degeneracy.

Motivation (smoke-test postmortem, deepseek_smoke 2026-06-09):
- With calib_window=1 the baseline anchored to a 19-token near-deterministic
  greeting (micro_raw = 0.231). Subsequent content turns (micro_raw = 1.486,
  0.819) registered relative excesses of 5.44x and 2.55x, hard-zeroing the
  legacy linear micro_health, saturating delta_instant at 1.0, and promoting a
  normal 3-turn conversation to IV_ENTROPIC_COLLAPSE.
- The calibration sensitivity harness ran windows [3, 5, 7] on a 3-turn
  trajectory: all windows clamp to the same effective window, so the
  "STABLE_CROSSED" verdict was one computation repeated three times.

These tests pin the fixes:
1. log_ratio micro_health preserves gradation and is multiplicatively symmetric.
2. The legacy linear mode remains the byte-identical default.
3. baseline_stat="median" and min_calib_tokens guard the calibration window.
4. The sensitivity harness refuses stability verdicts under degenerate windows.
"""

import importlib.util
import math
import unittest
from pathlib import Path

from aptadynamik.prama_problog_components import (
    baseline_from_turns,
    centered_health,
    log_ratio_health,
    measure,
    micro_health_from_mode,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "prama_components_runner.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("prama_components_runner", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


def synthetic_turn(turn_index: int, micro: float, mean_surprise: float = 4.0, tokens: int = 8):
    """Build a turn whose micro_raw (max-min surprise) equals `micro` exactly."""
    half = micro / 2.0
    assert mean_surprise - half >= 0.0, "surprises must stay non-negative"
    surprises = [mean_surprise - half, mean_surprise + half]
    surprises += [mean_surprise] * max(0, tokens - 2)
    return {
        "turn_index": turn_index,
        "tokens": [{"top1_logprob": -s} for s in surprises],
    }


def smoke_pattern_turns():
    """Replica of the deepseek_smoke amplitude pattern: greeting then content."""
    return [
        synthetic_turn(0, 0.231, mean_surprise=2.0, tokens=4),
        synthetic_turn(1, 1.486, mean_surprise=2.2, tokens=8),
        synthetic_turn(2, 0.819, mean_surprise=2.1, tokens=8),
    ]


class MicroHealthModeTests(unittest.TestCase):
    def test_linear_hard_zeros_log_ratio_preserves_gradation(self):
        baseline = 0.231
        # The two real content turns of the smoke test: 6.43x and 3.55x baseline.
        for micro in (1.486, 0.819):
            self.assertEqual(centered_health(micro, baseline), 0.0)
        h_far = log_ratio_health(1.486, baseline)
        h_near = log_ratio_health(0.819, baseline)
        self.assertGreater(h_near, h_far)
        self.assertGreater(h_far, 0.0)

    def test_log_ratio_multiplicative_symmetry(self):
        baseline = 0.8
        self.assertAlmostEqual(
            log_ratio_health(2.0 * baseline, baseline),
            log_ratio_health(baseline / 2.0, baseline),
            places=12,
        )
        # The legacy linear form is asymmetric in ratio space.
        self.assertNotAlmostEqual(
            centered_health(2.0 * baseline, baseline),
            centered_health(baseline / 2.0, baseline),
            places=3,
        )

    def test_log_ratio_scale_semantics(self):
        baseline = 1.0
        # health(e * b, b, scale=1) == exp(-1)
        self.assertAlmostEqual(
            log_ratio_health(math.e * baseline, baseline, scale=1.0),
            math.exp(-1.0),
            places=12,
        )
        # Larger scale -> more tolerant.
        self.assertGreater(
            log_ratio_health(2.0, baseline, scale=2.0),
            log_ratio_health(2.0, baseline, scale=1.0),
        )

    def test_log_ratio_never_hard_zero(self):
        baseline = 0.231
        for ratio in (3.0, 10.0, 100.0):
            self.assertGreater(log_ratio_health(ratio * baseline, baseline), 0.0)

    def test_mode_dispatcher_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            micro_health_from_mode(1.0, 1.0, mode="quadratic")


class MeasureModeTests(unittest.TestCase):
    def test_linear_mode_is_default_and_identical(self):
        turns = smoke_pattern_turns()
        default = measure(turns, calib_window=1)
        explicit = measure(turns, calib_window=1, micro_health_mode="linear")
        self.assertEqual(default["final_viability"], explicit["final_viability"])
        self.assertEqual(default["trajectory_assessment"], explicit["trajectory_assessment"])
        self.assertEqual(default["micro_health_mode"], "linear")
        self.assertIsNone(default["micro_health_scale"])

    def test_smoke_pattern_legacy_saturates_delta(self):
        turns = smoke_pattern_turns()
        result = measure(turns, calib_window=1)
        deltas = [row["delta_instant"] for row in result["turns"][1:]]
        # Legacy linear health hard-zeros both content turns -> delta saturated.
        for delta in deltas:
            self.assertAlmostEqual(delta, 1.0, places=6)

    def test_smoke_pattern_log_ratio_restores_gradation_and_margin(self):
        turns = smoke_pattern_turns()
        legacy = measure(turns, calib_window=1)
        patched = measure(
            turns,
            calib_window=3,
            micro_health_mode="log_ratio",
            baseline_stat="median",
        )
        deltas = [row["delta_instant"] for row in patched["turns"]]
        # Gradation: the strict ordering of amplitudes survives into delta.
        self.assertGreater(deltas[1], deltas[0])
        self.assertGreater(deltas[0], deltas[2])
        self.assertLess(max(deltas), 1.0)
        # The trajectory is no longer driven to its worst-possible margin.
        self.assertGreater(patched["final_viability"], legacy["final_viability"])
        self.assertEqual(patched["micro_health_mode"], "log_ratio")
        self.assertEqual(patched["baseline_stat"], "median")


class BaselineGuardTests(unittest.TestCase):
    def test_median_baseline(self):
        turns = smoke_pattern_turns()
        info = baseline_from_turns(turns, calib_window=3, baseline_stat="median")
        self.assertAlmostEqual(info["baseline_micro"], 0.819, places=6)
        self.assertEqual(info["baseline_contributing_turns"], 3)

    def test_min_calib_tokens_excludes_short_opening(self):
        turns = smoke_pattern_turns()  # turn 0 has 4 tokens, turns 1-2 have 8
        info = baseline_from_turns(turns, calib_window=3, min_calib_tokens=8)
        self.assertEqual(info["baseline_excluded_short_turns"], 1)
        self.assertEqual(info["baseline_contributing_turns"], 2)
        self.assertAlmostEqual(info["baseline_micro"], (1.486 + 0.819) / 2.0, places=6)
        self.assertIn("excluded", info["baseline_warning"])

    def test_single_contributor_baseline_warns(self):
        turns = smoke_pattern_turns()
        info = baseline_from_turns(turns, calib_window=1)
        self.assertEqual(info["baseline_contributing_turns"], 1)
        self.assertIsNotNone(info["baseline_warning"])
        self.assertIn("near-deterministic", info["baseline_warning"])

    def test_unknown_baseline_stat_rejected(self):
        with self.assertRaises(ValueError):
            baseline_from_turns(smoke_pattern_turns(), calib_window=3, baseline_stat="mode")


class DegenerateWindowTests(unittest.TestCase):
    def test_short_trajectory_windows_are_degenerate(self):
        turns = smoke_pattern_turns()  # 3 turns vs windows [3, 5, 7]
        payload = RUNNER.calibration_sensitivity_payload(turns, [3, 5, 7])
        self.assertEqual(payload["effective_windows"], [3, 3, 3])
        self.assertTrue(payload["windows_degenerate"])
        self.assertEqual(payload["crossing_stability"], "NOT_EVALUABLE_DEGENERATE_WINDOWS")

    def test_long_trajectory_windows_are_not_degenerate(self):
        turns = [
            synthetic_turn(i, 0.6 + 0.05 * i, mean_surprise=3.0 + 0.1 * i)
            for i in range(10)
        ]
        payload = RUNNER.calibration_sensitivity_payload(turns, [3, 5, 7])
        self.assertEqual(payload["effective_windows"], [3, 5, 7])
        self.assertFalse(payload["windows_degenerate"])
        self.assertIn(
            payload["crossing_stability"],
            {"STABLE_CROSSED", "STABLE_NOT_CROSSED", "SENSITIVE"},
        )

    def test_degenerate_report_carries_warning(self):
        turns = smoke_pattern_turns()
        payload = RUNNER.calibration_sensitivity_payload(turns, [3, 5, 7])
        report = RUNNER.calibration_sensitivity_report(payload, Path("raw.json"))
        self.assertIn("NOT evidence of robustness", report)
        self.assertIn("NOT_EVALUABLE_DEGENERATE_WINDOWS", report)


if __name__ == "__main__":
    unittest.main()
