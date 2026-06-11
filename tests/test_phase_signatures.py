import unittest

from aptadynamik.observer.phase_signatures import (
    critical_slowing_score,
    discontinuity_score,
    prama_phase_diagnostics,
    recovery_latency,
    rolling_lag1_autocorrelation,
    rolling_variance,
    transition_warning,
)


class PhaseSignatureObserverTests(unittest.TestCase):
    def test_discontinuity_detects_sharp_jump(self):
        smooth = [0.1, 0.12, 0.13, 0.14, 0.15]
        jump = [0.1, 0.12, 0.13, 1.2, 1.22]
        self.assertGreater(discontinuity_score(jump), discontinuity_score(smooth))
        self.assertGreater(discontinuity_score(jump), 2.0)

    def test_rolling_variance_returns_complete_windows(self):
        values = [1.0, 1.0, 1.0, 2.0, 3.0]
        out = rolling_variance(values, window=3)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], 0.0)
        self.assertGreater(out[-1], out[0])

    def test_rolling_lag1_autocorrelation_returns_values(self):
        values = [0.0, 0.1, 0.2, 0.3, 0.4]
        out = rolling_lag1_autocorrelation(values, window=4)
        self.assertEqual(len(out), 2)
        for value in out:
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)

    def test_critical_slowing_score_positive_for_noisy_tail(self):
        values = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 0.1, 0.7, 0.0]
        self.assertGreater(critical_slowing_score(values, window=5), 0.0)

    def test_recovery_latency_after_local_crossing(self):
        rows = [
            {"instant_viability_margin": 0.1, "instant_threshold_crossed": False},
            {"instant_viability_margin": -0.2, "instant_threshold_crossed": True},
            {"instant_viability_margin": -0.1, "instant_threshold_crossed": True},
            {"instant_viability_margin": 0.05, "instant_threshold_crossed": False},
        ]
        self.assertEqual(recovery_latency(rows), 2)

    def test_prama_phase_diagnostics_consumes_prama_output_fields(self):
        rows = []
        for i in range(8):
            rows.append(
                {
                    "delta_instant": 0.1 + 0.02 * i,
                    "xi_norm": 0.2 + 0.01 * i,
                    "viability_margin": 0.3 - 0.03 * i,
                    "instant_viability_margin": 0.2 if i < 4 else (-0.1 if i < 6 else 0.05),
                    "instant_threshold_crossed": 4 <= i < 6,
                    "rigidity": 0.5,
                    "entropy_norm": 0.2 + 0.01 * i,
                    "uncertainty": 0.1 + 0.005 * i,
                }
            )
        payload = prama_phase_diagnostics(rows, window=4)
        self.assertEqual(payload["status"], "experimental_observer_layer")
        self.assertEqual(payload["recovery_latency"], 2)
        for field in [
            "delta_instant",
            "xi_norm",
            "viability_margin",
            "instant_viability_margin",
            "rigidity",
            "entropy_norm",
            "uncertainty",
        ]:
            self.assertIn(field, payload["signals"])
            self.assertIn("rolling_variance", payload["signals"][field])
            self.assertIn("rolling_lag1_autocorrelation", payload["signals"][field])
        self.assertIn("hysteresis", payload["out_of_scope"])
        self.assertIn("structural_target", payload["out_of_scope"])
        self.assertIn("semantic_judges", payload["out_of_scope"])

    def test_transition_warning_is_diagnostic_only(self):
        payload = transition_warning([0.1, 0.1, 0.1, 1.0, 1.0], window=3)
        self.assertIn("transition_warning", payload)
        self.assertIn("not hysteresis", payload["note"])


if __name__ == "__main__":
    unittest.main()
