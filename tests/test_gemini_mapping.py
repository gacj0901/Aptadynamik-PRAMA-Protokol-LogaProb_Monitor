import unittest
from types import SimpleNamespace

from aptadynamik.pipelines.gemini import extract_signals, make_config, run_through_prama, window_aggregate


class GeminiMappingTests(unittest.TestCase):
    def test_logprob_signals_feed_prama_core(self):
        chosen = [
            SimpleNamespace(token="a", log_probability=-0.1),
            SimpleNamespace(token="b", log_probability=-0.3),
            SimpleNamespace(token="c", log_probability=-0.2),
            SimpleNamespace(token="d", log_probability=-0.4),
        ]
        top_candidates = [
            SimpleNamespace(candidates=[SimpleNamespace(log_probability=-0.1), SimpleNamespace(log_probability=-0.5)]),
            SimpleNamespace(candidates=[SimpleNamespace(log_probability=-0.3), SimpleNamespace(log_probability=-0.6)]),
            SimpleNamespace(candidates=[SimpleNamespace(log_probability=-0.2), SimpleNamespace(log_probability=-0.7)]),
            SimpleNamespace(candidates=[SimpleNamespace(log_probability=-0.4), SimpleNamespace(log_probability=-0.8)]),
        ]
        logprobs = SimpleNamespace(chosen_candidates=chosen, top_candidates=top_candidates)

        signals = extract_signals(logprobs)
        windows = window_aggregate(signals, window_size=2)
        result = run_through_prama("demo", "prompt", "abcd", logprobs, make_config())

        self.assertEqual(len(signals), 4)
        self.assertEqual(len(windows), 2)
        self.assertEqual(result["n_tokens"], 4)
        self.assertIsNotNone(result["final_integrity"])


if __name__ == "__main__":
    unittest.main()
