import unittest
from pathlib import Path

from aptadynamik.verification.scenarios import evaluate, run_all, write_results


class OfflineVerificationTests(unittest.TestCase):
    def test_offline_verification_passes(self):
        results = run_all()
        checks = evaluate(results)

        self.assertEqual(len(results), 4)
        self.assertTrue(all(passed for _, passed, *_ in checks))

    def test_results_are_written(self):
        output_dir = Path("results") / "test-output"
        output = write_results(run_all(), output_dir)
        self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
