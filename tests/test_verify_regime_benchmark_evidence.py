import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORTER_PATH = ROOT / "scripts" / "export_regime_benchmark_evidence.py"
EXPORTER_SPEC = importlib.util.spec_from_file_location("export_regime_benchmark_evidence", EXPORTER_PATH)
assert EXPORTER_SPEC is not None and EXPORTER_SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(EXPORTER_SPEC)
EXPORTER_SPEC.loader.exec_module(EXPORTER)

VERIFIER_PATH = ROOT / "scripts" / "verify_regime_benchmark_evidence.py"
VERIFIER_SPEC = importlib.util.spec_from_file_location("verify_regime_benchmark_evidence", VERIFIER_PATH)
assert VERIFIER_SPEC is not None and VERIFIER_SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(VERIFIER_SPEC)
VERIFIER_SPEC.loader.exec_module(VERIFIER)


class TestVerifyRegimeBenchmarkEvidence(unittest.TestCase):
    def test_verifier_accepts_current_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = EXPORTER.export_evidence(evidence_dir=Path(tmp) / "evidence", run_label="verify-current")
            verification = VERIFIER.verify_bundle(Path(result["bundle_dir"]))

            self.assertTrue(verification["passed"])
            self.assertEqual(verification["scenario_count"], 4)
            self.assertEqual(verification["errors"], [])

    def test_verifier_accepts_legacy_organized_equilibrium_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            export = EXPORTER.export_evidence(evidence_dir=Path(tmp) / "evidence", run_label="verify-legacy")
            bundle = Path(export["bundle_dir"])
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            for item in manifest["scenarios"]:
                if item["scenario_name"] != "organized_viability":
                    continue
                summary_path = Path(item["summary_path"])
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["expected_regime_label"] = "II_ORGANIZED_EQUILIBRIUM"
                summary["regime_label"] = "II_ORGANIZED_EQUILIBRIUM"
                summary["expected_trajectory_assessment"] = "VIABLE_ORGANIZED_EQUILIBRIUM"
                summary["trajectory_assessment"] = "VIABLE_ORGANIZED_EQUILIBRIUM"
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                item["expected_regime_label"] = "II_ORGANIZED_EQUILIBRIUM"
                item["observed_regime_label"] = "II_ORGANIZED_EQUILIBRIUM"
                item["expected_trajectory_assessment"] = "VIABLE_ORGANIZED_EQUILIBRIUM"
                item["observed_trajectory_assessment"] = "VIABLE_ORGANIZED_EQUILIBRIUM"
                item["result_hash"] = VERIFIER.benchmark.stable_summary_hash(summary)

            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            verification = VERIFIER.verify_bundle(bundle)

            self.assertTrue(verification["passed"])
            self.assertEqual(verification["errors"], [])


if __name__ == "__main__":
    unittest.main()
