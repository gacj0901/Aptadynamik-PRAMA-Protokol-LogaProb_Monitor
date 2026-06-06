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


class TestExportRegimeBenchmarkEvidence(unittest.TestCase):
    def test_exporter_generates_evidence_bundle_with_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence_dir = Path(tmp) / "evidence"
            result = EXPORTER.export_evidence(evidence_dir=evidence_dir, run_label="unit-test", include_raw=True)
            bundle = Path(result["bundle_dir"])
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual({path.name for path in evidence_dir.iterdir()}, {bundle.name})
            self.assertTrue((bundle / "manifest.json").exists())
            self.assertTrue((bundle / "aggregate_report.md").exists())
            self.assertTrue((bundle / "scenario_index.md").exists())
            self.assertTrue((bundle / "README.md").exists())
            self.assertEqual(manifest["scenario_count"], 4)
            self.assertTrue(all(item["passed"] for item in manifest["scenarios"]))

            scenarios_dir = bundle / "scenarios"
            self.assertEqual(
                {path.name for path in scenarios_dir.iterdir()},
                {
                    "short_calibrating_local_crossings",
                    "organized_viability",
                    "structural_pulsation",
                    "entropic_collapse",
                },
            )
            for item in manifest["scenarios"]:
                scenario_dir = scenarios_dir / item["scenario_name"]
                self.assertTrue((scenario_dir / "summary.json").exists())
                self.assertTrue((scenario_dir / "report.md").exists())
                self.assertTrue((scenario_dir / "raw.json").exists())

    def test_exporter_can_omit_raw_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence_dir = Path(tmp) / "evidence"
            result = EXPORTER.export_evidence(evidence_dir=evidence_dir, include_raw=False)
            bundle = Path(result["bundle_dir"])
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual({path.name for path in evidence_dir.iterdir()}, {bundle.name})
            self.assertEqual(manifest["scenario_count"], 4)
            self.assertTrue(all(item["passed"] for item in manifest["scenarios"]))
            for item in manifest["scenarios"]:
                scenario_dir = bundle / "scenarios" / item["scenario_name"]
                self.assertTrue((scenario_dir / "summary.json").exists())
                self.assertTrue((scenario_dir / "report.md").exists())
                self.assertFalse((scenario_dir / "raw.json").exists())
                self.assertIsNone(item["raw_path"])


if __name__ == "__main__":
    unittest.main()
