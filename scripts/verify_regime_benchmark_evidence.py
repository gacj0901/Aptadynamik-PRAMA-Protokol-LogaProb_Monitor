from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_regime_benchmark as benchmark  # noqa: E402
from aptadynamik.prama_protokol_core import (  # noqa: E402
    normalize_regime_label,
    normalize_trajectory_assessment,
)


REQUIRED_ROOT_FILES = [
    "manifest.json",
    "aggregate_report.md",
    "scenario_index.md",
    "README.md",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_match(expected: str | None, observed: str | None, kind: str) -> bool:
    if kind == "regime":
        return normalize_regime_label(expected) == normalize_regime_label(observed)
    if kind == "assessment":
        return normalize_trajectory_assessment(expected) == normalize_trajectory_assessment(observed)
    raise ValueError(f"Unknown comparison kind: {kind}")


def verify_bundle(bundle_dir: Path) -> Dict[str, Any]:
    errors: List[str] = []
    bundle = Path(bundle_dir)
    if not bundle.exists() or not bundle.is_dir():
        return {"bundle_dir": str(bundle), "passed": False, "errors": [f"Bundle directory not found: {bundle}"]}

    for name in REQUIRED_ROOT_FILES:
        if not (bundle / name).exists():
            errors.append(f"Missing root artifact: {name}")

    manifest_path = bundle / "manifest.json"
    if not manifest_path.exists():
        return {"bundle_dir": str(bundle), "passed": False, "errors": errors}

    manifest = load_json(manifest_path)
    scenarios = manifest.get("scenarios", [])
    if manifest.get("scenario_count") != len(scenarios):
        errors.append("scenario_count does not match manifest scenarios length")

    for item in scenarios:
        name = item.get("scenario_name")
        scenario_dir = bundle / "scenarios" / str(name)
        summary_path = Path(item.get("summary_path") or scenario_dir / "summary.json")
        report_path = Path(item.get("report_path") or scenario_dir / "report.md")
        raw_path_value = item.get("raw_path")
        raw_path = Path(raw_path_value) if raw_path_value else None

        if not summary_path.exists():
            errors.append(f"Missing summary for scenario {name}: {summary_path}")
            continue
        if not report_path.exists():
            errors.append(f"Missing report for scenario {name}: {report_path}")
        if raw_path is not None and not raw_path.exists():
            errors.append(f"Missing raw file for scenario {name}: {raw_path}")

        summary = load_json(summary_path)
        expected_regime = item.get("expected_regime_label")
        observed_regime = item.get("observed_regime_label")
        expected_assessment = item.get("expected_trajectory_assessment")
        observed_assessment = item.get("observed_trajectory_assessment")

        if not normalized_match(expected_regime, observed_regime, "regime"):
            errors.append(f"Regime mismatch for {name}: expected {expected_regime}, observed {observed_regime}")
        if not normalized_match(expected_assessment, observed_assessment, "assessment"):
            errors.append(
                f"Assessment mismatch for {name}: expected {expected_assessment}, observed {observed_assessment}"
            )
        if not bool(item.get("passed")):
            errors.append(f"Scenario did not pass according to manifest: {name}")
        expected_hash = item.get("result_hash")
        observed_hash = benchmark.stable_summary_hash(summary)
        if expected_hash != observed_hash:
            errors.append(f"Hash mismatch for {name}: expected {expected_hash}, observed {observed_hash}")

    return {
        "bundle_dir": str(bundle),
        "scenario_count": len(scenarios),
        "passed_scenarios": sum(1 for item in scenarios if item.get("passed")),
        "passed": not errors,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a PRAMA regime benchmark evidence bundle.")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_bundle(args.bundle_dir)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
