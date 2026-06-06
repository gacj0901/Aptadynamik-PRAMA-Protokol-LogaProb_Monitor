from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_regime_benchmark as benchmark  # noqa: E402


DEFAULT_EVIDENCE_DIR = Path(
    r"C:\Users\THINKPAD\Desktop\Documentación PRAMA Protokol ProbLogs Mónitor"
)
EVIDENCE_WARNING = "Generated evidence artifacts are intentionally kept outside the source repository."


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def sanitize_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "run"


def bundle_name(run_label: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%Mh%M%S")
    if run_label:
        return f"prama_regime_benchmark_{timestamp}_{sanitize_label(run_label)}"
    return f"prama_regime_benchmark_{timestamp}"


def copy_json(source: Path, target: Path, payload: Dict[str, Any] | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if payload is None:
        shutil.copy2(source, target)
    else:
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def scenario_index_text(manifest: Dict[str, Any], include_raw: bool) -> str:
    lines = [
        "# PRAMA Regime Benchmark Evidence Index",
        "",
        f"- generated_at: `{manifest.get('generated_at')}`",
        f"- git_commit_sha: `{manifest.get('git_commit_sha')}`",
        f"- git_branch: `{manifest.get('git_branch')}`",
        f"- benchmark_version: `{manifest.get('benchmark_version')}`",
        f"- output folder: `{manifest.get('output_dir')}`",
        "",
        "| scenario | observed_regime | observed_assessment | passed | result_hash | report path | summary path | raw path |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in manifest["scenarios"]:
        raw_path = item.get("raw_path") if include_raw else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("scenario_name")),
                    str(item.get("observed_regime_label")),
                    str(item.get("observed_trajectory_assessment")),
                    str(item.get("passed")),
                    str(item.get("result_hash")),
                    str(item.get("report_path")),
                    str(item.get("summary_path")),
                    str(raw_path),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def readme_text(manifest: Dict[str, Any], include_raw: bool) -> str:
    raw_line = "- scenario raw files, when included" if include_raw else "- scenario raw files omitted by request"
    return "\n".join(
        [
            "# PRAMA Regime Benchmark Evidence Bundle",
            "",
            "## Purpose",
            "",
            "This bundle contains externally exported deterministic evidence artifacts for the PRAMA regime benchmark.",
            "",
            "## Included Artifacts",
            "",
            "- manifest.json",
            "- aggregate_report.md",
            "- scenario_index.md",
            "- per-scenario summary.json files",
            "- per-scenario report.md files",
            raw_line,
            "",
            "## Methodological Note",
            "",
            benchmark.METHODOLOGICAL_NOTE,
            "",
            "## Repository Boundary",
            "",
            EVIDENCE_WARNING,
            "",
            f"Scenario count: `{manifest.get('scenario_count')}`",
            "",
        ]
    )


def rewrite_manifest_for_bundle(
    source_manifest: Dict[str, Any],
    bundle_dir: Path,
    include_raw: bool,
) -> Dict[str, Any]:
    manifest = dict(source_manifest)
    manifest["output_dir"] = str(bundle_dir)
    manifest["aggregate_report_path"] = str(bundle_dir / "aggregate_report.md")
    scenarios = []
    for item in source_manifest["scenarios"]:
        scenario_name = item["scenario_name"]
        scenario_dir = bundle_dir / "scenarios" / scenario_name
        updated = dict(item)
        updated["summary_path"] = str(scenario_dir / "summary.json")
        updated["report_path"] = str(scenario_dir / "report.md")
        updated["raw_path"] = str(scenario_dir / "raw.json") if include_raw else None
        scenarios.append(updated)
    manifest["scenarios"] = scenarios
    return manifest


def export_evidence(
    evidence_dir: Path | None = None,
    run_label: str | None = None,
    include_raw: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    root = Path(evidence_dir) if evidence_dir is not None else DEFAULT_EVIDENCE_DIR
    root.mkdir(parents=True, exist_ok=True)
    bundle_dir = root / bundle_name(run_label)
    if bundle_dir.exists() and not overwrite:
        raise FileExistsError(f"Evidence bundle already exists: {bundle_dir}")
    if bundle_dir.exists() and overwrite:
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp:
        benchmark_dir = Path(tmp) / "benchmark"
        benchmark_result = benchmark.run_benchmark(benchmark_dir)
        source_manifest = json.loads((benchmark_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest = rewrite_manifest_for_bundle(source_manifest, bundle_dir, include_raw)

        shutil.copy2(benchmark_dir / "aggregate_report.md", bundle_dir / "aggregate_report.md")
        copy_json(benchmark_dir / "manifest.json", bundle_dir / "manifest.json", manifest)
        (bundle_dir / "scenario_index.md").write_text(scenario_index_text(manifest, include_raw), encoding="utf-8")
        (bundle_dir / "README.md").write_text(readme_text(manifest, include_raw), encoding="utf-8")

        for item in source_manifest["scenarios"]:
            scenario_name = item["scenario_name"]
            source_dir = benchmark_dir / scenario_name
            target_dir = bundle_dir / "scenarios" / scenario_name
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_dir / "summary.json", target_dir / "summary.json")
            shutil.copy2(source_dir / "report.md", target_dir / "report.md")
            if include_raw:
                shutil.copy2(source_dir / "raw.json", target_dir / "raw.json")

    passed_count = sum(1 for item in manifest["scenarios"] if item.get("passed"))
    return {
        "bundle_dir": str(bundle_dir.resolve()),
        "scenario_count": manifest["scenario_count"],
        "passed_count": passed_count,
        "aggregate_report_path": str((bundle_dir / "aggregate_report.md").resolve()),
        "manifest_path": str((bundle_dir / "manifest.json").resolve()),
        "include_raw": include_raw,
        "benchmark_passed": benchmark_result["passed"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PRAMA regime benchmark evidence outside the source repo.")
    parser.add_argument("--evidence-dir", type=Path, default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--include-raw", type=parse_bool, default=True)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_evidence(
        evidence_dir=args.evidence_dir,
        run_label=args.run_label,
        include_raw=args.include_raw,
        overwrite=args.overwrite,
    )
    print(f"Evidence bundle: {result['bundle_dir']}")
    print(f"Scenario count: {result['scenario_count']}")
    print(f"Passed scenarios: {result['passed_count']}")
    print(f"Aggregate report: {result['aggregate_report_path']}")
    print(f"Manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
