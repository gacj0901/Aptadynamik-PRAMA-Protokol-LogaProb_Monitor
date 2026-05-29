import argparse
import csv
from collections import defaultdict
from pathlib import Path

from aptadynamik.observer.qualitative_signatures import (
    evaluate_series,
    load_sessions_from_path,
    rows_from_sessions,
    synthetic_validation,
)


def result_rows(label, model, results):
    return [
        {
            "source": label,
            "model": model,
            **result.as_dict(),
        }
        for result in results
    ]


def write_phase_signatures(path, rows):
    fields = ["source", "model", "signature", "triggered", "score", "threshold", "detail"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_comparative_summary(path, rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["signature"])].append(row)

    fields = ["model", "signature", "triggered_count", "total", "mean_score"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (model, signature), items in sorted(grouped.items()):
            writer.writerow(
                {
                    "model": model,
                    "signature": signature,
                    "triggered_count": sum(1 for item in items if str(item["triggered"]) == "True" or item["triggered"] is True),
                    "total": len(items),
                    "mean_score": round(sum(float(item["score"]) for item in items) / len(items), 6),
                }
            )


def write_report(path, mode, rows):
    lines = [
        "# PRAMA Phase Signature Report",
        "",
        "## Scope",
        "",
        "Synthetic validation is instrument validation, not model evidence.",
        "Model evidence begins only when drive_system is replaced by real PRAMA sessions.",
        "",
        f"- Mode: `{mode}`",
        f"- Evaluated signatures: `{len(rows)}`",
        "",
        "## Signature Results",
        "",
        "| source | model | signature | triggered | score | threshold |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['source']} | {row['model']} | {row['signature']} | "
            f"{row['triggered']} | {row['score']} | {row['threshold']} |"
        )
    lines.extend(
        [
            "",
            "## Methodological Note",
            "",
            "The initial observed-session viability proxy is `avg_rigidity - avg_uncertainty`.",
            "Synthetic fold/smooth systems validate whether the signature detectors behave as expected; they do not constitute evidence about any language model.",
            "Real-session evidence is produced only from PRAMA Monitor `session_*_raw.json` files.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_synthetic(output_dir):
    validation = synthetic_validation()
    rows = []
    for label, results in validation.items():
        rows.extend(result_rows(label, f"synthetic_{label}", results))
    write_outputs(output_dir, "synthetic", rows)
    return rows


def run_from_results(input_path, output_dir):
    sessions = load_sessions_from_path(input_path)
    rows = []
    for session in sessions:
        model = session.get("model", "unknown")
        session_id = session.get("session_id", "unknown")
        series = rows_from_sessions([session])
        if not series:
            continue
        rows.extend(result_rows(session_id, model, evaluate_series(series)))
    write_outputs(output_dir, "real-session", rows)
    return rows


def write_outputs(output_dir, mode, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_phase_signatures(output_dir / "phase_signatures.csv", rows)
    write_report(output_dir / "phase_report.md", mode, rows)
    models = {row["model"] for row in rows}
    if len(models) > 1:
        write_comparative_summary(output_dir / "comparative_phase_summary.csv", rows)


def main():
    parser = argparse.ArgumentParser(description="Run PRAMA qualitative phase-transition signatures.")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic fold/smooth validation.")
    parser.add_argument("--from-results", help="Analyze a directory or raw.json file from PRAMA Monitor.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to synthetic or input directory.")
    args = parser.parse_args()

    if args.synthetic == bool(args.from_results):
        parser.error("Choose exactly one mode: --synthetic or --from-results PATH")

    if args.synthetic:
        output_dir = Path(args.output_dir or "results/phase_synthetic")
        rows = run_synthetic(output_dir)
    else:
        input_path = Path(args.from_results)
        output_dir = Path(args.output_dir) if args.output_dir else (input_path if input_path.is_dir() else input_path.parent)
        rows = run_from_results(input_path, output_dir)

    print(f"Wrote {len(rows)} signature rows to {output_dir}")
    print(f"-> {output_dir / 'phase_signatures.csv'}")
    print(f"-> {output_dir / 'phase_report.md'}")
    if (output_dir / "comparative_phase_summary.csv").exists():
        print(f"-> {output_dir / 'comparative_phase_summary.csv'}")


if __name__ == "__main__":
    raise SystemExit(main())
