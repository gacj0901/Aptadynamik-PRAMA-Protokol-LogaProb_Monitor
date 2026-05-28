import csv
import glob
from pathlib import Path

RESULTS = Path("results")
files = sorted(glob.glob("results/v2_detail_*.csv"))

if not files:
    raise SystemExit("No se encontraron archivos results/v2_detail_*.csv")

latest = files[-1]

LEFT = "canonical_1"
RIGHT = "contradictory_1"

FIELDS = [
    "window",
    "dynamic_in",
    "symbolic_in",
    "avg_gap",
    "avg_entropy",
    "gap_norm",
    "entropy_norm",
    "rigidity",
    "uncertainty",
    "margin",
    "delta",
    "xi",
    "lambda",
    "theta_eff",
    "integrity",
    "anomaly",
]

def to_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def load(label):
    rows = []
    with open(latest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] == label:
                rows.append(row)
    return rows

def summarize(rows):
    summary = {}

    for key in FIELDS:
        if key == "window":
            continue

        vals = [to_float(r[key]) for r in rows if key in r and r[key] != ""]

        if vals:
            summary[key] = {
                "first": vals[0],
                "last": vals[-1],
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
                "range": max(vals) - min(vals),
            }

    return summary

left = load(LEFT)
right = load(RIGHT)

print(f"Using: {latest}")
print(f"{LEFT}: {len(left)} windows")
print(f"{RIGHT}: {len(right)} windows")

if not left:
    raise SystemExit(f"No se encontraron ventanas para {LEFT}")

if not right:
    raise SystemExit(f"No se encontraron ventanas para {RIGHT}")

left_summary = summarize(left)
right_summary = summarize(right)

print()
print("Metric comparison")
print("-" * 92)
print(
    f"{'metric':<18}"
    f"{'canon_mean':>12}"
    f"{'contr_mean':>12}"
    f"{'diff':>12}"
    f"{'canon_range':>14}"
    f"{'contr_range':>14}"
)
print("-" * 92)

for key in [
    "avg_gap",
    "avg_entropy",
    "rigidity",
    "uncertainty",
    "margin",
    "delta",
    "xi",
    "lambda",
    "integrity",
    "anomaly",
]:
    if key in left_summary and key in right_summary:
        diff = right_summary[key]["mean"] - left_summary[key]["mean"]
        print(
            f"{key:<18}"
            f"{left_summary[key]['mean']:>12.4f}"
            f"{right_summary[key]['mean']:>12.4f}"
            f"{diff:>12.4f}"
            f"{left_summary[key]['range']:>14.4f}"
            f"{right_summary[key]['range']:>14.4f}"
        )

out_path = RESULTS / "trajectory_compare_canonical1_vs_contradictory1.csv"

max_len = max(len(left), len(right))

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    header = ["window"]

    for prefix in ["canonical", "contradictory"]:
        for field in FIELDS[1:]:
            header.append(f"{prefix}_{field}")

    writer.writerow(header)

    for i in range(max_len):
        row = [i]

        lrow = left[i] if i < len(left) else {}
        rrow = right[i] if i < len(right) else {}

        for field in FIELDS[1:]:
            row.append(lrow.get(field, ""))

        for field in FIELDS[1:]:
            row.append(rrow.get(field, ""))

        writer.writerow(row)

print()
print(f"Exported CSV: {out_path}")

try:
    import matplotlib.pyplot as plt

    metrics_to_plot = [
        "delta",
        "xi",
        "lambda",
        "integrity",
        "rigidity",
        "uncertainty",
        "margin",
    ]

    for metric in metrics_to_plot:
        plt.figure()

        plt.plot(
            [int(r["window"]) for r in left],
            [to_float(r[metric]) for r in left],
            label=LEFT,
        )

        plt.plot(
            [int(r["window"]) for r in right],
            [to_float(r[metric]) for r in right],
            label=RIGHT,
        )

        plt.xlabel("window")
        plt.ylabel(metric)
        plt.title(f"{metric}: {LEFT} vs {RIGHT}")
        plt.legend()

        path = RESULTS / f"plot_{metric}_{LEFT}_vs_{RIGHT}.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        print(f"Plot: {path}")

except ImportError:
    print()
    print("matplotlib no esta instalado. El CSV comparativo ya fue generado.")
    print("Para generar graficas:")
    print("python -m pip install matplotlib")