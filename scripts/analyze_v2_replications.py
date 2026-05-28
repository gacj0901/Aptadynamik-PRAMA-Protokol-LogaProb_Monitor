import csv
import glob
import math
from collections import defaultdict
from pathlib import Path

FILES = sorted(glob.glob("results/v2_summary_*.csv"))

if not FILES:
    raise SystemExit("No se encontraron archivos results/v2_summary_*.csv")

FAMILIES = ["canonical", "fictional", "contradictory", "saturation"]

METRICS = [
    "avg_rigidity",
    "avg_uncertainty",
    "avg_margin",
    "xi_per_window",
    "final_integrity",
    "final_lambda",
    "avg_entropy",
    "avg_entropy_std",
    "max_entropy_std",
    "avg_entropy_range",
]

def f(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def load_file(path):
    by_family = defaultdict(list)

    with open(path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            fam = row.get("family", "")
            if fam in FAMILIES:
                by_family[fam].append(row)

    summary = {}

    for fam in FAMILIES:
        rows = by_family[fam]
        summary[fam] = {}

        for metric in METRICS:
            vals = [f(r.get(metric, "")) for r in rows]
            summary[fam][metric] = mean(vals)

    return summary

def test_run(summary):
    xi_c = summary["canonical"]["xi_per_window"]
    xi_f = summary["fictional"]["xi_per_window"]
    xi_k = summary["contradictory"]["xi_per_window"]
    xi_s = summary["saturation"]["xi_per_window"]

    margin_f = summary["fictional"]["avg_margin"]
    margin_k = summary["contradictory"]["avg_margin"]

    estd_c = summary["canonical"]["avg_entropy_std"]
    estd_s = summary["saturation"]["avg_entropy_std"]

    erange_c = summary["canonical"]["avg_entropy_range"]
    erange_s = summary["saturation"]["avg_entropy_range"]

    tests = {
        "T1_xi_saturation_gt_canonical": xi_s > xi_c,
        "T2_xi_structural_gt_semantic": max(xi_k, xi_s) > xi_f,
        "T3_entropy_std_saturation_gt_canonical": estd_s > estd_c,
        "T4_margin_contradictory_lt_fictional": margin_k < margin_f,
        "V1_entropy_range_saturation_gt_canonical": erange_s > erange_c,
    }

    return tests

runs = []

for path in FILES:
    summary = load_file(path)
    tests = test_run(summary)
    runs.append((path, summary, tests))

print("=" * 88)
print("PRAMA v2 replication analysis")
print("=" * 88)
print(f"Files analyzed: {len(runs)}")
print()

for path, summary, tests in runs:
    passed = sum(1 for v in tests.values() if v)

    print(Path(path).name)
    print("-" * 88)

    print(
        f"{'family':<16}"
        f"{'xi/w':>10}"
        f"{'margin':>10}"
        f"{'ent_std':>10}"
        f"{'max_std':>10}"
        f"{'ent_rng':>10}"
        f"{'rigid':>10}"
        f"{'lambda':>10}"
    )

    for fam in FAMILIES:
        s = summary[fam]
        print(
            f"{fam:<16}"
            f"{s['xi_per_window']:>10.4f}"
            f"{s['avg_margin']:>10.4f}"
            f"{s['avg_entropy_std']:>10.4f}"
            f"{s['max_entropy_std']:>10.4f}"
            f"{s['avg_entropy_range']:>10.4f}"
            f"{s['avg_rigidity']:>10.4f}"
            f"{s['final_lambda']:>10.4f}"
        )

    print()
    for name, value in tests.items():
        print(f"  {name:<46} {'PASS' if value else 'FAIL'}")

    print(f"  RESULT: {passed}/{len(tests)}")
    print()

# Aggregate across all runs
agg = {fam: {m: [] for m in METRICS} for fam in FAMILIES}
test_counts = defaultdict(int)

for _, summary, tests in runs:
    for fam in FAMILIES:
        for metric in METRICS:
            agg[fam][metric].append(summary[fam][metric])

    for name, value in tests.items():
        if value:
            test_counts[name] += 1

print("=" * 88)
print("Aggregate means across runs")
print("=" * 88)

print(
    f"{'family':<16}"
    f"{'xi/w':>10}"
    f"{'margin':>10}"
    f"{'ent_std':>10}"
    f"{'max_std':>10}"
    f"{'ent_rng':>10}"
    f"{'rigid':>10}"
    f"{'lambda':>10}"
)

for fam in FAMILIES:
    print(
        f"{fam:<16}"
        f"{mean(agg[fam]['xi_per_window']):>10.4f}"
        f"{mean(agg[fam]['avg_margin']):>10.4f}"
        f"{mean(agg[fam]['avg_entropy_std']):>10.4f}"
        f"{mean(agg[fam]['max_entropy_std']):>10.4f}"
        f"{mean(agg[fam]['avg_entropy_range']):>10.4f}"
        f"{mean(agg[fam]['avg_rigidity']):>10.4f}"
        f"{mean(agg[fam]['final_lambda']):>10.4f}"
    )

print()
print("=" * 88)
print("Test stability")
print("=" * 88)

for name in [
    "T1_xi_saturation_gt_canonical",
    "T2_xi_structural_gt_semantic",
    "T3_entropy_std_saturation_gt_canonical",
    "T4_margin_contradictory_lt_fictional",
    "V1_entropy_range_saturation_gt_canonical",
]:
    print(f"{name:<46} {test_counts[name]}/{len(runs)} runs")

print()
print("Interpretation rule:")
print("- 5/5 or 4/5 stable: strong preliminary signal")
print("- 3/5 unstable but present: needs more prompts or stricter definitions")
print("- 0/5 to 2/5 weak: definition probably wrong")