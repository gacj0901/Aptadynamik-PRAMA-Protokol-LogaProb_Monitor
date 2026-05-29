import csv
import glob
import math
from collections import defaultdict
from pathlib import Path

FILES = sorted(glob.glob("results/v2_summary_*.csv"))
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

GEOMETRY_TEST_ORDER = [
    "G1_entropy_std_saturation_gt_canonical",
    "G2_entropy_range_saturation_gt_canonical",
    "G3_structural_entropy_std_gt_semantic",
    "G4_structural_entropy_range_gt_semantic",
    "G5_canonical_rigidity_highest",
]

DYNAMICS_TEST_ORDER = [
    "D1_xi_contradictory_gt_canonical",
    "D2_xi_saturation_gt_canonical",
    "D3_xi_structural_gt_semantic",
    "D4_lambda_contradictory_lt_fictional",
]


def f(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stdev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def load_file(path):
    by_family = defaultdict(list)

    with open(path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            family = row.get("family", "")
            if family in FAMILIES:
                by_family[family].append(row)

    summary = {}
    for family in FAMILIES:
        rows = by_family[family]
        summary[family] = {}
        for metric in METRICS:
            summary[family][metric] = mean([f(row.get(metric, "")) for row in rows])

    return summary


def has_post_patch_entropy_metrics(summary):
    metrics = ("avg_entropy_std", "max_entropy_std", "avg_entropy_range")
    return any(summary[family][metric] != 0.0 for family in FAMILIES for metric in metrics)


def geometry_tests(summary):
    semantic_entropy_std = summary["fictional"]["avg_entropy_std"]
    semantic_entropy_range = summary["fictional"]["avg_entropy_range"]
    structural_entropy_std = (
        summary["contradictory"]["avg_entropy_std"] + summary["saturation"]["avg_entropy_std"]
    ) / 2.0
    structural_entropy_range = (
        summary["contradictory"]["avg_entropy_range"] + summary["saturation"]["avg_entropy_range"]
    ) / 2.0
    canonical_rigidity = summary["canonical"]["avg_rigidity"]

    return {
        "G1_entropy_std_saturation_gt_canonical": (
            summary["saturation"]["avg_entropy_std"] > summary["canonical"]["avg_entropy_std"]
        ),
        "G2_entropy_range_saturation_gt_canonical": (
            summary["saturation"]["avg_entropy_range"] > summary["canonical"]["avg_entropy_range"]
        ),
        "G3_structural_entropy_std_gt_semantic": structural_entropy_std > semantic_entropy_std,
        "G4_structural_entropy_range_gt_semantic": structural_entropy_range > semantic_entropy_range,
        "G5_canonical_rigidity_highest": all(
            canonical_rigidity > summary[family]["avg_rigidity"]
            for family in ("fictional", "contradictory", "saturation")
        ),
    }


def dynamics_tests(summary):
    xi_canonical = summary["canonical"]["xi_per_window"]
    xi_fictional = summary["fictional"]["xi_per_window"]
    xi_contradictory = summary["contradictory"]["xi_per_window"]
    xi_saturation = summary["saturation"]["xi_per_window"]

    return {
        "D1_xi_contradictory_gt_canonical": xi_contradictory > xi_canonical,
        "D2_xi_saturation_gt_canonical": xi_saturation > xi_canonical,
        "D3_xi_structural_gt_semantic": max(xi_contradictory, xi_saturation) > xi_fictional,
        "D4_lambda_contradictory_lt_fictional": (
            summary["contradictory"]["final_lambda"] < summary["fictional"]["final_lambda"]
        ),
    }


def print_summary_table(summary):
    print(
        f"{'family':<16}"
        f"{'rigid':>10}"
        f"{'uncert':>10}"
        f"{'margin':>10}"
        f"{'ent_std':>10}"
        f"{'max_std':>10}"
        f"{'ent_rng':>10}"
        f"{'xi/w':>10}"
        f"{'lambda':>10}"
    )

    for family in FAMILIES:
        s = summary[family]
        print(
            f"{family:<16}"
            f"{s['avg_rigidity']:>10.4f}"
            f"{s['avg_uncertainty']:>10.4f}"
            f"{s['avg_margin']:>10.4f}"
            f"{s['avg_entropy_std']:>10.4f}"
            f"{s['max_entropy_std']:>10.4f}"
            f"{s['avg_entropy_range']:>10.4f}"
            f"{s['xi_per_window']:>10.4f}"
            f"{s['final_lambda']:>10.4f}"
        )


def main():
    if not FILES:
        print("No se encontraron archivos results/v2_summary_*.csv")
        return 0

    runs = []
    skipped = []

    for path in FILES:
        summary = load_file(path)
        if not has_post_patch_entropy_metrics(summary):
            skipped.append(path)
            continue
        runs.append((path, summary, geometry_tests(summary), dynamics_tests(summary)))

    print("=" * 96)
    print("PRAMA v2 replication analysis")
    print("=" * 96)
    print(f"Files found: {len(FILES)}")
    print(f"Files analyzed: {len(runs)}")
    print(f"Files skipped as pre-patch: {len(skipped)}")
    print()

    if not runs:
        print("No post-patch v2 summaries available for aggregate analysis.")
        return 0

    for path, summary, g_tests, d_tests in runs:
        g_passed = sum(g_tests.values())
        d_passed = sum(d_tests.values())

        print(Path(path).name)
        print("-" * 96)
        print_summary_table(summary)

        print("\n  PRIMARY GEOMETRY TESTS")
        for name in GEOMETRY_TEST_ORDER:
            print(f"  {name:<50} {'PASS' if g_tests[name] else 'FAIL'}")

        print("\n  SECONDARY PRAMA DYNAMICS TESTS")
        for name in DYNAMICS_TEST_ORDER:
            print(f"  {name:<50} {'PASS' if d_tests[name] else 'FAIL'}")

        print(f"\n  GEOMETRY RESULT: {g_passed}/5")
        print(f"  PRAMA DYNAMICS RESULT: {d_passed}/4")
        print()

    agg = {family: {metric: [] for metric in METRICS} for family in FAMILIES}
    geometry_counts = defaultdict(int)
    dynamics_counts = defaultdict(int)

    for _, summary, g_tests, d_tests in runs:
        for family in FAMILIES:
            for metric in METRICS:
                agg[family][metric].append(summary[family][metric])

        for name, passed in g_tests.items():
            if passed:
                geometry_counts[name] += 1
        for name, passed in d_tests.items():
            if passed:
                dynamics_counts[name] += 1

    print("=" * 96)
    print("Aggregate means across post-patch runs")
    print("=" * 96)
    aggregate_summary = {
        family: {metric: mean(agg[family][metric]) for metric in METRICS}
        for family in FAMILIES
    }
    print_summary_table(aggregate_summary)

    print()
    print("=" * 96)
    print("Geometry test stability")
    print("=" * 96)
    for name in GEOMETRY_TEST_ORDER:
        print(f"{name:<50} {geometry_counts[name]}/{len(runs)} runs")

    print()
    print("=" * 96)
    print("PRAMA dynamics test stability")
    print("=" * 96)
    for name in DYNAMICS_TEST_ORDER:
        print(f"{name:<50} {dynamics_counts[name]}/{len(runs)} runs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
