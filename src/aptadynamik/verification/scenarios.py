"""
PRAMA Protokol - offline verification scenarios.

Four deterministic scenarios exercise the PRAMA core without API access.
"""
import json
from pathlib import Path

from aptadynamik.prama_core import CoreConfig, CoreState

STEPS = 80
SCENARIO_NAMES = ["A_varied", "B_monotonic", "C_dual", "D_partial"]


def run(name, input_fn):
    cfg = CoreConfig()
    cfg.regime_geometry.lambda_recovery = 0.10
    cfg.regime_geometry.lambda_gain = 0.03
    state = CoreState(cfg)
    series = []
    for t in range(STEPS):
        dyn, sym = input_fn(t)
        out = state.step(dyn, sym, cfg)
        series.append({
            "t": t,
            "delta": round(out["delta"], 4),
            "xi": round(out["xi"], 4),
            "lambda": round(out["lambda"], 4),
            "theta_eff": round(out["theta_eff"], 4),
            "lambda_sensitivity": round(out["lambda_sensitivity"], 4),
            "affectio": round(out["affectio"], 4),
            "anomaly": round(out["anomaly_index"], 4),
            "integrity": round(out["dominance"].integrity, 4),
            "mag_iota": round(out["dominance"].mag_iota, 4),
            "mag_kappa": round(out["dominance"].mag_kappa, 4),
            "mag_rho": round(out["dominance"].mag_rho, 4),
            "regime": out["regime"].name,
            "q_regime": out["quantized_regime"].name,
            "health": out["status"].health.name,
            "mode": out["dominance"].mode.name,
            "rotations": out["status"].rotation_count,
        })
    return series


def input_varied(t):
    phase = t % 6
    if phase < 2:
        return 0.4, 0.3
    if phase < 4:
        return 0.05, 0.05
    return 0.0, 0.0


def input_monotonic(t):
    return 2.0, 0.0


def input_dual_recovery(t):
    if t < 40:
        return 2.0, 0.0
    phase = (t - 40) % 6
    if phase < 2:
        return 0.1, 0.2
    if phase < 4:
        return 0.05, 0.05
    return 0.0, 0.0


def input_partial_recovery(t):
    if t < 40:
        return 2.0, 0.0
    return 1.5, 0.0


def run_all():
    return {
        "A_varied": run("A_varied", input_varied),
        "B_monotonic": run("B_monotonic", input_monotonic),
        "C_dual": run("C_dual", input_dual_recovery),
        "D_partial": run("D_partial", input_partial_recovery),
    }


def avg(series, key, a=60, b=80):
    values = [row[key] for row in series[a:b]]
    return sum(values) / len(values)


def evaluate(results):
    ia = avg(results["A_varied"], "integrity")
    ib = avg(results["B_monotonic"], "integrity")
    xa = avg(results["A_varied"], "xi")
    xb = avg(results["B_monotonic"], "xi")
    ic = avg(results["C_dual"], "integrity")
    id_ = avg(results["D_partial"], "integrity")
    lc = avg(results["C_dual"], "lambda")
    ld = avg(results["D_partial"], "lambda")

    checks = [
        ("T1", ia > 0.5 and ib < 0.05, ia, ib, ia / max(ib, 1e-10)),
        ("T2", xb > xa * 3, xa, xb, xb / max(xa, 1e-10)),
        ("T3", ic > id_ * 3, ic, id_, ic / max(id_, 1e-10)),
        ("T4", lc > ld, lc, ld, None),
    ]
    return checks


def print_report(results, checks):
    print("=" * 70)
    print("PRAMA PROTOKOL - VERIFICACION MINIMA")
    print("=" * 70)

    header = f"\n{'Metrica (avg t=60..80)':<28}{'A variado':>11}{'B monot.':>11}{'C dual':>11}{'D parcial':>11}"
    print(header)
    print("-" * 72)
    for label, key in [
        ("Integrity", "integrity"),
        ("Xi (tension acum.)", "xi"),
        ("Lambda (permisividad)", "lambda"),
        ("Anomaly index", "anomaly"),
        ("Theta_eff (umbral)", "theta_eff"),
    ]:
        values = [f"{avg(results[name], key):11.4f}" for name in SCENARIO_NAMES]
        print(f"{label:<28}{''.join(values)}")
    print("-" * 72)

    for name in SCENARIO_NAMES:
        row = results[name][-1]
        print(f"  {name:14s} health={row['health']:14s}  integrity={row['integrity']:.4f}")

    descriptions = {
        "T1": "input variado preserva integrity; monotono la reduce",
        "T2": "reduccion monotonica acumula mas tension",
        "T3": "recuperacion dual preserva mas integridad",
        "T4": "intervencion dual recupera mas permisividad",
    }

    print("\n" + "-" * 72)
    for name, passed, left, right, ratio in checks:
        status = "PASS" if passed else "FAIL"
        ratio_text = f" | ratio = {ratio:.1f}x" if ratio is not None else ""
        print(f"\n> {name}  {descriptions[name]}")
        print(f"      left = {left:.4f} | right = {right:.4f}{ratio_text}")
        print(f"      {status}")

    passed_count = sum(1 for _, passed, *_ in checks if passed)
    print(f"\n{'=' * 70}")
    print(f"  RESULTADO: {passed_count}/4 tests passed")
    print(f"{'=' * 70}")


def write_results(results, results_dir="results"):
    path = Path(results_dir)
    path.mkdir(exist_ok=True)
    output = path / "results.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return output


def main():
    results = run_all()
    checks = evaluate(results)
    print_report(results, checks)
    output = write_results(results)
    print(f"\n-> {output} exportado")
    return 0 if all(passed for _, passed, *_ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
