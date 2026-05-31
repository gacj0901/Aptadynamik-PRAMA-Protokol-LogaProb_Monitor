from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

from aptadynamik.observer.axial_inherence_adapters import MockAxialInherenceAdapters
from aptadynamik.observer.axial_inherence_monitor import (
    bootstrap_ci,
    dominant_axis,
    fatigue_series,
    run_axial_inherence_session,
)


TURN_FIELDS = [
    "session_id",
    "task_id",
    "condition",
    "turn",
    "i",
    "k",
    "r",
    "f",
    "dominant",
    "fatigue",
    "mean_logprob",
    "n_tokens",
    "finish_reason",
]


def _round(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 6)
    return value


def write_jsonl(path: Path, sessions: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for session in sessions:
            handle.write(json.dumps(session) + "\n")


def write_turns_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TURN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in TURN_FIELDS})


def axial_inherence_report_markdown(
    *,
    mode: str,
    sessions: Sequence[Dict[str, Any]],
    provisional: bool = False,
) -> str:
    final_fatigues = [session["fatigue"][-1] for session in sessions if session.get("fatigue")]
    leads = [session["precedence_lead"] for session in sessions if session.get("precedence_lead") is not None]
    ci = bootstrap_ci(leads)
    lines = [
        "# PRAMA Axial Inherence Monitor Report",
        "",
        "## Axiom 0 Trigon",
        "",
        "- iota: novelty",
        "- kappa: consistency",
        "- rho: retraction / release capacity",
        "",
        "## Axial Inherence",
        "",
        "Axial inherence refers to the operational rotation of the Axiom 0 trigon:",
        "ι, κ, and ρ. The monitor measures whether this rotation remains viable or",
        "stagnates before visible functional degradation.",
        "",
        "La inherencia axial refiere a la rotación operacional del trígono del Axioma 0:",
        "ι, κ y ρ. El monitor mide si esa rotación se mantiene viable o si se estanca",
        "antes de una degradación funcional visible.",
        "",
        "## Rotational Mobility",
        "",
        "Rotational mobility measures whether the system continues moving across iota/kappa/rho or becomes trapped in one dominant axis.",
        "",
        "## Axial Fatigue",
        "",
        "fatigue(t) = -log(M(t)) / log(100)",
        "",
        "where M(t) is rotational mobility.",
        "",
        "## Precedence Lead",
        "",
        "lead = t(function drops) - t(rotation stagnates)",
        "",
        "- lead > 0: trigon stagnation precedes functional loss",
        "- lead ~= 0: simultaneous",
        "- lead < 0: function drops before trigon stagnation",
        "- None: no measurable precedence",
        "",
        "## Exogenous Judge Constraint",
        "",
        "The exogenous judge must not receive or use iota/kappa/rho. Otherwise, the precedence test becomes circular.",
        "",
        "## Results",
        "",
        f"- mode: {mode}",
        f"- number of sessions: {len(sessions)}",
        f"- fatigue mean: {_round(mean(final_fatigues)) if final_fatigues else None}",
        f"- fatigue max: {_round(max(final_fatigues)) if final_fatigues else None}",
        f"- precedence_lead mean: {_round(ci['mean'])}",
        f"- bootstrap confidence interval: [{_round(ci['lo'])}, {_round(ci['hi'])}]",
        f"- number of valid leads: {ci['n']}",
    ]
    if provisional:
        lines.extend(
            [
                "",
                "Retrospective raw.json mode is provisional. It uses geometry-only proxies and must not be interpreted as true Axiom 0 trigon measurement.",
            ]
        )
    lines.extend(
        [
            "",
            "## Methodological Note",
            "",
            "The mock self-test validates pipeline wiring only. Empirical evidence begins only when the adapters are connected to real model generation, embeddings, ProbLog consistency/retraction, an exogenous judge, and an exogenous interlocutor.",
            "",
        ]
    )
    return "\n".join(lines)


def run_selftest(output_dir: Path = Path("results/axial_inherence_selftest")) -> Dict[str, Any]:
    task = {"task_id": "axial_inherence_mock", "prompt": "Maintain an explanatory session."}
    normal = run_axial_inherence_session(MockAxialInherenceAdapters(), task, condition="rotating", max_turns=36)
    echo = run_axial_inherence_session(MockAxialInherenceAdapters(), task, condition="stagnated_echo_decline", max_turns=36)
    injected_sessions = [
        run_axial_inherence_session(
            MockAxialInherenceAdapters(decline_delay=24),
            task,
            condition="stagnated_echo_decline",
            max_turns=36,
        )
        for _ in range(8)
    ]
    sessions = [normal, echo, *injected_sessions]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "axial_inherence_sessions.jsonl", sessions)
    (output_dir / "axial_inherence_report.md").write_text(
        axial_inherence_report_markdown(mode="mock self-test", sessions=sessions),
        encoding="utf-8",
    )
    ci = bootstrap_ci([session["precedence_lead"] for session in injected_sessions if session["precedence_lead"] is not None])
    print(f"final fatigue normal rotating condition: {normal['fatigue'][-1]:.6f}")
    print(f"final fatigue stagnated echo condition: {echo['fatigue'][-1]:.6f}")
    print(f"bootstrap CI for injected precedence: [{ci['lo']}, {ci['hi']}], mean={ci['mean']}, n={ci['n']}")
    return {"normal": normal, "echo": echo, "injected_ci": ci, "sessions": sessions}


def _raw_turn_proxy(raw: Dict[str, Any], turn: Dict[str, Any], idx: int) -> Dict[str, Any]:
    summary = turn.get("summary") if isinstance(turn.get("summary"), dict) else {}
    entropy = float(summary.get("avg_entropy_norm", 0.0))
    rigidity = float(summary.get("avg_rigidity", 0.0))
    uncertainty = float(summary.get("avg_uncertainty", 0.0))
    i = max(0.0, min(1.0, entropy))
    k = max(0.0, min(1.0, rigidity))
    r = max(0.0, min(1.0, uncertainty))
    finish_reason = turn.get("finish_reason")
    f = 0.65 if finish_reason == "length" else 1.0
    mean_logprob = turn.get("mean_logprob")
    return {
        "session_id": raw.get("session_id", "unknown"),
        "task_id": raw.get("session_id", "raw_session"),
        "condition": "provisional_geometry_proxy",
        "turn": int(turn.get("turn_index", idx)),
        "i": i,
        "k": k,
        "r": r,
        "f": f,
        "dominant": dominant_axis((i, k, r)),
        "mean_logprob": mean_logprob,
        "n_tokens": int(turn.get("token_count", 0)),
        "finish_reason": finish_reason,
    }


def run_from_raw(raw_path: Path, output_dir: Path | None = None) -> List[Dict[str, Any]]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    session_id = raw.get("session_id", raw_path.stem)
    target = output_dir or Path("results") / f"axial_inherence_analysis_{session_id}"
    turns = raw.get("turns", [])
    if not isinstance(turns, list):
        raise ValueError("raw.json missing list field 'turns'")
    rows = [_raw_turn_proxy(raw, turn, idx) for idx, turn in enumerate(turns)]
    ikr = [(row["i"], row["k"], row["r"]) for row in rows]
    fatigue = fatigue_series(ikr)
    for row, fatigue_value in zip(rows, fatigue):
        row["fatigue"] = fatigue_value
    session = {
        "task_id": session_id,
        "condition": "provisional_geometry_proxy",
        "turns": rows,
        "fatigue": fatigue,
        "precedence_lead": None,
    }
    write_turns_csv(target / "axial_inherence_turns.csv", rows)
    (target / "axial_inherence_report.md").write_text(
        axial_inherence_report_markdown(mode="retrospective raw.json geometry proxy", sessions=[session], provisional=True),
        encoding="utf-8",
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PRAMA Axial Inherence Monitor analyses.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selftest", action="store_true", help="Run mock adapter self-test.")
    group.add_argument("--from-raw", help="Run provisional retrospective raw.json analysis.")
    group.add_argument("--protocol", help="Protocol path for real adapter mode scaffold.")
    parser.add_argument("--adapter", default="mock", help="Adapter name. Use --adapter real to request real mode.")
    parser.add_argument("--output-dir", help="Optional output directory.")
    args = parser.parse_args()

    if args.selftest:
        run_selftest(Path(args.output_dir) if args.output_dir else Path("results/axial_inherence_selftest"))
        return 0

    if args.from_raw:
        rows = run_from_raw(Path(args.from_raw), Path(args.output_dir) if args.output_dir else None)
        print(f"Wrote {len(rows)} provisional axis rows")
        return 0

    if args.adapter == "real":
        raise SystemExit(
            "Real AxialInherenceAdapters are not implemented. Connect generate_turn, embed, kb_consistency, kb_retraction, exo_judge, and interlocutor_turn."
        )
    raise SystemExit("--protocol currently requires --adapter real, or use --selftest / --from-raw.")


if __name__ == "__main__":
    raise SystemExit(main())
