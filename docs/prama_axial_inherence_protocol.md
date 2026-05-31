# PRAMA Axial Inherence Monitor Protocol

## Axiom 0 Trigon

- ι: novelty
- κ: consistency
- ρ: retraction / release capacity

The empirical monitor tracks whether a session remains mobile across these three axes or becomes trapped in one dominant axis.

## Axial Inherence

Axial inherence refers to the operational rotation of the Axiom 0 trigon:
ι, κ, and ρ. The monitor measures whether this rotation remains viable or
stagnates before visible functional degradation.

La inherencia axial refiere a la rotación operacional del trígono del Axioma 0:
ι, κ y ρ. El monitor mide si esa rotación se mantiene viable o si se estanca
antes de una degradación funcional visible.

## Rotational Mobility

Rotational mobility measures whether the system continues moving across ι/κ/ρ or becomes trapped in one dominant axis.

The monitor first computes the dominant axis for each turn, then evaluates diversity and transition rate in a local window.

## Axial Fatigue

```text
fatigue(t) = -log(M(t)) / log(100)
```

where M(t) is rotational mobility.

## Precedence Lead

```text
lead = t(function drops) - t(rotation stagnates)
```

Interpretation:

- lead > 0: trigon stagnation precedes functional loss
- lead ~= 0: simultaneous
- lead < 0: function drops before trigon stagnation
- None: no measurable precedence

## Exogenous Judge Constraint

The exogenous judge must not receive or use ι/κ/ρ. Otherwise, the precedence test becomes circular.

Functional loss must be measured through a separate channel from the axis monitor.

## Modes

Self-test:

```bash
python scripts/prama_axial_inherence_runner.py --selftest
```

Retrospective raw.json mode:

```bash
python scripts/prama_axial_inherence_runner.py --from-raw results/session_<id>_raw.json
```

This mode is provisional. It computes a geometry-only proxy and must not be interpreted as true Axiom 0 trigon measurement.

Real adapter scaffold:

```bash
python scripts/prama_axial_inherence_runner.py --protocol docs/prama_axial_inherence_protocol.md --adapter real
```

Real mode requires connected generation, embeddings, ProbLog consistency/retraction, an exogenous judge, and an exogenous interlocutor.

## Outputs

Self-test writes:

- `results/axial_inherence_selftest/axial_inherence_report.md`
- `results/axial_inherence_selftest/axial_inherence_sessions.jsonl`

Retrospective raw.json mode writes:

- `results/axial_inherence_analysis_<session_id>/axial_inherence_report.md`
- `results/axial_inherence_analysis_<session_id>/axial_inherence_turns.csv`

## Methodological Note

The mock self-test validates pipeline wiring only. Empirical evidence begins only when the adapters are connected to real model generation, embeddings, ProbLog consistency/retraction, an exogenous judge, and an exogenous interlocutor.
