# PRAMA Regime Benchmark

## Purpose

This benchmark generates deterministic synthetic trajectories for PRAMA Components. It provides controlled evidence for the aptadynamic regime classifier without depending on live model output, API availability, prompt behavior, or sampling variance.

The benchmark uses `src/aptadynamik/prama_components.py::measure` directly. It does not replace live monitoring; it validates that the regime layer behaves consistently under known trajectory patterns.

## Local Threshold Crossing vs Trajectory Regime

A threshold crossing is a local viability event. It means a window or turn exceeded the current dynamic viability threshold.

A regime classification is a trajectory diagnosis. It requires enough valid history to distinguish an isolated local event from persistent drift, bounded recovery, or organized equilibrium.

This distinction is essential because a short response can show local crossings without providing enough evidence for terminal collapse.

## Why CALIBRATING Prevents False Positives

`CALIBRATING / INSUFFICIENT_HISTORY` is emitted when the trajectory is too short for regime diagnosis. Local threshold signals may still be recorded, but the classifier does not promote them to `IV_ENTROPIC_COLLAPSE`.

This protects PRAMA Monitor from treating a short token-window cascade inside a single answer as evidence of terminal regime collapse.

## Benchmark Scenarios

### short_calibrating_local_crossings

Few token windows, with possible local threshold crossings.

Expected result:

- `CALIBRATING`
- `INSUFFICIENT_HISTORY`

### organized_viability

Sufficient trajectory history with no formal threshold crossing.

Expected result:

- `II_ORGANIZED_STABILITY`
- `VIABLE_ORGANIZED_STABILITY`

The earlier label `II_ORGANIZED_EQUILIBRIUM` is retained only as a historical alias. It was replaced because Aptadynamia treats stability as organized dynamic viability, not static equilibrium.

### structural_pulsation

Sufficient trajectory history with threshold crossing followed by recovery.

Expected result:

- `III_STRUCTURAL_PULSATION`
- `THRESHOLD_CROSSED_STRUCTURAL_PULSATION`

### entropic_collapse

Sufficient trajectory history with persistent threshold crossing and no recovery.

Expected result:

- `IV_ENTROPIC_COLLAPSE`
- `ENTROPIC_COLLAPSE`

## Criteria

Regime II requires sufficient history and no formal crossing. It is named `II_ORGANIZED_STABILITY` because organized stability means dynamic, structured viability under flow rather than immobile equilibrium.

Regime III requires crossing plus recovery or bounded alternation after crossing.

Regime IV requires persistent crossing without recovery and negative final viability margin. It is not assigned from a short local cascade.

## Reproducibility Metadata

Each benchmark run writes a root-level `manifest.json` next to the aggregate `summary.json`.

The manifest records:

- generation timestamp
- Git commit SHA and branch when available
- Python version
- platform string
- benchmark version
- output directory
- scenario count
- per-scenario expected and observed labels
- artifact paths for `raw.json`, `report.md`, and `summary.json`
- parameter snapshot
- stable SHA256 hash of each normalized scenario `summary.json`

The per-scenario `result_hash` is computed from:

```python
json.dumps(summary, sort_keys=True, ensure_ascii=False)
```

This makes the benchmark suitable for comparing deterministic instrument behavior across commits, machines, and later protocol revisions.

## Aggregate Benchmark Report

Each run also writes `aggregate_report.md` in the benchmark output directory.

The aggregate report contains:

- run metadata from `manifest.json`
- a compact table of all scenarios
- expected and observed regime labels
- expected and observed trajectory assessments
- threshold and persistence ratios
- recovery flags
- pass/fail status
- result hashes
- one technical interpretation section per scenario
- the required methodological note on local threshold crossing versus trajectory regime classification

This file is intended as the human-readable companion to `manifest.json`. The manifest is the machine-readable reproducibility record; the aggregate report is the review artifact for protocol notes, audit trails, and technical discussion.

## Exporting Evidence Bundles

Use `scripts/export_regime_benchmark_evidence.py` to generate an external evidence bundle without committing generated artifacts to the source repository.

PowerShell example:

```powershell
python scripts\export_regime_benchmark_evidence.py --evidence-dir "C:\Users\THINKPAD\Desktop\Documentación PRAMA Protokol ProbLogs Mónitor" --run-label "initial-controlled-regimes"
```

The exporter runs the deterministic benchmark, then writes an external folder containing:

- `manifest.json`
- `aggregate_report.md`
- `scenario_index.md`
- `README.md`
- `scenarios/<scenario_name>/summary.json`
- `scenarios/<scenario_name>/report.md`
- `scenarios/<scenario_name>/raw.json` when `--include-raw true`

Generated evidence artifacts are intentionally kept outside the source repository.

## Legacy Regime Alias

The benchmark now emits:

- `II_ORGANIZED_STABILITY`
- `VIABLE_ORGANIZED_STABILITY`

The older names remain accepted by verification tools as legacy aliases:

- `II_ORGANIZED_EQUILIBRIUM` -> `II_ORGANIZED_STABILITY`
- `VIABLE_ORGANIZED_EQUILIBRIUM` -> `VIABLE_ORGANIZED_STABILITY`

This preserves validation of earlier evidence bundles while keeping new outputs conceptually aligned with Aptadynamia: stability is organized and dynamic, not static equilibrium.

## Limitations

The benchmark is synthetic instrument validation. It does not prove that a live model has entered any empirical regime. Live evidence requires recorded session trajectories, enough valid windows, and later comparison against external functional channels where appropriate.

The benchmark measures generative structure from logprobs. It does not measure semantic truth, intention, material cost, GPU load, energy, temperature, memory pressure, or latency.
