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

- `II_ORGANIZED_EQUILIBRIUM`
- `VIABLE_ORGANIZED_EQUILIBRIUM`

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

Regime II requires sufficient history and no formal crossing.

Regime III requires crossing plus recovery or bounded alternation after crossing.

Regime IV requires persistent crossing without recovery and negative final viability margin. It is not assigned from a short local cascade.

## Limitations

The benchmark is synthetic instrument validation. It does not prove that a live model has entered any empirical regime. Live evidence requires recorded session trajectories, enough valid windows, and later comparison against external functional channels where appropriate.

The benchmark measures generative structure from logprobs. It does not measure semantic truth, intention, material cost, GPU load, energy, temperature, memory pressure, or latency.
