# PRAMA Phase Signature Protocol

This observer module adds qualitative phase-transition signatures to PRAMA Monitor output. It operates over recorded assistant-output geometry and does not modify the PRAMA core.

## Viability Definition

The initial viability proxy is:

```text
viability = avg_rigidity - avg_uncertainty
```

The value is computed per turn from the compact turn summary stored in PRAMA Monitor `raw.json` files.

## Qualitative Observables

### Discontinuity

Discontinuity measures sharp adjacent drops or jumps in viability or rigidity. In real sessions it identifies the strongest transition turn, but it should be treated as a descriptive marker until paired with a controlled protocol.

### Hysteresis

Hysteresis compares separated up/down sweeps. It is meaningful when a diagnostic directory contains matched files such as `peripheral_up_raw.json` and `peripheral_down_raw.json`.

### Critical Slowing

Critical slowing is estimated with rising variance, rising lag-1 autocorrelation, or delayed recovery after a large viability drop.

### Structural Target Effect

Structural target effect compares peripheral and constitutive prompt trajectories. A positive effect means the constitutive trajectory transitions earlier, or at lower pressure, than the peripheral trajectory.

## Synthetic Validation

Run:

```bash
python scripts/prama_phase_signature_runner.py --synthetic
```

This writes:

```text
results/phase_synthetic_validation/phase_report.md
results/phase_synthetic_validation/phase_signatures.csv
```

Synthetic validation is instrument validation, not model evidence. It confirms that the detector responds to a known fold-like system and remains quiet for a smooth system.

## Single Raw Session Analysis

Run:

```bash
python scripts/prama_phase_signature_runner.py --from-raw results/session_<id>_raw.json
```

This extracts turns, computes viability per turn, detects local drops, estimates critical slowing, and writes:

```text
results/phase_analysis_<session_id>/phase_signatures.csv
results/phase_analysis_<session_id>/phase_report.md
```

Real raw.json analysis is empirical observation. It detects qualitative shifts in assistant-output geometry.

## Diagnostic Directory Analysis

Run:

```bash
python scripts/prama_phase_signature_runner.py --from-results results/diagnostic_001/
```

Expected optional structure:

```text
results/diagnostic_001/
  gpt-4o-mini/
    peripheral_up_raw.json
    peripheral_down_raw.json
    constitutive_up_raw.json
    constitutive_down_raw.json
```

For each model, the runner writes `phase_signatures.csv` and `phase_report.md`. At the diagnostic root it writes `comparative_phase_summary.csv` and `comparative_phase_report.md`.

## Methodological Note

This module measures qualitative transition signatures over PRAMA Monitor output geometry. It does not replace the future predictive-surprise module based on incoming user signs.
