# PRAMA Regime Classification v0.2.2

## 1. Purpose

PRAMA Components v0.2.2 separates threshold crossing from terminal collapse.

In this version, `threshold_crossed` does not automatically mean collapse. It means loss of point-regime viability: the trajectory no longer satisfies the operational viability condition at that point in the run. A trajectory can cross the threshold and later recover, alternate around the threshold, or remain persistently beyond it.

The aptadynamic regime layer therefore classifies not only whether a threshold was crossed, but how the trajectory behaves after the crossing. This distinction matters because a crossing can indicate structural pulsation rather than terminal degradation.

## 2. Formal Background

The operational stress accumulator is represented as:

```text
Ξ(t) = ∫ K(t - τ) Δ(τ) dτ
```

where `K(t - τ)` is the memory kernel and `Δ(τ)` is the instantaneous generative desacople incorporated into the accumulated trajectory signal.

The operational viability condition is:

```text
Ξ(t) ≤ Θ(λ(t))
```

The crossing criterion is:

```text
Ξ(t) > Θ(λ(t))
```

Here, `Θ(λ)` is the dynamic threshold modulated by remaining permissivity `λ`. In PRAMA Components, `viability_margin` is the signed operational margin between the dynamic threshold and normalized accumulated stress. A positive margin indicates that the point-regime condition is still satisfied. A negative margin indicates that the trajectory has crossed the point-regime threshold.

This crossing indicates loss of fixed-point viability. It does not, by itself, imply annihilation of the system, semantic failure, or terminal collapse.

## 3. Four Aptadynamic Regimes

### I_SUBCRITICAL_DISSOLUTION

`I_SUBCRITICAL_DISSOLUTION` indicates insufficient structural flow to sustain organization.

Operationally, this regime is reserved for cases with persistently low structural activity and persistently low raw acople. It is not intended to classify isolated acknowledgements or brief low-activity turns. The criteria are conservative: low average `activity_effective`, low average raw `acople`, and enough valid turns to avoid overreading a transient.

### II_ORGANIZED_EQUILIBRIUM

`II_ORGANIZED_EQUILIBRIUM` indicates no formal threshold crossing.

The trajectory conserves point-regime viability under the tested parameters. In this regime, the system remains within the operational inequality:

```text
Ξ(t) ≤ Θ(λ(t))
```

### III_STRUCTURAL_PULSATION

`III_STRUCTURAL_PULSATION` indicates formal threshold crossing with subsequent recovery or alternation.

The trajectory no longer sustains itself as a point-fixed regime, but it also does not show robust terminal collapse. Instead, it behaves as a bounded oscillatory pattern around the threshold. In this regime, the relevant empirical signature is crossing plus recovery, or crossing without sufficient evidence of persistent terminal drift.

### IV_ENTROPIC_COLLAPSE

`IV_ENTROPIC_COLLAPSE` indicates persistent crossing without observed recovery.

This regime is reserved for trajectories that cross the threshold, remain beyond it at a high post-crossing ratio, and end with negative `viability_margin`. It is an operational indication of terminal drift, not a claim about semantic truth or physical material cost.

## 4. Operational Criteria

The implemented regime layer reports the following fields:

- `regime_label`
- `regime_description`
- `recovery_observed`
- `first_crossing_turn`
- `threshold_crossing_ratio`
- `persistent_crossing_ratio`
- `post_crossing_recovery_turns`
- `trajectory_assessment`

The operational decision logic is:

```text
if no crossing:
    II_ORGANIZED_EQUILIBRIUM
elif crossing and recovery:
    III_STRUCTURAL_PULSATION
elif crossing and persistent ratio high and final margin negative:
    IV_ENTROPIC_COLLAPSE
else:
    III_STRUCTURAL_PULSATION
```

The `I_SUBCRITICAL_DISSOLUTION` branch is evaluated when no formal over-threshold crossing is present, but structural activity and raw acople are persistently low:

```text
avg_activity_effective < 0.05
avg_acople_raw < 0.25
valid_turns >= 3
```

This branch is intentionally conservative so isolated ACKs, short replies, or transient low-activity turns are not overdiagnosed.

## 5. Relation to Routh-Hurwitz and Hopf-Like Interpretation

The mathematical Aptadynamik treatment may use local stability conditions such as `p > 0`, `r > 0`, and `pq < r` to characterize pulsation in a formal dynamical analysis.

The current PRAMA Components runner does not calculate `p`, `q`, or `r`. It does not infer a formal local Jacobian, and it does not assert a mathematical Hopf bifurcation.

Instead, PRAMA Components v0.2.2 implements an operational analogy: loss of point-regime viability with observed recovery or alternation around the threshold is classified as structural pulsation.

PRAMA Components v0.2.2 does not prove a Hopf bifurcation; it detects a trajectory pattern operationally compatible with structural pulsation.

## 6. Parametric Sensitivity Result

The parametric sensitivity file examined was:

```text
results\session_20260602-0303h_gpt-4o-mini\v0.2.1_param\components_parametric_sensitivity.json
```

The grid evaluated:

- `theta0`: `0.35`, `0.5`, `0.75`, `1.0`
- `lambda0`: `1.0`
- `memory_beta`: `0.3`, `0.5`, `0.7`

This produced 12 parameter combinations.

The observed regime counts were:

- `III_STRUCTURAL_PULSATION`: 11
- `IV_ENTROPIC_COLLAPSE`: 1

The robust consensus fields were:

- `robust_regime_label`: `III_STRUCTURAL_PULSATION`
- `robust_trajectory_assessment`: `THRESHOLD_CROSSED_STRUCTURAL_PULSATION`

The first crossing occurred mostly at turn `2`. Under the most permissive tested threshold, `theta0 = 1.0`, the first crossing shifted to turn `4`.

Interpretation: the session is not in organized equilibrium under the tested parameter combinations, because every run includes a formal threshold crossing. However, the session also does not show robust terminal collapse. The robust diagnosis is structural pulsation: the trajectory crosses the point-regime threshold but shows recovery or bounded alternation in most tested configurations.

## 7. Methodological Limitation

This classification is structural-generative. It is not a semantic failure classifier.

PRAMA Components v0.2.2 does not measure:

- semantic truth
- intention
- physical material cost
- GPU load
- energy use
- temperature
- memory pressure
- latency
- cooling or infrastructure telemetry

The `compression_gap` field remains reserved for a future exogenous task channel measuring the gap between support required by the task and support expressed by the response. It is not the same as `viability_margin`.

The regime layer should therefore be read as an operational classification over logprob-derived trajectory geometry: `Ξ(t)`, `Θ(λ)`, `λ`, `viability_margin`, threshold crossing, recovery, persistence, and boundary orientation.

## 8. Version Notes

v0.2.1 corrected the ACK/activity artifact. In that version, raw activity is separated from structural activity so that acknowledgements or low-activity turns do not inject artificial accumulated stress.

v0.2.2 adds aptadynamic regime classification and parametric sensitivity. It distinguishes loss of point-regime viability from terminal collapse and reports whether the observed trajectory is better described as organized equilibrium, subcritical dissolution, structural pulsation, or entropic collapse.
