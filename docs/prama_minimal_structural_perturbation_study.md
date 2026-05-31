# PRAMA Minimal-Structural Perturbation Study

This scaffold defines a falsifiable study comparing concrete-content, abstract-content, minimal-structural, and neutral control perturbations.

## Study Question

The study asks whether minimal structural perturbations produce more real friction than perturbations that add absorbable narrative content.

## Independent Variable

- control_neutral
- concrete_content
- abstract_content
- minimal_structural

## Dependent Variables

- C_commitment_shift
- R_recombination
- S_surprise
- E_elaboration

## Initial Operational Definitions

Absorption:

```text
delta_C approximately 0
AND R high
AND S low
AND E increased
```

Real friction:

```text
delta_C nonzero
AND R decreases
AND S increases
```

## Hypotheses

H_A:
Greater absorbable narrative content in the perturbation produces less real friction.

H_B:
Minimal-structural perturbation produces more real friction than abstract-content perturbation.

H0:
Perturbation type explains no meaningful variance in C, R, or S after controls.

## Falsification Criteria

- reject the friction thesis if perturbation type explains approximately zero variance in C/R/S
- reject if abstract-content perturbations equal or exceed minimal-structural perturbations in friction
- reject if no perturbation exceeds the control arm drift

## Current Scaffold

The current implementation provides:

- a JSON-compatible YAML protocol
- a perturbation taxonomy
- dependency-free proxy metrics
- a dry-run enumerator
- a from-results metric extractor for available PRAMA Monitor raw files

## Methodological Note

This scaffold does not yet replace blinded human judges, preregistration, or the future ProbLog commitment tracker.
