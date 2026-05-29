\# Onset Experiment Protocol — Dynamic Ψ intervention in long LLM trajectories



\## Objective



This experiment tests whether PRAMA tension Ξ responds dynamically to an internal onset of environmental pressure Ψ inside a long generation trajectory.



Previous experiments compared different prompt families against each other. That design showed strong separation in Prompt Pressure and Logprob Geometry, but the relation between Ψ and accumulated PRAMA tension Ξ was partially confounded by response length.



This experiment changes the unit of analysis.



Instead of comparing short prompts, it introduces a known pressure onset inside a long trajectory and measures whether the slope of reconstructed Ξ changes after the onset.



\## Core hypothesis



Ψ does not map homogeneously to Ξ across all regimes.



Different forms of pressure should produce different dynamic signatures:



```text

Contradictory onset:

&#x20; sustained incompatible demand

&#x20; → increased post-onset growth rate of Ξ



Saturation onset:

&#x20; excessive simultaneous constraints

&#x20; → compression, truncation, or collapse of the trajectory

&#x20; → reduced post-onset survival rather than sustained Ξ growth


Experimental unit

One item is a long generation trajectory.

Each item has a neutral topic and is generated under three conditions:

1. control
2. contradictory_onset
3. saturation_onset
Trajectory structure

Each trajectory targets approximately 800 tokens.

The onset point is placed at the midpoint:

pre-onset:  approximately 400 tokens
post-onset: approximately 400 tokens

Because API generation cannot inject a new instruction into an already running call, the operational implementation uses two-stage continuation:

Stage 1:
  generate neutral prefix

Stage 2:
  continue from the prefix under one of three conditions:
    control continuation
    contradictory onset continuation
    saturation onset continuation

The analytical trajectory is the concatenation of Stage 1 and Stage 2.

Pressure function Ψ(t)

Ψ(t) is defined as a known step function.

For control:

Ψ(t) = low for the whole trajectory

For contradictory onset:

Ψ(t) = low before onset
Ψ(t) = high_contradictory after onset

For saturation onset:

Ψ(t) = low before onset
Ψ(t) = high_saturation after onset
Φ_A(t)

Φ_A(t) is measured per window from output logprob geometry.

Primary proxy:

Φ_A(t) = normalized entropy per window

If only top-logprobs are available, entropy must be described as truncated or estimated entropy.

Δ(t)
Δ(t) = |Φ_A(t) - Ψ(t)|
Ξ(t)

Ξ(t) is reconstructed using a heavy-tail memory kernel:

Ξ(t) = Σ K(t-τ) Δ(τ)
K(s) = (s + s0)^(-β)

The experiment should test multiple preregistered β values:

β ∈ {0.3, 0.6, 0.9}
Primary test

Difference-in-differences over Ξ slope:

DiD =
[slope_post(Ξ)_treatment - slope_pre(Ξ)_treatment]
-
[slope_post(Ξ)_control - slope_pre(Ξ)_control]

Primary contradictory prediction:

DiD_contradictory > 0
Saturation prediction

The saturation condition is not expected to behave like contradiction.

Primary saturation endpoints:

1. post-onset token survival
2. probability of early termination
3. post-onset length reduction relative to control
4. initial Δ increase followed by truncation/compression

Primary saturation prediction:

survival_saturation < survival_control

Secondary saturation prediction:

slope_post(Ξ)_saturation is not necessarily greater than control
Sample size

Minimum:

30 neutral topics × 3 conditions = 90 trajectories

Preferred:

50 neutral topics × 3 conditions = 150 trajectories
Main falsification criteria

The contradictory onset prediction fails if:

DiD_contradictory <= 0

across the preregistered sample.

The saturation prediction fails if:

survival_saturation is not lower than survival_control

and saturation behaves like sustained Ξ growth rather than compression/truncation.

Interpretation

If both predictions hold:

contradiction = accumulated internal dynamic tension
saturation = margin exhaustion / response compression / trajectory truncation

This would support a two-regime interpretation of prompt pressure in PRAMA:

Ψ does not produce one homogeneous Ξ response.
The form of pressure determines the dynamic regime.



