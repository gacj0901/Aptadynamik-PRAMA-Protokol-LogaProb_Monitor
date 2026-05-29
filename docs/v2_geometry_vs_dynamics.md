# PRAMA v2: Logprob Geometry vs Dynamics

PRAMA v2 separates four layers that should be read in order.

## 1. Prompt Pressure Layer Ψ

The prompt pressure layer is computed from the input prompt before generation. It estimates the environmental demand imposed on the model independently from the model's answer.

The current extractor reports:

- `psi`: total prompt pressure.
- `load_saturation`: density of simultaneous constraints.
- `load_contradiction`: incompatible demand load.
- `contra_weight`: weight used for contradiction load.

In this architecture, Ψ comes from the prompt and Φ comes from the output logprob geometry. Δ should therefore be interpreted as a decoupling between environmental pressure and generative support: the distance between what the prompt demands and what the generation trajectory can sustain.

The current integration is observational. It records Ψ alongside geometry and PRAMA dynamics, but it does not yet recalibrate or modify the PRAMA core coupling.

## 2. Logprob Geometry Layer

The logprob geometry layer is computed before the PRAMA core runs. It describes direct properties of the model's local token distribution:

- `rigidity`: concentration around a dominant continuation.
- `uncertainty`: entropy-weighted openness in the candidate distribution.
- `margin`: closeness to the configured entropy target.
- `avg_entropy_std`: average intra-window entropy variation.
- `max_entropy_std`: maximum intra-window entropy variation.
- `avg_entropy_range`: average intra-window entropy spread.

These metrics are primary for short LLM trajectories because they are direct measurements of the generation distribution. They do not depend on downstream dynamical integration.

## 3. PRAMA Dynamics Layer

The PRAMA dynamics layer receives geometry-derived window inputs and evolves the PRAMA state. It reports:

- `xi_per_window`
- `final_lambda`
- `final_integrity`
- `final_regime`
- `anomaly`

For short LLM trajectories, `xi_per_window` is a secondary metric. It can preserve or amplify a geometry signal, but it should not be treated as the primary discriminator when only a few windows are available.

## 4. Verification Layer

The verification layer tests whether prompt families separate in the expected way.

Prompt pressure tests verify that the input manipulation separates low-pressure families from contradictory and saturated prompts. Logprob geometry tests compare entropy variation, entropy range, and canonical rigidity across families. Secondary PRAMA dynamics tests compare whether the PRAMA state follows the geometry signal through `xi_per_window` and `lambda`.

## Semantic Stress vs Structural Stress

Semantic stress is represented by fictional prompts: the prompt asks about unsupported or invented entities. The model may still produce a locally smooth continuation, so semantic stress does not always imply high structural variation in the logprob trajectory.

Structural stress is represented by contradictory and saturation prompts. These prompts constrain the generation path itself: they create incompatible or overloaded requirements. In short trajectories, this often appears more directly as increased intra-window entropy variation and entropy range.

## Why Entropy Std and Entropy Range Are Primary

`avg_entropy_std` and `avg_entropy_range` measure how uneven the uncertainty profile is inside each generation window. A structurally stressed prompt can force the model to alternate between locally confident and locally unstable continuation points. That variation is a direct geometric property of the logprob distribution.

`avg_margin` remains useful descriptively, but it is not a primary stress test. Margin measures closeness to a target entropy level. A canonical prompt can have low margin because it is overly rigid, not because it is structurally stressed.

## Scope of the PRAMA Core

This update does not invalidate the PRAMA core. It delimits the regime of application. For short LLM trajectories, direct logprob geometry is the primary discrimination layer, and PRAMA dynamics is a secondary layer that may preserve, smooth, or amplify that signal.

`src/aptadynamik/prama_core.py` remains intact.
