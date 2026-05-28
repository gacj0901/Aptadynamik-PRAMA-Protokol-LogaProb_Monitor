# Empirical Result 001 - OpenAI logprobs -> PRAMA

Date: 2026-05-27  
Pipeline: OpenAI Chat Completions logprobs  
Model: gpt-4o-mini  
Top logprobs: 5  
Prompt limit: 2 clean + 2 stress  
Max tokens: 128  

## Raw summary

| Metric | Clean | Stress | Separation |
|---|---:|---:|---:|
| Integrity | 0.7838 | 0.8138 | 0.0300 |
| Xi tension | 3.5124 | 1.0875 | 2.4249 |
| Lambda permissivity | -0.2752 | 0.5889 | 0.8641 |
| Mean entropy | 0.2554 | 0.6371 | 0.3817 |

Combined separation: 2.4550

Result: PRAMA discriminates these trajectories.

## Technical interpretation

The motor discriminates. A combined separation of 2.4550 is not noise.

However, the polarity is inverted relative to the initial prediction. Clean prompts showed higher accumulated tension and lower lambda, while stress prompts showed lower accumulated tension and higher lambda.

The likely technical cause is that the current signal mapping treats high-confidence generation as structurally rigid. Clean prompts produce higher confidence, lower entropy, larger gaps, and longer responses. This produces more windows and more accumulated xi. Stress prompts, by contrast, often produce shorter, more uncertain, more cautious generations.

Thus the current pipeline is not detecting semantic truth or falsity. It is detecting probabilistic geometry: concentration, rigidity, entropy, accumulated tension, and loss of permissivity.

## Conceptual interpretation

The result does not invalidate PRAMA. It clarifies what PRAMA is measuring.

A semantically "stressful" prompt is not necessarily aptadynamically stressful. A fictional or unsupported prompt can increase uncertainty, but it may also preserve internal variation and avoid rigid collapse.

Aptadynamic stress should be defined structurally, not semantically:

- Semantic stress: the prompt asks for something false, fictional, or unsupported.
- Aptadynamic stress: the prompt reduces the model's operational degrees of freedom until it produces rigidity, contradiction, saturation, or loss of alternatives.

From this perspective, clean prompts may generate with excessive confidence and low variation. That can appear as rigidity rather than health. Stress prompts may preserve more internal variation because the model has to hedge, generalize, or navigate uncertainty.

## Working hypothesis

PRAMA does not measure hallucination directly.

PRAMA measures structural viability of generative trajectories: margin, concentration, bifurcation, accumulated tension, and loss of permissivity.

Viability is probably not maximal confidence and not maximal uncertainty. It is a dynamic zone between coherent form and adaptive variation.

## Next experimental step

The next prompt families should distinguish semantic stress from aptadynamic stress:

1. Canonical clean prompts.
2. Fictional or unsupported prompts.
3. Contradictory-constraint prompts.
4. Saturation or overconstraint prompts.

The expected aptadynamic stress should appear more strongly in groups 3 and 4 than in merely fictional prompts.

## Calibration note

Do not immediately invert the mapping.

First add derived metrics:

- xi_per_window
- rigidity_index
- uncertainty_index
- margin_index

Then compare regimes across prompt families before changing the PRAMA core.
