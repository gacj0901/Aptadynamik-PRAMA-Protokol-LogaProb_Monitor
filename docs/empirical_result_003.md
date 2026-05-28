# Empirical Result 003 - PRAMA v2 larger prompt-family sample

Date: 2026-05-27  
Pipeline: PRAMA v2 prompt-family pipeline  
Model: gpt-4o-mini  
Top logprobs: 5  
Prompt limit: 3 per family  
Total prompts: 12  
Max tokens: 128  

## Raw summary

| Metric | Canonical | Fictional | Contradictory | Saturation |
|---|---:|---:|---:|---:|
| Rigidity | 0.8619 | 0.6044 | 0.7601 | 0.2982 |
| Uncertainty | 0.0120 | 0.0821 | 0.0420 | 0.2810 |
| Margin | 0.6179 | 0.7700 | 0.6781 | 0.8655 |
| Xi / window | 0.1319 | 0.1342 | 0.1125 | 0.1221 |
| Integrity | 0.8920 | 0.8583 | 0.9013 | 0.9392 |
| Lambda | 0.2759 | 0.5238 | 0.5125 | 0.8019 |
| Raw entropy | 0.2737 | 0.6446 | 0.4354 | 1.1552 |

## Hypothesis tests

T1: xi/window(contradictory) > xi/window(canonical): FAIL  
T2: xi/window(saturation) > xi/window(canonical): FAIL  
T3: max(xi/window contradictory, saturation) > xi/window(fictional): FAIL  
T4: margin(contradictory/saturation) < margin(canonical): FAIL  

Result: 0/4 tests.

## Interpretation

The larger v2 sample does not support the current operational hypothesis that contradictory and saturation prompts automatically produce higher aptadynamic stress.

The strongest signal is not contradiction or saturation, but rigidity in canonical prompts. Canonical prompts generated with high confidence, low uncertainty, and high rigidity.

Fictional prompts produced more uncertainty without necessarily increasing structural collapse. This supports the distinction between semantic stress and aptadynamic stress.

Saturation prompts produced high uncertainty, high margin, high lambda, and relatively short outputs. This suggests that the model may discharge saturation through brevity, partial compliance, generalization, or evasion. In that case, logprob geometry alone reads the trajectory as flexible rather than collapsed.

## Revised hypothesis

Aptadynamic stress is not determined only by the semantic class of the prompt.

A prompt becomes aptadynamically stressful only when the model must sustain incompatible or excessive constraints without escaping through brevity, refusal, simplification, or noncompliance.

Therefore, saturation cannot be evaluated only with logprobs. It requires a compliance layer.

## Required next metrics

- xi_per_token
- output_length_ratio
- constraint_load_score
- constraint_compliance_score
- escape_index
- rigidity_pressure

## Working conclusion

Do not modify the PRAMA core yet.

The next correction belongs to the experimental mapping layer. PRAMA is detecting probabilistic geometry, but the experiment needs to distinguish sustained constraint load from constraint escape.
