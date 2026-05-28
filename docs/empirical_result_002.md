# Empirical Result 002 - PRAMA v2 prompt-family pipeline

Date: 2026-05-27  
Pipeline: PRAMA v2 prompt-family pipeline  
Model: gpt-4o-mini  
Top logprobs: 5  
Prompt limit: 1 per family  
Max tokens: 96  

## Families

1. Canonical clean: simple factual prompts.
2. Fictional / unsupported: semantic stress.
3. Contradictory constraint: aptadynamic stress by incompatible constraints.
4. Saturation / overconstraint: aptadynamic stress by simultaneous constraints.

## Raw summary

| Metric | Canonical | Fictional | Contradictory | Saturation |
|---|---:|---:|---:|---:|
| Rigidity | 0.8689 | 0.6807 | 0.6150 | 0.3060 |
| Uncertainty | 0.0089 | 0.0376 | 0.0455 | 0.2218 |
| Margin | 0.6134 | 0.7073 | 0.8118 | 0.9111 |
| Xi / window | 0.1405 | 0.0891 | 0.1898 | 0.1047 |
| Integrity | 0.9185 | 0.8711 | 0.6963 | 0.9692 |
| Lambda | 0.3140 | 0.7467 | 0.2050 | 0.8758 |
| Raw entropy | 0.2634 | 0.4814 | 0.7240 | 1.0656 |

## Hypothesis tests

T1: xi/window(contradictory) > xi/window(canonical): PASS  
T2: xi/window(saturation) > xi/window(canonical): FAIL  
T3: max(xi/window contradictory, saturation) > xi/window(fictional): PASS  
T4: margin(contradictory/saturation) < margin(canonical): FAIL  

Result: 2/4 tests.

## Interpretation

The v2 pipeline partially supports the revised aptadynamic hypothesis.

Contradictory prompts produced the strongest aptadynamic signal: highest xi per window, lowest integrity, and lowest lambda. This suggests that operational contradiction is more structurally stressful than fictional or unsupported content.

Fictional prompts did not behave as strongly stressful. This supports the distinction between semantic stress and aptadynamic stress.

Saturation did not behave as predicted in this minimal run. The model produced a short response, only 31 tokens, with high uncertainty and high margin. PRAMA therefore read the trajectory as flexible rather than collapsed. This suggests that overconstraint can be discharged by brevity or evasion unless compliance pressure is explicitly measured.

## Working conclusion

PRAMA v2 should not treat all difficult prompts as equivalent.

Contradiction appears to reduce operational viability more directly than semantic fiction. Saturation requires an additional compliance or constraint-load metric, because the model can escape saturation by shortening or simplifying its response.

## Next step

Run PRAMA v2 with PRAMA_PROMPT_LIMIT=3 or 5.

Add derived metrics:

- xi_per_token
- output_length_ratio
- constraint_escape_flag
- constraint_load_score

Do not modify the PRAMA core yet. The next correction belongs to the experimental mapping layer.
