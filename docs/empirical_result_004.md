# Empirical Result 004 - Intra-window entropy variance

Date: 2026-05-28  
Pipeline: PRAMA v2 prompt-family pipeline  
Model: gpt-4o-mini  
Top logprobs: 5  
Prompt limit: 3 per family  
Total prompts: 12  
Max tokens: 128  

## Raw summary

| Metric | Canonical | Fictional | Contradictory | Saturation |
|---|---:|---:|---:|---:|
| Rigidity | 0.8633 | 0.6471 | 0.7238 | 0.3625 |
| Uncertainty | 0.0139 | 0.0524 | 0.0471 | 0.2143 |
| Margin | 0.6140 | 0.7475 | 0.7082 | 0.8769 |
| Xi / window | 0.1301 | 0.1041 | 0.1309 | 0.1514 |
| Integrity | 0.8651 | 0.9578 | 0.9373 | 0.9159 |
| Lambda | 0.2750 | 0.5510 | 0.2701 | 0.7154 |
| Raw entropy | 0.2646 | 0.5745 | 0.4960 | 1.0781 |
| Entropy std | 0.3252 | 0.5758 | 0.4879 | 0.6651 |
| Max entropy std | 0.6719 | 0.7325 | 0.7527 | 0.8062 |
| Entropy range | 0.9043 | 1.6479 | 1.3612 | 1.8345 |

## Hypothesis tests

T1: xi/window(contradictory) > xi/window(canonical): PASS  
T2: xi/window(saturation) > xi/window(canonical): PASS  
T3: max(xi/window contradictory, saturation) > xi/window(fictional): PASS  
T4: margin(contradictory/saturation) < margin(canonical): FAIL  

Result: 3/4 tests.

## Interpretation

Adding intra-window entropy variance substantially improved the experimental signal.

The previous aggregation layer used average entropy per window. That erased local instability inside each window. After adding entropy standard deviation, maximum entropy standard deviation, and entropy range, saturation shows the strongest internal variability.

This supports the hypothesis that aptadynamic stress is not captured only by mean entropy or final accumulated xi. It appears as local instability: entropy jumps between neighboring tokens inside the same generative window.

Canonical prompts remain highly rigid and low-variance. Fictional prompts increase uncertainty and entropy variance, but saturation produces the strongest intra-window entropy instability.

## Corrected interpretation of margin

The failed margin test does not necessarily falsify the signal.

The current margin definition is:

margin = 1 - |entropy_norm - ENTROPY_TARGET|

Therefore, high margin means proximity to the target entropy zone, not absence of stress. Saturation can have high average margin while also showing high intra-window entropy variance. In that case, the trajectory remains near the viable entropy band but oscillates internally.

## Revised hypothesis

Aptadynamic stress is expressed less as mean entropy and more as entropy volatility within local generative windows.

Stable generation:
- low entropy variance
- low entropy range
- smooth delta trajectory

Internally stressed generation:
- high entropy variance
- high entropy range
- local jumps in delta and xi

Saturation is therefore better detected by intra-window entropy volatility than by margin loss alone.

## Next step

Do not modify the PRAMA core.

Update the hypothesis tests in v2:

- entropy_std(saturation) > entropy_std(canonical)
- max_entropy_std(contradictory or saturation) > max_entropy_std(canonical)
- entropy_range(saturation) > entropy_range(canonical)
- xi/window(aptadynamic families) > xi/window(fictional)

The margin test should be removed or reinterpreted.
